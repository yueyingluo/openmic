"""Independent jyb demonstration; the existing OpenMic page stays unchanged."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from openmic.autogen_v02 import AutoGenConfig
from openmic.models import ProjectRequest
from openmic.tts import SYSTEM_VOICES, extract_spoken_script
from openmic.workflow import OpenMicWorkflow
from openmic.tts_backends import SiliconFlowBackend, ToneBackend
from openmic.tts_plan import compile_script
from openmic.tts_render import render, save_result
from openmic.tts_workflow import JybWorkflow


ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "baseline": "普通朗读",
    "timing": "只加停顿",
    "emotion": "只加情绪",
    "performance": "完整表演",
}
FALLBACK = "<SCRIPT>\n[SETUP]我原以为网购是在省钱。\n[PUNCHLINE]后来发现，省下的是我出门的力气。\n</SCRIPT>"


def _sample() -> str:
    path = ROOT / "examples" / "jyb_demo.txt"
    return path.read_text(encoding="utf-8-sig") if path.is_file() else FALLBACK


def _workflow_script(result, backend: str) -> str:
    # The old mock coach emits an explanation rather than a marked script.
    role = "JokeWriter" if backend == "mock" else "PerformanceCoach"
    messages = [message.content for message in result.messages if message.agent == role]
    if not messages:
        raise ValueError(f"没有找到 {role} 脚本，请检查 Agent 轨迹")
    return extract_spoken_script(messages[-1])


def _fingerprint(script, backend, voice, fillers):
    value = json.dumps([script, backend, voice, fillers], ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    st.set_page_config(page_title="OpenMic · jyb 表演实验", page_icon="🎙️", layout="wide")
    st.title("🎙️ OpenMic · jyb 表演实验")
    st.caption("同一稿件、同一音色、同一分段，比较停顿、情绪与完整表演。")
    if "jyb_script" not in st.session_state:
        st.session_state["jyb_script"] = _sample()

    with st.sidebar:
        st.subheader("语音设置")
        backend_labels = {"测试音（非人声，无需 Key）": "tone", "CosyVoice2（真实语音）": "siliconflow"}
        backend_name = backend_labels[st.radio("语音后端", list(backend_labels), key="jyb_backend")]
        voice_label = st.selectbox("音色", list(SYSTEM_VOICES), disabled=backend_name == "tone")
        voice = "FunAudioLLM/CosyVoice2-0.5B:" + SYSTEM_VOICES[voice_label]
        use_cache = st.checkbox("使用片段缓存", value=False)
        fillers = st.checkbox("在部分长铺垫中加入语气词", value=False)
        st.caption("缓存适合演示；冷启动性能测试请关闭缓存。语气词会同时进入所有对照条件。")
        st.caption("真实 API 的 Key 从本地 .env 读取。")

    if backend_name == "tone":
        st.info("当前生成的是测试音，仅验证 WAV、分段和静音拼接；不能用于评判语音质量或模型实时率。")

    with st.expander("可选：从五 Agent 工作流生成新稿件"):
        with st.form("jyb_workflow_form"):
            agent_labels = {"Mock 短稿（无需 Key）": "mock", "真实 AutoGen": "autogen"}
            agent_backend = agent_labels[st.radio("脚本后端", list(agent_labels))]
            topic = st.text_input("主题", "我的网购经历")
            styles = {"观察": "observational", "自嘲": "self_deprecating", "吐槽": "roast"}
            style = styles[st.selectbox("表演风格", list(styles))]
            duration = st.selectbox("目标时长（分钟）", [3, 4, 5])
            audience = st.text_input("目标受众", "大学生")
            generate_script = st.form_submit_button("生成并载入脚本")
        if generate_script:
            try:
                request = ProjectRequest(topic, style, duration, audience)
                feed = st.container()

                def show_message(message):
                    with feed:
                        with st.chat_message("assistant"):
                            st.markdown("**" + message.agent + "**")
                            st.write(message.content)

                with st.spinner("五个 Agent 正在协作……"):
                    workflow = JybWorkflow(AutoGenConfig.from_env()) if agent_backend == "autogen" else OpenMicWorkflow()
                    result = workflow.run(request, on_message=show_message)
                st.session_state["jyb_script"] = _workflow_script(result, agent_backend)
                if not result.approved:
                    st.warning("质检未通过。请先审阅并修改下面的稿件，再手动生成音频。")
                if agent_backend == "mock":
                    st.caption("已载入 JokeWriter 的 Mock 短稿；这不是 3–5 分钟成品。")
                st.success("脚本已载入，请先预览表演计划。")
            except Exception as exc:
                st.error(f"脚本生成失败：{exc}")

    upload = st.file_uploader("也可以上传 UTF-8 编码的脚本", type=["txt"])
    if st.button("载入上传脚本", disabled=upload is None):
        try:
            st.session_state["jyb_script"] = upload.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error("文件不是 UTF-8 编码，请另存为 UTF-8 后重试。")
    script = st.text_area("带标记的脚本（可编辑）", key="jyb_script", height=300)
    st.caption("支持 SETUP / PUNCHLINE / CALLBACK、PAUSE、EMOTION、SPEED、EMPHASIS。未知标记会报错，请先修正。")
    fingerprint = _fingerprint(script, backend_name, voice, fillers)
    selected_labels = st.multiselect(
        "生成的对照条件", list(LABELS.values()), default=[LABELS[key] for key in ("baseline", "timing", "performance")],
    )
    conditions = [key for key, label in LABELS.items() if label in selected_labels]
    preview_col, render_col = st.columns(2)
    preview = preview_col.button("预览表演计划", use_container_width=True)
    generate_audio = render_col.button("生成对照音频", type="primary", use_container_width=True)

    if preview or generate_audio:
        if generate_audio:
            st.session_state["jyb_results"] = {}
        try:
            events = compile_script(script, add_fillers=fillers)
            st.session_state["jyb_plan"] = [asdict(event) for event in events]
            st.session_state["jyb_plan_fingerprint"] = fingerprint
            if generate_audio:
                if not conditions:
                    raise ValueError("请至少选择一个对照条件")
                backend = ToneBackend() if backend_name == "tone" else SiliconFlowBackend(voice=voice)
                st.session_state["jyb_result_fingerprint"] = fingerprint
                progress = st.progress(0.0)
                status = st.empty()
                chunk_player = st.empty()
                for index, condition in enumerate(conditions):
                    status.write(f"正在生成：{LABELS[condition]}")

                    def show_chunk(audio, row):
                        if row.get("kind") == "speech":
                            with chunk_player.container():
                                st.caption(f"{LABELS[condition]}：一个语音片段已就绪，可手动试听。")
                                st.audio(audio, format="audio/wav")

                    result = render(events, backend, condition=condition,
                                    cache_dir=ROOT / "artifacts" / "jyb" / "cache" if use_cache else None,
                                    on_chunk=show_chunk)
                    result.metrics["input_sha256"] = fingerprint
                    result.metrics["source_sha256"] = hashlib.sha256(script.encode("utf-8")).hexdigest()
                    output_dir = ROOT / "artifacts" / "jyb" / "ui" / condition
                    save_result(result, output_dir)
                    st.session_state["jyb_results"][condition] = result
                    progress.progress((index + 1) / len(conditions))
                status.success("音频已生成，完整结果见下方。")
                chunk_player.empty()
        except Exception as exc:
            st.error(f"预览或合成失败：{exc}")

    if st.session_state.get("jyb_plan_fingerprint") == fingerprint:
        with st.expander("表演计划：每段正文与计划静音", expanded=preview):
            st.json(st.session_state["jyb_plan"])

    results = st.session_state.get("jyb_results", {})
    if results and st.session_state.get("jyb_result_fingerprint") != fingerprint:
        st.warning("脚本或语音设置已修改。此前的结果不再代表当前输入，请重新生成。")
    elif results:
        st.subheader("对照结果")
        st.caption(f"输入指纹：{fingerprint[:12]}；输出目录：artifacts/jyb/ui/。分段就绪回调不等同于文本 token 流式生成或浏览器连续播放。")
        for tab, (condition, result) in zip(st.tabs([LABELS[key] for key in results]), results.items()):
            with tab:
                st.audio(result.audio, format="audio/wav")
                metrics = result.metrics
                cols = st.columns(3)
                cols[0].metric("音频时长（秒）", f"{metrics['duration_seconds']:.2f}")
                cols[1].metric("插入静音（秒）", f"{metrics['inserted_silence_seconds']:.2f}")
                cols[2].metric("总耗时（秒）", f"{metrics['wall_seconds']:.2f}")
                if metrics.get("valid_cold_tts_benchmark"):
                    st.caption(f"不计人工静音的生成速度：{metrics['speed_x_speech']:.2f}x；RTF：{metrics['rtf_speech']:.3f}。")
                else:
                    st.caption("本次为测试音或含缓存命中，不能作为冷启动 TTS 性能证据。")
                if not metrics.get("in_target_duration"):
                    st.caption("本段音频尚未达到课程要求的 3–5 分钟，请使用足够长度的正式稿件。")
                st.download_button("下载 WAV", result.audio, file_name=condition + ".wav", mime="audio/wav", key="audio_" + condition)
                st.download_button("下载性能与时间轴 JSON", json.dumps({"input_sha256": fingerprint, "metrics": metrics, "timeline": result.timeline}, ensure_ascii=False, indent=2),
                                   file_name=condition + ".json", mime="application/json", key="json_" + condition)
                with st.expander("完整指标与时间轴"):
                    st.json({"metrics": metrics, "timeline": result.timeline})


if __name__ == "__main__":
    main()
