"""Read-only player for completed jyb experiments; never calls model APIs."""

import json
import os
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPICS = {
    "shopping": "我的网购经历",
    "canteen": "大学食堂选择困难",
    "deadline": "期末作业与拖延症",
    "fitness": "办了健身卡之后",
    "groupwork": "小组合作的日常",
}
CONDITIONS = {
    "baseline": "普通朗读",
    "timing": "只加停顿",
    "emotion": "只加情绪",
    "performance": "完整表演",
}


def result_root():
    """Override points to the artifacts/jyb directory, not the source root."""
    value = os.getenv("OPENMIC_SHOWCASE_ROOT")
    return Path(value).expanduser().resolve() if value else PROJECT_ROOT / "artifacts" / "jyb"


def _json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _choices(root):
    entries = []
    demos = sorted(root.glob("real_demo_*"), reverse=True)
    generated = sorted(root.glob("real_autogen_*"), reverse=True)
    shows = sorted(root.glob("real_showcase_*"), reverse=True)
    if demos:
        entries.append(("短稿演示｜我的网购经历", demos[0] / "shopping", "demo"))
    if generated:
        entries.append(("真实五Agent稿｜我的网购经历", generated[0] / "shopping", "autogen"))
    if shows:
        for name, label in TOPICS.items():
            entries.append(("固定稿样例｜" + label, shows[0] / name, "fixed"))
    return entries


def _duration(seconds):
    return f"{int(seconds // 60)}分{seconds % 60:04.1f}秒"


def main():
    st.set_page_config(page_title="OpenMic · 已生成语音", page_icon="🎙️", layout="wide")
    st.title("🎙️ OpenMic · 已生成语音")
    st.caption("先听短稿对比，再查看真实五 Agent 稿或五份固定测试稿。此页面只读取已有文件，打开和播放都不调用 API。")
    st.info("本次为显式表演控制：精确停顿、情绪与重音提示。未训练 latent；表演效果仍待真人听评。")
    root = result_root()
    entries = _choices(root)
    if not entries:
        st.warning("尚未找到演示结果。生成中的文件准备好后，点击“刷新结果”。")
        st.caption("结果目录：" + str(root))
        if st.button("刷新结果"):
            st.rerun()
        return
    labels = [entry[0] for entry in entries]
    chosen = st.selectbox("选择主题", labels)
    _, topic_dir, source = entries[labels.index(chosen)]
    formal = source != "demo"
    if st.button("刷新结果"):
        st.rerun()
    if not formal:
        st.caption("短稿用于现场快速比较；3–5 分钟音频请切换其他样例。")
    elif source == "fixed":
        st.caption("这是为 TTS 对照准备的预置原创稿，不是五 Agent 生成结果。")
    else:
        st.caption("稿件来自真实五 Agent 工作流，仍待组内内容审核。模型质检分数不代表真人听评。")
        trace = _json(root / "execution" / "autogen_protocol" / "shopping.trace.json")
        if isinstance(trace, dict):
            with st.expander("查看五 Agent 轨迹与模型质检"):
                cols = st.columns(3)
                cols[0].metric("模型质检", "通过" if trace.get("approved") else "未通过")
                cols[1].metric("模型评分", f"{trace.get('score', 0):.2f}")
                cols[2].metric("稿件生成耗时", f"{trace.get('generation_seconds', 0):.2f} 秒")
                for message in trace.get("messages", []):
                    st.markdown("**" + str(message.get("agent", "Agent")) + "**")
                    st.write(message.get("content", ""))
    for tab, (condition, label) in zip(st.tabs(list(CONDITIONS.values())), CONDITIONS.items()):
        with tab:
            directory = topic_dir / condition
            audio_path = directory / "audio.wav"
            meta = _json(directory / "metrics.json")
            if not audio_path.is_file() or not isinstance(meta, dict):
                st.info(label + "：生成中，稍后点击“刷新结果”。")
                continue
            if meta.get("synthetic") is not False:
                st.warning("该文件未标记为真实语音，不能作为真实 TTS 成绩。")
            else:
                st.success("真实中文语音 · CosyVoice2 · 16 kHz WAV")
            try:
                audio = audio_path.read_bytes()
            except OSError:
                st.info("音频文件正在写入，请稍后刷新。")
                continue
            st.audio(audio, format="audio/wav")
            seconds = float(meta.get("duration_seconds", 0))
            cols = st.columns(3)
            cols[0].metric("实际时长", _duration(seconds))
            cols[1].metric("插入静音", f"{meta.get('inserted_silence_seconds', 0):.2f} 秒")
            cols[2].metric("生成耗时", f"{meta.get('wall_seconds', 0):.2f} 秒")
            if condition == "performance" and formal:
                if meta.get("in_target_duration"):
                    st.success("实际时长符合 3–5 分钟要求。")
                else:
                    st.warning("实际时长尚未符合 3–5 分钟要求，需继续调整。")
            if meta.get("valid_cold_tts_benchmark"):
                st.caption(f"本次无缓存命中；模型输出段生成速度 {meta.get('speed_x_speech', 0):.2f}x。此值不含 LLM 生成耗时。")
            else:
                st.caption(f"缓存命中 {meta.get('cache_hits', 0)} 段；该耗时用于展示，不能作为冷启动模型速度成绩。")
            st.download_button("下载这份 WAV", audio, file_name=topic_dir.name + "_" + condition + ".wav",
                               mime="audio/wav", key="download_" + condition)
            timeline = _json(directory / "timeline.json")
            with st.expander("查看指标与时间轴"):
                st.json({"metrics": meta, "timeline": timeline})
    script = topic_dir / "script.txt"
    if script.is_file():
        with st.expander("查看本主题共用的标记稿"):
            st.code(script.read_text(encoding="utf-8-sig"), language="text")
    acceptance = _json(root / "execution" / "acceptance" / "acceptance.json")
    with st.expander("验收与人工听评"):
        if isinstance(acceptance, dict):
            st.write("文件与时间轴硬检查：" + ("通过" if acceptance.get("hard_checks_passed") else "待处理"))
            st.write("已验收音频数：" + str(acceptance.get("sample_count", 0)))
        else:
            st.write("文件验收报告正在准备。")
        st.write("情绪、重音、发音和幽默感：待真人听评，未自动填分。")
        st.caption("短稿先导听评（四条件、八位听众）：" + str(root / "blind_demo_20261026" / "participant_package"))
        st.caption("固定稿听评材料：" + str(root / "blind_showcase_20261026" / "participant_package"))
        st.caption("原始结果：" + str(topic_dir))


if __name__ == "__main__":
    main()
