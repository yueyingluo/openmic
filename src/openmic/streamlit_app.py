"""Minimal Streamlit UI for the OpenMic text-to-audio pipeline."""

import streamlit as st

from openmic.autogen_v02 import AutoGenConfig, AutoGenV02Workflow
from openmic.models import ProjectRequest, WorkflowResult
from openmic.tts import SYSTEM_VOICES, SiliconFlowTTS, TTSConfig
from openmic.workflow import OpenMicWorkflow


STYLE_OPTIONS = {
    "观察类": "observational",
    "自嘲类": "self_deprecating",
    "吐槽类": "roast",
}


def _message(result: WorkflowResult, agent: str) -> str:
    matches = [item.content for item in result.messages if item.agent == agent]
    return matches[-1] if matches else ""


def _reset_audio() -> None:
    st.session_state.pop("audio", None)
    st.session_state.pop("audio_meta", None)


def main() -> None:
    st.set_page_config(page_title="OpenMic", page_icon="🎙️", layout="wide")
    st.title("🎙️ OpenMic")
    st.caption("AutoGen 0.2 多智能体中文脱口秀生成 + CosyVoice2 语音合成")

    with st.sidebar:
        st.header("运行设置")
        backend = st.radio(
            "Agent 后端",
            options=["autogen", "mock"],
            format_func=lambda value: "真实 AutoGen" if value == "autogen" else "Mock（不消耗额度）",
        )
        st.info("密钥仅从本地 .env 读取，不会显示在页面或写入 Git。")

    with st.form("generation_form"):
        topic = st.text_input("主题", value="我的网购经历")
        col1, col2, col3 = st.columns(3)
        with col1:
            style_label = st.selectbox("表演风格", list(STYLE_OPTIONS))
        with col2:
            duration = st.selectbox("目标时长（分钟）", [3, 4, 5])
        with col3:
            audience = st.text_input("目标受众", value="大学生")
        generate = st.form_submit_button("生成脱口秀脚本", type="primary")

    if generate:
        _reset_audio()
        request = ProjectRequest(
            topic=topic,
            style=STYLE_OPTIONS[style_label],
            duration_minutes=duration,
            audience=audience,
        )
        try:
            with st.spinner("五个 Agent 正在协作，真实模型可能需要 1–3 分钟……"):
                if backend == "autogen":
                    result = AutoGenV02Workflow(AutoGenConfig.from_env()).run(request)
                else:
                    result = OpenMicWorkflow().run(request)
            st.session_state["workflow_result"] = result
            st.session_state["performance_text"] = (
                _message(result, "PerformanceCoach")
                or _message(result, "JokeWriter")
            )
        except Exception as exc:
            st.error(f"脚本生成失败：{exc}")

    result = st.session_state.get("workflow_result")
    if result:
        status_col, score_col, revision_col = st.columns(3)
        status_col.metric("质检结果", "通过" if result.approved else "未通过")
        score_col.metric("平均分", f"{result.score:.2f}")
        revision_col.metric("返工次数", result.revision_count)

        st.subheader("Agent 协作轨迹")
        for item in result.messages:
            with st.expander(item.agent, expanded=item.agent in {"JokeWriter", "QualityController"}):
                st.markdown(item.content)

        st.subheader("语音合成")
        performance_text = st.text_area(
            "TTS 文本（可以手动修改）",
            key="performance_text",
            height=320,
        )
        voice_label = st.selectbox("音色", list(SYSTEM_VOICES), index=0)
        speed = st.slider("语速", 0.75, 1.50, 1.0, 0.05)

        if st.button("生成 16 kHz WAV", type="primary"):
            try:
                config = TTSConfig.from_env()
                voice = f"{config.model}:{SYSTEM_VOICES[voice_label]}"
                with st.spinner("CosyVoice2 正在合成语音……"):
                    audio = SiliconFlowTTS(config).synthesize(
                        performance_text,
                        voice=voice,
                        speed=speed,
                    )
                st.session_state["audio"] = audio.audio
                st.session_state["audio_meta"] = audio
            except Exception as exc:
                st.error(f"语音生成失败：{exc}")

        if st.session_state.get("audio"):
            audio = st.session_state["audio"]
            meta = st.session_state["audio_meta"]
            st.audio(audio, format=meta.media_type)
            st.download_button(
                "下载 WAV",
                data=audio,
                file_name="openmic.wav",
                mime=meta.media_type,
            )
            st.caption(
                f"模型：{meta.model} · 音色：{meta.voice} · 采样率：{meta.sample_rate} Hz"
            )


if __name__ == "__main__":
    main()
