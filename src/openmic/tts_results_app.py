"""Read-only viewer for this independent run; no model or credential imports."""

import json
import os
from pathlib import Path

import streamlit as st


TOPICS = {
    "shopping": "我的网购经历",
    "canteen": "大学食堂选择困难",
    "deadline": "期末作业与拖延症",
    "fitness": "办了健身卡之后",
    "groupwork": "小组合作的日常",
}
CONDITIONS = {"baseline": "普通朗读", "timing": "只加停顿",
              "emotion": "只加情绪", "performance": "完整表演"}


def run_root():
    override = os.getenv("JYB_RUN_ROOT")
    return Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[2]


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def duration(seconds):
    seconds = number(seconds)
    return f"{int(seconds // 60)}分{seconds % 60:04.1f}秒"


def find_trace(root, topic):
    candidates = [root / "outputs" / "scripts" / (topic + ".trace.json"),
                  root / "records" / "generation" / topic / "trace.json"]
    for path in candidates:
        value = read_json(path)
        if isinstance(value, dict):
            return value
    return None


def show_trace(trace):
    with st.expander("查看本主题的五 Agent 轨迹"):
        if not trace:
            st.info("轨迹文件尚未生成。")
            return
        cols = st.columns(3)
        cols[0].metric("模型质检", "通过" if trace.get("approved") else "未通过")
        cols[1].metric("模型评分", f"{number(trace.get('score')):.2f}")
        cols[2].metric("文本生成耗时", f"{number(trace.get('generation_seconds')):.2f} 秒")
        st.caption("模型评分是脚本质检，不是语音 MOS 或真人幽默评价。")
        for message in trace.get("messages", []):
            if isinstance(message, dict):
                st.markdown("**" + str(message.get("agent", "Agent")) + "**")
                st.write(message.get("content", ""))


def main():
    st.set_page_config(page_title="OpenMic · 本轮独立结果", page_icon="🎙️", layout="wide")
    st.title("🎙️ OpenMic · 本轮独立结果")
    st.caption("20261026_02：仅查看本目录中的结果。试听、切换和下载不调用 API。")
    root = run_root()
    outputs = root / "outputs"
    state = read_json(root / "state.json") or {}
    acceptance = read_json(outputs / "acceptance" / "acceptance.json")
    human = read_json(outputs / "study_summary/status.json") or {}
    human_complete = human.get("status") == "completed"
    st.info("本轮执行显式停顿、重音、语速与情绪控制；没有训练 latent。" +
            ("已收到真人听评，统计见 outputs/study_summary。" if human_complete else "真人听评尚待填写。"))
    if st.button("刷新结果"):
        st.rerun()
    if not outputs.is_dir():
        st.warning("本轮结果目录尚未创建，请稍后刷新。")
        return
    options = {"短稿对比｜我的网购经历": ("short_demo", "shopping")}
    options.update({"五 Agent 稿｜" + label: ("real", name) for name, label in TOPICS.items()})
    selected = st.selectbox("选择主题", list(options))
    group, topic = options[selected]
    topic_dir = outputs / group / topic
    if group == "short_demo":
        st.caption("短稿只用于快速比较，不作为 3–5 分钟正式样本。")
    else:
        st.caption("本轮重新运行真实五 Agent；稿件来源与模型质检见下方轨迹。")
        show_trace(find_trace(root, topic))
    for tab, (condition, label) in zip(st.tabs(list(CONDITIONS.values())), CONDITIONS.items()):
        with tab:
            folder = topic_dir / condition
            meta = read_json(folder / "metrics.json")
            audio_path = folder / "audio.wav"
            if not isinstance(meta, dict) or not audio_path.is_file():
                st.info(label + "：尚未完成，稍后点击“刷新结果”。")
                continue
            try:
                audio = audio_path.read_bytes()
            except OSError:
                st.info("音频正在写入，请稍后刷新。")
                continue
            if meta.get("synthetic") is not False:
                st.warning("此文件没有真实语音标记，不能作为真实 TTS 成绩。")
            else:
                st.success("已生成真实语音 · 16 kHz WAV")
            st.audio(audio, format="audio/wav")
            cols = st.columns(3)
            cols[0].metric("实际时长", duration(meta.get("duration_seconds")))
            cols[1].metric("插入静音", f"{number(meta.get('inserted_silence_seconds')):.2f} 秒")
            cols[2].metric("本次合成耗时", f"{number(meta.get('wall_seconds')):.2f} 秒")
            st.caption(f"第一完整语音段就绪：{number(meta.get('first_speech_chunk_seconds')):.2f} 秒；不等于首音频字节延迟。")
            if meta.get("valid_cold_tts_benchmark") is True:
                st.caption(f"无缓存：排除人工静音后的生成速度 {number(meta.get('speed_x_speech')):.2f}x；RTF {number(meta.get('rtf_speech')):.3f}。不含文本生成耗时。")
            else:
                st.caption(f"缓存命中 {meta.get('cache_hits', 0)} 段；本次耗时不作为无缓存模型速度成绩。")
            if group == "real" and condition == "performance":
                if meta.get("in_target_duration") is True:
                    st.success("表演版实际时长符合 3–5 分钟要求。")
                else:
                    st.warning("表演版实际时长尚未符合 3–5 分钟要求。")
            st.download_button("下载 WAV", audio, file_name=topic + "_" + condition + ".wav",
                               mime="audio/wav", key="wav_" + condition)
            with st.expander("查看指标与精确时间轴"):
                st.json({"metrics": meta, "timeline": read_json(folder / "timeline.json")})
    benchmark = read_json(outputs / "benchmark" / topic / "performance" / "metrics.json")
    if group == "real" and isinstance(benchmark, dict):
        with st.expander("本主题独立无缓存性能测试"):
            if benchmark.get("valid_cold_tts_benchmark") is True:
                st.write(f"模型输出段速度 {number(benchmark.get('speed_x_speech')):.2f}x，RTF {number(benchmark.get('rtf_speech')):.3f}。")
            else:
                st.warning("该测试未满足无缓存基准的记录条件。")
            st.json(benchmark)
    script = topic_dir / "script.txt"
    if script.is_file():
        with st.expander("查看四组共用的标记稿"):
            st.code(script.read_text(encoding="utf-8-sig"), language="text")
    with st.expander("本轮进度与验收"):
        st.write("执行状态：" + str(state.get("status", "记录准备中")))
        if isinstance(acceptance, dict):
            st.write("已有文件硬检查：" + ("通过" if acceptance.get("hard_checks_passed") else "待处理"))
            st.write("五主题交付齐全：" + ("是" if acceptance.get("five_topic_deliverable_passed") else "尚未齐全"))
            st.write("已检查音频数：" + str(acceptance.get("sample_count", 0)))
            st.json(acceptance.get("missing_conditions_by_topic", {}))
        else:
            st.write("最终验收报告尚未完成。")
        st.caption("进度记录：" + str(root / "PROGRESS.md"))
        st.caption("听评材料：" + str(outputs / "blind" / "participant_package"))
        st.caption("当前音频目录：" + str(topic_dir))
        st.write("情绪、重音、自然度和幽默感按实际真人听评判断，不自动填写 MOS。")


if __name__ == "__main__":
    main()
