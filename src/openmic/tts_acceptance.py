"""Validate saved speech experiments; never reads credentials or calls APIs."""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import struct
import wave
from pathlib import Path


RATE = 16000
DIMENSIONS = ("humor", "naturalness", "pause", "emphasis", "emotion")
EMOTIONS = {
    "neutral": "自然平静", "happy": "高兴", "sad": "悲伤", "angry": "轻微生气",
    "surprised": "惊讶", "calm": "冷静", "serious": "严肃", "self_deprecating": "带着自嘲的平静",
}


def _json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def _plain(text):
    return text.replace("<strong>", "").replace("</strong>", "")


def _pcm(path):
    data = Path(path).read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    if struct.unpack_from("<I", data, 4)[0] + 8 != len(data):
        raise ValueError("RIFF size does not match saved file length")
    offset, declared_pcm = 12, None
    while offset + 8 <= len(data):
        size = struct.unpack_from("<I", data, offset + 4)[0]
        if offset + 8 + size > len(data):
            raise ValueError("truncated WAV chunk")
        if data[offset:offset + 4] == b"data":
            declared_pcm = size
            break
        offset += 8 + size + size % 2
    if declared_pcm is None or declared_pcm == 0 or declared_pcm % 2:
        raise ValueError("missing, empty or partial 16-bit PCM samples")
    with wave.open(io.BytesIO(data), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getcomptype()) != (RATE, 1, 2, "NONE"):
            raise ValueError("expected 16000 Hz, mono, 16-bit PCM")
        frames = handle.getnframes()
        pcm = handle.readframes(frames)
    if len(pcm) != frames * 2 or len(pcm) != declared_pcm:
        raise ValueError("PCM payload length does not match declared frames")
    if not any(pcm):
        raise ValueError("audio is entirely digital silence")
    return pcm, frames


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _close(left, right, tolerance=1 / RATE):
    return _finite(left) and abs(left - right) <= tolerance


def _sample(metrics_path, runs, require_duration):
    directory = metrics_path.parent
    relative = directory.relative_to(runs)
    result = {"directory": str(directory), "topic": relative.parts[0] if len(relative.parts) >= 2 else "default_topic",
              "condition": directory.name, "passed": False, "checks": {}, "errors": []}
    speech_pcm = []

    def check(name, passed, message):
        result["checks"][name] = bool(passed)
        if not passed:
            result["errors"].append(message)

    try:
        metrics = json.loads(_json(metrics_path))
        condition = metrics["condition"]
        result["condition"] = condition
        check("known_condition", condition in {"baseline", "timing", "emotion", "performance"}, "unknown condition")
        check("declared_real_audio", metrics.get("synthetic") is False, "result is synthetic or lacks explicit real-audio metadata")
        result["backend"] = metrics.get("backend")
        for name in ("plan_sha256", "spoken_text_sha256"):
            result[name] = metrics.get(name)
            check(name + "_present", isinstance(metrics.get(name), str) and re.fullmatch(r"[0-9a-f]{64}", metrics[name]) is not None,
                  "missing or malformed " + name)
        check("backend_present", bool(result["backend"]), "missing model/voice backend identity")
        pcm, frame_count = _pcm(directory / "audio.wav")
        result.update(sample_rate=RATE, channels=1, sample_width=2, frames=frame_count,
                      duration_seconds=frame_count / RATE,
                      duration_in_target=180 <= frame_count / RATE <= 300)
        check("wav_format_and_payload", True, "")
        check("duration_metadata", _close(metrics.get("duration_seconds"), frame_count / RATE), "duration metadata differs from actual WAV")
        if require_duration and condition == "performance":
            check("required_duration", result["duration_in_target"], "performance is outside 180–300 seconds")
        timeline = json.loads(_json(directory / "timeline.json"))
        if not isinstance(timeline, list) or not timeline:
            raise ValueError("timeline must be a nonempty array")
        position = speech_frames = pause_frames = pause_08 = pause_20 = 0
        payload_ok = boundaries_ok = pause_ok = True
        for row in timeline:
            start, end = row.get("start_frame"), row.get("end_frame")
            if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
                raise ValueError("timeline frame boundaries must be integers")
            if start != position or end < start or end > frame_count:
                boundaries_ok = False
            if start < 0 or end < start or end > frame_count:
                raise ValueError("timeline has out-of-bounds frame slices")
            position = end
            frames = end - start
            fragment = pcm[start * 2:end * 2]
            if not _close(row.get("duration_seconds"), frames / RATE):
                boundaries_ok = False
            if row.get("kind") == "pause":
                seconds = row.get("requested_seconds", row.get("seconds"))
                if not _finite(seconds) or not 0 <= seconds <= 10:
                    pause_ok = False
                elif frames != round(seconds * RATE):
                    pause_ok = False
                if any(fragment):
                    pause_ok = False
                pause_frames += frames
                pause_08 += frames == 12800
                pause_20 += frames == 32000
            elif row.get("kind") == "speech":
                speech_frames += frames
                speech_pcm.append(fragment)
                expected_text = row["text"] if condition == "performance" else _plain(row["text"])
                expected_emotion = row["emotion"] if condition in {"emotion", "performance"} else "neutral"
                expected_speed = row["speed"] if condition == "performance" else 1.0
                expected_instruction = "请用" + EMOTIONS[expected_emotion] + "的语气说。"
                if row.get("actual_input") != expected_text or row.get("actual_instruction") != expected_instruction or not _close(row.get("actual_speed"), expected_speed, 1e-9):
                    payload_ok = False
                if frames == 0:
                    payload_ok = False
            else:
                raise ValueError("timeline contains unknown event kind")
        check("timeline_contiguous", boundaries_ok and position == frame_count, "timeline is not contiguous or does not cover actual WAV")
        check("exact_zero_sample_pauses", pause_ok, "pause samples or duration do not match requested PCM silence")
        check("declared_control_payloads", payload_ok, "saved segment controls do not match the experimental condition")
        check("speech_duration_metadata", _close(metrics.get("speech_seconds"), speech_frames / RATE), "speech_seconds differs from timeline")
        check("silence_duration_metadata", _close(metrics.get("inserted_silence_seconds"), pause_frames / RATE), "inserted_silence_seconds differs from timeline")
        result.update(speech_segments=len(speech_pcm), pause_0_8_count=pause_08, pause_2_0_count=pause_20)
        plan_path = directory.parent / "plan.json"
        if plan_path.is_file():
            plan = json.loads(_json(plan_path))
            expected_hash = hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            text_hash = hashlib.sha256("\n".join(_plain(event["text"]) for event in plan if "text" in event).encode("utf-8")).hexdigest()
            check("saved_plan_hash", metrics.get("plan_sha256") == expected_hash, "saved plan differs from plan_sha256")
            check("saved_plan_text_hash", metrics.get("spoken_text_sha256") == text_hash, "saved plan differs from spoken_text_sha256")
    except (OSError, ValueError, KeyError, TypeError, wave.Error, EOFError) as exc:
        result["errors"].append(str(exc))
    result["passed"] = not result["errors"]
    return result, speech_pcm


def _benchmarks(path):
    if path is None:
        return {"status": "not_provided", "target_passed": None, "samples": []}
    samples = []
    for file in sorted(Path(path).rglob("metrics.json")):
        row = {"directory": str(file.parent), "valid_cold_benchmark": False, "errors": []}
        try:
            metrics = json.loads(_json(file))
            pcm, frames = _pcm(file.with_name("audio.wav"))
            wall, speech = metrics.get("wall_seconds"), metrics.get("speech_seconds")
            eligible = (metrics.get("synthetic") is False and metrics.get("cache_hits") == 0
                        and metrics.get("valid_cold_tts_benchmark") is True
                        and isinstance(metrics.get("api_calls"), int) and metrics["api_calls"] > 0
                        and _finite(wall) and wall > 0 and _finite(speech) and speech > 0
                        and speech <= frames / RATE + 1 / RATE)
            row["valid_cold_benchmark"] = bool(eligible)
            if not eligible:
                row["errors"].append("not an uncached real-backend run with positive timings")
            else:
                speed, rtf = speech / wall, wall / speech
                row.update(wall_seconds=wall, speech_seconds=speech,
                           speed_x_speech=speed, rtf_speech=rtf,
                           first_speech_chunk_seconds=metrics.get("first_speech_chunk_seconds"),
                           meets_speed_target=speed > 1.0)
                if not _close(metrics.get("speed_x_speech"), speed, 1e-6) or not _close(metrics.get("rtf_speech"), rtf, 1e-6):
                    row["errors"].append("reported speed/RTF arithmetic mismatch")
                    row["valid_cold_benchmark"] = False
        except (OSError, ValueError, KeyError, TypeError, wave.Error, EOFError) as exc:
            row["errors"].append(str(exc))
        samples.append(row)
    return {"status": "checked" if samples else "no_benchmarks_found",
            "target_passed": bool(samples) and all(row["valid_cold_benchmark"] and row.get("meets_speed_target") for row in samples),
            "samples": samples}


def _blind(path):
    if path is None:
        return {"status": "not_provided", "human_review_pending": True}
    root = Path(path)
    package = root / "participant_package"
    result = {"status": "checked", "passed": False, "human_review_pending": True, "errors": []}
    try:
        manifest = json.loads(_json(package / "manifest.json"))
        if manifest.get("synthetic_practice_only") is not False:
            result["errors"].append("blind package is synthetic practice or has no real-audio declaration")
        audio = sorted((package / "audio").glob("*.wav"))
        if not audio or len(audio) != manifest.get("samples"):
            result["errors"].append("blind audio count differs from manifest")
        for file in audio:
            if not re.fullmatch(r"sample[0-9]{3,}\.wav", file.name):
                result["errors"].append("audio filename leaks a condition or is not an anonymous sample ID")
            _pcm(file)
        if (package / "key.csv").exists() or not (root / "investigator" / "key.csv").is_file():
            result["errors"].append("answer key must be outside participant_package")
        filled = 0
        forms = sorted(package.glob("p*_ratings.csv"))
        for file in forms:
            with file.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    filled += any((row.get(name) or "").strip() for name in DIMENSIONS)
        result.update(samples=len(audio), rating_forms=len(forms), filled_rating_rows=filled,
                      rating_templates_blank=filled == 0)
        if not forms:
            result["errors"].append("no listener rating forms")
    except (OSError, ValueError, TypeError, wave.Error, EOFError) as exc:
        result["errors"].append(str(exc))
    result["passed"] = not result["errors"]
    return result


def _design_status(samples, hard_checks_passed):
    """Separate integrity of existing files from completeness of a whole study."""
    groups = {}
    for row in samples:
        groups.setdefault(row["topic"], []).append(row)
    expected = {"baseline", "timing", "emotion", "performance"}
    required_deliverable = {"baseline", "timing", "performance"}
    present = {topic: {row["condition"] for row in rows} for topic, rows in groups.items()}
    missing = {topic: sorted(expected - conditions) for topic, conditions in present.items()
               if expected - conditions}
    all_performance = bool(groups) and all("performance" in conditions for conditions in present.values())
    performance_rows = [row for row in samples if row["condition"] == "performance"]
    duration_passed = all_performance and all(row.get("duration_in_target") is True for row in performance_rows)
    mandatory_conditions = bool(groups) and all(required_deliverable <= conditions for conditions in present.values())
    return {
        "complete_four_condition_design": bool(groups) and not missing,
        "missing_conditions_by_topic": missing,
        "all_topics_have_performance": all_performance,
        "performance_duration_target_passed": duration_passed,
        "five_topic_deliverable_passed": (len(groups) >= 5 and mandatory_conditions
                                          and hard_checks_passed and duration_passed),
    }


def inspect_runs(runs, *, require_duration=False, benchmarks=None, blind=None):
    runs = Path(runs).resolve()
    samples, speech = [], {}
    for path in sorted(runs.rglob("metrics.json")):
        row, pcm = _sample(path, runs, require_duration)
        samples.append(row)
        speech[(row["topic"], row["condition"])] = pcm
    groups = {}
    for row in samples:
        groups.setdefault(row["topic"], []).append(row)
    matches = []
    for topic, rows in groups.items():
        errors = []
        conditions = [row["condition"] for row in rows]
        if len(conditions) != len(set(conditions)):
            errors.append("duplicate condition for topic")
        for field in ("plan_sha256", "spoken_text_sha256", "backend"):
            serialized = {json.dumps(row.get(field), ensure_ascii=False, sort_keys=True) for row in rows}
            if len(serialized) != 1:
                errors.append("conditions differ in " + field)
        pcm_status = "not_available"
        if "baseline" in conditions and "timing" in conditions:
            left, right = speech[(topic, "baseline")], speech[(topic, "timing")]
            identical = bool(left) and left == right
            pcm_status = "identical" if identical else "different"
            if not identical:
                errors.append("baseline/timing speech PCM is not identical; regenerate with shared cache for the timing-only comparison")
        matches.append({"topic": topic, "conditions": conditions, "passed": not errors,
                        "baseline_timing_speech_pcm": pcm_status, "errors": errors})
    hard_checks_passed = bool(samples) and all(row["passed"] for row in samples) and all(row["passed"] for row in matches)
    report = {
        "schema_version": 1, "runs": str(runs), "duration_required": require_duration,
        "hard_checks_passed": hard_checks_passed,
        "sample_count": len(samples), "topic_count": len(groups),
        "five_topics_present": len(groups) >= 5,
        **_design_status(samples, hard_checks_passed),
        "samples": samples, "matched_topics": matches,
        "benchmarks": _benchmarks(benchmarks), "blind_package": _blind(blind),
        "human_review_pending": True,
        "evidence_limits": [
            "This checks local files and declared request metadata; it does not independently authenticate provider provenance.",
            "Precise PCM silence is checked; perceptual phoneme-to-phoneme gaps may also include natural low-energy audio.",
            "Emotion, emphasis, pronunciation, humor and MOS need listening review; no improvement is inferred from a valid payload.",
            "A valid benchmark is a local uncached measurement, not proof of end-to-end LLM latency or general service performance.",
        ],
    }
    if not samples:
        report["errors"] = ["no metrics.json/audio.wav experiment pairs found"]
    return report


def save_report(report, out):
    out = Path(out)
    if out.suffix.lower() == ".json":
        json_path, directory = out, out.parent / (out.stem + "_tables")
    else:
        directory, json_path = out, out / "acceptance.json"
    directory.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ("topic", "condition", "passed", "duration_seconds", "duration_in_target", "sample_rate",
               "channels", "sample_width", "pause_0_8_count", "pause_2_0_count", "errors")
    with (directory / "samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report["samples"]:
            writer.writerow({**{name: row.get(name, "") for name in columns}, "errors": "; ".join(row["errors"])})
    lines = ["# jyb 文件验收结果", "", f"已有文件硬检查通过：{report['hard_checks_passed']}",
             "上述结果只表示已经存在的文件有效，不代表计划中的音频都已生成。",
             f"主题：{report['topic_count']}；音频：{report['sample_count']}",
             f"四条件设计齐全：{report['complete_four_condition_design']}",
             f"每个主题都有表演版：{report['all_topics_have_performance']}",
             f"每主题均有表演版且均在180–300秒：{report['performance_duration_target_passed']}",
             f"五主题音频交付齐全且硬检查通过：{report['five_topic_deliverable_passed']}",
             "五主题音频交付要求每主题至少有普通朗读、仅停顿和完整表演；仅情绪条件另计入四条件完整性。",
             f"缺少条件：{json.dumps(report['missing_conditions_by_topic'], ensure_ascii=False)}",
             f"已提供冷测试的实时速度目标通过：{report['benchmarks']['target_passed']}",
             "", "人工试听待完成：情绪、重音、发音、幽默感与MOS没有自动判定为合格。", "",
             "详细检查与失败原因见 acceptance.json / samples.csv。"]
    (directory / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-duration", action="store_true", help="require performance clips to be 180–300 seconds")
    parser.add_argument("--benchmarks", type=Path, help="separate uncached benchmark output root")
    parser.add_argument("--blind", type=Path, help="blind package root containing participant_package/investigator")
    args = parser.parse_args(argv)
    report = inspect_runs(args.runs, require_duration=args.require_duration, benchmarks=args.benchmarks, blind=args.blind)
    save_report(report, args.out)
    print(f"Local file checks passed={report['hard_checks_passed']}; topics={report['topic_count']}; samples={report['sample_count']}; human_review_pending=True")
    return 0 if report["hard_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
