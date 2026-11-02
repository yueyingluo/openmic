"""CPU acoustic diagnostics, global loudness matching and blank stage AB forms.

These measurements are not humor, naturalness, speaker-identity or MOS scores.
Original audio is preserved. No model, API, GPU or credentials are used.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import struct
import tempfile
import wave

import numpy as np


CONDITIONS = ("legacy", "stage_prompt", "stage_course", "stage_club")
DIMENSIONS = ("stage_like", "audience_address", "naturalness", "voice_consistency")
PREFERENCES = ("A", "B", "tie", "unsure")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _read_pcm(path):
    path = Path(path)
    audio = path.read_bytes()
    if (len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE"
            or struct.unpack_from("<I", audio, 4)[0] != len(audio) - 8):
        raise ValueError("Expected a complete RIFF/WAVE file with exact RIFF size")
    offset = 12
    while offset + 8 <= len(audio):
        size = struct.unpack_from("<I", audio, offset + 4)[0]
        offset += 8 + size + size % 2
        if offset > len(audio):
            raise ValueError("Truncated WAV chunk")
    if offset != len(audio):
        raise ValueError("Invalid WAV chunk boundary")
    try:
        with wave.open(str(path), "rb") as reader:
            if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(),
                    reader.getcomptype()) != (16000, 1, 2, "NONE"):
                raise ValueError("Expected 16000 Hz mono 16-bit PCM WAV")
            frames = reader.getnframes()
            pcm = reader.readframes(frames)
            if frames <= 0 or len(pcm) != frames * 2:
                raise ValueError("Empty or truncated WAV data")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid PCM WAV") from exc
    return np.frombuffer(pcm, dtype="<i2").copy()


def _db(value):
    return float(20 * math.log10(value)) if value > 0 else None


def _longest_run(mask):
    extended = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    boundaries = np.flatnonzero(extended[1:] != extended[:-1])
    return int((boundaries[1::2] - boundaries[::2]).max()) if len(boundaries) else 0


def analyze_file(path):
    """Measure a complete 16 kHz mono PCM file; missing dependencies are errors.

    LUFS uses pyloudnorm's BS.1770-4 implementation. Four-times polyphase
    oversampling is an approximate true peak, not certified BS.1770 true peak.
    Fixed-gate short-time RMS spread is not EBU Loudness Range (LRA).
    """
    import pyloudnorm
    from scipy.signal import resample_poly

    path = Path(path).resolve()
    pcm = _read_pcm(path)
    samples = pcm.astype(np.float64) / 32768.0
    duration = len(samples) / 16000
    peak = float(np.max(np.abs(samples)))
    approximate_peak = max(peak, float(np.max(np.abs(resample_poly(samples, 4, 1)))))
    meter = pyloudnorm.Meter(16000, filter_class="K-weighting", block_size=0.400)
    if duration < 0.400:
        loudness, loudness_status = None, "unsupported_duration_under_400_ms"
    else:
        measured = float(meter.integrated_loudness(samples))
        loudness = measured if math.isfinite(measured) else None
        loudness_status = "measured" if loudness is not None else "unmeasurable_silence"
    starts = np.arange(0, len(samples), 160)
    ends = np.minimum(starts + 400, len(samples))
    sums = np.concatenate(([0.0], np.cumsum(samples * samples)))
    frame_rms = np.sqrt(np.maximum(0.0, (sums[ends] - sums[starts]) / (ends - starts)))
    frame_db = 20 * np.log10(np.maximum(frame_rms, 1e-12))
    active = frame_db > -45.0
    selected = frame_db[active]
    percentiles = np.quantile(selected, [0.1, 0.5, 0.9]).tolist() if len(selected) else None
    adjacent_active = active[:-1] & active[1:]
    differences = np.abs(np.diff(frame_db))[adjacent_active]
    rails = (pcm == -32768) | (pcm == 32767)
    zero = pcm == 0
    artifacts = {}
    for name in ("metrics.json", "timeline.json", "reference_provenance.json"):
        artifact = path.parent / name
        if artifact.is_file():
            artifacts[name] = {"path": str(artifact), "sha256": _sha(artifact)}
    return {
        "audio_path": str(path), "audio_sha256": _sha(path),
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        "frames": len(pcm), "sample_rate": 16000, "channels": 1, "sample_width": 2,
        "duration_seconds": duration, "integrated_lufs": loudness,
        "loudness_status": loudness_status,
        "loudness_method": "pyloudnorm BS.1770-4 library implementation; K-weighting; 400 ms blocks",
        "pyloudnorm_version": importlib.metadata.version("pyloudnorm"),
        "sample_peak_dbfs": _db(peak), "approx_true_peak_dbtp": _db(approximate_peak),
        "true_peak_method": "Approximation: scipy.signal.resample_poly 4x, default Kaiser 5 filter; not certified BS.1770 true-peak metering",
        "rms_dbfs_including_silence": _db(float(np.sqrt(np.mean(samples * samples)))),
        "dc_mean": float(samples.mean()),
        "samples_at_digital_rails": int(rails.sum()),
        "samples_at_digital_rails_fraction": float(rails.mean()),
        "near_full_scale_fraction_abs_ge_0_999": float((np.abs(samples) >= 0.999).mean()),
        "longest_digital_rail_run_samples": _longest_run(rails),
        "clipping_scope": "Samples at/near digital rails are diagnostics; they do not establish the cause or prove that all clipping is detectable",
        "exact_zero_samples": int(zero.sum()), "exact_zero_fraction": float(zero.mean()),
        "longest_exact_zero_seconds": _longest_run(zero) / 16000,
        "rms_frames": {
            "window_samples": 400, "hop_samples": 160, "tail_included": True,
            "active_gate_dbfs_exclusive": -45.0, "total_frames": len(starts),
            "active_frames": int(active.sum()), "inactive_frame_fraction": float((~active).mean()),
            "active_p10_p50_p90_dbfs": percentiles,
            "active_p90_minus_p10_db": float(percentiles[2] - percentiles[0]) if percentiles else None,
            "adjacent_active_abs_difference_median_db": float(np.median(differences)) if len(differences) else None,
            "scope": "Fixed energy gate, not VAD; spread is not EBU LRA, perceived expressiveness, or a quality score",
        },
        "f0_measured": False, "perceptual_quality_estimated": False,
        "source_artifacts": artifacts,
    }


def _provenance(topicdir):
    for candidate in (topicdir / "legacy/reference_provenance.json", topicdir / "reference_provenance.json"):
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("reference_provenance.json must be an object")
            return {"path": str(candidate.resolve()), "sha256": _sha(candidate), "record": value}
    raise FileNotFoundError("Legacy copied audio requires reference_provenance.json")


def match_topic(topicdir):
    """Create whole-file constant-gain copies, preserving all original audio.wav.

    Common target <= -20 LUFS, with -1.1 dB approximate true-peak headroom:
    the extra 0.1 dB is reserved for quantization/numeric differences. No limiter,
    compressor, per-utterance normalization, speed or pitch processing is applied.
    """
    topicdir = Path(topicdir).resolve()
    provenance = _provenance(topicdir)
    original = {}
    for condition in CONDITIONS:
        directory = topicdir / condition
        for name in ("audio.wav", "metrics.json", "timeline.json"):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        original[condition] = analyze_file(directory / "audio.wav")
        if original[condition]["integrated_lufs"] is None or original[condition]["approx_true_peak_dbtp"] is None:
            raise ValueError(f"Cannot loudness-match {condition}: unmeasurable loudness/peak")
    target = min(-20.0, *(row["integrated_lufs"] - 1.1 - row["approx_true_peak_dbtp"]
                          for row in original.values()))
    report = {"schema": "openmic.stage-levels/1", "topic": topicdir.name,
              "target_lufs": target, "initial_target_lufs": -20.0,
              "target_is_experiment_setting_not_stage_standard": True,
              "peak_headroom_before_quantization_db": 1.1,
              "quantization_headroom_margin_db": 0.1,
              "method": "One fixed linear gain per whole waveform; no compression, limiting, per-sentence normalization, pitch or speed changes",
              "original_audio_preserved": True, "reference_provenance": provenance,
              "legacy_new_tts_request_this_run": False,
              "legacy_scope": "Legacy is copied reference audio; its historical API counts are not new requests or this run's synthesis timings",
              "perceptual_quality_estimated": False, "conditions": {}}
    for condition in CONDITIONS:
        directory = topicdir / condition
        raw = original[condition]
        gain_db = target - raw["integrated_lufs"]
        gain = 10 ** (gain_db / 20)
        scaled = _read_pcm(directory / "audio.wav").astype(np.float64) * gain
        rounded = np.rint(scaled)
        if np.max(rounded) > 32767 or np.min(rounded) < -32768:
            raise ValueError("Linear gain would clip; refusing hidden limiting")
        matched_pcm = rounded.astype("<i2")
        descriptor, temporary = tempfile.mkstemp(prefix=".matched-", suffix=".wav", dir=directory)
        import os
        os.close(descriptor)
        temporary = Path(temporary)
        try:
            with wave.open(str(temporary), "wb") as writer:
                writer.setparams((1, 2, 16000, len(matched_pcm), "NONE", "not compressed"))
                writer.writeframes(matched_pcm.tobytes())
            matched = analyze_file(temporary)
            loudness_error = abs(matched["integrated_lufs"] - target) if matched["integrated_lufs"] is not None else float("inf")
            if loudness_error > 0.12:
                raise ValueError(f"Quantized LUFS error exceeds 0.12 dB: {loudness_error}")
            if matched["approx_true_peak_dbtp"] is None or matched["approx_true_peak_dbtp"] > -1.0:
                raise ValueError("Quantized approximate true peak exceeds -1 dBTP")
            destination = directory / "matched.wav"
            temporary.replace(destination)
            matched["audio_path"] = str(destination)
        finally:
            temporary.unlink(missing_ok=True)
        if _sha(directory / "audio.wav") != raw["audio_sha256"]:
            raise ValueError("Original waveform changed during matching")
        row = {"condition": condition, "original": raw, "matched": matched,
               "gain_db": gain_db, "linear_gain": gain, "lufs_error_db": loudness_error,
               "samples_unchanged_in_count": matched["frames"] == raw["frames"],
               "original_sha256_verified": True,
               "new_tts_request_this_run": False if condition == "legacy" else None,
               "request_scope": "Copied historical reference; no new TTS request" if condition == "legacy" else
                                "This run's condition; request count must be read from its metrics because cached conditions may make no HTTP call"}
        _json(directory / "levels.json", row)
        report["conditions"][condition] = row
    report["status"] = "passed"
    _json(topicdir / "levels.json", report)
    return report


def _topics(runs):
    runs = Path(runs).resolve()
    topics = sorted(path for path in runs.iterdir() if path.is_dir()) if runs.is_dir() else []
    if not topics:
        raise ValueError("No comparison topics; missing data cannot be substituted")
    return topics


def build_blind(runs, out, listeners=8, seed=20261101):
    """Build blank four-dimension AB packets: legacy versus stage_club.

    Only out/public is distributed. out/investigator contains the private key.
    Empty preferences remain empty; this function produces no human results.
    """
    if type(listeners) is not int or not 1 <= listeners <= 1000:
        raise ValueError("listeners must be an integer in 1..1000")
    topics = _topics(runs)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new blind directory; existing materials are preserved")
    if Path(runs).resolve() == out or Path(runs).resolve() in out.parents:
        raise ValueError("Blind output must be outside the comparison runs")
    pairs = []
    for topic in topics:
        report_path = topic / "levels.json"
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        if report.get("status") != "passed":
            raise ValueError("Topic has not passed loudness matching")
        pair = {}
        for condition in ("legacy", "stage_club"):
            row = report["conditions"][condition]
            source, matched = topic / condition / "audio.wav", topic / condition / "matched.wav"
            if _sha(source) != row["original"]["audio_sha256"] or _sha(matched) != row["matched"]["audio_sha256"]:
                raise ValueError("Waveform changed after matching")
            pair[condition] = {"path": matched, "sha256": _sha(matched)}
        pairs.append({"topic": topic.name, "files": pair, "levels_sha256": _sha(report_path)})
    rng = random.Random(seed)
    public, investigator = out / "public", out / "investigator"
    (public / "audio").mkdir(parents=True)
    investigator.mkdir()
    identifiers = list(range(1, len(pairs) * 2 + 1))
    rng.shuffle(identifiers)
    for pair in pairs:
        for file in pair["files"].values():
            name = f"clip-{identifiers.pop():03d}.wav"
            shutil.copyfile(file["path"], public / "audio" / name)
            if _sha(public / "audio" / name) != file["sha256"]:
                raise ValueError("Blind copy hash mismatch")
            file["anonymous_audio"] = "../../audio/" + name
    orientation = []
    for _ in pairs:
        sides = [i % 2 == 0 for i in range(listeners)]
        rng.shuffle(sides)
        orientation.append(sides)
    private_rows = []
    fields = ["listener_id", "item_id", "audio_a", "audio_b", *DIMENSIONS, "comment"]
    instructions = (
        "# 同稿声音对照\n\n请戴耳机，在安静环境独立试听 A、B。两段文字相同，可重听，"
        "请不要查看身份映射或与其他听者讨论。播放器音量设定保持一致。\n\n"
        "在 ratings.csv 中逐题填写 A、B、tie（接近）或 unsure（无法判断）；初值全为空白。\n\n"
        "- stage_like：哪条更像面对观众进行脱口秀表演？本项为主要比较。\n"
        "- audience_address：哪条更像在向观众讲述、留回应空间？\n"
        "- naturalness：哪条更连贯自然，较少机械或做作感？\n"
        "- voice_consistency：哪条更像同一说话者，较少突然变成另一种音色？\n\n"
        "四项分别判断，不需要选同一条；更大声不自动等于更好。备注可记录突变时间。"
        "可以留空，不允许由工具代填。请只交回自己的真实评分。\n\n"
        "音频在 ../../audio/，保留完整 public 目录。本表比较表达，不据重复听同一包袱"
        "推断更好笑；若另行评价趣味，应记录熟悉情况并主要分析首次接触的版本。\n"
    )
    for listener in range(listeners):
        listener_id = f"listener-{listener + 1:02d}"
        folder = public / "listeners" / listener_id
        folder.mkdir(parents=True)
        order = list(range(len(pairs)))
        rng.shuffle(order)
        with (folder / "ratings.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index in order:
                pair = pairs[index]
                a, b = ("legacy", "stage_club") if orientation[index][listener] else ("stage_club", "legacy")
                item_id = f"item-{index + 1:02d}"
                writer.writerow({"listener_id": listener_id, "item_id": item_id,
                                 "audio_a": pair["files"][a]["anonymous_audio"],
                                 "audio_b": pair["files"][b]["anonymous_audio"],
                                 **{name: "" for name in (*DIMENSIONS, "comment")}})
                private_rows.append({"listener_id": listener_id, "item_id": item_id,
                                     "topic": pair["topic"], "A": a, "B": b,
                                     "audio_a_sha256": pair["files"][a]["sha256"],
                                     "audio_b_sha256": pair["files"][b]["sha256"],
                                     "levels_sha256": pair["levels_sha256"]})
        (folder / "README.md").write_text(instructions, encoding="utf-8")
    (public / "README.md").write_text(
        "# 听评材料\n\n每位听者打开 listeners 下自己的目录，按 README 试听并填写空白 CSV。"
        "只分发本 public 文件夹。不要查看研究者映射。四维分开，允许接近、无法判断和留空。\n",
        encoding="utf-8")
    _json(investigator / "answer_key.json", {"seed": seed, "primary_endpoint": "stage_like",
          "comparison": ["legacy", "stage_club"], "rows": private_rows,
          "analysis_note": "No scores collected. Aggregate per listener across fixed topics; repeated ratings/frames are not independent observations. Funniness needs first-exposure control."})
    summary = {"status": "blank_materials_ready", "topics": len(pairs), "listeners": listeners,
               "pairs_per_listener": len(pairs), "dimensions": list(DIMENSIONS),
               "primary_endpoint": "stage_like", "all_preferences_initially_blank": True,
               "human_ratings_collected": 0, "model_scores_computed": False,
               "distribute_only": str(public), "private_key": str(investigator / "answer_key.json"),
               "balanced_A_B_per_topic_difference_at_most_one": True}
    _json(investigator / "pack_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blind-out", type=Path)
    parser.add_argument("--listeners", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    report = {"schema": "openmic.stage-acoustic-analysis/1", "topics": [], "errors": [],
              "quality_improvement_established": False, "human_ratings_collected": 0}
    try:
        topics = _topics(args.runs)
        for topic in topics:
            try:
                report["topics"].append(match_topic(topic))
            except (OSError, ValueError, ImportError, KeyError) as exc:
                report["errors"].append({"topic": topic.name, "type": type(exc).__name__, "message": str(exc)})
        if not report["errors"] and args.blind_out:
            report["blind"] = build_blind(args.runs, args.blind_out, args.listeners, args.seed)
    except (OSError, ValueError, ImportError, KeyError) as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    report["status"] = "passed" if not report["errors"] else "failed"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _json(args.out, report)
    print(json.dumps({"status": report["status"], "topics_analyzed": len(report["topics"]),
                      "errors": report["errors"], "human_ratings_collected": 0}, ensure_ascii=False))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
