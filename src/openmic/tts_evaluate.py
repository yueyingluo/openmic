"""Package anonymous listening trials and summarize real human ratings.

Only Python's standard library is needed. Naturalness is a project-specific
1--5 subjective MOS-style score; it is not an automatic objective MOS metric.
"""

import argparse
import csv
import json
import random
import re
import shutil
import statistics
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


DIMENSIONS = ("humor", "naturalness", "pause", "emphasis", "emotion")
RATING_COLUMNS = ("participant_id", "sample_id", "order", "topic", *DIMENSIONS, "comment")
KEY_COLUMNS = ("sample_id", "topic", "condition", "backend", "synthetic",
               "in_target_duration", "spoken_text_sha256", "plan_sha256", "source_file")
NOTE = ("Naturalness is a project-specific subjective 1-5 MOS-style listening score, "
        "not an automatically estimated objective MOS. Sample variance uses ddof=1; "
        "it is null when n<2. Ratings are human observations, not generated results.")


def _write_csv(path: Path, columns, rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as handle:
            if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth(),
                    handle.getcomptype()) != (16000, 1, 2, "NONE"):
                raise ValueError(f"{path}: 盲评要求 16 kHz、单声道、16-bit PCM WAV")
            count = handle.getnframes()
            if count == 0 or len(handle.readframes(count)) != count * 2:
                raise ValueError(f"{path}: 音频为空或数据截断")
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{path}: 无效 WAV") from exc


def pack(runs: Path, out: Path, *, listeners: int = 8, seed: int = 42,
         allow_synthetic: bool = False) -> Dict[str, Any]:
    """Copy opaque WAV names; keep condition mapping outside the listener folder."""
    runs, out = Path(runs).resolve(), Path(out).resolve()
    if not runs.is_dir():
        raise ValueError(f"运行目录不存在: {runs}")
    if not 1 <= listeners <= 1000:
        raise ValueError("listeners 必须是 1--1000 的整数")
    if out == runs or runs in out.parents:
        raise ValueError("盲评输出目录必须位于 runs 目录之外，避免重复收集")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("输出目录非空，请选择一个新的盲评目录")

    samples: List[Dict[str, Any]] = []
    for metrics_path in sorted(runs.rglob("metrics.json")):
        audio_path = metrics_path.with_name("audio.wav")
        if not audio_path.is_file():
            raise ValueError(f"缺少配对音频: {audio_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        condition = metrics.get("condition")
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(f"{metrics_path}: 缺少 condition")
        condition = condition.strip()
        if not isinstance(metrics.get("synthetic"), bool):
            raise ValueError(f"{metrics_path}: synthetic 必须明确为 true/false")
        if "backend" not in metrics or not metrics["backend"]:
            raise ValueError(f"{metrics_path}: 缺少 backend")
        if metrics["synthetic"] and not allow_synthetic:
            raise ValueError("包含 synthetic 测试音频，不能用于人工实评；练习时显式加 --allow-synthetic")
        _validate_wav(audio_path)
        relative = metrics_path.parent.relative_to(runs)
        topic = relative.parts[0] if len(relative.parts) >= 2 else "default_topic"
        backend = metrics["backend"]
        if not isinstance(backend, str):
            backend = json.dumps(backend, ensure_ascii=False, sort_keys=True)
        text_hash = metrics.get("spoken_text_sha256")
        if text_hash is not None and not isinstance(text_hash, str):
            raise ValueError(f"{metrics_path}: spoken_text_sha256 必须是字符串")
        text_hash = (text_hash or "").strip()
        samples.append({"topic": topic, "condition": condition,
                        "backend": backend, "synthetic": metrics["synthetic"],
                        "in_target_duration": metrics.get("in_target_duration"),
                        "spoken_text_sha256": text_hash,
                        "plan_sha256": metrics.get("plan_sha256", ""),
                        "source_file": str(audio_path)})
    if not samples:
        raise ValueError("runs 下未找到 metrics.json/audio.wav 结果对")

    # Validate before writing anything. Independent reruns or changed voices/text
    # must not silently become a nominally matched listening comparison.
    topic_groups = defaultdict(list)
    seen_conditions = set()
    for row in samples:
        pair = row["topic"], row["condition"]
        if pair in seen_conditions:
            raise ValueError(f"重复 topic/condition {pair}，请将不同运行分开打包")
        seen_conditions.add(pair)
        topic_groups[row["topic"]].append(row)
    topic_validation = {}
    for topic, rows in topic_groups.items():
        if len({row["backend"] for row in rows}) > 1:
            raise ValueError(f"主题 {topic!r} 的 backend identity 不同，不能混合不同模型/音色做对照")
        available_hashes = {row["spoken_text_sha256"] for row in rows if row["spoken_text_sha256"]}
        if len(available_hashes) > 1:
            raise ValueError(f"主题 {topic!r} 的 spoken_text_sha256 冲突，不是同稿对照")
        missing = sum(not row["spoken_text_sha256"] for row in rows)
        topic_validation[topic] = {
            "conditions_count": len(rows), "missing_text_hash_count": missing,
            "same_text_verified": len(rows) >= 2 and missing == 0,
        }

    rng = random.Random(seed)
    rng.shuffle(samples)  # sample IDs themselves must not encode condition order.
    for index, row in enumerate(samples, 1):
        row["sample_id"] = f"sample{index:03d}"
    participant_dir = out / "participant_package"
    investigator_dir = out / "investigator"
    audio_dir = participant_dir / "audio"
    audio_dir.mkdir(parents=True)
    investigator_dir.mkdir(parents=True)
    for row in samples:
        shutil.copyfile(row["source_file"], audio_dir / (row["sample_id"] + ".wav"))
    _write_csv(investigator_dir / "key.csv", KEY_COLUMNS, samples)

    # A randomized cyclic order and its reverse balance serial position over
    # listeners without assigning a meaningful fixed ordering to sample IDs.
    base = list(samples)
    rng.shuffle(base)
    for listener in range(listeners):
        offset = (listener // 2) % len(base)
        ordered = base[offset:] + base[:offset]
        if listener % 2:
            ordered = list(reversed(ordered))
        participant_id = f"p{listener + 1:03d}"
        rating_rows, playlist_rows = [], []
        for position, row in enumerate(ordered, 1):
            public = {"participant_id": participant_id,
                      "sample_id": row["sample_id"], "order": position,
                      "topic": row["topic"]}
            rating_rows.append({**public, **{name: "" for name in DIMENSIONS}, "comment": ""})
            playlist_rows.append({**public, "audio_file": "audio/" + row["sample_id"] + ".wav"})
        _write_csv(participant_dir / (participant_id + "_ratings.csv"), RATING_COLUMNS, rating_rows)
        _write_csv(participant_dir / (participant_id + "_playlist.csv"),
                   ("participant_id", "sample_id", "order", "topic", "audio_file"), playlist_rows)
    practice = any(row["synthetic"] for row in samples)
    manifest = {"seed": seed, "listeners": listeners, "samples": len(samples),
                "synthetic_practice_only": practice,
                "missing_text_hash_count": sum(not row["spoken_text_sha256"] for row in samples),
                "topic_match_validation": topic_validation,
                "text_hash_note": "Missing hashes are accepted for legacy data, but same-text matching is then unverified.",
                "not_in_target_duration_count": sum(row["in_target_duration"] is not True for row in samples),
                "duration_note": "Non-target-duration clips are valid short trials, not 3-5 minute course deliverables.",
                "key_file": "investigator/key.csv", "rating_note": NOTE}
    _json_write(investigator_dir / "manifest.json", manifest)
    # This public manifest deliberately contains no conditions or source paths.
    _json_write(participant_dir / "manifest.json",
                {"samples": len(samples), "listeners": listeners,
                 "synthetic_practice_only": practice, "rating_note": NOTE})
    instructions = (
        "按本人匿名ID对应的 playlist.csv 顺序播放 audio/ 中的文件，填写同名 ratings.csv。\n"
        "不要查看 investigator 文件夹；组织者仅分发 participant_package。\n"
        "所有维度填写 1--5 的整数（1 最差、5 最好）；可留空未完成项，勿填估计结果。\n"
        "humor=幽默感，naturalness=语音自然度，pause=停顿合理性，"
        "emphasis=重音表达，emotion=情感表达。按听到的效果评分，不推测系统条件。\n"
        "不同条件使用同一文本，重复听取可能有熟悉效应；保持音量、设备、环境一致。\n"
        "自然度为本项目主观 MOS 形式评分，不是自动客观 MOS。\n"
    )
    if practice:
        instructions = "练习包：包含合成测试音，不能作为真实 TTS 人工评价结果。\n" + instructions
    (participant_dir / "README.txt").write_text(instructions, encoding="utf-8")
    return manifest


def _stats(values: List[int], listeners=None) -> Dict[str, Any]:
    result = {"n": len(values),
              "mean": statistics.mean(values) if values else None,
              "sample_variance": statistics.variance(values) if len(values) > 1 else None}
    if listeners is not None:
        result["listeners"] = len(set(listeners))
    return result


def summarize(ratings: Path, key: Path, out: Path) -> Dict[str, Any]:
    """Aggregate dimensions separately and pair differences within listener/topic."""
    ratings, key, out = Path(ratings), Path(key), Path(out)
    key_fields, key_rows = _read_csv(key)
    if not {"sample_id", "topic", "condition"}.issubset(key_fields):
        raise ValueError("key.csv 缺少 sample_id/topic/condition")
    lookup = {}
    for row in key_rows:
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in lookup or not row["condition"].strip():
            raise ValueError("key.csv 样本ID为空、重复或缺少 condition")
        lookup[sample_id] = row
    if not lookup:
        raise ValueError("key.csv 没有样本")
    paths = sorted(ratings.rglob("*.csv")) if ratings.is_dir() else [ratings]
    observations, seen = [], set()
    ignored_files = []
    required = {"participant_id", "sample_id", *DIMENSIONS}
    for path in paths:
        fields, rows = _read_csv(path)
        if not required.issubset(fields):
            if ratings.is_dir():
                ignored_files.append(str(path))
                continue
            raise ValueError(f"{path}: 缺少评分表字段 {sorted(required - set(fields))}")
        for line, row in enumerate(rows, 2):
            context = f"{path}:{line}"
            if None in row:
                raise ValueError(f"{context}: CSV 列数超出表头，请将含逗号的 comment 加引号")
            scores = {name: (row.get(name) or "").strip() for name in DIMENSIONS}
            if not any(scores.values()):
                continue  # Blank/template rows are not ratings.
            participant = (row.get("participant_id") or "").strip().lower()
            if not re.fullmatch(r"p[0-9]{3,}", participant) or int(participant[1:]) == 0:
                raise ValueError(f"{context}: 使用匿名 participant_id，例如 p001")
            sample = (row.get("sample_id") or "").strip()
            if sample not in lookup:
                raise ValueError(f"{context}: 未知 sample_id {sample!r}")
            pair = participant, sample
            if pair in seen:
                raise ValueError(f"{context}: 重复 participant/sample {pair}")
            seen.add(pair)
            parsed = {}
            for name, score in scores.items():
                if score:
                    if not re.fullmatch(r"[1-5]", score):
                        raise ValueError(f"{context}: {name} 必须为 1--5 的整数或留空")
                    parsed[name] = int(score)
            meta = lookup[sample]
            observations.append({"participant_id": participant, "sample_id": sample,
                                 "topic": meta["topic"], "condition": meta["condition"],
                                 "scores": parsed})
    if not observations:
        raise ValueError("没有已填写的有效评分；请先完成真实人工听评，不要使用自动填充数据")

    conditions = sorted({row["condition"] for row in lookup.values()})
    per_condition = {}
    summary_rows = []
    for condition in conditions:
        relevant = [row for row in observations if row["condition"] == condition]
        condition_stats = {"ratings_count": len(relevant),
                           "listeners": len({row["participant_id"] for row in relevant}),
                           "dimensions": {}}
        for dimension in DIMENSIONS:
            filled = [row for row in relevant if dimension in row["scores"]]
            stats = _stats([row["scores"][dimension] for row in filled],
                           [row["participant_id"] for row in filled])
            condition_stats["dimensions"][dimension] = stats
            summary_rows.append({"condition": condition, "dimension": dimension,
                                 "ratings_count": len(relevant), **stats})
        per_condition[condition] = condition_stats

    # Repeated topics/conditions are first averaged within each listener/topic.
    # Only topics with both conditions contribute; then each listener has ONE
    # difference, avoiding treating a listener's many scores as independent people.
    cells = defaultdict(list)
    for row in observations:
        for dimension, score in row["scores"].items():
            cells[(row["participant_id"], row["topic"], row["condition"], dimension)].append(score)
    participants = sorted({row["participant_id"] for row in observations})
    topics = sorted({row["topic"] for row in lookup.values()})
    paired, paired_rows = {}, []
    for dimension in DIMENSIONS:
        listener_differences = []
        topic_pairs = 0
        for participant in participants:
            differences = []
            for topic in topics:
                baseline = cells.get((participant, topic, "baseline", dimension), [])
                performance = cells.get((participant, topic, "performance", dimension), [])
                if baseline and performance:
                    differences.append(statistics.mean(performance) - statistics.mean(baseline))
            if differences:
                difference = statistics.mean(differences)
                listener_differences.append(difference)
                topic_pairs += len(differences)
                paired_rows.append({"participant_id": participant, "dimension": dimension,
                                    "paired_topics": len(differences),
                                    "performance_minus_baseline": difference})
        paired[dimension] = {**_stats(listener_differences),
                             "n_listeners": len(listener_differences),
                             "n_listener_topic_pairs": topic_pairs}
    synthetic = any(str(row.get("synthetic", "")).lower() == "true" for row in key_rows)
    report = {"rating_note": NOTE, "synthetic_practice_only": synthetic,
              "ratings_count": len(observations), "listeners": len(participants),
              "per_condition": per_condition, "paired_performance_minus_baseline": paired,
              "pairing_note": "Mean per matched topic within listener; then one mean difference per listener. "
                              "No confidence intervals/p-values are claimed. Missing dimensions use available observations.",
              "ignored_nonrating_csv_files": ignored_files}
    if out.suffix.lower() == ".json":
        out.parent.mkdir(parents=True, exist_ok=True)
        json_path, directory = out, out.parent / (out.stem + "_tables")
    else:
        directory, json_path = out, out / "summary.json"
    directory.mkdir(parents=True, exist_ok=True)
    _json_write(json_path, report)
    _write_csv(directory / "summary.csv",
               ("condition", "dimension", "ratings_count", "n", "mean", "sample_variance", "listeners"),
               summary_rows)
    _write_csv(directory / "paired_by_listener.csv",
               ("participant_id", "dimension", "paired_topics", "performance_minus_baseline"), paired_rows)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack_parser = commands.add_parser("pack", help="创建匿名盲评材料（答案单独存放）")
    pack_parser.add_argument("--runs", type=Path, required=True)
    pack_parser.add_argument("--out", type=Path, required=True)
    pack_parser.add_argument("--listeners", type=int, default=8)
    pack_parser.add_argument("--seed", type=int, default=42)
    pack_parser.add_argument("--allow-synthetic", action="store_true", help="仅练习允许 synthetic 测试音")
    summary_parser = commands.add_parser("summarize", help="汇总人工填写的 CSV")
    summary_parser.add_argument("--ratings", type=Path, required=True)
    summary_parser.add_argument("--key", type=Path, required=True)
    summary_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            result = pack(args.runs, args.out, listeners=args.listeners, seed=args.seed,
                          allow_synthetic=args.allow_synthetic)
            print(f"Created {result['samples']} anonymous samples for {result['listeners']} listeners: {args.out}")
            print("Only distribute participant_package; keep investigator/key.csv private.")
        else:
            result = summarize(args.ratings, args.key, args.out)
            print(f"Summarized {result['ratings_count']} ratings from {result['listeners']} listeners: {args.out}")
        if result.get("synthetic_practice_only"):
            print("PRACTICE ONLY: synthetic test signals are not TTS listening-study evidence.")
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
