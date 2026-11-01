"""TMHINT-QI human intelligibility with content-group holdouts, not quality MOS."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from graded_text import read_lines, write_json, write_lines, metrics
from v3_text import digest

METHODS = ["waveform", "audio_embedding", "audio_and_waveform"]
EXPECTED_SHA256 = "cda6581c9fb0ea634b8bac4c6503e772feb080629a5d95eb84f3d1888876912b"


def content_id(utterance):
    match = re.fullmatch(r"TMHINT_[gb]\d+_(\d+)_(\d+)", utterance)
    if not match:
        raise ValueError("Unknown sentence identity: " + utterance)
    # The same text read by different speakers and all enhanced/noise variants
    # stay together. This is more conservative than speaker-specific IDs.
    return match.group(1) + "_" + match.group(2)


def prepare(args):
    import soundfile as sf

    args.output.mkdir(parents=True, exist_ok=False)
    if args.archive:
        if (
            args.archive.stat().st_size != 2278236226
            or digest(args.archive) != EXPECTED_SHA256
        ):
            raise ValueError("TMHINT-QI archive hash/size mismatch")
        args.root.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(args.archive) as source:
            if sum(i.file_size for i in source.infolist()) > 4 * 1024**3:
                raise ValueError("Unexpected expanded archive size")
            for info in source.infolist():
                if info.filename.startswith("__MACOSX/") or Path(
                    info.filename
                ).name.startswith("."):
                    continue
                path = (args.root / info.filename).resolve()
                if args.root.resolve() not in path.parents:
                    raise ValueError("Unsafe ZIP path")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("ZIP symlinks are not supported")
                if info.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                if path.suffix.lower() not in (".wav", ".csv", ".txt"):
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                # ZipExtFile verifies CRC on complete read.
                with source.open(info) as src, path.open("xb") as dest:
                    for chunk in iter(lambda: src.read(1024 * 1024), b""):
                        dest.write(chunk)
    csvs = [p for p in args.root.rglob("raw_data.csv") if "__MACOSX" not in p.parts]
    if len(csvs) != 1:
        raise ValueError("Expected one raw_data.csv")
    annotations = list(csv.DictReader(csvs[0].open(encoding="utf-8-sig")))
    files = defaultdict(list)
    for path in args.root.rglob("*.wav"):
        files[path.stem].append(path)
    ratings = defaultdict(list)
    metadata = {}
    exclusions = Counter()
    for row in annotations:
        # The quality_score column deliberately never enters the regression target.
        try:
            score = float(row["intelligibility_score"])
        except (ValueError, TypeError):
            exclusions["invalid_intelligibility_score"] += 1
            continue
        if not np.isfinite(score) or not 0 <= score <= 10:
            exclusions["invalid_intelligibility_score"] += 1
            continue
        filename = Path(row["file_name"]).stem
        if filename not in files:
            exclusions["rating_without_released_waveform"] += 1
            continue
        if len(files[filename]) != 1:
            raise ValueError("Duplicate file identity: audit source")
        group = content_id(row["uttr"])
        if filename in metadata and metadata[filename]["group"] != group:
            raise ValueError("Conflicting content identity")
        metadata[filename] = {
            "group": group,
            "utterance": row["uttr"],
            "speaker": row["uttr"].split("_")[1],
        }
        ratings[filename].append((str(row["idx"]), score))
    ordered = sorted(
        {r["group"] for r in metadata.values()},
        key=lambda s: hashlib.sha256(("tmhint-v3-" + s).encode()).hexdigest(),
    )
    a = int(0.7 * len(ordered))
    b = int(0.85 * len(ordered))
    assignments = {
        g: ("train" if i < a else "validation" if i < b else "confirmation")
        for i, g in enumerate(ordered)
    }
    records = []
    for filename, scores in sorted(ratings.items()):
        by_rater = defaultdict(list)
        for who, value in scores:
            by_rater[who].append(value)
        exclusions["repeated_subject_rows_averaged_within_rater"] += len(scores) - len(
            by_rater
        )
        rater_scores = [float(np.mean(v)) for v in by_rater.values()]
        path = files[filename][0].resolve()
        info = sf.info(path)
        meta = metadata[filename]
        if not 0 < info.duration <= 30:
            raise ValueError("Unexpected complete utterance length")
        records.append(
            {
                "sample_id": hashlib.sha256(filename.encode()).hexdigest()[:20],
                "group": meta["group"],
                "split": assignments[meta["group"]],
                "audio_path": str(path),
                "audio_sha256": digest(path),
                "duration_seconds": info.duration,
                "text": "",
                "target": float(np.mean(rater_scores)),
                "reference_count": len(rater_scores),
                "speaker": meta["speaker"],
                "original_partition": path.parent.name,
            }
        )
    write_lines(
        args.output / "inputs.private.jsonl",
        [
            {k: v for k, v in r.items() if k not in ("target", "reference_count")}
            for r in records
        ],
    )
    for split in ("train", "validation", "confirmation"):
        write_lines(
            args.output / (split + ".private.jsonl"),
            [r for r in records if r["split"] == split],
        )
    original = defaultdict(set)
    for r in records:
        original[r["original_partition"]].add(r["group"])
    pairs = [
        len(original[a] & original[b])
        for i, a in enumerate(sorted(original))
        for b in sorted(original)[i + 1 :]
    ]
    report = {
        "source": "author public TMHINT-QI release",
        "archive_sha256": EXPECTED_SHA256,
        "archive_size": 2278236226,
        "raw_csv_sha256": digest(csvs[0]),
        "raw_annotation_rows": len(annotations),
        "waveforms": len(records),
        "reference_counts": dict(Counter(r["reference_count"] for r in records)),
        "unique_content_groups": len(ordered),
        "counts": dict(Counter(r["split"] for r in records)),
        "groups_by_split": {
            s: len({r["group"] for r in records if r["split"] == s})
            for s in ("train", "validation", "confirmation")
        },
        "speakers_by_split": {
            s: sorted({r["speaker"] for r in records if r["split"] == s})
            for s in ("train", "validation", "confirmation")
        },
        "original_partition_content_overlaps": pairs,
        "exclusions": dict(exclusions),
        "labels": "human intelligibility_score 0..10, NOT quality_score",
        "text_only_control": "unavailable: released package has sentence IDs but no grounded transcripts; do not invent them",
        "plan_sha256": digest(args.plan),
        "inputs_sha256": digest(args.output / "inputs.private.jsonl"),
        "split_hashes": {
            s: digest(args.output / (s + ".private.jsonl"))
            for s in ("train", "validation", "confirmation")
        },
    }
    write_json(args.output / "preparation.json", report)
    print(json.dumps(report), flush=True)


def matrices(args):
    rows = read_lines(args.data / "inputs.private.jsonl")
    f = np.load(args.output / "features.private.npz")
    # High-dimensional standardized embeddings can be ill-conditioned in float32.
    # This changes numerical precision only, not alpha, features, split, or target.
    x = {k: f[k].astype(np.float64) for k in ("waveform", "audio_embedding")}
    x["audio_and_waveform"] = np.concatenate(
        [x["audio_embedding"], x["waveform"]], axis=1
    )
    if not all(len(v) == len(rows) and np.isfinite(v).all() for v in x.values()):
        raise ValueError("Invalid features")
    return rows, x


def select(args):
    rows, x = matrices(args)
    fitting = read_lines(args.data / "train.private.jsonl") + read_lines(
        args.data / "validation.private.jsonl"
    )
    labels = {r["sample_id"]: r["target"] for r in fitting}
    ix = {
        s: np.array([i for i, r in enumerate(rows) if r["split"] == s])
        for s in ("train", "validation")
    }
    y = {s: np.array([labels[rows[i]["sample_id"]] for i in j]) for s, j in ix.items()}
    candidates = []
    for method in METHODS:
        for alpha in [1, 10, 100, 1000]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[method][ix["train"]], y["train"])
            p = np.clip(model.predict(x[method][ix["validation"]]), 0, 10)
            candidate = {
                "method": method,
                "alpha": alpha,
                "validation": metrics(y["validation"], p),
            }
            candidates.append(candidate)
            joblib.dump(model, args.output / (method + f"-{alpha}.joblib"))
    key = lambda c: (
        c["validation"]["spearman"] if c["validation"]["spearman"] is not None else -2,
        -c["validation"]["mae"],
    )
    fixed = {
        m: max((c for c in candidates if c["method"] == m), key=key) for m in METHODS
    }
    chosen = max(fixed.values(), key=key)
    write_json(
        args.output / "selection.lock.json",
        {
            "chosen": chosen,
            "per_method": fixed,
            "candidates": candidates,
            "train_mean": float(y["train"].mean()),
            "plan_sha256": digest(args.plan),
            "preparation_sha256": digest(args.data / "preparation.json"),
            "features_sha256": digest(args.output / "features.private.npz"),
            "selection_read_confirmation_labels": False,
        },
    )


def report(args):
    if (args.output / "confirmation.json").exists():
        raise ValueError("Confirmation already reported")
    lock = json.loads((args.output / "selection.lock.json").read_text())
    prep = json.loads((args.data / "preparation.json").read_text())
    if (
        lock["plan_sha256"] != digest(args.plan)
        or lock["preparation_sha256"] != digest(args.data / "preparation.json")
        or lock["features_sha256"] != digest(args.output / "features.private.npz")
    ):
        raise ValueError("Frozen configuration changed")
    if (
        digest(args.data / "confirmation.private.jsonl")
        != prep["split_hashes"]["confirmation"]
    ):
        raise ValueError("Confirmation changed")
    rows, x = matrices(args)
    refs = {
        r["sample_id"]: r["target"]
        for r in read_lines(args.data / "confirmation.private.jsonl")
    }
    ix = np.array([i for i, r in enumerate(rows) if r["split"] == "confirmation"])
    y = np.array([refs[rows[i]["sample_id"]] for i in ix])
    groups = np.array([rows[i]["group"] for i in ix])
    predictions = {}
    for method, c in lock["per_method"].items():
        model = joblib.load(args.output / (method + f'-{c["alpha"]}.joblib'))
        predictions[method] = np.clip(model.predict(x[method][ix]), 0, 10)
    predictions["train_mean"] = np.repeat(lock["train_mean"], len(y))
    results = {m: metrics(y, p) for m, p in predictions.items()}
    boot = {m: [] for m in predictions}
    deltas = []
    unique = sorted(set(groups))
    buckets = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(20261026)
    chosen = lock["chosen"]["method"]
    for _ in range(5000):
        j = np.concatenate(
            [buckets[g] for g in rng.choice(unique, len(unique), replace=True)]
        )
        current = {}
        for m, p in predictions.items():
            if len(set(p[j])) > 1 and len(set(y[j])) > 1:
                current[m] = float(spearmanr(y[j], p[j]).statistic)
                boot[m].append(current[m])
        if chosen in current and "waveform" in current:
            deltas.append(current[chosen] - current["waveform"])
    for m, v in boot.items():
        results[m]["spearman_ci95"] = (
            np.quantile(v, [0.025, 0.975]).tolist() if v else None
        )
    result = {
        "selected": chosen,
        "n": len(y),
        "content_groups": len(unique),
        "scale": [0, 10],
        "results": results,
        "selected_minus_waveform_spearman_ci95": (
            np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else None
        ),
        "bootstrap_unit": "underlying sentence, all noisy/enhanced/speaker variants together",
        "resamples": 5000,
        "selection_lock_sha256": digest(args.output / "selection.lock.json"),
        "human_intelligibility_label": True,
        "full_work_intelligibility_human_validation": False,
        "text_control": prep["text_only_control"],
        "meaning": "Chinese short noisy/enhanced sentence intelligibility, not stand-up humor, long-work clarity, or quality MOS",
    }
    write_json(args.output / "confirmation.json", result)
    write_lines(
        args.output / "confirmation_predictions.private.jsonl",
        [
            {
                "sample_id": rows[i]["sample_id"],
                "group": rows[i]["group"],
                "target": float(t),
                "predictions": {m: float(p[j]) for m, p in predictions.items()},
            }
            for j, (i, t) in enumerate(zip(ix, y))
        ],
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "select", "report"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args)
