"""Human expressive-label regression using fresh story holdouts and complete waveforms."""

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_data import story_splits
from graded_text import read_lines, write_json, write_lines, metrics
from omni_judge import assert_idle_gpu
from v3_text import digest, embed_texts
from openmic.evaluation_audio import (
    load_waveform,
    waveform_descriptors,
    load_audio_encoder,
    audio_embeddings,
    DESCRIPTORS,
)

METHODS = ["waveform", "audio_embedding", "text_embedding", "audio_and_text"]


def fresh_story_splits(stories):
    old = story_splits(stories)
    fresh = sorted(
        (s for s in old if old[s] == "train"),
        key=lambda s: hashlib.sha256(("audio-v3-" + s).encode()).hexdigest(),
    )
    a = int(0.7 * len(fresh))
    b = int(0.85 * len(fresh))
    return {
        s: ("train" if i < a else "validation" if i < b else "confirmation")
        for i, s in enumerate(fresh)
    }


def prepare(args):
    import soundfile as sf

    args.output.mkdir(parents=True, exist_ok=False)
    files = list(args.root.rglob("story_infos.json"))
    if len(files) != 1:
        raise ValueError("Expected one story_infos.json")
    source = files[0]
    base = source.parent
    data = json.loads(source.read_text())
    assignments = fresh_story_splits([entry["story_id"] for entry in data.values()])
    rows = []
    excluded = Counter()
    for file_id, entry in sorted(data.items()):
        story = entry["story_id"]
        if story not in assignments:
            continue
        if entry.get("error") not in (None, False, "", "无", "none", "None"):
            excluded["source_error_files"] += 1
            continue
        for i, seg in enumerate(entry["segment_infos"]["sentence_segments"]):
            text = seg.get("content", "").strip()
            values = seg.get("score", [])
            if isinstance(values, (int, float)):
                values = [values]
            if (
                not text
                or not values
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    or not 0 <= v <= 5
                    for v in values
                )
            ):
                excluded["missing_label_or_text"] += 1
                continue
            path = (base / seg["audio_path"]).resolve()
            if base.resolve() not in path.parents or not path.is_file():
                raise ValueError("Missing or unsafe waveform")
            info = sf.info(path)
            if not 3 <= info.duration <= 20:
                excluded["outside_3_20_seconds"] += 1
                continue
            ident = hashlib.sha256(
                (str(file_id) + "/" + str(i) + "/" + seg["audio_path"]).encode()
            ).hexdigest()[:20]
            rows.append(
                {
                    "sample_id": ident,
                    "group": story,
                    "split": assignments[story],
                    "text": text,
                    "audio_path": str(path),
                    "audio_sha256": digest(path),
                    "duration_seconds": info.duration,
                    "target": sum(values) / len(values),
                }
            )
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate audio identity")
    write_lines(
        args.output / "inputs.private.jsonl",
        [{k: v for k, v in r.items() if k != "target"} for r in rows],
    )
    for split in ("train", "validation", "confirmation"):
        write_lines(
            args.output / (split + ".private.jsonl"),
            [r for r in rows if r["split"] == split],
        )
    write_json(
        args.output / "preparation.json",
        {
            "metadata_sha256": digest(source),
            "plan_sha256": digest(args.plan),
            "counts": dict(Counter(r["split"] for r in rows)),
            "story_counts": {
                s: len({r["group"] for r in rows if r["split"] == s})
                for s in ("train", "validation", "confirmation")
            },
            "exclusions": dict(excluded),
            "retired_all_v2_validation_and_test_stories": True,
            "split_hashes": {
                s: digest(args.output / (s + ".private.jsonl"))
                for s in ("train", "validation", "confirmation")
            },
            "inputs_sha256": digest(args.output / "inputs.private.jsonl"),
            "target": "human expressive appropriateness 0..5, not humor or intelligibility",
        },
    )


def features(args):
    import torch
    from transformers import Qwen2_5OmniProcessor

    assert_idle_gpu(args.gpu_uuid)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = read_lines(args.data / "inputs.private.jsonl")
    vectors = (
        np.zeros((len(rows), 0), dtype=np.float32)
        if args.skip_text_embeddings
        else embed_texts([r["text"] for r in rows], args.embedding_model)
    )
    encoder = load_audio_encoder(args.model)
    extractor = Qwen2_5OmniProcessor.from_pretrained(
        args.model, local_files_only=True
    ).feature_extractor
    encoder.save_pretrained(args.output / "audio_encoder", safe_serialization=True)
    extractor.save_pretrained(args.output / "audio_extractor")
    acoustic = []
    embedding = []
    coverage = []
    for start in range(0, len(rows), 8):
        if time.monotonic() - started > args.seconds:
            raise TimeoutError("Feature extraction budget exhausted")
        batch = rows[start : start + 8]
        waves = []
        infos = []
        for row in batch:
            if digest(row["audio_path"]) != row["audio_sha256"]:
                raise ValueError("Waveform changed")
            wave, info = load_waveform(row["audio_path"])
            waves.append(wave)
            infos.append(info)
            acoustic.append(waveform_descriptors(wave))
        emb, frames = audio_embeddings(waves, extractor, encoder)
        embedding.extend(emb)
        for row, info, n in zip(batch, infos, frames):
            coverage.append(
                {"sample_id": row["sample_id"], **info, "audio_feature_frames": n}
            )
        if start % 80 == 0:
            print(
                json.dumps({"complete": start + len(batch), "total": len(rows)}),
                flush=True,
            )
    np.savez(
        args.output / "features.private.npz",
        waveform=np.array(acoustic),
        audio_embedding=np.array(embedding),
        text_embedding=vectors,
    )
    write_lines(args.output / "coverage.private.jsonl", coverage)
    write_json(
        args.output / "features.json",
        {
            "input_sha256": digest(args.data / "inputs.private.jsonl"),
            "features_sha256": digest(args.output / "features.private.npz"),
            "n": len(rows),
            "dimensions": {
                "waveform": len(DESCRIPTORS),
                "audio_embedding": len(embedding[0]),
                "text_embedding": vectors.shape[1],
            },
            "complete_waveform_decode_fraction_min": min(
                c["decoded_fraction"] for c in coverage
            ),
            "coverage_checks_passed": len(coverage),
            "frozen_audio_encoder": "Qwen/Qwen2.5-Omni-7B@ae9e1690543ffd5c0221dc27f79834d0294cba00 audio_tower",
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def matrices(args):
    rows = read_lines(args.data / "inputs.private.jsonl")
    f = np.load(args.output / "features.private.npz")
    x = {k: f[k] for k in ["waveform", "audio_embedding", "text_embedding"]}
    x["audio_and_text"] = np.concatenate(
        [x["audio_embedding"], x["text_embedding"]], axis=1
    )
    if not all(len(v) == len(rows) and np.isfinite(v).all() for v in x.values()):
        raise ValueError("Invalid feature matrix")
    return rows, x


def select(args):
    rows, x = matrices(args)
    train = read_lines(args.data / "train.private.jsonl")
    dev = read_lines(args.data / "validation.private.jsonl")
    refs = {r["sample_id"]: r["target"] for r in train + dev}
    indices = {
        s: np.array([i for i, r in enumerate(rows) if r["split"] == s])
        for s in ("train", "validation")
    }
    y = {
        s: np.array([refs[rows[i]["sample_id"]] for i in ix])
        for s, ix in indices.items()
    }
    candidates = []
    for method in METHODS:
        for alpha in [1, 10, 100, 1000]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[method][indices["train"]], y["train"])
            p = np.clip(model.predict(x[method][indices["validation"]]), 0, 5)
            m = metrics(y["validation"], p)
            candidates.append({"method": method, "alpha": alpha, "validation": m})
            joblib.dump(model, args.output / (method + f"-{alpha}.joblib"))
    key = lambda c: (
        c["validation"]["spearman"] if c["validation"]["spearman"] is not None else -2,
        -c["validation"]["mae"],
    )
    per_method = {
        m: max([c for c in candidates if c["method"] == m], key=key) for m in METHODS
    }
    chosen = max(per_method.values(), key=key)
    write_json(
        args.output / "selection.lock.json",
        {
            "chosen": chosen,
            "per_method": per_method,
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
    if (
        lock["plan_sha256"] != digest(args.plan)
        or lock["preparation_sha256"] != digest(args.data / "preparation.json")
        or lock["features_sha256"] != digest(args.output / "features.private.npz")
    ):
        raise ValueError("Run changed after selection")
    rows, x = matrices(args)
    refs = read_lines(args.data / "confirmation.private.jsonl")
    prep = json.loads((args.data / "preparation.json").read_text())
    if (
        digest(args.data / "confirmation.private.jsonl")
        != prep["split_hashes"]["confirmation"]
    ):
        raise ValueError("Confirmation changed")
    lookup = {r["sample_id"]: r["target"] for r in refs}
    ix = np.array([i for i, r in enumerate(rows) if r["split"] == "confirmation"])
    y = np.array([lookup[rows[i]["sample_id"]] for i in ix])
    groups = np.array([rows[i]["group"] for i in ix])
    predictions = {}
    for method, c in lock["per_method"].items():
        model = joblib.load(args.output / (method + f'-{c["alpha"]}.joblib'))
        predictions[method] = np.clip(model.predict(x[method][ix]), 0, 5)
    predictions["train_mean"] = np.repeat(lock["train_mean"], len(y))
    results = {m: metrics(y, p) for m, p in predictions.items()}
    rng = np.random.default_rng(20261026)
    unique = sorted(set(groups))
    buckets = {g: np.flatnonzero(groups == g) for g in unique}
    chosen = lock["chosen"]["method"]
    boot = {m: [] for m in predictions}
    delta = []
    for _ in range(5000):
        j = np.concatenate(
            [buckets[g] for g in rng.choice(unique, len(unique), replace=True)]
        )
        current = {}
        for m, p in predictions.items():
            if len(set(p[j])) > 1 and len(set(y[j])) > 1:
                current[m] = float(spearmanr(y[j], p[j]).statistic)
                boot[m].append(current[m])
        if chosen in current and "text_embedding" in current:
            delta.append(current[chosen] - current["text_embedding"])
    for m, values in boot.items():
        results[m]["spearman_ci95"] = (
            np.quantile(values, [0.025, 0.975]).tolist() if values else None
        )
    result = {
        "selected": chosen,
        "n": len(y),
        "stories": len(unique),
        "results": results,
        "selected_minus_text_spearman_ci95": (
            np.quantile(delta, [0.025, 0.975]).tolist() if delta else None
        ),
        "bootstrap_unit": "story",
        "resamples": 5000,
        "selection_lock_sha256": digest(args.output / "selection.lock.json"),
        "project_humor_alignment": False,
        "intelligibility_human_validation": False,
        "meaning": "new story confirmation of human expressive appropriateness; audiobook domain only",
    }
    write_json(args.output / "confirmation.json", result)
    write_lines(
        args.output / "confirmation_predictions.private.jsonl",
        [
            {
                "sample_id": rows[i]["sample_id"],
                "group": rows[i]["group"],
                "target": float(target),
                "predictions": {m: float(p[j]) for m, p in predictions.items()},
            }
            for j, (i, target) in enumerate(zip(ix, y))
        ],
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "features", "select", "report"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--embedding-model", type=Path)
    parser.add_argument(
        "--skip-text-embeddings",
        action="store_true",
        help="No transcript features for datasets without grounded human transcripts",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    globals()[args.stage](args)
