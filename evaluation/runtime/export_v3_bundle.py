"""Export only the frozen selected models to a portable, local-only API bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "version": "openmic-evaluation-v3",
        "network_calls": False,
        "raw_data_included": False,
    }
    if args.text_run:
        lock = json.loads((args.text_run / "selection.lock.json").read_text())
        report = json.loads((args.text_run / "confirmation.json").read_text())
        shutil.copytree(args.text_run / "tokenizer", args.output / "text/tokenizer")
        kind = lock.get("model_type", "encoder")
        if kind == "qwen_lora":
            if not args.qwen_base:
                raise ValueError("--qwen-base is required for a LoRA bundle")
            shutil.copytree(args.qwen_base, args.output / "text/base")
        checkpoints = []
        for seed in lock["seeds"]:
            relative = f"text/seed-{seed}"
            shutil.copytree(
                args.text_run / lock["chosen"] / f"seed-{seed}", args.output / relative
            )
            checkpoints.append(relative)
        plan = json.loads(args.text_plan.read_text()) if args.text_plan else {}
        config["text"] = {
            "tokenizer": "text/tokenizer",
            "checkpoints": checkpoints,
            "model_id": lock["chosen"] + "-ensemble-17-29-43",
            "model_type": kind,
            "base_path": "text/base" if kind == "qwen_lora" else None,
            "max_tokens": lock.get("inference_max_tokens", 512),
            "training_max_observed_tokens": lock.get(
                "training_max_observed_tokens", 512
            ),
            "base_model": plan.get("text", {}).get(
                "initialization", "see frozen execution plan"
            ),
            "validation": {
                "task": "CMHL five-grade single-annotator labels",
                "confirmation": report["neural"],
                "uncertainty": report["uncertainty"],
                "independence_limit": report["independence_limit"],
                "inference_device": report.get("inference_device", "cuda"),
                "inference_precision": report.get(
                    "inference_precision", "bfloat16_autocast"
                ),
                "project_full_script_human_validation": False,
            },
            "selection_lock_sha256": sha(args.text_run / "selection.lock.json"),
        }
    if args.audio_run:
        lock = json.loads((args.audio_run / "selection.lock.json").read_text())
        report = json.loads((args.audio_run / "confirmation.json").read_text())
        method = lock["chosen"]["method"]
        alpha = lock["chosen"]["alpha"]
        (args.output / "audio").mkdir()
        shutil.copy2(
            args.audio_run / (method + f"-{alpha}.joblib"),
            args.output / "audio/regressor.joblib",
        )
        if method in ("audio_embedding", "audio_and_text"):
            for name in ("audio_encoder", "audio_extractor"):
                shutil.copytree(args.audio_run / name, args.output / "audio" / name)
        if method == "audio_and_text":
            if not args.embedding_model:
                raise ValueError("--embedding-model required for fusion model")
            shutil.copytree(
                args.embedding_model, args.output / "audio/embedding_encoder"
            )
        config["audio"] = {
            "method": method,
            "regressor": "audio/regressor.joblib",
            "encoder": "audio/audio_encoder",
            "extractor": "audio/audio_extractor",
            "embedding_encoder": "audio/embedding_encoder",
            "model_id": f"ceaeval-ridge-{method}-{alpha}",
            "validation": {
                "task": "CEAEval human expressive appropriateness, 0..5",
                "confirmation": report["results"][method],
                "stories": report["stories"],
                "scope": "Mandarin audiobook clips, complete 3..20 seconds",
                "project_standup_validation": False,
            },
            "selection_lock_sha256": sha(args.audio_run / "selection.lock.json"),
        }
    if args.intelligibility_run:
        root = args.intelligibility_run
        lock = json.loads((root / "selection.lock.json").read_text())
        report = json.loads((root / "confirmation.json").read_text())
        method = lock["chosen"]["method"]
        alpha = lock["chosen"]["alpha"]
        (args.output / "intelligibility").mkdir()
        shutil.copy2(
            root / (method + f"-{alpha}.joblib"),
            args.output / "intelligibility/regressor.joblib",
        )
        if method != "waveform":
            for name in ("audio_encoder", "audio_extractor"):
                shutil.copytree(root / name, args.output / "intelligibility" / name)
        config["intelligibility"] = {
            "method": method,
            "regressor": "intelligibility/regressor.joblib",
            "encoder": "intelligibility/audio_encoder",
            "extractor": "intelligibility/audio_extractor",
            "model_id": f"tmhintqi-frozen-omni-ridge-{method}-{alpha}",
            "validation": {
                "task": "TMHINT-QI native 0..10 intelligibility_score",
                "confirmation": report["results"][method],
                "content_groups": report["content_groups"],
                "long_work_human_validation": False,
                "same_speaker_may_appear_across_splits": True,
                "feature_precision": "Published confirmation uses CUDA BF16 autocast; CPU feature precision is not separately benchmarked",
                "scope": "Chinese short clean/noisy/enhanced sentence recognition, not quality MOS",
            },
            "selection_lock_sha256": sha(root / "selection.lock.json"),
        }
    if not (args.text_run or args.audio_run or args.intelligibility_run):
        raise ValueError("At least one model run required")
    manifest = {
        str(p.relative_to(args.output)): sha(p)
        for p in sorted(args.output.rglob("*"))
        if p.is_file()
    }
    config["bundle_id"] = hashlib.sha256(
        json.dumps({"files": manifest, "config": config}, sort_keys=True).encode()
    ).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "bundle.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "bundle_id": config["bundle_id"],
                "files": len(manifest),
                "text": bool(args.text_run),
                "audio": bool(args.audio_run),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-run", type=Path)
    parser.add_argument("--text-plan", type=Path)
    parser.add_argument("--qwen-base", type=Path)
    parser.add_argument("--audio-run", type=Path)
    parser.add_argument("--intelligibility-run", type=Path)
    parser.add_argument("--embedding-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args())
