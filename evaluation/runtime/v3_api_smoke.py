"""Actual unified API integration on all fixed complete pilot works, not validity proof."""

import argparse
import json
from pathlib import Path
import time

from openmic.evaluation import OfflineEvaluator
from graded_text import read_lines, write_json, write_lines
from omni_judge import assert_idle_gpu
from v3_text import digest


def run(args):
    if args.device == "cuda":
        assert_idle_gpu(args.gpu_uuid)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = read_lines(args.inputs)
    evaluator = OfflineEvaluator(
        args.bundle, args.device, args.omni_model, args.asr_model
    )
    verified = evaluator.verify_assets()
    records = []
    for row in rows:
        if digest(row["audio_path"]) != row["audio_sha256"]:
            raise ValueError("Pilot waveform changed")
        result = evaluator.evaluate(
            text=row["text"], audio_path=row["audio_path"], evidence=False
        )
        records.append({"sample_id": row["sample_id"], "result": result})
        write_lines(args.output / "predictions.private.jsonl", records)
        print(
            json.dumps(
                {
                    "complete": len(records),
                    "total": len(rows),
                    "status": result["status"],
                    "failures": len(result["failures"]),
                }
            ),
            flush=True,
        )
    results = [r["result"] for r in records]
    coverage = [r.get("audio_evidence", {}).get("coverage", {}) for r in results]
    full = [r.get("experimental_full_work_scores", {}) for r in results]
    summary = {
        "n": len(rows),
        "input_sha256": digest(args.inputs),
        "plan_sha256": digest(args.plan),
        "bundle": verified,
        "decoded_full_waveforms": sum(c.get("decoded_fraction") == 1 for c in coverage),
        "duration_seconds_total": sum(c.get("duration_seconds", 0) for c in coverage),
        "text_outputs": {
            s: sum(r["scores"]["text_humor"]["status"] == s for r in results)
            for s in sorted({r["scores"]["text_humor"]["status"] for r in results})
        },
        "full_audio_humor_numeric_outputs": sum(
            r["scores"]["whole_audio_humor"]["continuous"] is not None for r in results
        ),
        "full_audio_expression_numeric_outputs": sum(
            r["scores"]["expression"]["continuous"] is not None for r in results
        ),
        "full_audio_intelligibility_numeric_outputs": sum(
            r["scores"]["intelligibility"]["continuous"] is not None for r in results
        ),
        "full_omni_feature_coverage_checks": sum(
            v.get("coverage", {}).get("scored_fraction") == 1
            for f in full
            for v in f.values()
        ),
        "intelligibility_segment_coverage_checks": sum(
            r["scores"]["intelligibility"].get("coverage", {}).get("processed_fraction")
            == 1
            for r in results
        ),
        "failures": sum(len(r["failures"]) for r in results),
        "elapsed_seconds": time.monotonic() - started,
        "actual_model_calls": True,
        "scientific_full_work_human_validation": False,
        "meaning": "Engineering integration and complete-input checks only; no human ratings, no generation-condition comparison, no validity claim",
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if summary["failures"] or summary["decoded_full_waveforms"] != len(rows):
        raise RuntimeError(
            "Unified API integration has failures; inspect preserved outputs"
        )
    if args.omni_model and any(
        summary[k] != len(rows)
        for k in (
            "full_audio_humor_numeric_outputs",
            "full_audio_expression_numeric_outputs",
            "full_audio_intelligibility_numeric_outputs",
        )
    ):
        raise RuntimeError("Full-work scoring did not cover every frozen pilot input")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--omni-model", type=Path)
    parser.add_argument("--asr-model", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
