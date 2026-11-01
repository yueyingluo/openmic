"""Fixed ASR baseline on human transcripts and complete project pilot waveforms."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from openmic.evaluation_asr import WhisperAudit, character_error
from openmic.evaluation_audio import load_waveform
from graded_text import read_lines, write_json, write_lines
from omni_judge import assert_idle_gpu
from v3_text import digest


def run(args):
    assert_idle_gpu(args.gpu_uuid)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = read_lines(args.inputs)
    if args.test_only:
        rows = [r for r in rows if r["split"] == "test"]
    audit = WhisperAudit(args.model)
    results = []
    start = time.monotonic()
    with (args.output / "predictions.private.jsonl").open("x") as target:
        for i, row in enumerate(rows):
            if time.monotonic() - start > args.seconds:
                raise TimeoutError("ASR budget exhausted")
            if (
                row.get("audio_sha256")
                and digest(row["audio_path"]) != row["audio_sha256"]
            ):
                raise ValueError("Waveform changed")
            wave, coverage = load_waveform(row["audio_path"])
            result = {
                "sample_id": row["sample_id"],
                "group": row.get("group", row.get("topic_id")),
                **audit.transcribe(wave),
                "source_coverage": coverage,
                "reference_metrics": None,
            }
            if row.get("text"):
                result["reference_metrics"] = character_error(
                    row["text"], result["transcript"]
                )
            results.append(result)
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
            print(
                json.dumps(
                    {"complete": i + 1, "total": len(rows), "status": result["status"]}
                ),
                flush=True,
            )
    measured = [
        r["reference_metrics"] for r in results if r["reference_metrics"] is not None
    ]
    total_chars = sum(r["reference_characters"] for r in measured)
    total_errors = sum(r["edit_distance"] for r in measured)
    report = {
        "n": len(rows),
        "transcribed": sum(r["status"] == "transcribed" for r in results),
        "decoder_limit": sum(r["status"] != "transcribed" for r in results),
        "total_duration_seconds": sum(
            r["source_coverage"]["duration_seconds"] for r in results
        ),
        "min_complete_waveform_processed_fraction": min(
            r["coverage"]["processed_fraction"] for r in results
        ),
        "reference_characters": total_chars,
        "edit_distance": total_errors,
        "micro_cer": total_errors / total_chars if total_chars else None,
        "mean_clip_cer": (
            float(np.mean([r["character_error_rate"] for r in measured]))
            if measured
            else None
        ),
        "empty_transcript_baseline_cer": 1.0 if total_chars else None,
        "plan_sha256": digest(args.plan),
        "input_sha256": digest(args.inputs),
        "elapsed_seconds": time.monotonic() - start,
        "human_intelligibility_validation": False,
        "meaning": "Fixed ASR transcript fidelity audit. Human transcripts on CEAEval; authored scripts on project pilot, not new human listener judgments.",
    }
    write_json(args.output / "summary.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    run(parser.parse_args())
