"""Blank three-dimension listening pack and explicit, local human-data import.

No model scores enter the rater pack. Playback metadata is a coverage check,
not proof of human identity, attention, independence, or truthful responding.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

from analysis import analyze
from common import read_jsonl

PROTOCOL = "openmic-listening-pilot-v3"
DIMENSIONS = ("humor", "expression", "intelligibility")
MIN_PLAYED_FRACTION = 0.95


def identity(mapping):
    return hashlib.sha256(
        json.dumps({"protocol": PROTOCOL, "mapping": mapping}, sort_keys=True).encode()
    ).hexdigest()


def build(manifest, output):
    rows = read_jsonl(manifest)
    ids = [r["sample_id"] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Empty or duplicate manifest")
    # Check every source before creating the pack.
    for row in rows:
        if (
            hashlib.sha256(Path(row["audio_path"]).read_bytes()).hexdigest()
            != row["audio_sha256"]
        ):
            raise ValueError("Audio hash mismatch")
        if not row.get("topic_id", row.get("group")):
            raise ValueError("Missing topic identity")
    targets = [
        output,
        output.with_name(output.name + ".mapping.private.json"),
        output.with_name(output.name + ".samples.private.jsonl"),
    ]
    if any(p.exists() for p in targets):
        raise ValueError("Use a new pack and private mapping path")
    output.mkdir(parents=True, exist_ok=False)
    (output / "audio").mkdir()
    random.Random(20261027).shuffle(rows)
    clips, mapping, samples = [], [], []
    for index, row in enumerate(rows, 1):
        blind = f"clip-{index:02d}"
        relative = f"audio/{blind}.wav"
        shutil.copyfile(row["audio_path"], output / relative)
        clips.append({"sample_id": blind, "audio": relative})
        topic = row.get("topic_id", row.get("group"))
        mapping.append(
            {
                "blind_id": blind,
                "sample_id": row["sample_id"],
                "topic_id": topic,
                "audio_sha256": row["audio_sha256"],
                "formal_test_eligible": False,
            }
        )
        samples.append(
            {
                "sample_id": row["sample_id"],
                "topic_id": topic,
                "story_family_id": topic,
                "audio_sha256": row["audio_sha256"],
                "split": "pilot",
                "formal_test_eligible": False,
            }
        )
    dataset = identity(mapping)
    for sample in samples:
        sample["dataset_id"] = dataset
    payload = json.dumps(
        {"dataset_id": dataset, "protocol": PROTOCOL, "clips": clips},
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    template = (Path(__file__).parent.parent / "listening_v3.html").read_text()
    (output / "index.html").write_text(template.replace("__PACK_JSON__", payload))
    (output / "README.txt").write_text(
        "打开 index.html，独立听完整段后分别填写三个维度；没有默认分数。\n"
        "可留空、说明无法判断。原来的评分表不受影响。\n"
        "音频播放覆盖只是技术记录，不证明身份、注意力或独立性。\n"
        "完成后导出自己的 JSON，勿参考模型分数；不要把评分发到公共仓库。\n"
        "这些作品用于开发试评，不是新的正式盲测。\n"
    )
    for suffix, value in (
        (".mapping.private.json", mapping),
        (".samples.private.jsonl", samples),
    ):
        target = output.with_name(output.name + suffix)
        with target.open("x") as stream:
            if suffix.endswith("jsonl"):
                stream.write("".join(json.dumps(r) + "\n" for r in value))
            else:
                stream.write(json.dumps(value, indent=2) + "\n")
    return {
        "protocol": PROTOCOL,
        "dataset_id": dataset,
        "clips": len(clips),
        "dimensions": list(DIMENSIONS),
        "contains_model_scores": False,
        "contains_human_scores": False,
        "mapping_outside_pack": True,
        "formal_test_eligible": False,
    }


def convert(export, mapping):
    if export.get("protocol") != PROTOCOL or export.get("dataset_id") != identity(
        mapping
    ):
        raise ValueError("Ratings belong to a different listening pack/protocol")
    lookup = {r["blind_id"]: r["sample_id"] for r in mapping}
    if len(lookup) != len(mapping):
        raise ValueError("Duplicate mapping")
    ratings = export["ratings"]
    if len(ratings) != len(lookup) or {r["sample_id"] for r in ratings} != set(lookup):
        raise ValueError("Export must preserve every planned item, including blanks")
    raters = {r["rater_id"] for r in ratings}
    if len(raters) != 1 or not all(isinstance(r, str) and r.strip() for r in raters):
        raise ValueError("One nonempty anonymous rater ID required")
    orders = [r["presentation_order"] for r in ratings]
    if any(type(i) is not int for i in orders) or set(orders) != set(
        range(1, len(lookup) + 1)
    ):
        raise ValueError("Invalid recorded presentation order")
    result = []
    for row in ratings:
        understood = row.get("understood")
        if understood not in ("yes", "partial", "no", None):
            raise ValueError("Invalid comprehension value")
        fraction = row.get("played_fraction")
        if (
            type(fraction) not in (int, float)
            or not 0 <= fraction <= 1
            or type(row.get("full_listened")) is not bool
            or type(row.get("previously_heard")) is not bool
            or not isinstance(row.get("comment", ""), str)
        ):
            raise ValueError("Invalid listening metadata")
        scored = False
        for dimension in DIMENSIONS:
            score = row.get(dimension + "_score")
            if score is not None:
                scored = True
                if type(score) is not int or not 1 <= score <= 5:
                    raise ValueError("Human score must be an integer 1..5 or null")
                if dimension != "intelligibility" and understood not in (
                    "yes",
                    "partial",
                ):
                    raise ValueError(
                        "Humor/expression scores require yes/partial comprehension"
                    )
        if scored and (not row["full_listened"] or fraction < MIN_PLAYED_FRACTION):
            raise ValueError(
                "Scored items require declared full listening and >=95% playback"
            )
        result.append(
            {
                "sample_id": lookup[row["sample_id"]],
                "rater_id": row["rater_id"].strip(),
                "annotation_round": PROTOCOL,
                **{d + "_score": row.get(d + "_score") for d in DIMENSIONS},
                "understood": understood,
                "played_fraction": float(fraction),
                "full_listened": row["full_listened"],
                "previously_heard": row["previously_heard"],
                "presentation_order": row["presentation_order"],
                "comment": row.get("comment", ""),
                "dataset_id": export["dataset_id"],
            }
        )
    return result


def evaluate_humans(samples, ratings, predictions, min_raters=3, n_bootstrap=5000):
    datasets = {r.get("dataset_id") for r in samples}
    if len(datasets) != 1 or None in datasets:
        raise ValueError("Use the generated samples from exactly one listening pack")
    if ratings and {r.get("dataset_id") for r in ratings} != datasets:
        raise ValueError("Do not pool ratings from different listening packs")
    if any(r.get("annotation_round") != PROTOCOL for r in ratings):
        raise ValueError("Choose only explicitly imported v3 listener ratings")
    if any(r.get("dimension") not in DIMENSIONS for r in predictions):
        raise ValueError("Unknown prediction dimension")
    audio_hashes = {r["sample_id"]: r["audio_sha256"] for r in samples}
    for row in predictions:
        if row.get("score") is not None and (
            not row.get("audio_sha256")
            or row["audio_sha256"] != audio_hashes.get(row["sample_id"])
        ):
            raise ValueError("Prediction audio does not match the rated waveform")
    reports = {}
    for dimension in DIMENSIONS:
        current = [r for r in predictions if r["dimension"] == dimension]
        reports[dimension] = analyze(
            samples,
            ratings,
            current,
            split="pilot",
            min_raters=min_raters,
            n_bootstrap=n_bootstrap,
            score_field=dimension + "_score",
            comprehension_required=dimension != "intelligibility",
        )
    return {
        "protocol": PROTOCOL,
        "dataset_id": next(iter(datasets)),
        "dimensions": reports,
        "formal_test": False,
        "limitations": [
            "Existing pilot works only; not a new independent project test",
            "Few works/topics; correlations and intervals are exploratory",
            "Playback coverage does not establish human identity, attention or independent responding",
            "Listener intelligibility uses 1..5 perceived word clarity, not TMHINT-QI 0..10 recognized characters",
            "Ranks only: native model scales differ; do not compare absolute scores or infer exact five-class accuracy",
        ],
    }


def extract_predictions(rows):
    if len({row["result"].get("bundle_id") for row in rows}) > 1:
        raise ValueError("Do not pool evaluator bundles")
    output = []
    for row in rows:
        result = row["result"]
        for dimension in DIMENSIONS:
            key = "whole_audio_humor" if dimension == "humor" else dimension
            score = result["scores"][key]
            output.append(
                {
                    "sample_id": row["sample_id"],
                    "dimension": dimension,
                    "model_id": score.get(
                        "model_id",
                        (
                            score.get("basis", {}).get("model_id")
                            if isinstance(score.get("basis"), dict)
                            else None
                        ),
                    )
                    or result["evaluator_version"] + "-" + key,
                    "score": score.get("continuous"),
                    "source_scale": score.get("scale"),
                    "source_status": score["status"],
                    "bundle_id": result.get("bundle_id"),
                    "audio_sha256": result.get("audio_evidence", {})
                    .get("coverage", {})
                    .get("audio_sha256"),
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("build")
    make.add_argument("--manifest", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    load = sub.add_parser("import")
    load.add_argument("--export", type=Path, required=True)
    load.add_argument("--mapping", type=Path, required=True)
    load.add_argument("--output", type=Path, required=True)
    pred = sub.add_parser("predictions")
    pred.add_argument("--results", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("analyze")
    for name in ("samples", "ratings", "predictions", "output"):
        report.add_argument("--" + name, type=Path, required=True)
    report.add_argument("--min-raters", type=int, default=3)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.manifest, args.output)))
        return
    if args.command == "import":
        data = convert(
            json.loads(args.export.read_text()), json.loads(args.mapping.read_text())
        )
    elif args.command == "predictions":
        data = extract_predictions(read_jsonl(args.results))
    else:
        data = evaluate_humans(
            read_jsonl(args.samples),
            read_jsonl(args.ratings),
            read_jsonl(args.predictions),
            args.min_raters,
        )
    with args.output.open("x") as stream:
        if isinstance(data, list):
            stream.write(
                "".join(
                    json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                    for r in data
                )
            )
        else:
            stream.write(
                json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            )


if __name__ == "__main__":
    main()
