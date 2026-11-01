"""Compare development locks before the single final confirmation evaluation."""

import argparse
import hashlib
import json
from pathlib import Path


def freeze(runs, output):
    candidates = []
    for root in runs:
        if (root / "confirmation.json").exists():
            raise ValueError(
                "A confirmation result already exists; do not select after seeing it"
            )
        source = root / "selection.lock.json"
        lock = json.loads(source.read_text())
        candidates.append(
            {
                "run": str(root),
                "chosen": lock["chosen"],
                "validation": lock["validation"],
                "preparation_sha256": lock["preparation_sha256"],
                "selection_lock_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
            }
        )
    if len({c["preparation_sha256"] for c in candidates}) != 1:
        raise ValueError("Candidate runs use different partitions")
    key = lambda c: (
        c["validation"]["exact_accuracy"],
        c["validation"]["spearman"] if c["validation"]["spearman"] is not None else -2,
    )
    chosen = max(candidates, key=key)
    result = {
        "candidates": candidates,
        "selected_run": chosen["run"],
        "rule": "validation exact accuracy, then Spearman, then input run order",
        "confirmation_results_exist": False,
        "no_confirmation_targets_read": True,
    }
    with output.open("x") as target:
        target.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.run, args.output)
