"""Supervised causal-model development extension; no confirmation access in select."""

import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np

from graded_text import read_lines, write_json, write_lines
from omni_judge import assert_idle_gpu
from v3_text import digest, class_metrics, selection_key, cpu_baseline, bootstrap
from openmic.evaluation_qwen import encode_text, QwenGradeEnsemble, PREFIX


def probabilities(model, items, collator, batch_size=8):
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    out = []
    with torch.inference_mode():
        for batch in DataLoader(items, batch_size=batch_size, collate_fn=collator):
            batch.pop("labels", None)
            out.extend(
                model(**{k: v.to("cuda") for k, v in batch.items()})
                .logits.float()
                .softmax(-1)
                .cpu()
                .numpy()
            )
    return np.array(out)


def select(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        set_seed,
        get_linear_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    assert_idle_gpu(args.gpu_uuid)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    cfg = plan["text"]
    rows = {
        s: read_lines(args.data / (s + ".private.jsonl"))
        for s in ("train", "validation")
    }
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.save_pretrained(args.output / "tokenizer")
    items = {
        s: [{**encode_text(tok, r["text"]), "labels": r["target"] - 1} for r in rs]
        for s, rs in rows.items()
    }
    max_tokens = max(len(x["input_ids"]) for part in items.values() for x in part)
    if max_tokens > cfg["training_max_tokens"]:
        raise ValueError("Configured training context exceeded; no truncation")
    collator = DataCollatorWithPadding(tok)
    y = [r["target"] for r in rows["validation"]]
    baseline, cpu_trials = cpu_baseline(rows["train"], rows["validation"])
    joblib.dump((baseline[0], baseline[1]), args.output / "lightweight.joblib")
    candidates = []
    for lr in cfg["learning_rates"]:
        name = f"qwen-lora-ce-lr{lr:g}"
        predictions = []
        histories = []
        for seed in cfg["seeds"]:
            assert_idle_gpu(args.gpu_uuid)
            set_seed(seed)
            base, info = AutoModelForSequenceClassification.from_pretrained(
                args.model,
                num_labels=5,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                output_loading_info=True,
            )
            if any(k != "score.weight" for k in info["missing_keys"]) or info.get(
                "mismatched_keys"
            ):
                raise ValueError("Unexpected missing pretrained weights")
            base.config.pad_token_id = tok.pad_token_id
            base.config.use_cache = False
            lora = cfg["lora"]
            model = get_peft_model(
                base,
                LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=lora["rank"],
                    lora_alpha=lora["alpha"],
                    lora_dropout=lora["dropout"],
                    target_modules=lora["target_modules"],
                    bias="none",
                ),
            ).to("cuda")
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            loader = DataLoader(
                items["train"],
                batch_size=cfg["batch_size"],
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
                collate_fn=collator,
            )
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=lr,
                weight_decay=0.01,
            )
            accumulation = cfg["gradient_accumulation"]
            steps = math.ceil(len(loader) / accumulation) * cfg["epochs"]
            schedule = get_linear_schedule_with_warmup(
                optimizer, int(0.1 * steps), steps
            )
            best = None
            history = []
            checkpoint = args.output / name / f"seed-{seed}"
            for epoch in range(1, cfg["epochs"] + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for n, batch in enumerate(loader):
                    if time.monotonic() - started > args.seconds:
                        raise TimeoutError("Qwen training budget exceeded")
                    labels = batch.pop("labels").to("cuda")
                    logits = model(
                        **{k: v.to("cuda") for k, v in batch.items()}
                    ).logits.float()
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    # Last partial accumulation has its actual number of batches.
                    group_start = (n // accumulation) * accumulation
                    divisor = min(accumulation, len(loader) - group_start)
                    (loss / divisor).backward()
                    losses.append(loss.item())
                    if (n + 1) % accumulation == 0 or n + 1 == len(loader):
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in model.parameters() if p.requires_grad), 1
                        )
                        optimizer.step()
                        schedule.step()
                        optimizer.zero_grad(set_to_none=True)
                p = probabilities(model, items["validation"], collator)
                m = class_metrics(y, p)
                history.append(
                    {
                        "epoch": epoch,
                        "training_loss": float(np.mean(losses)),
                        "validation": m,
                    }
                )
                if best is None or selection_key(m) > selection_key(best[0]):
                    best = (m, p.copy(), epoch)
                    model.save_pretrained(checkpoint, safe_serialization=True)
                print(
                    json.dumps({"candidate": name, "seed": seed, **history[-1]}),
                    flush=True,
                )
            predictions.append(best[1])
            histories.append({"seed": seed, "best_epoch": best[2], "history": history})
            del model, base, optimizer, schedule
            gc.collect()
            torch.cuda.empty_cache()
        mean = np.mean(predictions, axis=0)
        m = class_metrics(y, mean)
        candidates.append({"name": name, "validation": m, "seeds": histories})
        np.save(args.output / (name + ".validation.private.npy"), mean)
        write_json(
            args.output / "selection_progress.json",
            {"candidates": candidates, "cpu_trials": cpu_trials},
        )
    chosen = max(candidates, key=lambda c: selection_key(c["validation"]))
    lock = {
        "protocol": plan["protocol"],
        "model_type": "qwen_lora",
        "base_model": str(args.model),
        "base_model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "plan_sha256": digest(args.plan),
        "preparation_sha256": digest(args.data / "preparation.json"),
        "chosen": chosen["name"],
        "seeds": cfg["seeds"],
        "validation": chosen["validation"],
        "training_max_observed_tokens": max_tokens,
        "inference_max_tokens": cfg["inference_max_tokens"],
        "prefix": PREFIX,
        "cpu_C": baseline[3],
        "cpu_validation": baseline[2],
        "train_majority": Counter(r["target"] for r in rows["train"]).most_common(1)[0][
            0
        ],
        "checkpoints": {
            str(s): digest(
                args.output / chosen["name"] / f"seed-{s}" / "adapter_model.safetensors"
            )
            for s in cfg["seeds"]
        },
        "elapsed_seconds": time.monotonic() - started,
        "selection_read_confirmation_labels": False,
    }
    write_json(args.output / "selection.lock.json", lock)
    print(json.dumps(lock), flush=True)


def report(args):
    from transformers import AutoTokenizer

    assert_idle_gpu(args.gpu_uuid)
    if (args.output / "confirmation.json").exists():
        raise ValueError("Confirmation already reported")
    lock = json.loads((args.output / "selection.lock.json").read_text())
    if lock["plan_sha256"] != digest(args.plan) or lock["preparation_sha256"] != digest(
        args.data / "preparation.json"
    ):
        raise ValueError("Protocol changed")
    rows = read_lines(args.data / "confirmation.private.jsonl")
    prep = json.loads((args.data / "preparation.json").read_text())
    if (
        digest(args.data / "confirmation.private.jsonl")
        != prep["split_hashes"]["confirmation"]
    ):
        raise ValueError("Confirmation changed")
    tok = AutoTokenizer.from_pretrained(
        args.output / "tokenizer", local_files_only=True
    )
    model = QwenGradeEnsemble(
        args.model,
        [args.output / lock["chosen"] / f"seed-{s}" for s in lock["seeds"]],
        tok,
    )
    p = np.concatenate(
        [
            model.probabilities([r["text"] for r in rows[i : i + 8]])
            for i in range(0, len(rows), 8)
        ]
    )
    y = np.array([r["target"] for r in rows])
    vec, clf = joblib.load(args.output / "lightweight.joblib")
    base = np.zeros_like(p)
    base[:, clf.classes_ - 1] = clf.predict_proba(
        vec.transform([r["text"] for r in rows])
    )
    majority = np.zeros_like(p)
    majority[:, lock["train_majority"] - 1] = 1
    result = {
        "selection_lock_sha256": digest(args.output / "selection.lock.json"),
        "chosen": lock["chosen"],
        "model_type": "qwen_lora",
        "confirmation_n": len(y),
        "confirmation_groups": len({r["group"] for r in rows}),
        "class_counts": dict(Counter(map(int, y))),
        "neural": class_metrics(y, p),
        "lightweight": class_metrics(y, base),
        "train_majority": class_metrics(y, majority),
        "uncertainty": bootstrap(y, p, [r["group"] for r in rows], base),
        "target_accuracy": 0.5,
        "achieved": bool(np.mean(p.argmax(1) + 1 == y) >= 0.5),
        "independence_limit": json.loads(args.plan.read_text())["independence_limit"],
        "project_human_alignment": False,
    }
    write_json(args.output / "confirmation.json", result)
    write_lines(
        args.output / "confirmation_predictions.private.jsonl",
        [
            {
                "sample_id": r["sample_id"],
                "group": r["group"],
                "probabilities": prob.tolist(),
                "lightweight_probabilities": bp.tolist(),
                "target": r["target"],
            }
            for r, prob, bp in zip(rows, p, base)
        ],
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["select", "report"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    globals()[args.stage](args)
