"""Accuracy-led text experiment. Separate prepare, select, and confirmation stages."""

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from graded_text import read_lines, write_json, write_lines, metrics, canonical
from omni_judge import assert_idle_gpu


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def class_metrics(y, probabilities):
    y = np.asarray(y)
    p = np.asarray(probabilities)
    if (
        p.shape != (len(y), 5)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(1), 1, atol=1e-5)
    ):
        raise ValueError("Expected finite normalized five-class probabilities")
    grade = p.argmax(1) + 1
    return {
        **metrics(y, p @ np.arange(1, 6)),
        "correct": int((grade == y).sum()),
        "exact_accuracy": float((grade == y).mean()),
        "macro_f1": float(
            f1_score(y, grade, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y, grade, labels=[1, 2, 3, 4, 5]).tolist(),
    }


def selection_key(result):
    return (
        result["exact_accuracy"],
        result["spearman"] if result["spearman"] is not None else -2,
    )


def topic_groups(rows, embeddings, n_topics=80):
    """Label-independent semantic topics, joined by lexical/semantic duplicate edges."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    vectors = vectors / np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
    )
    topics = KMeans(
        n_clusters=min(n_topics, len(rows)), random_state=20261026, n_init=10
    ).fit_predict(vectors)
    parent = list(range(int(topics.max()) + 1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(int(topics[j]))] = find(int(topics[i]))

    original = defaultdict(list)
    for i, row in enumerate(rows):
        original[row["group"]].append(i)
    for members in original.values():
        for i in members[1:]:
            union(members[0], i)
    lexical = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit_transform(
        [canonical(r["text"]) for r in rows]
    )
    close = (cosine_similarity(lexical) >= 0.70) | ((vectors @ vectors.T) >= 0.92)
    left, right = np.where(np.triu(close, k=1))
    for i, j in zip(left, right):
        union(i, j)
    members = defaultdict(list)
    for i, t in enumerate(topics):
        members[find(int(t))].append(i)
    identities = {
        k: hashlib.sha256(
            "\n".join(sorted(rows[i]["sample_id"] for i in v)).encode()
        ).hexdigest()
        for k, v in members.items()
    }
    ordered = sorted(identities, key=lambda k: identities[k])
    # Exact topic counts, not label stratification. Small fixture support only.
    a = max(1, int(0.6 * len(ordered)))
    b = max(a + 1, int(0.8 * len(ordered)))
    assignments = {
        k: ("train" if n < a else "validation" if n < b else "confirmation")
        for n, k in enumerate(ordered)
    }
    output = []
    for row, t in zip(rows, topics):
        root = find(int(t))
        output.append(
            {
                **row,
                "previous_split": row["split"],
                "split": assignments[root],
                "group": identities[root],
            }
        )
    return output, {
        "initial_topics": int(topics.max()) + 1,
        "merged_topic_groups": len(members),
        "near_duplicate_edges": len(left),
        "largest_group": max(map(len, members.values())),
    }


def embed_texts(texts, model_path, device="cuda"):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = (
        AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    )
    output = []
    with torch.inference_mode():
        for text in texts:
            tokens = tok.encode(text, add_special_tokens=False, truncation=False)
            chunks = [tokens[i : i + 510] for i in range(0, len(tokens), 510)] or [[]]
            items = [tok.prepare_for_model(c, truncation=False) for c in chunks]
            batch = tok.pad(items, padding=True, return_tensors="pt").to(device)
            vectors = model(**batch).last_hidden_state[:, 0].float()
            weights = torch.tensor([max(1, len(c)) for c in chunks], device=device)
            vector = (vectors * weights[:, None]).sum(0) / weights.sum()
            output.append(torch.nn.functional.normalize(vector, dim=0).cpu().numpy())
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.array(output)


def prepare(args):
    assert_idle_gpu(args.gpu_uuid)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [r for r in read_lines(args.records) if r["split"] != "test"]
    embeddings = embed_texts([r["text"] for r in rows], args.embedding_model)
    rows, info = topic_groups(rows, embeddings)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    kept = []
    excluded = Counter()
    for row in rows:
        if len(tok.encode(row["text"], truncation=False)) > 512:
            excluded[row["split"]] += 1
        else:
            kept.append(row)
    for split in ("train", "validation", "confirmation"):
        selected = [r for r in kept if r["split"] == split]
        if not selected:
            raise ValueError("Empty partition; inspect grouping before any training")
        write_lines(args.output / (split + ".private.jsonl"), selected)
    info.update(
        {
            "source_sha256": digest(args.records),
            "retired_v2_test": sum(
                r["split"] == "test" for r in read_lines(args.records)
            ),
            "plan_sha256": digest(args.plan),
            "counts": dict(Counter(r["split"] for r in kept)),
            "excluded_overlength": dict(excluded),
            "class_counts_train_and_validation": {
                s: dict(Counter(r["target"] for r in kept if r["split"] == s))
                for s in ("train", "validation")
            },
            "confirmation_targets_not_used_for_grouping_or_stratification": True,
            "split_hashes": {
                s: digest(args.output / (s + ".private.jsonl"))
                for s in ("train", "validation", "confirmation")
            },
        }
    )
    write_json(args.output / "preparation.json", info)
    print(json.dumps(info), flush=True)


def predict(model, items, collator, device="cuda", batch_size=32):
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    out = []
    with torch.inference_mode():
        for batch in DataLoader(items, batch_size=batch_size, collate_fn=collator):
            batch.pop("labels", None)
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
            ):
                logits = model(**inputs)
            out.extend(model.probabilities(logits).cpu().numpy())
    return np.array(out)


def cpu_baseline(train, validation):
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=150000,
    )
    x = vec.fit_transform([r["text"] for r in train])
    v = vec.transform([r["text"] for r in validation])
    best = None
    trials = []
    for c in [0.1, 1, 10]:
        clf = LogisticRegression(C=c, max_iter=1000, random_state=20261026).fit(
            x, [r["target"] for r in train]
        )
        p = np.zeros((len(validation), 5))
        p[:, clf.classes_ - 1] = clf.predict_proba(v)
        m = class_metrics([r["target"] for r in validation], p)
        trials.append({"C": c, "validation": m})
        if best is None or selection_key(m) > selection_key(best[2]):
            best = (vec, clf, m, c)
    return best, trials


def select(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModel,
        AutoTokenizer,
        DataCollatorWithPadding,
        set_seed,
        get_linear_schedule_with_warmup,
    )
    from openmic.evaluation_neural import GradedEncoder

    assert_idle_gpu(args.gpu_uuid)
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    cfg = plan["text"]
    parts = {
        s: read_lines(args.data / (s + ".private.jsonl"))
        for s in ("train", "validation")
    }
    # Deliberately no confirmation file read in selection.
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.save_pretrained(args.output / "tokenizer")
    items = {
        s: [
            {**tok(r["text"], truncation=False), "labels": r["target"] - 1}
            for r in rows
        ]
        for s, rows in parts.items()
    }
    if any(len(r["input_ids"]) > 512 for rows in items.values() for r in rows):
        raise ValueError("No truncation allowed")
    collator = DataCollatorWithPadding(tok)
    y = [r["target"] for r in parts["validation"]]
    baseline, trials = cpu_baseline(parts["train"], parts["validation"])
    joblib.dump((baseline[0], baseline[1]), args.output / "lightweight.joblib")
    candidates = []
    for objective in cfg["objectives"]:
        for lr in cfg["learning_rates"]:
            name = f"{objective}-lr{lr:g}"
            seed_predictions = []
            histories = []
            for seed in cfg["seeds"]:
                set_seed(seed)
                assert_idle_gpu(args.gpu_uuid)
                model = GradedEncoder(
                    AutoModel.from_pretrained(args.model, local_files_only=True),
                    objective,
                ).to("cuda")
                loader = DataLoader(
                    items["train"],
                    batch_size=cfg["batch_size"],
                    shuffle=True,
                    generator=torch.Generator().manual_seed(seed),
                    collate_fn=collator,
                )
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=0.01
                )
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    int(0.1 * len(loader) * cfg["epochs"]),
                    len(loader) * cfg["epochs"],
                )
                best = None
                history = []
                checkpoint = args.output / name / f"seed-{seed}"
                for epoch in range(1, cfg["epochs"] + 1):
                    model.train()
                    losses = []
                    for batch in loader:
                        if time.monotonic() - started > args.seconds:
                            raise TimeoutError("Selection budget exhausted")
                        optimizer.zero_grad(set_to_none=True)
                        labels = batch.pop("labels").to("cuda")
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = model.loss(
                                model(**{k: v.to("cuda") for k, v in batch.items()}),
                                labels,
                            )
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        scheduler.step()
                        losses.append(loss.item())
                    p = predict(model, items["validation"], collator)
                    m = class_metrics(y, p)
                    history.append(
                        {
                            "epoch": epoch,
                            "mean_training_loss": float(np.mean(losses)),
                            "validation": m,
                        }
                    )
                    if best is None or selection_key(m) > selection_key(best[0]):
                        best = (m, p.copy(), epoch)
                        model.save(checkpoint)
                    print(
                        json.dumps({"candidate": name, "seed": seed, **history[-1]}),
                        flush=True,
                    )
                seed_predictions.append(best[1])
                histories.append(
                    {"seed": seed, "best_epoch": best[2], "history": history}
                )
                del model, optimizer, scheduler
                gc.collect()
                torch.cuda.empty_cache()
            p = np.mean(seed_predictions, axis=0)
            m = class_metrics(y, p)
            np.save(args.output / (name + ".validation.private.npy"), p)
            candidates.append(
                {
                    "name": name,
                    "objective": objective,
                    "learning_rate": lr,
                    "validation": m,
                    "seeds": histories,
                }
            )
            write_json(
                args.output / "selection_progress.json",
                {"candidates": candidates, "cpu_trials": trials},
            )
    chosen = max(candidates, key=lambda c: selection_key(c["validation"]))
    lock = {
        "protocol": plan["protocol"],
        "plan_sha256": digest(args.plan),
        "preparation_sha256": digest(args.data / "preparation.json"),
        "chosen": chosen["name"],
        "objective": chosen["objective"],
        "seeds": cfg["seeds"],
        "validation": chosen["validation"],
        "cpu_C": baseline[3],
        "cpu_validation": baseline[2],
        "train_majority": Counter(r["target"] for r in parts["train"]).most_common(1)[
            0
        ][0],
        "classification_rule": "argmax(mean_seed_probabilities), lowest grade breaks ties",
        "checkpoints": {
            str(seed): digest(
                args.output / chosen["name"] / f"seed-{seed}" / "head.safetensors"
            )
            for seed in cfg["seeds"]
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(args.output / "selection.lock.json", lock)
    print(json.dumps(lock), flush=True)


def bootstrap(y, probabilities, groups, baseline, reps=5000):
    rng = np.random.default_rng(20261026)
    unique = sorted(set(groups))
    buckets = {g: np.flatnonzero(np.array(groups) == g) for g in unique}
    grade = probabilities.argmax(1) + 1
    base_grade = baseline.argmax(1) + 1
    accuracy = []
    delta = []
    rho = []
    for _ in range(reps):
        ix = np.concatenate(
            [buckets[g] for g in rng.choice(unique, len(unique), replace=True)]
        )
        accuracy.append(np.mean(grade[ix] == y[ix]))
        delta.append(np.mean(grade[ix] == y[ix]) - np.mean(base_grade[ix] == y[ix]))
        scores = probabilities[ix] @ np.arange(1, 6)
        if len(set(scores)) > 1 and len(set(y[ix])) > 1:
            rho.append(float(spearmanr(y[ix], scores).statistic))
    ci = lambda x: np.quantile(x, [0.025, 0.975]).tolist() if x else None
    return {
        "exact_accuracy_ci95": ci(accuracy),
        "accuracy_minus_lightweight_ci95": ci(delta),
        "spearman_ci95": ci(rho),
        "unit": "semantic topic group",
        "resamples": reps,
    }


def report(args):
    import torch
    from transformers import AutoTokenizer, DataCollatorWithPadding
    from openmic.evaluation_neural import GradedEncoder

    if args.device == "cuda":
        assert_idle_gpu(args.gpu_uuid)
    lock = json.loads((args.output / "selection.lock.json").read_text())
    if (args.output / "confirmation.json").exists():
        raise ValueError("Confirmation already reported; freeze this run")
    if lock["plan_sha256"] != digest(args.plan) or lock["preparation_sha256"] != digest(
        args.data / "preparation.json"
    ):
        raise ValueError("Protocol changed after selection")
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
    items = [tok(r["text"], truncation=False) for r in rows]
    if any(len(r["input_ids"]) > 512 for r in items):
        raise ValueError("Overlength confirmation")
    probabilities = []
    for seed in lock["seeds"]:
        model = GradedEncoder.load(args.output / lock["chosen"] / f"seed-{seed}").to(
            args.device
        )
        probabilities.append(
            predict(model, items, DataCollatorWithPadding(tok), device=args.device)
        )
        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    p = np.mean(probabilities, axis=0)
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
        "inference_device": args.device,
        "inference_precision": (
            "float32" if args.device == "cpu" else "bfloat16_autocast"
        ),
    }
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
    write_json(args.output / "confirmation.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "select", "report"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--embedding-model", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Confirmation inference device; prepare/select require approved CUDA",
    )
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    {"prepare": prepare, "select": select, "report": report}[args.stage](args)
