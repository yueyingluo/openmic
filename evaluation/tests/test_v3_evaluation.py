import json
from pathlib import Path

import numpy as np
import pytest

from openmic.evaluation import OfflineEvaluator, probability_summary
from openmic.evaluation_audio import load_waveform, waveform_descriptors, DESCRIPTORS
from v3_text import class_metrics, topic_groups
from v3_audio import fresh_story_splits
from openmic.evaluation_asr import character_error
from v3_finalize_selection import freeze
from v3_intelligibility import content_id, prepare as prepare_intelligibility


def test_unified_schema_requires_real_numeric_scores_for_scored_status():
    import jsonschema

    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schemas/evaluation_response.schema.json"
        ).read_text()
    )
    result = OfflineEvaluator().evaluate(text="未配置模型的接口检查")
    jsonschema.validate(result, schema)
    result["scores"]["text_humor"] = {
        "status": "scored",
        "grade": 4,
        "continuous": 3.1,
        "scale": [1, 5],
    }
    jsonschema.validate(result, schema)
    result["scores"]["text_humor"]["continuous"] = None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(result, schema)


def test_argmax_grade_is_not_rounded_expected_score():
    p = [0.05, 0.25, 0.30, 0.38, 0.02]
    result = probability_summary(p)
    assert result["grade"] == 4
    assert round(result["continuous"]) == 3
    assert class_metrics([4], np.array([p]))["correct"] == 1
    assert probability_summary([0, 0, 0, 0, 1 + 1e-7])["continuous"] == 5
    for invalid in ([1, 0, 0], [1, 0, 0, 0, float("nan")], [-1, 1, 1, 0, 0]):
        with pytest.raises(ValueError):
            probability_summary(invalid)


def test_cer_normalization_and_insertions_are_not_capped():
    assert character_error("這是Ａ，Ｂ！", "这是ab")["character_error_rate"] == 0
    assert character_error("甲", "甲乙丙丁")["character_error_rate"] == 3
    with pytest.raises(ValueError):
        character_error("。！", "任何内容")


def test_final_selection_refuses_existing_confirmation_result(tmp_path):
    roots = []
    for i, acc in enumerate([0.4, 0.45]):
        root = tmp_path / str(i)
        root.mkdir()
        roots.append(root)
        (root / "selection.lock.json").write_text(
            json.dumps(
                {
                    "chosen": "candidate",
                    "validation": {"exact_accuracy": acc, "spearman": 0.3},
                    "preparation_sha256": "same",
                }
            )
        )
    freeze(roots, tmp_path / "lock.json")
    assert json.loads((tmp_path / "lock.json").read_text())["selected_run"] == str(
        roots[1]
    )
    (roots[0] / "confirmation.json").write_text("{}")
    with pytest.raises(ValueError, match="confirmation result already"):
        freeze(roots, tmp_path / "must-not-exist.json")


def test_topic_and_duplicate_groups_are_indivisible():
    rows = [
        {"sample_id": str(i), "split": "train", "group": str(i), "text": text}
        for i, text in enumerate(
            [
                "同一个很长的故事。",
                "同一个很长的故事！",
                "完全不同的苹果音乐主题",
                "另外一只熊猫吃竹子",
                "宇宙星空的月亮",
            ]
        )
    ]
    # Put lexical duplicates into very different embedding topics.
    grouped, _ = topic_groups(rows, np.eye(5), n_topics=5)
    assert grouped[0]["group"] == grouped[1]["group"]
    assert grouped[0]["split"] == grouped[1]["split"]
    for group in {r["group"] for r in grouped}:
        assert len({r["split"] for r in grouped if r["group"] == group}) == 1


def test_fresh_audio_confirmation_excludes_all_previous_test_stories():
    from audio_data import story_splits

    stories = [f"story-{i}" for i in range(82)]
    old = story_splits(stories)
    new = fresh_story_splits(stories)
    assert all(old[s] == "train" for s in new)
    assert set(new.values()) == {"train", "validation", "confirmation"}
    assert new == fresh_story_splits(stories[::-1])


def test_intelligibility_groups_cross_speakers_and_never_use_quality(tmp_path):
    import csv
    import soundfile as sf
    from types import SimpleNamespace

    assert content_id("TMHINT_b2_22_03") == content_id("TMHINT_g4_22_03")
    with pytest.raises(ValueError):
        content_id("unknown-format")
    root = tmp_path / "data"
    root.mkdir()
    rows = []
    for i in range(10):
        utterance = f"TMHINT_b2_{i+1:02d}_01"
        filename = "clean_" + utterance
        sf.write(root / (filename + ".wav"), np.zeros(16000, dtype="float32"), 16000)
        rows.append(
            {
                "idx": 1,
                "uttr": utterance,
                "quality_score": 1,
                "intelligibility_score": 9,
                "file_name": filename,
            }
        )
    with (root / "raw_data.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    out = tmp_path / "out"
    prepare_intelligibility(
        SimpleNamespace(output=out, archive=None, root=root, plan=plan)
    )
    retained = [
        json.loads(s)
        for p in out.glob("*.private.jsonl")
        if p.name != "inputs.private.jsonl"
        for s in p.read_text().splitlines()
    ]
    assert len(retained) == 10
    assert {r["target"] for r in retained} == {9}
    assert all(
        "target" not in json.loads(s)
        for s in (out / "inputs.private.jsonl").read_text().splitlines()
    )


def test_complete_audio_tail_and_failure_states(tmp_path):
    import soundfile as sf

    rate = 22050
    wave = np.zeros(rate * 4, dtype=np.float32)
    wave[-rate:] = 0.2 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
    path = tmp_path / "tail.wav"
    sf.write(path, wave, rate)
    decoded, coverage = load_waveform(path)
    assert coverage["source_frames"] == 4 * rate
    assert coverage["decoded_fraction"] == 1
    assert len(decoded) == 4 * 16000
    descriptors = dict(zip(DESCRIPTORS, waveform_descriptors(decoded)))
    assert descriptors["peak"] > 0.19  # Last second was read.
    result = OfflineEvaluator().evaluate(audio_path=path)
    assert result["audio_evidence"]["coverage"]["duration_seconds"] == 4
    assert result["scores"]["whole_audio_humor"]["continuous"] is None
    assert result["scores"]["intelligibility"]["continuous"] is None
    assert result["scores"]["expression"]["status"] == "unconfigured"
    failed = OfflineEvaluator().evaluate(audio_path=tmp_path / "missing.wav")
    assert failed["scores"]["expression"]["status"] == "failed"
    assert failed["failures"][0]["modality"] == "audio"


def test_long_intelligibility_aggregate_covers_tail_and_keeps_unvalidated_scope(
    tmp_path,
):
    import joblib
    import soundfile as sf
    from sklearn.dummy import DummyRegressor

    model = DummyRegressor(strategy="constant", constant=7).fit(
        np.zeros((2, len(DESCRIPTORS))), [7, 7]
    )
    joblib.dump(model, tmp_path / "regressor.joblib")
    wave = 0.1 * np.sin(2 * np.pi * 440 * np.arange(33 * 16000) / 16000)
    path = tmp_path / "long.wav"
    sf.write(path, wave, 16000)
    evaluator = OfflineEvaluator()
    evaluator.root = tmp_path
    evaluator.config = {
        "intelligibility": {
            "method": "waveform",
            "regressor": "regressor.joblib",
            "model_id": "fixture",
            "validation": {"long_work_human_validation": False},
        }
    }
    result = evaluator.evaluate(audio_path=path)["scores"]["intelligibility"]
    assert result["continuous"] == pytest.approx(7)
    assert result["status"] == "experimental_aggregate"
    assert result["coverage"]["processed_fraction"] == 1
    assert result["coverage"]["score_coverage_fraction"] == 1
    assert result["segments"][0]["start_frame"] == 0
    assert result["segments"][-1]["end_frame"] == len(wave)
    assert all(
        a["end_frame"] == b["start_frame"]
        for a, b in zip(result["segments"], result["segments"][1:])
    )
    assert result["validation"]["long_work_human_validation"] is False


@pytest.mark.parametrize("failed_component", ["full_work", "transcript_fidelity"])
def test_optional_audio_failure_keeps_completed_expression(tmp_path, failed_component):
    import soundfile as sf

    path = tmp_path / "clip.wav"
    sf.write(path, np.ones(16000, dtype="float32") * 0.1, 16000)
    evaluator = OfflineEvaluator()
    evaluator._audio = lambda *args: {
        "coverage": {"decoded_fraction": 1},
        "expression": {"status": "experimental_score", "continuous": 3, "grade": None},
    }

    class Broken:
        def score(self, **kwargs):
            raise RuntimeError("fixture failure")

        def transcribe(self, *args):
            raise RuntimeError("fixture failure")

    if failed_component == "full_work":
        evaluator.full_work = Broken()
    else:
        evaluator.asr_model = "fixture"
        evaluator.asr = Broken()
    result = evaluator.evaluate(audio_path=path)
    assert result["scores"]["expression"]["continuous"] == 3
    assert result["failures"][0]["component"] == failed_component
    assert result["audio_evidence"]["coverage"]["decoded_fraction"] == 1


def test_overlength_rejected_before_any_inference():
    evaluator = OfflineEvaluator()
    evaluator.config = {"text": {"model_id": "test"}}

    class Tokenizer:
        def encode(self, *args, **kwargs):
            return list(range(513))

    evaluator.tokenizer = Tokenizer()
    evaluator._load_text = lambda: None

    def must_not_infer(*args):
        raise AssertionError("Overlength must never be scored")

    evaluator._text_probabilities = must_not_infer
    result = evaluator.evaluate(text="长稿")
    assert result["scores"]["text_humor"]["status"] == "unsupported_length"
    assert result["scores"]["text_humor"]["coverage"]["truncated"] is False


def test_evidence_is_measured_perturbation_not_a_generated_reason():
    evaluator = OfflineEvaluator()
    evaluator.config = {
        "text": {"model_id": "fixture", "validation": {"scope": "fixture"}}
    }

    class Tokenizer:
        def encode(self, *args, **kwargs):
            return [1, 2, 3]

    evaluator.tokenizer = Tokenizer()
    evaluator._load_text = lambda: None
    evaluator._text_probabilities = lambda texts: np.array(
        [[0, 0, 1, 0, 0] if "揭底" in t else [0, 1, 0, 0, 0] for t in texts],
        dtype=float,
    )
    text = "这是铺垫。这是揭底！"
    result = evaluator.evaluate(text=text, evidence=True)["scores"]["text_humor"]
    for e in result["evidence"]:
        assert text[e["start"] : e["end"]] == e["quote"]
        assert e["difference"] == pytest.approx(
            e["original_score"] - e["score_without_span"]
        )
        assert e["type"] == "sentence_deletion_sensitivity"


def test_corn_probabilities_and_losses_have_finite_gradients():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from openmic.evaluation_neural import GradedEncoder
    from torch import nn

    class Backbone(nn.Module):
        config = type("Config", (), {"hidden_size": 4})()

    for objective in ("ce", "corn", "mse", "ce_rank"):
        model = GradedEncoder(Backbone(), objective)
        logits = torch.randn(5, 4 if objective == "corn" else 5, requires_grad=True)
        probabilities = model.probabilities(logits)
        assert torch.all(probabilities >= 0)
        assert torch.allclose(probabilities.sum(1), torch.ones(5))
        loss = model.loss(logits, torch.arange(5))
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_lora_adapter_roundtrip_uses_each_saved_classification_head(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from openmic.evaluation_qwen import QwenGradeEnsemble, encode_text
    import copy

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            ids = [1] + [4 + ord(c) % 25 for c in text[-5:]] + [2]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        def pad(self, items, **kwargs):
            width = max(len(i["input_ids"]) for i in items)
            return transformers.BatchEncoding(
                {
                    k: torch.tensor([[0] * (width - len(x[k])) + x[k] for x in items])
                    for k in ("input_ids", "attention_mask")
                }
            )

    tok = Tokenizer()
    torch.manual_seed(12)
    cfg = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_labels=5,
        pad_token_id=0,
        head_dim=16,
    )
    base = transformers.Qwen3ForSequenceClassification(cfg)
    base.save_pretrained(tmp_path / "base", safe_serialization=True)
    model = peft.get_peft_model(
        copy.deepcopy(base),
        peft.LoraConfig(
            task_type=peft.TaskType.SEQ_CLS,
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    head = [
        p
        for name, p in model.named_parameters()
        if name.endswith("score.modules_to_save.default.weight")
    ]
    assert len(head) == 1
    texts = ["例子甲。", "另外一个例子乙！"]
    batch = tok.pad([encode_text(tok, t) for t in texts])
    expected = []
    for i, sign in enumerate([1, -1]):
        with torch.no_grad():
            head[0].zero_()
            head[0][0, 0] = sign * 3
        model.eval()
        with torch.inference_mode():
            expected.append(model(**batch).logits.float().softmax(-1).numpy())
        model.save_pretrained(tmp_path / f"adapter-{i}", safe_serialization=True)
    assert not np.allclose(expected[0], expected[1])
    restored = QwenGradeEnsemble(
        tmp_path / "base",
        [tmp_path / "adapter-0", tmp_path / "adapter-1"],
        tok,
        device="cpu",
    )
    assert np.allclose(
        restored.probabilities(texts), np.mean(expected, axis=0), atol=1e-6
    )
