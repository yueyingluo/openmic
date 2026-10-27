"""Synthetic fixtures test validation only; they are never human observations."""

import copy
import hashlib
import json

import pytest

from v3_listening import (
    DIMENSIONS,
    PROTOCOL,
    build,
    convert,
    evaluate_humans,
    extract_predictions,
    identity,
)


def fixture():
    mapping = [
        {
            "blind_id": f"clip-{i:02d}",
            "sample_id": f"synthetic-{i}",
            "topic_id": str(i // 2),
            "audio_sha256": "synthetic",
            "formal_test_eligible": False,
        }
        for i in range(1, 5)
    ]
    export = {
        "protocol": PROTOCOL,
        "dataset_id": identity(mapping),
        "ratings": [
            {
                "sample_id": row["blind_id"],
                "rater_id": "synthetic-listener",
                "presentation_order": index,
                **{d + "_score": None for d in DIMENSIONS},
                "understood": None,
                "played_fraction": 0,
                "full_listened": False,
                "previously_heard": False,
                "comment": "",
            }
            for index, row in enumerate(mapping, 1)
        ],
    }
    return mapping, export


def test_blank_exports_stay_blank_and_cannot_create_validation():
    mapping, export = fixture()
    rows = convert(export, mapping)
    assert all(r[d + "_score"] is None for r in rows for d in DIMENSIONS)
    samples = [
        {
            "sample_id": r["sample_id"],
            "topic_id": r["topic_id"],
            "split": "pilot",
            "dataset_id": identity(mapping),
            "audio_sha256": r["audio_sha256"],
        }
        for r in mapping
    ]
    report = evaluate_humans(samples, rows, [], n_bootstrap=20)
    assert report["formal_test"] is False
    for value in report["dimensions"].values():
        assert value["valid_human_samples"] == 0
        assert value["status"] == "awaiting_human_ratings_or_predictions"


def test_scored_items_require_complete_playback_and_explicit_declaration():
    mapping, export = fixture()
    row = export["ratings"][0]
    row.update(
        humor_score=2, understood="yes", played_fraction=0.94, full_listened=True
    )
    with pytest.raises(ValueError, match="95%"):
        convert(export, mapping)
    row.update(played_fraction=1, full_listened=False)
    with pytest.raises(ValueError, match="full listening"):
        convert(export, mapping)
    row["full_listened"] = True
    assert convert(export, mapping)[0]["humor_score"] == 2


def test_word_clarity_can_be_rated_without_content_comprehension():
    mapping, export = fixture()
    row = export["ratings"][0]
    row.update(
        intelligibility_score=4, understood="no", played_fraction=1, full_listened=True
    )
    assert convert(export, mapping)[0]["intelligibility_score"] == 4
    row["expression_score"] = 3
    with pytest.raises(ValueError, match="comprehension"):
        convert(export, mapping)
    row["expression_score"] = None
    row["humor_score"] = 3
    with pytest.raises(ValueError, match="comprehension"):
        convert(export, mapping)


def test_wrong_pack_missing_items_and_mixed_raters_rejected():
    mapping, original = fixture()
    for mutation in (
        lambda e: e.update(dataset_id="wrong"),
        lambda e: e["ratings"].pop(),
        lambda e: e["ratings"][0].update(rater_id="someone-else"),
        lambda e: e["ratings"][0].update(presentation_order=2),
        lambda e: e["ratings"][0].update(played_fraction=float("nan")),
    ):
        export = copy.deepcopy(original)
        mutation(export)
        with pytest.raises(ValueError):
            convert(export, mapping)
    for invalid in (True, 0, 6, 2.0):
        export = copy.deepcopy(original)
        export["ratings"][0].update(
            humor_score=invalid, understood="yes", played_fraction=1, full_listened=True
        )
        with pytest.raises(ValueError, match="integer"):
            convert(export, mapping)


def test_three_dimensions_and_rater_counts_remain_separate():
    mapping, export = fixture()
    for i, row in enumerate(export["ratings"], 1):
        row.update(
            humor_score=i,
            expression_score=5 - i,
            intelligibility_score=i,
            understood="yes",
            played_fraction=1,
            full_listened=True,
        )
    ratings = convert(export, mapping)
    samples = [
        {
            "sample_id": r["sample_id"],
            "topic_id": r["topic_id"],
            "split": "pilot",
            "dataset_id": identity(mapping),
            "audio_sha256": r["audio_sha256"],
        }
        for r in mapping
    ]
    predictions = [
        {
            "sample_id": row["sample_id"],
            "dimension": dimension,
            "model_id": "synthetic-fixed-model",
            "score": index,
            "audio_sha256": row["audio_sha256"],
        }
        for dimension in DIMENSIONS
        for index, row in enumerate(mapping, 1)
    ]
    # One person cannot pass the default three-person requirement.
    default = evaluate_humans(samples, ratings, predictions, n_bootstrap=20)
    assert all(r["valid_human_samples"] == 0 for r in default["dimensions"].values())
    personal = evaluate_humans(
        samples, ratings, predictions, min_raters=1, n_bootstrap=20
    )
    for dimension, expected in (
        ("humor", 1),
        ("expression", -1),
        ("intelligibility", 1),
    ):
        result = personal["dimensions"][dimension]
        assert result["inference_scope"] == "individual_preference_only"
        assert result["models"]["synthetic-fixed-model"]["common_valid_set"][
            "spearman"
        ] == pytest.approx(expected)
    with pytest.raises(ValueError, match="Duplicate key"):
        evaluate_humans(samples, ratings + ratings, predictions, n_bootstrap=20)
    mixed = copy.deepcopy(ratings)
    mixed[0]["dataset_id"] = "different"
    with pytest.raises(ValueError, match="different listening packs"):
        evaluate_humans(samples, mixed, predictions, n_bootstrap=20)
    wrong_audio = copy.deepcopy(predictions)
    wrong_audio[0]["audio_sha256"] = "different"
    with pytest.raises(ValueError, match="rated waveform"):
        evaluate_humans(samples, ratings, wrong_audio, n_bootstrap=20)
    predictions[0]["dimension"] = "quality"
    with pytest.raises(ValueError, match="dimension"):
        evaluate_humans(samples, ratings, predictions, n_bootstrap=20)


def test_pack_hides_text_conditions_scores_and_keeps_mapping_outside(tmp_path):
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"synthetic audio fixture")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "private-original-id",
                "topic_id": "private-topic",
                "audio_path": str(audio),
                "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "text": "private script",
                "condition": "private-condition",
                "score": 4,
            }
        )
        + "\n"
    )
    output = tmp_path / "pack"
    info = build(manifest, output)
    html = (output / "index.html").read_text()
    assert "__PACK_JSON__" not in html
    assert all(
        s not in html
        for s in ("private-original-id", "private script", "private-condition")
    )
    assert info["contains_human_scores"] is False
    assert info["formal_test_eligible"] is False
    assert (tmp_path / "pack.mapping.private.json").exists()
    assert not list(output.glob("*private*"))
    with pytest.raises(ValueError, match="new pack"):
        build(manifest, output)
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        build(manifest, tmp_path / "other")
    assert not (tmp_path / "other").exists()


def test_prediction_extraction_keeps_failures_and_native_scale():
    rows = [
        {
            "sample_id": "synthetic-1",
            "result": {
                "evaluator_version": "synthetic",
                "bundle_id": "synthetic-bundle",
                "scores": {
                    "whole_audio_humor": {
                        "status": "experimental_score",
                        "continuous": 2.4,
                        "scale": [1, 5],
                    },
                    "expression": {"status": "failed", "continuous": None},
                    "intelligibility": {
                        "status": "experimental_aggregate",
                        "continuous": 7.2,
                        "scale": [0, 10],
                        "model_id": "synthetic-intelligibility",
                    },
                },
            },
        }
    ]
    predictions = extract_predictions(rows)
    assert len(predictions) == 3
    assert predictions[1]["score"] is None
    assert predictions[2]["source_scale"] == [0, 10]
    assert predictions[2]["score"] == 7.2
    other = copy.deepcopy(rows[0])
    other["result"]["bundle_id"] = "different"
    with pytest.raises(ValueError, match="bundles"):
        extract_predictions(rows + [other])
