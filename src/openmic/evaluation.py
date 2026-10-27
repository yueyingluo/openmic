"""Offline, versioned evaluation API. Scores carry coverage and validation scope.

Optional neural dependencies load only when a configured model is called.
Public human labels validate separate tasks, never an invented overall audio score.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

VERSION = "openmic-evaluation-v3"


def probability_summary(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if (
        p.shape != (5,)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.isclose(p.sum(), 1, atol=1e-5)
    ):
        raise ValueError("Expected five finite, normalized class probabilities")
    p = p / p.sum()
    return {
        "grade": int(p.argmax() + 1),
        "continuous": float(p @ np.arange(1, 6)),
        "scale": [1, 5],
        "classification_rule": "argmax of mean class probabilities; lowest grade breaks ties",
        "continuous_rule": "expectation of grade probabilities",
        "probabilities": p.tolist(),
        "probability_calibration": "not established; model probabilities are not reader-consensus confidence",
    }


def missing(status, reason):
    return {"status": status, "grade": None, "continuous": None, "reason": reason}


class OfflineEvaluator:
    """Reuse one instance for repeated calls; all models are local and read-only.

    A bundle is created by evaluation/runtime/export_v3_bundle.py. Do not load
    untrusted joblib model files. No network calls, ASR, generation, or paid APIs.
    """

    def __init__(
        self, bundle=None, device="cpu", experimental_omni_model=None, asr_model=None
    ):
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        self.device = device
        self.root = Path(bundle).resolve() if bundle else None
        self.config = (
            json.loads((self.root / "bundle.json").read_text()) if self.root else {}
        )
        if self.config and self.config.get("version") != VERSION:
            raise ValueError("Unsupported bundle version")
        if self.config:
            manifest = json.loads((self.root / "manifest.json").read_text())
            bound = {k: v for k, v in self.config.items() if k != "bundle_id"}
            identity = hashlib.sha256(
                json.dumps(
                    {"files": manifest, "config": bound}, sort_keys=True
                ).encode()
            ).hexdigest()
            if identity != self.config.get("bundle_id"):
                raise ValueError("Bundle configuration/manifest identity mismatch")
        self.text_models = None
        self.tokenizer = None
        self.audio_encoder = None
        self.full_work = None
        self.asr_model = asr_model
        self.asr = None
        if experimental_omni_model:
            from .evaluation_full_work import FullWorkOmni

            self.full_work = FullWorkOmni(experimental_omni_model, device)

    def _asset(self, relative):
        path = (self.root / relative).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("Bundle asset escapes root")
        return path

    def verify_assets(self):
        if not self.root:
            raise ValueError("No bundle configured")
        manifest = json.loads((self.root / "manifest.json").read_text())
        for relative, expected in manifest.items():
            h = hashlib.sha256()
            with self._asset(relative).open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError("Bundle asset hash mismatch: " + relative)
        return {"verified_files": len(manifest), "bundle_id": self.config["bundle_id"]}

    def _load_text(self):
        if self.text_models is None:
            from transformers import AutoTokenizer

            cfg = self.config["text"]
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._asset(cfg["tokenizer"]), local_files_only=True
            )
            if cfg.get("model_type") == "qwen_lora":
                from .evaluation_qwen import QwenGradeEnsemble

                self.qwen = QwenGradeEnsemble(
                    self._asset(cfg["base_path"]),
                    [self._asset(p) for p in cfg["checkpoints"]],
                    self.tokenizer,
                    self.device,
                )
                self.text_models = []
            else:
                from .evaluation_neural import GradedEncoder

                self.text_models = [
                    GradedEncoder.load(self._asset(p)).to(self.device).eval()
                    for p in cfg["checkpoints"]
                ]

    def _text_probabilities(self, texts):
        import torch

        self._load_text()
        if self.config["text"].get("model_type") == "qwen_lora":
            return self.qwen.probabilities(texts)
        encoded = [self.tokenizer(t, truncation=False) for t in texts]
        if any(len(e["input_ids"]) > 512 for e in encoded):
            raise ValueError("No text truncation allowed")
        batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device,
                dtype=torch.bfloat16,
                enabled=self.device == "cuda",
            ):
                predictions = [
                    m.probabilities(m(**batch)).cpu().numpy() for m in self.text_models
                ]
        return np.mean(predictions, axis=0)

    def _text(self, text, evidence):
        if not text.strip():
            return missing("invalid_input", "Text is empty")
        if not self.config.get("text"):
            return missing("unconfigured", "A local text model bundle is required")
        self._load_text()
        cfg = self.config["text"]
        limit = cfg.get("max_tokens", 512)
        if cfg.get("model_type") == "qwen_lora":
            from .evaluation_qwen import encode_text

            tokens = len(encode_text(self.tokenizer, text)["input_ids"])
        else:
            tokens = len(self.tokenizer.encode(text, truncation=False))
        coverage = {
            "characters": len(text),
            "tokens": tokens,
            "model_limit_tokens": limit,
            "truncated": False,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        if tokens > limit:
            if self.full_work is not None:
                return self.full_work.score(text=text)["text_humor"]
            return {
                **missing(
                    "unsupported_length",
                    f"Complete text exceeds {limit} tokens; no score was computed",
                ),
                "coverage": {**coverage, "scored_fraction": 0.0},
            }
        p = self._text_probabilities([text])[0]
        result = {
            "status": "scored",
            **probability_summary(p),
            "coverage": {**coverage, "scored_fraction": 1.0},
            "validation": self.config["text"]["validation"],
            "basis": {
                "type": "supervised_grade_prediction",
                "source": "CMHL single-annotator 1..5 labels",
                "model_id": self.config["text"]["model_id"],
                "probabilities_are_not_explanations": True,
            },
            "evidence": [],
        }
        if tokens > cfg.get("training_max_observed_tokens", 512):
            result["status"] = "experimental_score"
            result["length_validation"] = (
                "Complete input scored; longer than all training/development examples and not validated on full-work human labels"
            )
        if evidence:
            spans = list(re.finditer(r"[^。！？!?；;\n]+[。！？!?；;\n]*", text))
            variants = []
            kept = []
            for span in spans[:12]:
                variant = text[: span.start()] + text[span.end() :]
                if variant.strip():
                    variants.append(variant)
                    kept.append(span)
            if variants:
                ablated = self._text_probabilities(variants) @ np.arange(1, 6)
                changes = []
                for span, score in zip(kept, ablated):
                    changes.append(
                        {
                            "type": "sentence_deletion_sensitivity",
                            "start": span.start(),
                            "end": span.end(),
                            "quote": span.group(),
                            "original_score": result["continuous"],
                            "score_without_span": float(score),
                            "difference": float(result["continuous"] - score),
                        }
                    )
                result["evidence"] = sorted(
                    changes, key=lambda e: abs(e["difference"]), reverse=True
                )[:3]
            result["evidence_scope"] = {
                "evaluated_spans": len(variants),
                "total_spans": len(spans),
                "meaning": "Measured model sensitivity to deletion; not a causal account of humor or a human explanation",
            }
        return result

    def _audio(self, path, text):
        from .evaluation_audio import load_waveform, waveform_descriptors, DESCRIPTORS

        wave, coverage = load_waveform(path)
        descriptors = waveform_descriptors(wave)
        measurements = dict(zip(DESCRIPTORS, map(float, descriptors)))
        cfg = self.config.get("audio")
        result = {
            "coverage": {
                **coverage,
                "audio_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            },
            "measurements": measurements,
            "measurement_scope": "Signal measurements do not establish intelligibility or humor",
            "expression": missing(
                "unconfigured", "A local expressive-label model bundle is required"
            ),
        }
        if not cfg:
            return result
        if not 3 <= coverage["duration_seconds"] <= 20:
            result["expression"] = missing(
                "unsupported_duration",
                "Expression model was evaluated on complete 3..20 second clips; whole waveform measurements are available",
            )
            return result
        method = cfg["method"]
        if method == "text_embedding":
            result["expression"] = missing(
                "no_validated_audio_model",
                "Text-only control won development selection; it cannot establish a waveform-based expression score",
            )
            return result
        if method == "audio_and_text" and (text is None or not text.strip()):
            result["expression"] = missing(
                "missing_transcript",
                "Selected model requires the actual transcript in addition to waveform",
            )
            return result
        import joblib

        if method == "waveform":
            features = descriptors[None, :]
        else:
            from transformers import AutoFeatureExtractor
            from .evaluation_audio import load_audio_encoder, audio_embeddings

            if self.audio_encoder is None:
                self.audio_encoder = load_audio_encoder(
                    self._asset(cfg["encoder"]), self.device, standalone=True
                )
                self.audio_extractor = AutoFeatureExtractor.from_pretrained(
                    self._asset(cfg["extractor"]), local_files_only=True
                )
            features, frames = audio_embeddings(
                [wave], self.audio_extractor, self.audio_encoder, self.device
            )
            result["coverage"]["audio_feature_frames"] = frames[0]
            if method == "audio_and_text":
                from transformers import AutoTokenizer, AutoModel
                import torch

                path = self._asset(cfg["embedding_encoder"])
                tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
                model = (
                    AutoModel.from_pretrained(path, local_files_only=True)
                    .to(self.device)
                    .eval()
                )
                tokens = tok.encode(text, add_special_tokens=False, truncation=False)
                chunks = [tokens[i : i + 510] for i in range(0, len(tokens), 510)] or [
                    []
                ]
                batch = tok.pad(
                    [tok.prepare_for_model(c) for c in chunks],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.inference_mode():
                    vectors = model(**batch).last_hidden_state[:, 0].float()
                    weights = torch.tensor(
                        [max(1, len(c)) for c in chunks], device=self.device
                    )
                    vector = (vectors * weights[:, None]).sum(0) / weights.sum()
                    vector = (
                        torch.nn.functional.normalize(vector, dim=0)
                        .cpu()
                        .numpy()[None, :]
                    )
                features = np.concatenate([features, vector], axis=1)
        regressor = joblib.load(self._asset(cfg["regressor"]))
        score = float(np.clip(regressor.predict(features)[0], 0, 5))
        result["expression"] = {
            "status": "experimental_score",
            "continuous": score,
            "grade": None,
            "scale": [0, 5],
            "task": "ceaeval_contextual_appropriateness",
            "validation": cfg["validation"],
            "model_id": cfg["model_id"],
            "basis": "Ridge regression fitted only to human expressive appropriateness labels; Mandarin audiobook clips",
            "limitations": [
                "No project stand-up listener validation",
                "Not a humor score",
                "Not an intelligibility score",
            ],
        }
        return result

    def _intelligibility(self, path):
        from .evaluation_audio import (
            load_waveform,
            waveform_descriptors,
            audio_embeddings,
            load_audio_encoder,
        )
        import joblib

        cfg = self.config["intelligibility"]
        wave, coverage = load_waveform(path)
        # Native utterances are kept complete. Long works use an explicitly
        # experimental time-weighted aggregate with every source frame covered.
        ranges = [(0, len(wave))]
        if len(wave) > 30 * 16000:
            ranges = [
                (i, min(i + 4 * 16000, len(wave)))
                for i in range(0, len(wave), 4 * 16000)
            ]
            if len(ranges) > 1 and ranges[-1][1] - ranges[-1][0] < 2 * 16000:
                ranges[-2] = (ranges[-2][0], ranges[-1][1])
                ranges.pop()
        if sum(b - a for a, b in ranges) != len(wave):
            raise ValueError("Incomplete intelligibility segment coverage")
        method = cfg["method"]
        model = joblib.load(self._asset(cfg["regressor"]))
        if method != "waveform" and not hasattr(self, "intell_encoder"):
            from transformers import AutoFeatureExtractor

            self.intell_encoder = load_audio_encoder(
                self._asset(cfg["encoder"]), self.device, standalone=True
            )
            self.intell_extractor = AutoFeatureExtractor.from_pretrained(
                self._asset(cfg["extractor"]), local_files_only=True
            )
        segments = []
        for start in range(0, len(ranges), 8):
            batch = ranges[start : start + 8]
            valid = [(a, b) for a, b in batch if np.max(np.abs(wave[a:b])) > 1e-6]
            if valid:
                waves = [wave[a:b] for a, b in valid]
                descriptors = np.stack([waveform_descriptors(w) for w in waves])
                if method == "waveform":
                    x = descriptors
                else:
                    x, _ = audio_embeddings(
                        waves, self.intell_extractor, self.intell_encoder, self.device
                    )
                    if method == "audio_and_waveform":
                        x = np.concatenate([x, descriptors], axis=1)
                predictions = np.clip(model.predict(x), 0, 10)
                measured = {
                    interval: float(score)
                    for interval, score in zip(valid, predictions)
                }
            else:
                measured = {}
            for a, b in batch:
                segments.append(
                    {
                        "start_frame": a,
                        "end_frame": b,
                        "sample_rate": 16000,
                        "continuous": measured.get((a, b)),
                        "status": "scored" if (a, b) in measured else "no_signal",
                    }
                )
        scored = [s for s in segments if s["continuous"] is not None]
        if not scored:
            return missing(
                "invalid_input", "No nonzero signal for intelligibility prediction"
            )
        weights = np.array([s["end_frame"] - s["start_frame"] for s in scored])
        values = np.array([s["continuous"] for s in scored])
        result = {
            "status": (
                "experimental_aggregate" if len(ranges) > 1 else "experimental_score"
            ),
            "grade": None,
            "continuous": float(np.average(values, weights=weights)),
            "scale": [0, 10],
            "task": "tmhintqi_character_recognition_intelligibility",
            "model_id": cfg["model_id"],
            "validation": cfg["validation"],
            "coverage": {
                **coverage,
                "processed_fraction": 1.0,
                "score_coverage_fraction": float(weights.sum() / len(wave)),
                "truncated": False,
            },
            "segments": segments,
            "basis": "Regression fitted to human intelligibility_score; complete waveform features; native target differs from quality MOS",
            "explanation": None,
            "minimum_segment_score": float(values.min()),
            "aggregation": (
                "complete native clip"
                if len(ranges) == 1
                else "4 second nonoverlapping windows, tail under 2 seconds joined to previous; time-weighted mean"
            ),
            "limitations": [
                "No full-work human intelligibility validation",
                "Short noisy/enhanced sentence domain; speaker/content/style shift remains",
            ],
        }
        return result

    def evaluate(self, *, text=None, audio_path=None, evidence=False):
        if text is None and audio_path is None:
            raise ValueError("Provide text or audio_path")
        if text is not None and not isinstance(text, str):
            raise TypeError("text must be a string")
        scores = {
            "text_humor": missing("not_requested", "No text supplied"),
            "whole_audio_humor": (
                missing(
                    "human_validation_pending",
                    "Full-work human humor ratings are not available",
                )
                if audio_path is not None
                else missing("not_requested", "No audio supplied")
            ),
            "expression": missing("not_requested", "No audio supplied"),
            "intelligibility": (
                missing(
                    "human_validation_pending",
                    "No validated human intelligibility model; signal measurements cannot replace listener judgments",
                )
                if audio_path is not None
                else missing("not_requested", "No audio supplied")
            ),
        }
        result = {
            "schema_version": "1.0",
            "evaluator_version": VERSION,
            "bundle_id": self.config.get("bundle_id"),
            "runtime_device": self.device,
            "scores": scores,
            "failures": [],
        }
        if text is not None:
            try:
                if not self.config.get("text") and self.full_work is not None:
                    scores["text_humor"] = self.full_work.score(text=text)["text_humor"]
                else:
                    scores["text_humor"] = self._text(text, evidence)
            except (ValueError, OSError, RuntimeError, ImportError) as exc:
                scores["text_humor"] = missing("failed", str(exc))
                result["failures"].append(
                    {
                        "modality": "text",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        if audio_path is not None:
            errors = (ValueError, OSError, RuntimeError, ImportError)

            def audio_failure(component, exc):
                result["failures"].append(
                    {
                        "modality": "audio",
                        "component": component,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

            decoded = False
            try:
                from .evaluation_audio import load_waveform

                wave, coverage = load_waveform(audio_path)
                del wave
                result["audio_evidence"] = {"coverage": coverage}
                decoded = True
            except errors as exc:
                for key in ("whole_audio_humor", "expression", "intelligibility"):
                    scores[key] = missing("failed", str(exc))
                audio_failure("decode", exc)
            if decoded:
                try:
                    audio = self._audio(audio_path, text)
                    scores["expression"] = audio.pop("expression")
                    result["audio_evidence"] = audio
                except errors as exc:
                    scores["expression"] = missing("failed", str(exc))
                    audio_failure("expression", exc)
            if decoded and self.config.get("intelligibility"):
                try:
                    scores["intelligibility"] = self._intelligibility(audio_path)
                except errors as exc:
                    scores["intelligibility"] = missing("failed", str(exc))
                    audio_failure("intelligibility", exc)
            if decoded and self.full_work is not None:
                try:
                    full = self.full_work.score(text=text or "", audio_path=audio_path)
                    result["experimental_full_work_scores"] = full
                    scores["whole_audio_humor"] = full["whole_audio_humor"]
                    if not self.config.get("intelligibility"):
                        scores["intelligibility"] = full["intelligibility"]
                    if scores["expression"]["continuous"] is None:
                        scores["expression"] = full["expression"]
                except errors as exc:
                    scores["whole_audio_humor"] = missing("failed", str(exc))
                    audio_failure("full_work", exc)
            if decoded and self.asr_model is not None:
                try:
                    from .evaluation_asr import WhisperAudit, character_error

                    if self.asr is None:
                        self.asr = WhisperAudit(self.asr_model, self.device)
                    wave, _ = load_waveform(audio_path)
                    fidelity = self.asr.transcribe(wave)
                    if text and text.strip():
                        fidelity["reference_metrics"] = character_error(
                            text, fidelity["transcript"]
                        )
                    result["transcript_fidelity"] = fidelity
                except errors as exc:
                    result["transcript_fidelity"] = {
                        "status": "failed",
                        "reason": str(exc),
                    }
                    audio_failure("transcript_fidelity", exc)
        statuses = [
            scores[k]["status"]
            for k in (
                ["text_humor"]
                if audio_path is None
                else ["whole_audio_humor", "expression", "intelligibility"]
                + (["text_humor"] if text is not None else [])
            )
        ]
        result["status"] = (
            "scored"
            if all(s == "scored" for s in statuses)
            else (
                "failed"
                if result["failures"] and all(s == "failed" for s in statuses)
                else "partial"
            )
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text")
    group.add_argument("--text-file", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--experimental-omni-model",
        type=Path,
        help="Opt in to unvalidated full-work text/audio scores using local Qwen2.5-Omni weights",
    )
    parser.add_argument(
        "--asr-model",
        type=Path,
        help="Optional local Whisper transcript fidelity audit; not a human intelligibility score",
    )
    parser.add_argument(
        "--verify-bundle",
        action="store_true",
        help="Verify all local model files against the bundle SHA256 manifest",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="Measured sentence deletion sensitivity; no generated reasons",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = args.text_file.read_text() if args.text_file else args.text
    if text is None and args.audio is None:
        parser.error("Provide text or audio")
    if args.device == "cuda":
        import os
        import subprocess

        uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not uuid or not uuid.startswith("GPU-"):
            parser.error("CUDA_VISIBLE_DEVICES must name one approved idle GPU UUID")
        active = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader",
            ],
            text=True,
        )
        if any(line.split(",")[0].strip() == uuid for line in active.splitlines()):
            parser.error("Selected GPU already has a compute process")
    evaluator = OfflineEvaluator(
        args.bundle, args.device, args.experimental_omni_model, args.asr_model
    )
    verification = evaluator.verify_assets() if args.verify_bundle else None
    result = evaluator.evaluate(
        text=text, audio_path=args.audio, evidence=args.evidence
    )
    if verification:
        result["bundle_verification"] = verification
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as target:
            target.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
