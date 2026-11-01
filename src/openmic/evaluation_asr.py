"""Full waveform ASR audit. Character fidelity is not human intelligibility."""

import unicodedata

import numpy as np


def normalize_transcript(text):
    from opencc import OpenCC

    text = OpenCC("t2s").convert(unicodedata.normalize("NFKC", text)).lower()
    return "".join(c for c in text if c.isalnum())


def character_error(reference, hypothesis):
    ref = normalize_transcript(reference)
    hyp = normalize_transcript(hypothesis)
    if not ref:
        raise ValueError("Normalized reference transcript is empty")
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(
                min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return {
        "reference_characters": len(ref),
        "edit_distance": previous[-1],
        "character_error_rate": previous[-1] / len(ref),
        "normalization": "NFKC, simplified Chinese, lowercase, alphanumeric characters only",
        "meaning": "ASR transcript fidelity; not a listener intelligibility score",
    }


class WhisperAudit:
    def __init__(self, model_path, device="cuda"):
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = (
            AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=self.dtype,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )

    def transcribe(self, wave):
        import torch

        texts = []
        chunks = []
        cap = 25 * 16000
        for start in range(0, len(wave), cap):
            chunk = wave[start : start + cap]
            inputs = self.processor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs = inputs.to(self.device)
            inputs["input_features"] = inputs["input_features"].to(self.dtype)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    language="zh",
                    task="transcribe",
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=440,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            # Whisper's default tensor-return path removes EOS. Inspect the
            # underlying generation output, not the postprocessed transcript IDs.
            ids = generated.sequences
            steps = len(generated.scores)
            ended = bool(ids[0, -1].item() == self.model.generation_config.eos_token_id)
            hit_limit = steps >= 440 and not ended
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
            texts.append(text)
            chunks.append(
                {
                    "start_frame": start,
                    "end_frame": start + len(chunk),
                    "sample_rate": 16000,
                    "decoder_ended": ended,
                    "decoder_hit_limit": hit_limit,
                    "generated_steps": steps,
                    "output_tokens": len(ids[0]),
                    "text": text,
                }
            )
        return {
            "transcript": "".join(texts),
            "status": (
                "transcribed"
                if not any(c["decoder_hit_limit"] for c in chunks)
                else "decoder_limit"
            ),
            "chunks": chunks,
            "coverage": {
                "waveform_frames": len(wave),
                "processed_source_frames": sum(
                    c["end_frame"] - c["start_frame"] for c in chunks
                ),
                "processed_fraction": 1.0,
                "sample_rate": 16000,
                "chunk_seconds": 25,
                "overlap_seconds": 0,
            },
            "model_id": "openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
            "limitations": [
                "Chunk boundaries may cause transcription errors",
                "Known/unknown pretraining overlap not excluded",
                "Not a human intelligibility rating",
            ],
        }
