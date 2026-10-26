"""Optional real ASR content check. This is not a MOS or human listening test."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import requests

from openmic.tts import TTSConfig
from openmic.tts_plan import Speech, compile_script, plain_text


def normalize_text(text):
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text)).lower()


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, first in enumerate(left, 1):
        current = [i]
        for j, second in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (first != second)))
        previous = current
    return previous[-1]


def load_reference(metrics_path, metadata):
    """Use the rendered plan, including optional inserted fillers, as reference.

    The raw script is a legacy fallback only. Hash validation prevents scoring
    an audio clip against text that belongs to another or pre-filler plan.
    """
    directory = Path(metrics_path).parent
    plan_path = directory.parent / "plan.json"
    timeline_path = directory / "timeline.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        if not isinstance(plan, list):
            raise ValueError("plan.json必须是事件数组")
        texts = [plain_text(event["text"]) for event in plan
                 if isinstance(event, dict) and isinstance(event.get("text"), str)]
        plan_hash = hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if metadata.get("plan_sha256") and metadata["plan_sha256"] != plan_hash:
            raise ValueError("ASR参考计划与metrics的plan_sha256不一致")
        source = str(plan_path)
    elif timeline_path.is_file():
        timeline = json.loads(timeline_path.read_text(encoding="utf-8-sig"))
        texts = [plain_text(event["text"]) for event in timeline
                 if event.get("kind") == "speech"]
        source = str(timeline_path)
    else:
        script_path = directory.parent / "script.txt"
        if not script_path.is_file():
            raise ValueError("找不到ASR参考plan、timeline或script")
        texts = [plain_text(event.text) for event in
                 compile_script(script_path.read_text(encoding="utf-8-sig"))
                 if isinstance(event, Speech)]
        source = str(script_path)
    if not texts:
        raise ValueError("ASR参考没有可朗读正文")
    text_hash = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
    if metadata.get("spoken_text_sha256") and metadata["spoken_text_sha256"] != text_hash:
        raise ValueError("ASR参考正文与metrics的spoken_text_sha256不一致，拒绝计算错误CER")
    return {"expected_text": "".join(texts), "reference_source": source,
            "reference_sha256": text_hash,
            "reference_verified": bool(metadata.get("spoken_text_sha256"))}


def find_cached_transcription(rows, audio_hash, reference_hash, model, condition):
    """Content identity survives moving this independent run to Linux."""
    return next((row for row in rows if row.get("audio_sha256") == audio_hash
                 and row.get("reference_sha256") == reference_hash
                 and row.get("model") == model and row.get("condition") == condition
                 and "error" not in row), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+", default=["performance"])
    parser.add_argument("--model", default="FunAudioLLM/SenseVoiceSmall")
    args = parser.parse_args()
    config = TTSConfig.from_env()
    rows = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else []
    for metric_path in sorted(args.runs.rglob("metrics.json")):
        meta = json.loads(metric_path.read_text(encoding="utf-8"))
        if meta.get("condition") not in args.conditions or meta.get("synthetic"):
            continue
        audio_path = metric_path.with_name("audio.wav")
        audio_hash = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        reference_info = load_reference(metric_path, meta)
        cached = find_cached_transcription(rows, audio_hash, reference_info["reference_sha256"],
                                           args.model, meta["condition"])
        if cached is not None:
            if cached.get("audio") != str(audio_path):
                cached.setdefault("original_audio_path", cached.get("audio"))
                cached.update(audio=str(audio_path), reference_source=reference_info["reference_source"])
                args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Already transcribed " + str(audio_path), flush=True)
            continue
        expected = reference_info["expected_text"]
        started = time.perf_counter()
        print("Transcribing " + str(audio_path), flush=True)
        try:
            for attempt in range(3):
                with audio_path.open("rb") as audio:
                    response = requests.post(config.base_url.rstrip("/") + "/audio/transcriptions",
                                             headers={"Authorization": "Bearer " + config.api_key},
                                             files={"file": ("audio.wav", audio, "audio/wav")},
                                             data={"model": args.model}, timeout=(15, 180))
                if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                    break
                print(f"ASR HTTP {response.status_code}; retry {attempt + 1}/2", flush=True)
                time.sleep(2)
            if response.status_code != 200:
                raise RuntimeError(f"ASR HTTP {response.status_code}")
            result = response.json()
            transcript = result.get("text", "")
            reference, hypothesis = normalize_text(expected), normalize_text(transcript)
            row = {"audio": str(audio_path), "audio_sha256": audio_hash, "condition": meta["condition"],
                   **reference_info, "transcript": transcript,
                   "language": result.get("language"), "model": args.model,
                   "reference_characters": len(reference), "hypothesis_characters": len(hypothesis),
                   "character_error_rate": edit_distance(reference, hypothesis) / max(1, len(reference)),
                   "asr_seconds": time.perf_counter() - started,
                   "human_review_pending": True,
                   "note": "ASR proxy: punctuation removed; Chinese/Arabic number writing is NOT normalized. "
                           "Differences may be ASR errors or notation differences, not necessarily TTS errors. "
                           "Not MOS or proof of emotion/emphasis quality"}
        except requests.RequestException:
            row = {"audio": str(audio_path), "error": "ASR network failure", "human_review_pending": True}
        except (RuntimeError, ValueError) as exc:
            row = {"audio": str(audio_path), "error": str(exc).replace(config.api_key, "[REDACTED]"),
                   "human_review_pending": True}
        rows = [old for old in rows if old.get("audio") != str(audio_path)]
        rows.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in row.items()
                          if key not in {"expected_text", "transcript"}}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
