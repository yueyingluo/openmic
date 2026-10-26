"""Render matched conditions and record sample-accurate injected silence."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from openmic.tts_backends import SAMPLE_RATE, decode_wav, encode_wav
from openmic.tts_plan import EMOTIONS, Event, Pause, Speech, plain_text


CONDITIONS = ("baseline", "timing", "emotion", "performance")


@dataclass
class RenderResult:
    audio: bytes
    metrics: Dict[str, Any]
    timeline: List[Dict[str, Any]]


def _preprocess(samples: np.ndarray) -> np.ndarray:
    # Trim only exterior low-energy samples, identically in all conditions.
    # This defines an engineering boundary, not a forced-aligned phoneme edge.
    active = np.flatnonzero(np.abs(samples.astype(np.int32)) > 104)  # about -50 dBFS
    if not len(active):
        raise ValueError("服务返回全静音/极低音量片段，不能当作成功语音")
    result = samples[active[0]:active[-1] + 1].copy()
    ramp = min(48, len(result) // 2)  # 3 ms, no crossfade across planned pauses
    if ramp:
        result[:ramp] = (result[:ramp] * np.linspace(0, 1, ramp)).astype("<i2")
        result[-ramp:] = (result[-ramp:] * np.linspace(1, 0, ramp)).astype("<i2")
    return result


def render(events: List[Event], backend: Any, *, condition: str = "performance",
           cache_dir: Optional[Path] = None,
           on_chunk: Optional[Callable[[bytes, Dict[str, Any]], None]] = None) -> RenderResult:
    if condition not in CONDITIONS:
        raise ValueError(f"condition 必须是 {CONDITIONS}")
    if not any(isinstance(e, Speech) for e in events):
        raise ValueError("计划没有正文")
    # Give the neutral segmented baseline ordinary short boundaries, rather
    # than making it unnaturally concatenate fully trimmed sentences.
    effective = []
    source = events if condition in {"timing", "performance"} else [e for e in events if isinstance(e, Speech)]
    for event in source:
        if effective and isinstance(effective[-1], Speech) and isinstance(event, Speech):
            effective.append(Pause(0.12, "ordinary_boundary"))
        effective.append(event)
    started = time.perf_counter()
    pieces, timeline = [], []
    position = speech_frames = pause_frames = requests_count = hits = 0
    request_seconds = 0.0
    first_chunk_seconds = None
    for event in effective:
        row = asdict(event)
        if isinstance(event, Pause):
            samples = np.zeros(round(event.seconds * SAMPLE_RATE), dtype="<i2")
            pause_frames += len(samples)
            row.update(kind="pause", requested_seconds=event.seconds)
        else:
            text = event.text if condition == "performance" else plain_text(event.text)
            speed = event.speed if condition == "performance" else 1.0
            emotion = event.emotion if condition in {"emotion", "performance"} else "neutral"
            instruction = f"请用{EMOTIONS[emotion]}的语气说。"
            args = {"text": text, "instruction": instruction, "speed": speed}
            key = hashlib.sha256(json.dumps({"backend": backend.identity, **args},
                                           sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            cache = Path(cache_dir) / (key + ".wav") if cache_dir is not None else None
            cached = cache is not None and cache.is_file()
            samples = None
            if cached:
                try:
                    samples = decode_wav(cache.read_bytes())
                    _preprocess(samples)
                except (ValueError, OSError):
                    cached = False
            if not cached:
                call_started = time.perf_counter()
                audio = backend.synthesize(**args)
                request_seconds += time.perf_counter() - call_started
                requests_count += 1
                samples = decode_wav(audio)
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    # Every cached object is independently playable, not a WAV byte concatenation.
                    temporary = cache.with_suffix(".tmp")
                    temporary.write_bytes(encode_wav(samples))
                    temporary.replace(cache)
            else:
                hits += 1
            original_frames = len(samples)
            samples = _preprocess(samples)
            speech_frames += len(samples)
            row.update(kind="speech", cache_hit=cached, actual_input=text,
                       actual_instruction=instruction, actual_speed=speed,
                       trimmed_frames=original_frames - len(samples))
        row.update(start_frame=position, end_frame=position + len(samples),
                   start_seconds=position / SAMPLE_RATE,
                   end_seconds=(position + len(samples)) / SAMPLE_RATE,
                   duration_seconds=len(samples) / SAMPLE_RATE)
        pieces.append(samples)
        timeline.append(row)
        position += len(samples)
        if row["kind"] == "speech" and first_chunk_seconds is None:
            first_chunk_seconds = time.perf_counter() - started
        if on_chunk and len(samples):
            on_chunk(encode_wav(samples), dict(row))
    audio = encode_wav(np.concatenate(pieces))
    elapsed = time.perf_counter() - started
    speech_duration = speech_frames / SAMPLE_RATE
    synthetic = bool(backend.synthetic)
    metrics = {
        "condition": condition, "backend": backend.identity, "synthetic": synthetic,
        "plan_sha256": hashlib.sha256(json.dumps([asdict(e) for e in events],
                                                  ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "spoken_text_sha256": hashlib.sha256("\n".join(plain_text(e.text) for e in events
                                                      if isinstance(e, Speech)).encode("utf-8")).hexdigest(),
        "sample_rate": SAMPLE_RATE, "channels": 1, "sample_width": 2,
        "duration_seconds": position / SAMPLE_RATE,
        "speech_seconds": speech_duration, "inserted_silence_seconds": pause_frames / SAMPLE_RATE,
        "wall_seconds": elapsed, "api_seconds": request_seconds,
        "first_speech_chunk_seconds": first_chunk_seconds,
        "speech_segments": sum(isinstance(e, Speech) for e in events),
        "api_calls": requests_count, "cache_hits": hits,
        "all_cache_miss": hits == 0,
        # RTF < 1 means faster than real time; speed_x > 1 means the same.
        # A synthetic run or a mixed/warm cache run is NOT a model benchmark.
        "rtf_speech": elapsed / speech_duration if speech_duration else None,
        "speed_x_speech": speech_duration / elapsed if elapsed else None,
        "speed_x_including_pauses": position / SAMPLE_RATE / elapsed if elapsed else None,
        "valid_cold_tts_benchmark": not synthetic and hits == 0,
        "in_target_duration": 180 <= position / SAMPLE_RATE <= 300,
        "pause_definition": "inserted PCM zero samples; perceptual gap also depends on model/internal silences",
        "ordinary_boundary_seconds": 0.12,
        "streaming_scope": "completed segment callbacks; not token streaming or continuous browser playback",
    }
    return RenderResult(audio, metrics, timeline)


def save_result(result: RenderResult, out_dir: Path) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "audio.wav").write_bytes(result.audio)
    (target / "metrics.json").write_text(json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "timeline.json").write_text(json.dumps(result.timeline, ensure_ascii=False, indent=2), encoding="utf-8")
