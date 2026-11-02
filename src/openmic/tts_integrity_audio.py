"""Preserve accepted raw speech PCM and insert explicit, measured silence.

This module never sends a request. The edge screen rejects suspicious model
endpoints; adding silence cannot recover words that a model did not synthesize.
"""

import hashlib
import json
import math
import struct
from typing import Any, Dict

import numpy as np

from openmic.tts_backends import SAMPLE_RATE, encode_wav
from openmic.tts_render import RenderResult


EDGE_WINDOW_SECONDS = 0.010
HEAD_RMS_LIMIT_DBFS = -35.0
TAIL_RMS_LIMIT_DBFS = -40.0
START_GUARD_SECONDS = 0.120
END_GUARD_SECONDS = 0.400
SENTENCE_PAUSE_FLOOR_SECONDS = 0.360
PROFILES = ("safe_course", "safe_club")
BOUNDARY_SECONDS = {
    "safe_course": {"turn": 0.800, "reaction": 2.000},
    "safe_club": {"turn": 0.480, "reaction": 0.850},
}
ADAPTER = "openmic.integrity-audio/1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_pcm(audio: bytes) -> np.ndarray:
    """Decode one complete PCM data chunk, without repairing or dropping bytes."""
    if not isinstance(audio, bytes):
        raise ValueError("audio must be immutable WAV bytes")
    if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise ValueError("audio is not a RIFF/WAVE file")
    if struct.unpack_from("<I", audio, 4)[0] != len(audio) - 8:
        raise ValueError("WAV RIFF length differs from the complete response")
    offset, fmt, pcm = 12, None, None
    while offset < len(audio):
        if offset + 8 > len(audio):
            raise ValueError("WAV ends inside a chunk header")
        kind, size = audio[offset:offset + 4], struct.unpack_from("<I", audio, offset + 4)[0]
        start, end = offset + 8, offset + 8 + size
        if end + (size % 2) > len(audio):
            raise ValueError("WAV chunk or its padding is truncated")
        if kind == b"fmt ":
            if fmt is not None or size < 16:
                raise ValueError("WAV must contain one complete PCM format chunk")
            fmt = struct.unpack_from("<HHIIHH", audio, start)
        elif kind == b"data":
            if pcm is not None:
                raise ValueError("multiple PCM data chunks are not supported")
            if size == 0 or size % 2:
                raise ValueError("WAV PCM data is empty or contains a partial frame")
            pcm = audio[start:end]
        offset = end + (size % 2)
    if fmt != (1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16):
        raise ValueError("audio must be 16000 Hz, mono, uncompressed 16-bit PCM")
    if pcm is None:
        raise ValueError("WAV has no PCM data chunk")
    return np.frombuffer(pcm, dtype="<i2").copy()


def _rms(samples: np.ndarray):
    value = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return value, 20 * math.log10(value / 32768.0) if value else None


def inspect_edges(audio: bytes) -> Dict[str, Any]:
    """Return a conservative endpoint screen; this is not a transcript check.

    The dBFS reference is 32768. Digital silence has rms=0 and rms_dbfs=None,
    keeping reports valid JSON rather than serializing negative infinity.
    """
    report = {
        "passed": False, "errors": [], "warnings": [],
        "raw_pcm_sha256": None, "nframes": 0,
        "sample_rate": SAMPLE_RATE, "channels": 1, "sample_width_bytes": 2,
        "head_rms": None, "tail_rms": None,
        "head_rms_dbfs": None, "tail_rms_dbfs": None,
        "window_seconds": EDGE_WINDOW_SECONDS,
        "head_rms_limit_dbfs": HEAD_RMS_LIMIT_DBFS,
        "tail_rms_limit_dbfs": TAIL_RMS_LIMIT_DBFS,
        "content_completeness_verified": False,
    }
    try:
        pcm = _strict_pcm(audio)
    except (ValueError, struct.error, TypeError) as error:
        report["errors"].append(str(error))
        return report
    window = min(len(pcm), round(EDGE_WINDOW_SECONDS * SAMPLE_RATE))
    head, head_dbfs = _rms(pcm[:window])
    tail, tail_dbfs = _rms(pcm[-window:])
    peak = int(np.max(np.abs(pcm.astype(np.int32))))
    report.update(raw_pcm_sha256=_sha(pcm.tobytes()), nframes=len(pcm),
                  duration_seconds=len(pcm) / SAMPLE_RATE,
                  window_frames=window, head_rms=head, tail_rms=tail,
                  head_rms_dbfs=head_dbfs, tail_rms_dbfs=tail_dbfs,
                  sample_peak=peak)
    if peak == 0:
        report["errors"].append("PCM is entirely digital silence")
    elif peak <= 104:
        report["warnings"].append("Entire PCM has very low peak amplitude; check speech audibility")
    if head_dbfs is not None and head_dbfs > HEAD_RMS_LIMIT_DBFS:
        report["errors"].append("Head 10 ms RMS exceeds -35 dBFS: possible clipped speech onset")
    if tail_dbfs is not None and tail_dbfs > TAIL_RMS_LIMIT_DBFS:
        report["errors"].append("Tail 10 ms RMS exceeds -40 dBFS: possible clipped speech ending")
    report["warnings"].append("Endpoint energy cannot certify that every source word was synthesized")
    report["passed"] = not report["errors"]
    return report


def _seconds(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be a finite nonnegative number")
    return float(value)


def render_integrity(plan: Dict[str, Any], audio_by_id: Dict[int, bytes],
                     profile: str = "safe_course") -> RenderResult:
    """Assemble validated raw clips in plan order, with no changes to speech.

    Existing source pauses are lower bounds. Zero source pauses remain in the
    timeline, but the new minimum sentence gap still applies. Guard, reaction
    and source durations use max(), never an added stack of separate silences.
    """
    if profile not in PROFILES:
        raise ValueError("profile must be safe_course or safe_club")
    from openmic.tts_integrity_plan import validate_execution_integrity_plan
    validation = validate_execution_integrity_plan(plan)
    if not validation.get("passed"):
        raise ValueError("Invalid integrity plan: " + "; ".join(validation.get("errors", [])))
    units = plan["units"]
    if not isinstance(audio_by_id, dict) or any(type(key) is not int for key in audio_by_id):
        raise ValueError("audio_by_id must be a dict with integer unit IDs")
    expected_ids = {unit["id"] for unit in units}
    if set(audio_by_id) != expected_ids:
        raise ValueError("audio_by_id must cover every plan unit exactly once, with no extra IDs")
    prepared = []
    for unit in units:
        audio = audio_by_id[unit["id"]]
        check = inspect_edges(audio)
        if not check["passed"]:
            raise ValueError("Rejected raw audio for unit " + str(unit["id"]) + ": " + "; ".join(check["errors"]))
        prepared.append((_strict_pcm(audio), check))
    # All inputs have passed before the first output buffer is assembled.
    pieces, speech_parts, timeline = [], [], []
    position = speech_frames = silence_frames = 0
    durations = BOUNDARY_SECONDS[profile]

    def pause(seconds: float, pause_type: str, **details: Any) -> None:
        nonlocal position, silence_frames
        seconds = _seconds(seconds, pause_type)
        count = round(seconds * SAMPLE_RATE)
        samples = np.zeros(count, dtype="<i2")
        timeline.append({"kind": "pause", "pause_type": pause_type,
                         "requested_seconds": seconds, "nframes": count,
                         "start_frame": position, "end_frame": position + count,
                         "start_seconds": position / SAMPLE_RATE,
                         "end_seconds": (position + count) / SAMPLE_RATE,
                         "duration_seconds": count / SAMPLE_RATE, **details})
        pieces.append(samples)
        position += count
        silence_frames += count

    first_source = _seconds(units[0]["source_pause_before_seconds"], "source pause")
    pause(max(START_GUARD_SECONDS, first_source), "start_guard",
          guard_floor_seconds=START_GUARD_SECONDS, source_pause_before_seconds=first_source,
          reason="max_of_start_guard_and_original_source_pause")
    previous = None
    for unit, (pcm, check) in zip(units, prepared):
        if previous is not None:
            base_changed = unit["base_unit_index"] != previous["base_unit_index"]
            requested_boundary = durations.get(unit["boundary_before"], 0.0) if base_changed else 0.0
            reaction = durations["reaction"] if base_changed and previous["base_unit_end"] and previous["intent"] in {"reveal", "button"} else 0.0
            source_pause = _seconds(unit["source_pause_before_seconds"], "source pause")
            seconds = max(SENTENCE_PAUSE_FLOOR_SECONDS, requested_boundary, reaction, source_pause)
            pause(seconds, "boundary", before_unit_id=unit["id"],
                  boundary_before=unit["boundary_before"], base_unit_changed=base_changed,
                  floor_seconds=SENTENCE_PAUSE_FLOOR_SECONDS,
                  planned_boundary_seconds=requested_boundary, reaction_seconds=reaction,
                  source_pause_before_seconds=source_pause,
                  reason="max_of_sentence_floor_boundary_previous_base_reaction_and_source_pause")
        count = len(pcm)
        row = {
            "kind": "speech", "unit_id": unit["id"],
            "base_unit_index": unit["base_unit_index"],
            "base_unit_start": unit["base_unit_start"], "base_unit_end": unit["base_unit_end"],
            "source_index": unit["source_index"], "source_start": unit["start"], "source_end": unit["end"],
            "text": unit["text"], "intent": unit["intent"], "focus": unit["focus"],
            "role": unit["role"], "beat": unit["beat"],
            "nframes": count, "raw_frames": count,
            "raw_pcm_sha256": check["raw_pcm_sha256"],
            "start_frame": position, "end_frame": position + count,
            "start_seconds": position / SAMPLE_RATE, "end_seconds": (position + count) / SAMPLE_RATE,
            "duration_seconds": count / SAMPLE_RATE,
            "trimmed_frames": 0, "fade_frames": 0, "local_gain_db": 0.0,
            "global_headroom_gain": 1.0, "edge_check": check,
            "synthesis_parameters_source": "accepted candidate provenance saved by calling pipeline",
        }
        timeline.append(row)
        # This is deliberately the complete original array, not a slice,
        # envelope, crossfade, normalization or a floating-point round trip.
        pieces.append(pcm)
        speech_parts.append(pcm)
        speech_frames += count
        position += count
        previous = unit
    source_tail = _seconds(plan["source_tail_pause_seconds"], "source tail pause")
    tail_reaction = durations["reaction"] if previous["base_unit_end"] and previous["intent"] in {"reveal", "button"} else 0.0
    pause(max(END_GUARD_SECONDS, source_tail, tail_reaction), "end_guard",
          guard_floor_seconds=END_GUARD_SECONDS, source_tail_pause_seconds=source_tail,
          reaction_seconds=tail_reaction, reason="max_of_end_guard_original_tail_pause_and_final_base_reaction")
    combined = np.concatenate(pieces).astype("<i2", copy=False)
    speech_pcm = b"".join(pcm.tobytes() for pcm in speech_parts)
    plan_json = json.dumps(plan, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    audio = encode_wav(combined)
    metrics = {
        "adapter": ADAPTER, "profile": profile, "condition": profile,
        "source_kind": "provided_pcm", "synthetic": None,
        "sample_rate": SAMPLE_RATE, "channels": 1, "sample_width_bytes": 2,
        "nframes": len(combined), "speech_frames": speech_frames, "pause_frames": silence_frames,
        "duration_seconds": len(combined) / SAMPLE_RATE,
        "speech_seconds": speech_frames / SAMPLE_RATE,
        "injected_silence_seconds": silence_frames / SAMPLE_RATE,
        "speech_pcm_sha256": _sha(speech_pcm), "audio_sha256": _sha(audio),
        "source_script_sha256": plan["source_script_sha256"],
        "spoken_text_sha256": plan["spoken_text_sha256"],
        "base_stage_plan_sha256": plan["base_stage_plan_sha256"],
        "plan_sha256": _sha(plan_json),
        "unit_count": len(units), "speech_count": len(units),
        "edge_window_seconds": EDGE_WINDOW_SECONDS,
        "head_rms_limit_dbfs": HEAD_RMS_LIMIT_DBFS, "tail_rms_limit_dbfs": TAIL_RMS_LIMIT_DBFS,
        "edge_guard_start_seconds": START_GUARD_SECONDS, "edge_guard_end_seconds": END_GUARD_SECONDS,
        "sentence_pause_floor_seconds": SENTENCE_PAUSE_FLOOR_SECONDS,
        "boundary_seconds": dict(durations),
        "pause_policy": "maximum of guard/floor, base-boundary, final-base reaction, and original source pause",
        "source_zero_pause_policy": "recorded as zero; the new sentence floor and reaction policy may be longer",
        "speech_processing": "complete raw PCM copied without trimming, fades, overlap, pitch shift or gain",
        "original_stage_intent_gain_executed": False,
        "per_sentence_normalization_executed": False,
        "global_headroom_gain": 1.0,
        "api_calls": 0, "api_scope": "renderer only; candidate request counts belong to calling pipeline",
        "original_script_verified": validation.get("original_script_verified", False),
        "native_phoneme_completeness_verified": False,
        "quality_note": "Structural PCM preservation and conservative edge screening do not prove source-word completeness or listening quality",
    }
    return RenderResult(audio=audio, metrics=metrics, timeline=timeline)
