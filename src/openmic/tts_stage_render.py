"""Render fixed-persona stage delivery without altering the source words.

stage_prompt changes only the old performance instruction. The two full stage
profiles share speech requests, linear gains and whole-speech headroom; only
their inserted silence differs. Native focus remains a soft synthesis request,
not evidence of a verified word-level acoustic emphasis.
"""

import hashlib
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from openmic.tts_backends import SAMPLE_RATE, decode_wav, encode_wav
from openmic.tts_plan import plain_text
from openmic.tts_render import RenderResult, _preprocess


STAGE_INSTRUCTION = "请轻快地说。"
PROFILES = ("stage_prompt", "stage_course", "stage_club")
ADAPTER = "openmic.stage-render/1"
INTENT_SPEEDS = {
    "setup": 1.12, "audience_question": 1.06, "self_question": 1.06,
    "quoted_question": 1.07, "contrast": 1.08, "reveal": 1.02,
    "aside": 1.12, "button": 1.04,
}
INTENT_GAINS_DB = {
    "setup": 0.0, "audience_question": 0.3, "self_question": 0.3,
    "quoted_question": 0.0, "contrast": 0.4, "reveal": 0.8,
    "aside": -0.8, "button": 1.2,
}
PAUSE_SECONDS = {
    "stage_course": {"flow": 0.08, "breath": 0.15, "turn": 0.8, "reaction": 2.0},
    "stage_club": {"flow": 0.08, "breath": 0.15, "turn": 0.32, "reaction": 0.65},
}
HEADROOM_DBFS = -1.0
# A floor avoids a rounding step overshooting the declared peak ceiling.
HEADROOM_PEAK = math.floor(32767 * 10 ** (HEADROOM_DBFS / 20))


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@contextmanager
def _publication_lock(cache):
    """Brief per-key cross-process lock; never hold it during a paid request."""
    lock = cache.with_suffix(".lock")
    deadline = time.monotonic() + 10
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError("缓存发布锁等待超时；请检查本轮残留 .lock 文件后重试") from None
            time.sleep(0.02)
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _cache_audio(path):
    samples = decode_wav(path.read_bytes())
    return samples, _preprocess(samples)


def _pause(seconds, pause_type, **metadata):
    return {"kind": "pause", "requested_seconds": seconds,
            "pause_type": pause_type, **metadata}


def _jobs(plan, profile):
    jobs = []
    if profile == "stage_prompt":
        source_index = 0
        for source_event_index, event in enumerate(plan["source_events"]):
            if jobs and jobs[-1]["kind"] == "speech" and event["kind"] == "speech":
                jobs.append(_pause(0.12, "ordinary_boundary", reason="ordinary_boundary"))
            if event["kind"] == "pause":
                jobs.append(_pause(event["seconds"], "source_pause", reason=event.get("reason", "explicit"),
                                   source_event_index=source_event_index))
                continue
            jobs.append({"kind": "speech", "text": plain_text(event["text"]),
                         "actual_input": event["text"], "actual_speed": event["speed"],
                         "actual_instruction": STAGE_INSTRUCTION, "actual_gain_db": 0.0,
                         "intent": "source_protocol", "focus": None, "unit_index": None,
                         "source_index": source_index, "source_event_index": source_event_index,
                         "source_start": 0, "source_end": len(plain_text(event["text"])),
                         "role": event["role"], "beat": event["beat"],
                         "original_text": event["text"], "original_emotion": event["emotion"],
                         "original_speed": event["speed"], "original_emphasis": event["emphasis"]})
            source_index += 1
        return jobs
    durations = PAUSE_SECONDS[profile]
    sources = [event for event in plan["source_events"] if event["kind"] == "speech"]
    previous = None
    for unit in plan["units"]:
        if unit["intent"] not in INTENT_SPEEDS or unit["boundary_before"] not in durations:
            raise ValueError("计划包含未支持的舞台意图或停顿类型")
        if previous is not None:
            pause_type = unit["boundary_before"]
            seconds = durations[pause_type]
            needs_reaction = previous["intent"] in {"reveal", "button"}
            if needs_reaction and durations["reaction"] >= seconds:
                seconds, pause_type = durations["reaction"], "reaction"
            jobs.append(_pause(seconds, pause_type, boundary_before=unit["boundary_before"],
                               after_intent=previous["intent"], before_unit_index=unit["index"],
                               reason="max_of_boundary_and_previous_reaction"))
        text, focus = unit["text"], unit["focus"]
        actual_input = text.replace(focus, "<strong>" + focus + "</strong>", 1) if focus else text
        source = sources[unit["source_index"]]
        jobs.append({"kind": "speech", "text": text, "actual_input": actual_input,
                     "actual_speed": INTENT_SPEEDS[unit["intent"]],
                     "actual_instruction": STAGE_INSTRUCTION,
                     "actual_gain_db": INTENT_GAINS_DB[unit["intent"]],
                     "intent": unit["intent"], "focus": focus,
                     "unit_index": unit["index"], "source_index": unit["source_index"],
                     "source_start": unit["start"], "source_end": unit["end"],
                     "role": unit["role"], "beat": unit["beat"],
                     "original_emotion": source["emotion"], "original_speed": source["speed"],
                     "original_emphasis": source["emphasis"]})
        previous = unit
    if previous and previous["intent"] in {"reveal", "button"}:
        jobs.append(_pause(durations["reaction"], "reaction", after_intent=previous["intent"],
                           reason="final_reaction"))
    return jobs


def _check_inputs(plan, backend, profile, cache_dir, on_chunk):
    if profile not in PROFILES:
        raise ValueError("profile 必须为 " + ", ".join(PROFILES))
    from openmic.tts_stage_plan import validate_execution_plan

    validation = validate_execution_plan(plan)
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("舞台计划未通过执行前校验：" + "; ".join(
            str(value) for value in (validation.get("errors", []) if isinstance(validation, dict) else [])))
    if on_chunk is not None and not callable(on_chunk):
        raise ValueError("on_chunk 必须可调用或为 None")
    if not callable(getattr(backend, "synthesize", None)):
        raise ValueError("backend 必须提供 synthesize")
    if not isinstance(getattr(backend, "synthetic", None), bool):
        raise ValueError("backend 必须明确声明 synthetic")
    # Freeze the declared identity; no secret-bearing configuration is saved.
    identity = json.loads(_json(getattr(backend, "identity", None)))
    if not isinstance(identity, dict) or not identity:
        raise ValueError("backend 必须提供非空 identity")
    if identity.get("sample_rate", SAMPLE_RATE) != SAMPLE_RATE:
        raise ValueError("舞台渲染只接受 16000 Hz 后端")
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None and cache.exists() and not cache.is_dir():
        raise ValueError("cache_dir 必须为目录")
    jobs = _jobs(plan, profile)
    if not any(job["kind"] == "speech" for job in jobs):
        raise ValueError("舞台计划没有可执行语音")
    return identity, cache, jobs


def _speech(job, backend, identity, cache_dir):
    arguments = {"text": job["actual_input"], "instruction": job["actual_instruction"],
                 "speed": job["actual_speed"]}
    key = _hash(_json({"adapter": ADAPTER, "backend": identity, **arguments}))
    cache = cache_dir / (key + ".wav") if cache_dir is not None else None
    samples = processed = None
    cache_hit = False
    if cache is not None and cache.is_file():
        try:
            samples, processed = _cache_audio(cache)
            cache_hit = True
        except (OSError, ValueError):
            samples = processed = None
    api_seconds = 0.0
    request_made = not cache_hit
    concurrent_winner_reused = False
    if not cache_hit:
        started = time.perf_counter()
        audio = backend.synthesize(**arguments)
        api_seconds = time.perf_counter() - started
        samples = decode_wav(audio)
        processed = _preprocess(samples)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name(cache.stem + "." + uuid.uuid4().hex + ".tmp")
            try:
                temporary.write_bytes(encode_wav(samples))
                with _publication_lock(cache):
                    winner = None
                    if cache.is_file():
                        try:
                            winner = _cache_audio(cache)
                        except (OSError, ValueError):
                            cache.unlink(missing_ok=True)
                    if winner is not None:
                        # Reuse the first valid publisher, even if our own
                        # stochastic HTTP response arrived a little later.
                        samples, processed = winner
                        concurrent_winner_reused = True
                    else:
                        # Same-volume hard linking atomically exposes a whole
                        # WAV and refuses to overwrite an existing winner.
                        try:
                            os.link(temporary, cache)
                        except FileExistsError:
                            samples, processed = _cache_audio(cache)
                            concurrent_winner_reused = True
            finally:
                temporary.unlink(missing_ok=True)
    return (processed, len(samples) - len(processed), cache_hit, api_seconds, key,
            request_made, concurrent_winner_reused)


def render_stage(plan, backend, profile, cache_dir=None, on_chunk=None):
    """Validate before synthesis; return the old RenderResult container.

    Full profiles buffer all speech before emitting final-PCM callbacks, because
    one global headroom scale depends on the entire speech peak. Run course and
    club serially with the SAME cache to obtain an exact timing-only comparison.
    """
    identity, cache, jobs = _check_inputs(plan, backend, profile, cache_dir, on_chunk)
    started = time.perf_counter()
    prepared = {}
    hits = calls = winner_reuses = 0
    api_seconds = 0.0
    first_model = first_raw = first_chunk = None
    pieces, timeline, speech_parts = [], [], []
    position = speech_frames = pause_frames = 0

    def append(job, samples, headroom):
        nonlocal position, speech_frames, pause_frames, first_chunk
        row = dict(job)
        if row["kind"] == "speech":
            speech_frames += len(samples)
            speech_parts.append(samples)
            row.update(global_headroom_gain=headroom,
                       global_headroom_gain_db=20 * math.log10(headroom),
                       effective_gain_db=row["actual_gain_db"] + 20 * math.log10(headroom),
                       focus_execution="native <strong> synthesis hint; word timing and acoustic emphasis not independently verified",
                       emotion_execution="fixed global stage persona; original emotion is not executed")
            if first_chunk is None:
                first_chunk = time.perf_counter() - started
        else:
            pause_frames += len(samples)
        row.update(start_frame=position, end_frame=position + len(samples),
                   start_seconds=position / SAMPLE_RATE,
                   end_seconds=(position + len(samples)) / SAMPLE_RATE,
                   duration_seconds=len(samples) / SAMPLE_RATE)
        position += len(samples)
        pieces.append(samples)
        timeline.append(row)
        if on_chunk is not None and len(samples):
            on_chunk(encode_wav(samples), dict(row))

    for index, job in enumerate(jobs):
        if job["kind"] == "pause":
            if profile == "stage_prompt":
                append(job, np.zeros(round(job["requested_seconds"] * SAMPLE_RATE), dtype="<i2"), 1.0)
            continue
        samples, trimmed, hit, seconds, key, requested, race = _speech(job, backend, identity, cache)
        ready = time.perf_counter() - started
        if first_raw is None:
            first_raw = ready
        if requested and first_model is None:
            first_model = ready
        hits += int(hit)
        calls += int(requested)
        winner_reuses += int(race)
        api_seconds += seconds
        job.update(cache_hit=hit, cache_key=key, backend_request_made=requested,
                   concurrent_cache_winner_reused=race, trimmed_frames=trimmed,
                   pre_gain_peak=int(np.max(np.abs(samples.astype(np.int32)))))
        if profile == "stage_prompt":
            append(job, samples, 1.0)
        else:
            # Keep float precision until the shared whole-speech headroom step.
            prepared[index] = samples.astype(np.float64) * 10 ** (job["actual_gain_db"] / 20)

    headroom = 1.0
    if profile != "stage_prompt":
        peak = max(float(np.max(np.abs(samples))) for samples in prepared.values())
        headroom = min(1.0, HEADROOM_PEAK / peak)
        for index, job in enumerate(jobs):
            if job["kind"] == "pause":
                samples = np.zeros(round(job["requested_seconds"] * SAMPLE_RATE), dtype="<i2")
            else:
                # A constant scalar for the entire speech: no per-unit RMS
                # normalization, compressor, pitch shift or phoneme-tail crop.
                samples = np.rint(prepared[index] * headroom).astype("<i2")
                if int(np.max(np.abs(samples.astype(np.int32)))) > HEADROOM_PEAK:
                    raise RuntimeError("内部错误：全篇 headroom 未能满足峰值约束")
            append(job, samples, headroom)
    if json.loads(_json(backend.identity)) != identity:
        raise ValueError("渲染过程中后端 identity 改变，不能声称固定模型和音色")
    combined = np.concatenate(pieces)
    speech_pcm = np.concatenate(speech_parts).astype("<i2", copy=False).tobytes()
    audio = encode_wav(combined)
    elapsed = time.perf_counter() - started
    speech_duration = speech_frames / SAMPLE_RATE
    peak = int(np.max(np.abs(combined.astype(np.int32))))
    legacy_events = [{key: value for key, value in event.items() if key != "kind"}
                     for event in plan["source_events"]]
    source_texts = [plain_text(event["text"]) for event in plan["source_events"] if event["kind"] == "speech"]
    metrics = {
        "profile": profile, "condition": profile, "adapter": ADAPTER,
        "backend": identity, "synthetic": backend.synthetic,
        "source_script_sha256": plan["source_script_sha256"],
        "plan_sha256": _hash(_json(plan)), "source_plan_sha256": _hash(_json(legacy_events)),
        "spoken_text_sha256": plan["spoken_text_sha256"],
        "legacy_spoken_text_sha256": _hash("\n".join(source_texts)),
        "speech_pcm_sha256": hashlib.sha256(speech_pcm).hexdigest(),
        "sample_rate": SAMPLE_RATE, "channels": 1, "sample_width": 2,
        "duration_seconds": position / SAMPLE_RATE, "speech_seconds": speech_duration,
        "inserted_silence_seconds": pause_frames / SAMPLE_RATE,
        "speech_segments": len(speech_parts), "source_speech_segments": len(source_texts),
        "wall_seconds": elapsed, "api_seconds": api_seconds, "api_calls": calls, "cache_hits": hits,
        "cache_winner_reuses": winner_reuses,
        "all_cache_miss": hits == 0 and winner_reuses == 0,
        "valid_cold_tts_benchmark": not backend.synthetic and hits == 0 and winner_reuses == 0,
        "rtf_speech": elapsed / speech_duration, "speed_x_speech": speech_duration / elapsed,
        "speed_x_including_pauses": position / SAMPLE_RATE / elapsed,
        "first_model_segment_seconds": first_model, "first_raw_speech_ready_seconds": first_raw,
        "first_speech_chunk_seconds": first_chunk,
        "in_target_duration": 180 <= position / SAMPLE_RATE <= 300,
        "actual_instruction": STAGE_INSTRUCTION,
        "global_headroom_gain": headroom, "global_headroom_gain_db": 20 * math.log10(headroom),
        "headroom_policy": "not applied: prompt-only comparison" if profile == "stage_prompt" else "one whole-speech attenuation to at most -1 dBFS; no per-unit leveling",
        "sample_peak": peak, "sample_peak_dbfs": 20 * math.log10(peak / 32768) if peak else None,
        "pause_policy": "unchanged legacy performance pauses and ordinary boundaries" if profile == "stage_prompt" else PAUSE_SECONDS[profile],
        "pause_definition": "exact inserted PCM zero samples; model/internal silence can extend perceived gaps",
        "gain_definition": "actual_gain_db is the unit's fixed linear gain; effective_gain_db additionally includes global headroom attenuation",
        "cache_definition": "cache_hits means reuse before a request; cache_winner_reuses separately counts a first-writer winner reused after our own successful request; api_calls still counts that request",
        "streaming_scope": "completed final-PCM callbacks; full stage profiles buffer all speech for global headroom; not token streaming",
        "focus_note": "native <strong> hint, not verified word-level timing or acoustic emphasis",
        "content_note": "source emotions retained only as provenance; all requests use one fixed stage instruction",
        "quality_note": "different acoustic controls do not establish improved stage delivery, timbre stability or MOS",
        "plan_validation": {"execution_structure_verified": True, "original_script_verified": False,
                            "note": "caller must separately validate_plan(plan, original_script) to verify original script hash"},
    }
    return RenderResult(audio, metrics, timeline)
