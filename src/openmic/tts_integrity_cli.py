"""Bounded sentence synthesis with strict defaults and explicit review authority."""
import argparse
import copy
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

import requests
from openmic.tts import TTSConfig
from openmic.tts_backends import SiliconFlowBackend
from openmic.tts_render import save_result
from openmic.tts_stage_cli import source_review
from openmic.tts_stage_render import STAGE_INSTRUCTION, INTENT_SPEEDS
from openmic.tts_integrity_plan import build_integrity_plan, validate_integrity_plan, validate_execution_integrity_plan
from openmic.tts_integrity_gate import assess_content
from openmic.tts_integrity_audio import inspect_edges
from openmic.tts_integrity_listening import (
    ReviewAuthority, canonical_sha, human_accepted, load_authority, render_reviewed,
)


class IntegrityError(RuntimeError): pass
class FatalServiceError(IntegrityError): pass
class ASRUnavailable(FatalServiceError): pass

DEFAULT_VERBATIM_INSTRUCTION = "请逐字朗读。"


def canonical(value): return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
def digest(value): return hashlib.sha256(value).hexdigest()
def file_sha(path): return digest(Path(path).read_bytes())
def load(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def policy_sha():
    folder = Path(__file__).parent
    return digest(b"integrity-content-and-edge-v1" + (folder / "tts_integrity_gate.py").read_bytes()
                  + (folder / "tts_integrity_audio.py").read_bytes())


def orthography_review(expected, primary, secondary, audio_sha256, *, authority=None, entry=None):
    """A retained strict failure needs an explicitly approved exact-case proof."""
    if not isinstance(authority, ReviewAuthority) or not isinstance(entry, dict):
        return {"passed": False, "reason": "explicit_external_authority_required"}
    proof = authority.proof_for(entry, "orthography")
    if proof is None or proof["text"] != expected or proof["audio_sha256"] != audio_sha256 \
            or proof["primary_transcript"] != primary or proof["secondary_transcript"] != secondary:
        return {"passed": False, "reason": "no_exact_authorized_case"}
    return proof


def content_accepted(entry, *, authority=None):
    if not isinstance(entry, dict):
        return False
    if isinstance(entry.get("content"), dict) and entry["content"].get("passed") is True:
        return True
    if human_accepted(entry, authority=authority):
        return True
    if not isinstance(authority, ReviewAuthority):
        return False
    proof = authority.proof_for(entry, "orthography")
    return proof is not None and entry.get("orthography_review") == proof


def transcribe(audio, config, model="FunAudioLLM/SenseVoiceSmall"):
    """Retry transport/server failures only, never a disagreeing transcript."""
    record = {"model": model, "api_calls": 0}
    started = time.perf_counter()
    for attempt in range(3):
        try:
            record["api_calls"] += 1
            response = requests.post(config.base_url.rstrip("/") + "/audio/transcriptions",
                headers={"Authorization": "Bearer " + config.api_key},
                files={"file": ("sentence.wav", audio, "audio/wav")}, data={"model": model}, timeout=(15, 120))
            if response.status_code in {401, 402, 403}:
                raise FatalServiceError("ASR HTTP " + str(response.status_code))
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(1)
                continue
            if response.status_code != 200:
                raise ASRUnavailable("ASR HTTP " + str(response.status_code))
            try:
                text = response.json().get("text")
            except (ValueError, AttributeError):
                raise ASRUnavailable("ASR response format unavailable") from None
            if not isinstance(text, str) or not text.strip():
                raise ASRUnavailable("ASR returned no text")
            record.update(transcript=text, seconds=time.perf_counter() - started)
            return record
        except requests.RequestException as error:
            if attempt == 2:
                raise ASRUnavailable("ASR transport " + type(error).__name__) from None
            time.sleep(1)
    raise ASRUnavailable("ASR unavailable")


def parameters(unit, backend, attempt, *, trial=None):
    """IDs 0--4 are bounded controls; 5 is opt-in; 6/7 require a trial object."""
    if type(attempt) is not int or not 0 <= attempt <= 7:
        raise ValueError("Unknown candidate attempt")
    if attempt >= 6:
        if not isinstance(trial, dict) or trial.get("unit_id") != unit["id"] or trial.get("attempt") != attempt:
            raise ValueError("Optional input trials require exact external authority")
        if trial["voice"] != backend.identity.get("voice"):
            raise ValueError("Trial authority specifies a different voice")
        return {"backend": backend.identity, "text": trial["actual_input"],
                "instruction": trial["instruction"], "speed": trial["speed"]}
    text = unit["text"]
    if unit["focus"] and attempt <= 2:
        text = text.replace(unit["focus"], "<strong>" + unit["focus"] + "</strong>", 1)
    return {"backend": backend.identity, "text": text,
            "instruction": DEFAULT_VERBATIM_INSTRUCTION if attempt == 5 else STAGE_INSTRUCTION,
            "speed": 1.0 if attempt >= 3 else INTENT_SPEEDS[unit["intent"]]}


def _inside(root, value):
    if not isinstance(value, (str, Path)):
        raise IntegrityError("Artifact path is missing or invalid")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise IntegrityError("Artifact path escapes assets root")
    return path


def _entry(record, path, root, plan):
    return {"unit_id": record["unit_id"], "text": record["text"],
            "audio_path": (path.parent / "audio.wav").relative_to(root).as_posix(),
            "audio_sha256": record["audio_sha256"], "params": record["params"],
            "content": copy.deepcopy(record.get("content")), "edge": copy.deepcopy(record["edge"]),
            "origin": record.get("origin", "existing_candidate"),
            "candidate_record": path.relative_to(root).as_posix(), "policy_sha256": record["policy_sha256"],
            "plan_sha256": canonical_sha(plan), "source_script_sha256": plan["source_script_sha256"],
            "control_fallback": record.get("control_fallback", False),
            "verbatim_rescue": record.get("verbatim_rescue", False),
            "input_transform": copy.deepcopy(record.get("input_transform")),
            "trial_authority_sha256": record.get("trial_authority_sha256"), "synthetic": record["synthetic"]}


def _attach_proofs(entry, authority):
    if authority is not None:
        for kind, field in (("human_listening", "human_review"), ("orthography", "orthography_review")):
            proof = authority.proof_for(entry, kind)
            if proof is not None:
                entry[field] = proof
    return entry


def _retained_trial(record, candidate, plan, root):
    """Verify a past trial's saved authorization; never grant a new request."""
    attempt = record.get("attempt")
    if type(attempt) is not int or attempt < 6:
        return None
    authority_sha = record.get("trial_authority_sha256")
    if not isinstance(authority_sha, str) or len(authority_sha) != 64:
        raise IntegrityError("Optional trial has no retained authority digest")
    snapshot = candidate.parents[3] / "authorities" / (authority_sha + ".json")
    retained = load_authority(snapshot, authority_sha, plan, root)
    trial = retained.trial_for(record["unit_id"], attempt)
    params = record.get("params", {})
    if trial is None or any(params.get(key) != trial[key] for key in ("instruction", "speed")) \
            or params.get("text") != trial["actual_input"] or params.get("backend", {}).get("voice") != trial["voice"]:
        raise IntegrityError("Optional trial differs from its retained authorization")
    return trial


def validate_accepted(entry, plan, assets_root, *, authority=None):
    """Recheck bytes, original reports and evidence locally; no service calls."""
    root = Path(assets_root).resolve()
    index = entry.get("unit_id")
    if type(index) is not int or not 0 <= index < len(plan["units"]):
        raise IntegrityError("Accepted sentence ID is invalid")
    if entry.get("plan_sha256") != canonical_sha(plan) or entry.get("source_script_sha256") != plan["source_script_sha256"]:
        raise IntegrityError("Accepted sentence belongs to another plan/source")
    path = _inside(root, entry["candidate_record"])
    record, audio_path = load(path), _inside(root, entry["audio_path"])
    if audio_path != path.parent / "audio.wav":
        raise IntegrityError("Accepted candidate and audio paths disagree")
    audio = audio_path.read_bytes()
    if entry.get("audio_sha256") != file_sha(audio_path) or record.get("audio_sha256") != entry["audio_sha256"]:
        raise IntegrityError("Accepted audio changed")
    if entry.get("text") != plan["units"][index]["text"] or any(entry.get(key) != record.get(key) for key in ("unit_id", "text", "params", "content", "edge", "synthetic")):
        raise IntegrityError("Accepted source or original candidate reports changed")
    if (record.get("plan_sha256") != entry["plan_sha256"]
            or record.get("source_script_sha256") != entry["source_script_sha256"]
            or record.get("reference_sha256") != digest(entry["text"].encode("utf-8"))):
        raise IntegrityError("Original candidate plan/source/sentence binding changed")
    _retained_trial(record, path, plan, root)
    if entry.get("policy_sha256") != policy_sha() or record.get("policy_sha256") != policy_sha():
        raise IntegrityError("Accepted candidate has not passed the current machine policy")
    edge = inspect_edges(audio)
    if edge != entry["edge"]:
        raise IntegrityError("Accepted endpoint metadata is stale")
    content = record.get("content")
    if content is not None:
        recognition = record.get("asr", {})
        reference_sha = digest(entry["text"].encode("utf-8"))
        if recognition.get("audio_sha256") != entry["audio_sha256"] or recognition.get("reference_sha256") != reference_sha:
            raise IntegrityError("ASR is not bound to the accepted WAV and original sentence")
        if not isinstance(recognition.get("model"), str) or not recognition["model"].strip():
            raise IntegrityError("ASR record has no recognizer identity")
        if content != assess_content(entry["text"], recognition.get("transcript")):
            raise IntegrityError("Accepted content metadata is stale")
    if authority is not None:
        authority.validate_plan(plan)
    reviewed = human_accepted(entry, authority=authority)
    if not (edge["passed"] or reviewed) or not content_accepted(entry, authority=authority):
        raise IntegrityError("Sentence requires strict checks or explicit matching external review authority")
    return audio


def ensure_units(plan, backend, out, assets_root, reuse_index=None, unit_ids=None, asr_function=None,
                 terminal_control_approved=False, *, authority=None, verbatim_rescue=False):
    """Two controlled and two plain native requests maximum per new sentence.

    A fifth verbatim request requires explicit opt-in. Extra input transforms
    require the separately loaded manifest; a boolean cannot authorize them.
    Failed or unknown-outcome requests consume their persistent attempt IDs.
    """
    validation = validate_execution_integrity_plan(plan)
    if not validation["passed"]:
        raise ValueError("Invalid integrity plan")
    if authority is not None and not isinstance(authority, ReviewAuthority):
        raise ValueError("authority must come from load_authority")
    if terminal_control_approved:
        raise ValueError("A boolean is not authorization; supply a SHA256-bound authority manifest")
    root, out = Path(assets_root).resolve(), Path(out).resolve()
    if type(getattr(backend, "synthetic", None)) is not bool:
        raise ValueError("Backend must declare whether it is synthetic")
    if not out.is_relative_to(root):
        raise ValueError("Output must stay inside assets root")
    if unit_ids is not None and (any(type(value) is not int for value in unit_ids) or not set(unit_ids) <= set(range(len(plan["units"])))):
        raise ValueError("Unknown sentence IDs")
    if authority is not None:
        authority.validate_plan(plan)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".integrity.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        if (out / "plan.json").exists() and load(out / "plan.json") != plan:
            raise IntegrityError("Stored plan differs; use a new output directory")
        dump(out / "plan.json", plan)
        if authority is not None:
            authority.snapshot_to(out)
        return _ensure(plan, backend, out, root, reuse_index or {}, unit_ids, asr_function, authority, verbatim_rescue)
    finally:
        lock.unlink(missing_ok=True)


def _ensure(plan, backend, out, root, reuse, unit_ids, asr_function, authority, verbatim_rescue):
    policy, manifest = policy_sha(), out / "accepted.json"
    selected = set(range(len(plan["units"]))) if unit_ids is None else set(unit_ids)
    accepted = {}
    saved_entries = load(manifest) if manifest.exists() else []
    for entry in saved_entries:
        index = entry["unit_id"]
        record = load(_inside(root, entry["candidate_record"]))
        attempt = record["attempt"]
        trial = _retained_trial(record, _inside(root, entry["candidate_record"]), plan, root) if attempt >= 6 else None
        entry = _attach_proofs(copy.deepcopy(entry), authority)
        has_review = entry.get("human_review") or entry.get("orthography_review")
        expected = record["params"] if has_review else parameters(plan["units"][index], backend, attempt, trial=trial)
        if entry["params"] != expected or expected.get("backend") != backend.identity or entry["synthetic"] != backend.synthetic:
            raise IntegrityError("Accepted parameters/backend changed")
        validate_accepted(entry, plan, root, authority=authority)
        if index in accepted:
            raise IntegrityError("Duplicate accepted sentence ID")
        accepted[index] = entry
    rebound_entries = [accepted[index] for index in sorted(accepted)]
    if rebound_entries != saved_entries:
        dump(manifest, rebound_entries)
    failed = []
    for unit in plan["units"]:
        index = unit["id"]
        if index not in selected or index in accepted:
            continue
        if authority is not None:
            for approved_path in authority.candidate_paths(index):
                reviewed_record = load(approved_path)
                reviewed = _attach_proofs(_entry(reviewed_record, approved_path, root, plan), authority)
                if reviewed["params"].get("backend") != backend.identity or reviewed["synthetic"] != backend.synthetic:
                    raise IntegrityError("Reviewed candidate uses another backend identity")
                validate_accepted(reviewed, plan, root, authority=authority)
                accepted[index] = reviewed
                dump(manifest, [accepted[key] for key in sorted(accepted)])
                break
            if index in accepted:
                continue
        attempts = list(range(5)) + ([5] if verbatim_rescue else [])
        if authority is not None:
            attempts += authority.trial_ids(index)
        for attempt in attempts:
            trial = authority.trial_for(index, attempt) if authority is not None and attempt >= 6 else None
            params = parameters(unit, backend, attempt, trial=trial)
            identity, reference = digest(canonical(params).encode("utf-8")), digest(unit["text"].encode("utf-8"))
            if attempt == 0 and identity not in reuse:
                continue
            candidate = out / "candidates" / f"{index:03d}" / f"{attempt:02d}"
            candidate.mkdir(parents=True, exist_ok=True)
            meta_path, audio_path = candidate / "record.json", candidate / "audio.wav"
            record = load(meta_path) if meta_path.exists() else None
            if record is not None:
                if record.get("params") != params or record.get("reference_sha256") != reference:
                    raise IntegrityError("Candidate parameters/reference changed; use a new output directory")
                if audio_path.exists() and authority is not None:
                    reviewed = _attach_proofs(_entry(record, meta_path, root, plan), authority)
                    if reviewed.get("human_review") or reviewed.get("orthography_review"):
                        validate_accepted(reviewed, plan, root, authority=authority)
                        accepted[index] = reviewed
                        dump(manifest, [accepted[key] for key in sorted(accepted)])
                        break
                if record.get("status") not in {"passed", "running", "downloaded", "checked", "service_blocked"}:
                    continue
                if attempt > 0 and record.get("status") == "running" and not audio_path.exists():
                    record.update(status="unknown_native_outcome")
                    dump(meta_path, record)
                    continue
            else:
                record = {"status": "running", "unit_id": index, "text": unit["text"],
                          "reference_sha256": reference, "params": params, "policy_sha256": policy,
                          "plan_sha256": canonical_sha(plan), "source_script_sha256": plan["source_script_sha256"],
                          "attempt": attempt, "new_tts_api_calls": int(attempt > 0),
                          "origin": "previous_exact_parameter_raw" if attempt == 0 else "new_native_request",
                          "control_fallback": attempt >= 3, "verbatim_rescue": attempt == 5,
                          "input_transform": trial["transform"] if trial else None,
                          "trial_authority_sha256": authority.sha256 if trial else None, "synthetic": backend.synthetic}
                dump(meta_path, record)
            try:
                if not audio_path.exists():
                    if attempt == 0:
                        original = _inside(root, reuse[identity]["audio_path"])
                        if file_sha(original) != reuse[identity]["sha256"]:
                            raise IntegrityError("Reusable source cache changed")
                        audio = original.read_bytes()
                    else:
                        audio = backend.synthesize(text=params["text"], instruction=params["instruction"], speed=params["speed"])
                    audio_path.write_bytes(audio)
                    record.update(status="downloaded", audio_sha256=file_sha(audio_path))
                    dump(meta_path, record)
                audio, audio_hash = audio_path.read_bytes(), file_sha(audio_path)
                if record.get("audio_sha256", audio_hash) != audio_hash:
                    raise IntegrityError("Candidate WAV changed")
                record.update(audio_sha256=audio_hash, policy_sha256=policy)
                record["edge"] = inspect_edges(audio)
                if not record["edge"]["passed"]:
                    record["status"] = "edge_rejected"
                    dump(meta_path, record)
                    continue
                if "asr" not in record:
                    record["asr"] = asr_function(audio) if asr_function else transcribe(audio, TTSConfig.from_env())
                record["asr"].update(audio_sha256=audio_hash, reference_sha256=reference)
                if backend.synthetic:
                    record["asr"].setdefault("model", "synthetic-fixture-asr")
                record["content"] = assess_content(unit["text"], record["asr"]["transcript"])
                record["status"] = "passed" if record["content"]["passed"] else "content_rejected"
                dump(meta_path, record)
                if record["status"] != "passed":
                    continue
                entry = _entry(record, meta_path, root, plan)
                validate_accepted(entry, plan, root)
                accepted[index] = entry
                dump(manifest, [accepted[key] for key in sorted(accepted)])
                break
            except FatalServiceError:
                record.update(status="service_blocked")
                dump(meta_path, record)
                raise
            except IntegrityError:
                raise
            except Exception as error:
                status = re.search(r"HTTP (\d{3})", str(error))
                record.update(status="request_failed", error_type=type(error).__name__,
                              http_status=int(status[1]) if status else None)
                dump(meta_path, record)
                if status and int(status[1]) in {401, 402, 403}:
                    raise FatalServiceError("TTS HTTP " + status[1]) from None
        if index not in accepted:
            failed.append(index)
    dump(out / "sentence_progress.json", {"accepted": sorted(accepted), "requested": sorted(selected),
                                         "failed": failed, "total": len(plan["units"])})
    if failed:
        raise IntegrityError("Sentences failed bounded screening: " + ",".join(map(str, failed)))
    return [accepted[index] for index in sorted(accepted)]


def assemble(plan, accepted, assets_root, out, *, authority=None):
    root, out = Path(assets_root).resolve(), Path(out).resolve()
    if not out.is_relative_to(root):
        raise ValueError("Output must stay inside assets root")
    ids = [entry.get("unit_id") for entry in accepted]
    if any(type(index) is not int for index in ids) or sorted(ids) != list(range(len(plan["units"]))):
        raise IntegrityError("Incomplete or duplicate sentence coverage; no full audio written")
    by_id = {entry["unit_id"]: entry for entry in accepted}
    samples = {index: validate_accepted(by_id[index], plan, root, authority=authority) for index in sorted(by_id)}
    for profile in ("safe_course", "safe_club"):
        target = out / profile
        human = {index: entry for index, entry in by_id.items() if entry.get("human_review")}
        result = render_reviewed(plan, samples, profile=profile, human_reviews=human, authority=authority)
        result.metrics.update(accepted_manifest_sha256=file_sha(out / "accepted.json"),
                              synthetic=any(entry["synthetic"] for entry in accepted),
                              backend=accepted[0]["params"]["backend"], all_sentences_screened=True,
                              authority_sha256=authority.sha256 if authority else None,
                              content_note="Machine checks and explicit reviews are retained separately; no human MOS or quality gain is inferred")
        for row in result.timeline:
            if row["kind"] == "speech":
                row["accepted_parameters"] = by_id[row["unit_id"]]["params"]
        files = [target / name for name in ("audio.wav", "metrics.json", "timeline.json")]
        if all(path.is_file() for path in files) and load(target / "metrics.json") == result.metrics \
                and load(target / "timeline.json") == result.timeline and (target / "audio.wav").read_bytes() == result.audio:
            dump(target / "completion.json", {"audio_sha256": file_sha(target / "audio.wav"),
                 "plan_sha256": result.metrics["plan_sha256"], "timeline_sha256": file_sha(target / "timeline.json")})
            continue
        if any(path.exists() for path in files):
            backup = target / "partial_recovery" / uuid.uuid4().hex[:12]
            backup.mkdir(parents=True)
            if not backup.resolve().is_relative_to(root):
                raise IntegrityError("Recovery target escapes assets root")
            for path in files:
                if path.exists():
                    path.rename(backup / path.name)
        save_result(result, target)
        dump(target / "completion.json", {"audio_sha256": file_sha(target / "audio.wav"),
             "plan_sha256": result.metrics["plan_sha256"], "timeline_sha256": file_sha(target / "timeline.json")})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Strict sentence screening; optional reviews need external SHA256-bound authority")
    parser.add_argument("command", choices=["plan", "synth"])
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--stage-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--manual-script", action="store_true")
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--reuse-index", type=Path)
    parser.add_argument("--voice")
    parser.add_argument("--verbatim-rescue", action="store_true", help="Explicitly allow one additional plain-text 1.0-speed request")
    parser.add_argument("--authority", type=Path, help="External review/trial manifest; entries cannot self-authorize")
    parser.add_argument("--authority-sha256", help="Independently approved SHA256 of that exact manifest")
    args = parser.parse_args(argv)
    if bool(args.authority) != bool(args.authority_sha256):
        parser.error("--authority and --authority-sha256 must be provided together")
    script = args.script.read_text(encoding="utf-8-sig")
    stage = load(args.stage_plan)
    review = source_review(script, args.trace, manual=args.manual_script)
    plan = build_integrity_plan(script, stage)
    if not validate_integrity_plan(plan, script, stage)["passed"]:
        raise IntegrityError("Invalid integrity plan")
    if args.command == "plan":
        dump(args.out, plan)
        return
    root = args.assets_root or args.out.resolve().parent
    authority = load_authority(args.authority, args.authority_sha256, plan, root) if args.authority else None
    accepted = ensure_units(plan, SiliconFlowBackend(voice=args.voice), args.out, root,
                            load(args.reuse_index) if args.reuse_index else {},
                            authority=authority, verbatim_rescue=args.verbatim_rescue)
    dump(args.out / "source_review.json", review)
    assemble(plan, accepted, root, args.out, authority=authority)


if __name__ == "__main__":
    main()
