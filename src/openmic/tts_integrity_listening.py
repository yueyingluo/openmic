"""Explicit, file-bound review authority; original machine reports stay intact.

An entry cannot authorize itself. Review exceptions and optional input trials
require an independently supplied manifest and its caller-approved SHA256.
Nothing in this module requests models or claims that software can verify a
reviewer's actual listening or linguistic judgment.
"""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from openmic.tts_backends import SAMPLE_RATE, encode_wav
from openmic.tts_render import RenderResult
from openmic.tts_integrity_plan import validate_execution_integrity_plan
from openmic.tts_integrity_audio import (
    ADAPTER as BASE_ADAPTER, BOUNDARY_SECONDS, EDGE_WINDOW_SECONDS,
    END_GUARD_SECONDS, HEAD_RMS_LIMIT_DBFS, PROFILES,
    SENTENCE_PAUSE_FLOOR_SECONDS, START_GUARD_SECONDS, TAIL_RMS_LIMIT_DBFS,
    _strict_pcm, inspect_edges, render_integrity,
)


ADAPTER = "openmic.integrity-listening/2"
HEAD_ERROR = "Head 10 ms RMS exceeds -35 dBFS: possible clipped speech onset"
AUTHORITY_SCHEMA = "openmic.integrity-authority/1"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _finite(value):
    return type(value) in {int, float} and math.isfinite(value)


def _hash(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _same(left, right):
    """Allow only harmless platform rounding in saved floating-point reports."""
    if type(left) is dict and type(right) is dict:
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if type(left) is list and type(right) is list:
        return len(left) == len(right) and all(_same(a,b) for a,b in zip(left,right))
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if _finite(left) and _finite(right):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-9)
    return type(left) is type(right) and left == right


def _head_only(edge):
    if type(edge) is not dict or edge.get("passed") is not False or edge.get("errors") != [HEAD_ERROR]:
        return False
    if any(type(edge.get(key)) is not int or edge[key] != value
           for key,value in (("sample_rate",16000),("channels",1),("sample_width_bytes",2))):
        return False
    if type(edge.get("nframes")) is not int or edge["nframes"] <= 0 or not _hash(edge.get("raw_pcm_sha256")):
        return False
    if type(edge.get("sample_peak")) is not int or edge["sample_peak"] <= 0:
        return False
    if edge.get("window_seconds") != EDGE_WINDOW_SECONDS \
            or edge.get("head_rms_limit_dbfs") != HEAD_RMS_LIMIT_DBFS \
            or edge.get("tail_rms_limit_dbfs") != TAIL_RMS_LIMIT_DBFS:
        return False
    if not _finite(edge.get("head_rms_dbfs")) or edge["head_rms_dbfs"] <= HEAD_RMS_LIMIT_DBFS:
        return False
    tail, tail_db = edge.get("tail_rms"), edge.get("tail_rms_dbfs")
    if not _finite(tail) or tail < 0:
        return False
    # The frozen inspector represents digital silence (-infinity dBFS) as null.
    if tail == 0:
        return tail_db is None
    return _finite(tail_db) and tail_db <= TAIL_RMS_LIMIT_DBFS


def canonical_sha(value):
    """SHA256 of sorted compact UTF-8 JSON, without nonfinite numbers."""
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _inside(root, value):
    if not isinstance(value, (str, Path)):
        raise ValueError("An artifact path must be a string or Path")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Review artifact path escapes assets_root")
    return path


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a nonempty string")
    return value


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key in authority evidence")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("Nonfinite JSON number in authority evidence")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique, parse_constant=invalid)


class ReviewAuthority:
    """Validated immutable-file authority, created through load_authority().

    The caller explicitly approves the file digest. Its declarations express
    the human review; digest validation authenticates the chosen file, not the
    truth of a reviewer's statements. Source candidates are never edited.
    """
    def __init__(self, path, raw, plan, assets_root):
        self.path, self.sha256, self.assets_root = path, _sha(raw), assets_root
        self.plan_sha256 = canonical_sha(plan)
        self.source_script_sha256 = plan["source_script_sha256"]
        self._raw, self._reviews, self._trials = raw, {}, {}

    def assert_current(self):
        if _sha(self.path.read_bytes()) != self.sha256:
            raise ValueError("Authority manifest changed after explicit approval")

    def validate_plan(self, plan):
        self.assert_current()
        if canonical_sha(plan) != self.plan_sha256 or plan["source_script_sha256"] != self.source_script_sha256:
            raise ValueError("Authority belongs to another plan/source")

    def snapshot_to(self, out):
        self.assert_current()
        target = Path(out) / "authorities" / (self.sha256 + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != self._raw:
            raise ValueError("Retained authority snapshot changed")
        if not target.exists():
            target.write_bytes(self._raw)

    def proof_for(self, entry, kind):
        self.assert_current()
        if type(entry) is not dict or type(entry.get("unit_id")) is not int:
            return None
        item = self._reviews.get((entry["unit_id"], kind))
        if item is None:
            return None
        if entry.get("plan_sha256") != self.plan_sha256 or entry.get("source_script_sha256") != self.source_script_sha256:
            return None
        if _sha(item["path"].read_bytes()) != item["proof"]["candidate_record_sha256"]:
            raise ValueError("Reviewed candidate record changed")
        if _sha(item["audio_path"].read_bytes()) != item["proof"]["audio_sha256"]:
            raise ValueError("Reviewed WAV changed")
        if _inside(self.assets_root, entry.get("candidate_record", "__missing__")) != item["path"]:
            return None
        if not all(_same(entry.get(key), item["record"].get(key)) for key in ("unit_id", "text", "audio_sha256", "params", "edge", "content")):
            return None
        if item.get("secondary_path") is not None and _sha(item["secondary_path"].read_bytes()) != item["proof"]["secondary_record_sha256"]:
            raise ValueError("Independent ASR evidence changed")
        return copy.deepcopy(item["proof"])

    def trial_for(self, unit_id, attempt):
        self.assert_current()
        return copy.deepcopy(self._trials.get((unit_id, attempt)))

    def trial_ids(self, unit_id):
        self.assert_current()
        return sorted(attempt for index, attempt in self._trials if index == unit_id)

    def candidate_paths(self, unit_id):
        self.assert_current()
        return sorted({item["path"] for (index, _), item in self._reviews.items() if index == unit_id})


def load_authority(path, expected_sha256, plan, assets_root):
    """Load opt-in reviews/trials only after an independently approved digest.

    Reviews bind an existing candidate record and WAV. Trials authorize one
    future request each; they never preapprove its audio or relax any gate.
    """
    from openmic.tts_integrity_gate import assess_content
    validation = validate_execution_integrity_plan(plan)
    if not validation["passed"]:
        raise ValueError("Cannot authorize an invalid plan")
    path, root = Path(path).resolve(), Path(assets_root).resolve()
    raw = path.read_bytes()
    if not _hash(expected_sha256) or _sha(raw) != expected_sha256:
        raise ValueError("Authority file does not match the explicitly supplied SHA256")
    data = _read_json(path)
    if type(data) is not dict or data.get("schema") != AUTHORITY_SCHEMA:
        raise ValueError("Unsupported authority manifest schema")
    if set(data) - {"schema", "plan_sha256", "source_script_sha256", "reviews", "trials", "maximum_new_tts_calls"}:
        raise ValueError("Unknown authority manifest field")
    authority = ReviewAuthority(path, raw, plan, root)
    if data.get("plan_sha256") != authority.plan_sha256 or data.get("source_script_sha256") != authority.source_script_sha256:
        raise ValueError("Authority manifest does not match this plan and source")
    reviews, trials = data.get("reviews", []), data.get("trials", [])
    if type(reviews) is not list or type(trials) is not list:
        raise ValueError("Authority reviews and trials must be lists")
    units, used_ids = plan["units"], set()
    for review in reviews:
        if type(review) is not dict:
            raise ValueError("Authority review must be an object")
        kind, index = review.get("kind"), review.get("unit_id")
        if kind not in {"human_listening", "orthography"} or type(index) is not int or not 0 <= index < len(units):
            raise ValueError("Unknown review kind or unit ID")
        key = (index, kind)
        identifier = _nonempty(review.get("id"), "review id")
        if key in authority._reviews or identifier in used_ids:
            raise ValueError("Duplicate authority review")
        used_ids.add(identifier)
        if review.get("decision") != "accept" or review.get("text") != units[index]["text"]:
            raise ValueError("Review must explicitly accept the exact source sentence")
        reviewer = _nonempty(review.get("reviewer"), "reviewer")
        review_text = _nonempty(review.get("review_text"), "review_text")
        candidate = _inside(root, review.get("candidate_record", "__missing__"))
        record = _read_json(candidate)
        if _sha(candidate.read_bytes()) != review.get("candidate_record_sha256"):
            raise ValueError("Reviewed candidate record digest mismatch")
        audio_path = candidate.parent / "audio.wav"
        audio = audio_path.read_bytes()
        if record.get("unit_id") != index or type(record.get("unit_id")) is not int or record.get("text") != units[index]["text"]:
            raise ValueError("Reviewed candidate belongs to another sentence")
        if (record.get("plan_sha256") != authority.plan_sha256
                or record.get("source_script_sha256") != authority.source_script_sha256
                or record.get("reference_sha256") != _sha(record["text"].encode("utf-8"))):
            raise ValueError("Reviewed candidate does not bind this plan/source/sentence")
        if record.get("audio_sha256") != review.get("audio_sha256") or _sha(audio) != review.get("audio_sha256"):
            raise ValueError("Review belongs to another WAV")
        edge = inspect_edges(audio)
        if not _same(record.get("edge"), edge) or canonical_sha(edge) != review.get("machine_edge_sha256"):
            raise ValueError("Original machine endpoint report changed")
        content = record.get("content")
        if canonical_sha(content) != review.get("machine_content_sha256"):
            raise ValueError("Original machine content report changed")
        if content is not None and content != assess_content(record["text"], record.get("asr", {}).get("transcript")):
            raise ValueError("Original content report does not match its stored ASR")
        proof = {"passed": True, "kind": kind, "review_id": identifier,
                 "authority_sha256": authority.sha256, "plan_sha256": authority.plan_sha256,
                 "source_script_sha256": authority.source_script_sha256,
                 "unit_id": index, "text": record["text"], "audio_sha256": review["audio_sha256"],
                 "candidate_record": candidate.relative_to(root).as_posix(),
                 "candidate_record_sha256": review["candidate_record_sha256"],
                 "machine_edge_sha256": review["machine_edge_sha256"],
                 "machine_content_sha256": review["machine_content_sha256"],
                 "reviewer": reviewer, "review_text": review_text}
        item = {"path": candidate, "audio_path": audio_path, "record": record, "proof": proof}
        if kind == "human_listening":
            if review.get("head_only_exemption") is not True or not _head_only(edge):
                raise ValueError("Human listening exemption is limited to the original head-only risk")
            proof.update(head_only_exemption=True, machine_edge=copy.deepcopy(edge))
        else:
            if not edge["passed"] or not isinstance(content, dict) or content.get("passed") is not False:
                raise ValueError("Orthography review requires safe endpoints and a retained strict content failure")
            secondary_path = _inside(root, review.get("secondary_asr_record", "__missing__"))
            secondary = _read_json(secondary_path)
            if _sha(secondary_path.read_bytes()) != review.get("secondary_record_sha256"):
                raise ValueError("Independent ASR record digest mismatch")
            primary = record.get("asr", {})
            reference_sha = _sha(record["text"].encode("utf-8"))
            for recognition in (primary, secondary):
                _nonempty(recognition.get("model"), "ASR model")
                if recognition.get("audio_sha256") != review["audio_sha256"] or recognition.get("reference_sha256") != reference_sha:
                    raise ValueError("Both recognizers must be bound to this WAV and source sentence")
            other = assess_content(record["text"], secondary.get("transcript"))
            substitution = review.get("substitution", {})
            if type(substitution) is not dict:
                raise ValueError("Orthographic substitution must be an explicit object")
            expected_edit = {"op": "substitute", "reference_index": substitution.get("reference_index"),
                             "hypothesis_index": substitution.get("reference_index"),
                             "reference": substitution.get("reference"), "hypothesis": substitution.get("hypothesis")}
            if (primary["model"] == secondary["model"] or type(substitution.get("reference_index")) is not int
                    or any(row["deletions"] != 0 or row["insertions"] != 0 or row["substitutions"] != 1 for row in (content, other))
                    or content["hypothesis_normalized"] != other["hypothesis_normalized"]
                    or content["edits"] != [expected_edit] or other["edits"] != [expected_edit]):
                raise ValueError("Only the explicitly reviewed single substitution with two distinct agreeing recognizers is allowed")
            proof.update(rule="explicit-single-substitution/1", primary_transcript=primary["transcript"],
                         secondary_transcript=secondary["transcript"], primary_model=primary["model"],
                         secondary_model=secondary["model"], substitution=copy.deepcopy(substitution),
                         secondary_record_sha256=review["secondary_record_sha256"])
            item["secondary_path"] = secondary_path
        authority._reviews[key] = item
    if type(data.get("maximum_new_tts_calls", 0)) is not int or data.get("maximum_new_tts_calls", 0) != len(trials):
        raise ValueError("Trial authority must state its exact bounded request budget")
    for trial in trials:
        if type(trial) is not dict:
            raise ValueError("Authorized trial must be an object")
        index, attempt = trial.get("unit_id"), trial.get("attempt")
        if type(index) is not int or not 0 <= index < len(units) or type(attempt) is not int or attempt not in {6, 7}:
            raise ValueError("Optional trial IDs are limited to 6/7 for an exact plan unit")
        if (index, attempt) in authority._trials or type(trial.get("maximum_new_tts_calls")) is not int or trial.get("maximum_new_tts_calls") != 1:
            raise ValueError("Each authorized trial permits exactly one new request")
        if trial.get("text") != units[index]["text"]:
            raise ValueError("Trial source sentence changed")
        _nonempty(trial.get("reviewer"), "trial reviewer")
        _nonempty(trial.get("review_text"), "trial authorization text")
        instruction = _nonempty(trial.get("instruction"), "trial instruction")
        if len(instruction) > 80 or "<" in instruction or ">" in instruction:
            raise ValueError("Trial instruction must be short plain text")
        _nonempty(trial.get("voice"), "trial voice")
        if not _finite(trial.get("speed")) or not .75 <= trial["speed"] <= 1.5:
            raise ValueError("Trial speed must be in .75..1.5")
        transform, text = trial.get("transform", {}), trial["text"]
        if transform.get("kind") == "terminal_pause" and transform.get("suffix") == "……":
            actual_input = text + transform["suffix"]
        elif transform.get("kind") == "focus_spans":
            spans = transform.get("spans")
            if type(spans) is not list or not spans:
                raise ValueError("Focus trial needs explicit nonoverlapping source offsets")
            previous, parts = 0, []
            for span in spans:
                if type(span) is not list or len(span) != 2 or any(type(value) is not int for value in span):
                    raise ValueError("Each focus span must be [start, end]")
                start, end = span
                if not previous <= start < end <= len(text) or not any(char.isalnum() for char in text[start:end]):
                    raise ValueError("Focus spans overlap, leave the source, or contain no spoken characters")
                parts.extend((text[previous:start], "<strong>" + text[start:end] + "</strong>"))
                previous = end
            actual_input = "".join(parts) + text[previous:]
        else:
            raise ValueError("Unknown or invalid explicit input transform")
        value = copy.deepcopy(trial)
        value.update(actual_input=actual_input, authority_sha256=authority.sha256)
        authority._trials[(index, attempt)] = value
    return authority


def human_accepted(entry, *, authority=None):
    """No externally loaded authority means no human-review exception."""
    if not isinstance(authority, ReviewAuthority):
        return False
    proof = authority.proof_for(entry, "human_listening")
    return proof is not None and entry.get("human_review") == proof and _head_only(entry.get("edge"))


def render_reviewed(plan, audio_by_id, profile="safe_course", human_reviews=None, *, authority=None):
    """Render unchanged raw PCM with explicit externally authorized head reviews.

    human_reviews maps integer unit IDs to complete accepted-entry records.
    With no reviewed entries this delegates directly to the frozen renderer.
    """
    if human_reviews is None:
        human_reviews = {}
    if type(human_reviews) is not dict:
        raise ValueError("human_reviews must be a dict keyed by integer unit IDs")
    if not human_reviews:
        return render_integrity(plan,audio_by_id,profile=profile)
    if not isinstance(authority, ReviewAuthority):
        raise ValueError("Human-reviewed audio requires explicit external authority")
    authority.validate_plan(plan)
    if profile not in PROFILES:
        raise ValueError("profile must be safe_course or safe_club")
    validation = validate_execution_integrity_plan(plan)
    if not validation["passed"]:
        raise ValueError("Invalid integrity plan: " + "; ".join(validation["errors"]))
    units = plan["units"]
    expected_ids = {unit["id"] for unit in units}
    if type(audio_by_id) is not dict or any(type(key) is not int for key in audio_by_id) or set(audio_by_id) != expected_ids:
        raise ValueError("audio_by_id must cover every plan unit exactly once")
    if any(type(key) is not int for key in human_reviews) or not set(human_reviews) <= expected_ids:
        raise ValueError("Unknown or non-integer reviewed unit ID")
    if any(not human_accepted(entry, authority=authority) or entry["unit_id"] != index for index,entry in human_reviews.items()):
        raise ValueError("Every listed human-review entry must carry its exact authorization")
    prepared = []
    for unit in units:
        index, audio = unit["id"], audio_by_id[unit["id"]]
        check = inspect_edges(audio)
        entry = human_reviews.get(index)
        if entry is not None:
            if not human_accepted(entry, authority=authority) or entry["unit_id"] != index or entry["text"] != unit["text"]:
                raise ValueError("Invalid user-listening authorization for unit " + str(index))
            if not isinstance(audio,bytes) or _sha(audio) != entry["audio_sha256"] or not _same(check,entry["edge"]):
                raise ValueError("Reviewed audio/report no longer matches the listened WAV for unit " + str(index))
        elif not check["passed"]:
            raise ValueError("Rejected unreviewed raw audio for unit " + str(index))
        prepared.append((_strict_pcm(audio),check,entry))
    pieces, speech_parts, timeline = [], [], []
    position = speech_frames = silence_frames = 0
    durations = BOUNDARY_SECONDS[profile]

    def pause(seconds,pause_type,**details):
        nonlocal position,silence_frames
        if not _finite(seconds) or seconds < 0:
            raise ValueError("Pause must be finite and nonnegative")
        count = round(seconds*SAMPLE_RATE)
        timeline.append({"kind":"pause","pause_type":pause_type,"requested_seconds":seconds,"nframes":count,
                         "start_frame":position,"end_frame":position+count,
                         "start_seconds":position/SAMPLE_RATE,"end_seconds":(position+count)/SAMPLE_RATE,
                         "duration_seconds":count/SAMPLE_RATE,**details})
        pieces.append(np.zeros(count,dtype="<i2"))
        position += count
        silence_frames += count

    first_source = float(units[0]["source_pause_before_seconds"])
    pause(max(START_GUARD_SECONDS,first_source),"start_guard",guard_floor_seconds=START_GUARD_SECONDS,
          source_pause_before_seconds=first_source,reason="max_of_start_guard_and_original_source_pause")
    previous = None
    for unit,(pcm,check,entry) in zip(units,prepared):
        if previous is not None:
            changed = unit["base_unit_index"] != previous["base_unit_index"]
            boundary = durations.get(unit["boundary_before"],0.0) if changed else 0.0
            reaction = durations["reaction"] if changed and previous["base_unit_end"] and previous["intent"] in {"reveal","button"} else 0.0
            source_pause = float(unit["source_pause_before_seconds"])
            pause(max(SENTENCE_PAUSE_FLOOR_SECONDS,boundary,reaction,source_pause),"boundary",
                  before_unit_id=unit["id"],boundary_before=unit["boundary_before"],base_unit_changed=changed,
                  floor_seconds=SENTENCE_PAUSE_FLOOR_SECONDS,planned_boundary_seconds=boundary,
                  reaction_seconds=reaction,source_pause_before_seconds=source_pause,
                  reason="max_of_sentence_floor_boundary_previous_base_reaction_and_source_pause")
        count = len(pcm)
        row = {"kind":"speech","unit_id":unit["id"],"base_unit_index":unit["base_unit_index"],
               "base_unit_start":unit["base_unit_start"],"base_unit_end":unit["base_unit_end"],
               "source_index":unit["source_index"],"source_start":unit["start"],"source_end":unit["end"],
               "text":unit["text"],"intent":unit["intent"],"focus":unit["focus"],"role":unit["role"],"beat":unit["beat"],
               "nframes":count,"raw_frames":count,"raw_pcm_sha256":check["raw_pcm_sha256"],
               "start_frame":position,"end_frame":position+count,"start_seconds":position/SAMPLE_RATE,
               "end_seconds":(position+count)/SAMPLE_RATE,"duration_seconds":count/SAMPLE_RATE,
               "trimmed_frames":0,"fade_frames":0,"local_gain_db":0.0,"global_headroom_gain":1.0,
               "edge_check":check,"human_review_passed":entry is not None,
               "synthesis_parameters_source":"accepted candidate provenance saved by calling pipeline"}
        if entry is not None:
            row["human_review"] = copy.deepcopy(entry["human_review"])
            row["machine_content_check"] = copy.deepcopy(entry.get("content"))
        timeline.append(row)
        pieces.append(pcm)
        speech_parts.append(pcm)
        speech_frames += count
        position += count
        previous = unit
    source_tail = float(plan["source_tail_pause_seconds"])
    tail_reaction = durations["reaction"] if previous["base_unit_end"] and previous["intent"] in {"reveal","button"} else 0.0
    pause(max(END_GUARD_SECONDS,source_tail,tail_reaction),"end_guard",guard_floor_seconds=END_GUARD_SECONDS,
          source_tail_pause_seconds=source_tail,reaction_seconds=tail_reaction,
          reason="max_of_end_guard_original_tail_pause_and_final_base_reaction")
    audio = encode_wav(np.concatenate(pieces).astype("<i2",copy=False))
    plan_bytes = json.dumps(plan,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode("utf-8")
    reviewed_ids = sorted(human_reviews)
    metrics = {"adapter":ADAPTER,"base_adapter":BASE_ADAPTER,"profile":profile,"condition":profile,
               "source_kind":"provided_pcm_with_explicit_user_review","synthetic":None,
               "sample_rate":SAMPLE_RATE,"channels":1,"sample_width_bytes":2,
               "nframes":position,"speech_frames":speech_frames,"pause_frames":silence_frames,
               "duration_seconds":position/SAMPLE_RATE,"speech_seconds":speech_frames/SAMPLE_RATE,
               "injected_silence_seconds":silence_frames/SAMPLE_RATE,
               "speech_pcm_sha256":_sha(b"".join(pcm.tobytes() for pcm in speech_parts)),"audio_sha256":_sha(audio),
               "source_script_sha256":plan["source_script_sha256"],"spoken_text_sha256":plan["spoken_text_sha256"],
               "base_stage_plan_sha256":plan["base_stage_plan_sha256"],"plan_sha256":_sha(plan_bytes),
               "unit_count":len(units),"speech_count":len(units),
               "edge_window_seconds":EDGE_WINDOW_SECONDS,"head_rms_limit_dbfs":HEAD_RMS_LIMIT_DBFS,
               "tail_rms_limit_dbfs":TAIL_RMS_LIMIT_DBFS,"edge_guard_start_seconds":START_GUARD_SECONDS,
               "edge_guard_end_seconds":END_GUARD_SECONDS,"sentence_pause_floor_seconds":SENTENCE_PAUSE_FLOOR_SECONDS,
               "boundary_seconds":dict(durations),
               "pause_policy":"maximum of guard/floor, base-boundary, final-base reaction, and original source pause",
               "source_zero_pause_policy":"recorded as zero; the new sentence floor and reaction policy may be longer",
               "speech_processing":"complete raw PCM copied without trimming, fades, overlap, pitch shift or gain",
               "original_stage_intent_gain_executed":False,"per_sentence_normalization_executed":False,
               "global_headroom_gain":1.0,"api_calls":0,"api_scope":"renderer only; no synthesis or ASR request",
               "original_script_verified":False,"native_phoneme_completeness_verified":False,
               "automatic_edge_pass_unit_ids":[unit["id"] for unit in units if unit["id"] not in human_reviews],
               "user_reviewed_unit_ids":reviewed_ids,"all_machine_edges_passed":False,
               "human_listening_acceptance_used":True,"human_review_count":len(reviewed_ids),
               "authority_sha256":authority.sha256,
               "review_scope":"Only exact candidate WAVs in the explicitly loaded review manifest; original machine checks remain unchanged",
               "automatic_ids_scope":"Machine edge pass only; content/orthographic acceptance is recorded by the calling pipeline",
               "quality_note":"The user confirmed these exact recordings; this is not a general endpoint waiver or proof of human MOS"}
    return RenderResult(audio=audio,metrics=metrics,timeline=timeline)
