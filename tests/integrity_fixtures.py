"""Synthetic local evidence. Fixtures never assert actual user listening."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from openmic.tts_backends import encode_wav
from openmic.tts_integrity_audio import inspect_edges
from openmic.tts_integrity_cli import parameters, policy_sha
from openmic.tts_integrity_gate import assess_content
from openmic.tts_integrity_listening import AUTHORITY_SCHEMA, canonical_sha, load_authority
from openmic.tts_integrity_plan import build_integrity_plan
from openmic.tts_stage_plan import build_plan

SCRIPT = "<SCRIPT>[SETUP]今天我们先把这句话完整说出来。</SCRIPT>"

def digest(data): return hashlib.sha256(data).hexdigest()

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def build_fixture_plan(script=SCRIPT):
    return build_integrity_plan(script, build_plan(script))

def audio_fixture(hot_head=False, hot_tail=False):
    head = np.full(160, 1800 if hot_head else 0, dtype="<i2")
    body = np.tile(np.array([4000, -4000, 20, -20], dtype="<i2"), 160)
    tail = np.full(200, 1800 if hot_tail else 0, dtype="<i2")
    return encode_wav(np.concatenate((head, body, tail)))

class FixtureBackend:
    synthetic = True
    identity = {"backend": "synthetic-fixture", "voice": "fixture:stable", "sample_rate": 16000}
    def __init__(self, audio=None):
        self.calls, self.requests, self.audio = 0, [], audio or audio_fixture()
    def synthesize(self, text, *, instruction, speed):
        self.calls += 1
        self.requests.append({"text": text, "instruction": instruction, "speed": speed})
        return self.audio

def record_fixture(root, plan, *, audio=None, transcript=None, attempt=1, index=0):
    root = Path(root)
    unit, backend = plan["units"][index], FixtureBackend()
    audio = audio or audio_fixture(hot_head=True)
    folder = root / "run/candidates" / f"{index:03d}" / f"{attempt:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "audio.wav").write_bytes(audio)
    edge = inspect_edges(audio)
    record = {"unit_id": index, "text": unit["text"], "audio_sha256": digest(audio),
              "reference_sha256": digest(unit["text"].encode("utf-8")), "edge": edge,
              "params": parameters(unit, backend, attempt), "policy_sha256": policy_sha(),
              "plan_sha256": canonical_sha(plan), "source_script_sha256": plan["source_script_sha256"],
              "attempt": attempt, "new_tts_api_calls": 1, "origin": "synthetic-fixture",
              "synthetic": True, "control_fallback": attempt >= 3, "content": None,
              "status": "edge_rejected" if not edge["passed"] else "downloaded"}
    if transcript is not None:
        record["asr"] = {"model": "fixture-primary", "transcript": transcript,
                         "audio_sha256": digest(audio), "reference_sha256": record["reference_sha256"]}
        record["content"] = assess_content(unit["text"], transcript)
        record["status"] = "passed" if edge["passed"] and record["content"]["passed"] else "content_rejected"
    path = folder / "record.json"
    write_json(path, record)
    return path, record

def entry_fixture(root, plan, path, record):
    return {"unit_id": record["unit_id"], "text": record["text"], "audio_sha256": record["audio_sha256"],
            "params": copy.deepcopy(record["params"]), "content": copy.deepcopy(record.get("content")),
            "edge": copy.deepcopy(record["edge"]), "policy_sha256": record["policy_sha256"],
            "synthetic": True, "plan_sha256": canonical_sha(plan),
            "source_script_sha256": plan["source_script_sha256"],
            "candidate_record": Path(path).relative_to(root).as_posix(),
            "audio_path": (Path(path).parent / "audio.wav").relative_to(root).as_posix()}

def authority_fixture(root, plan, path=None, record=None, *, kind="human_listening", trials=None):
    root, reviews = Path(root), []
    if path is not None:
        review = {"id": "synthetic-review", "kind": kind, "unit_id": record["unit_id"],
                  "text": record["text"], "audio_sha256": record["audio_sha256"], "decision": "accept",
                  "reviewer": "unit-test fixture, not a real listener",
                  "review_text": "Synthetic authorization to test record and PCM integrity only.",
                  "candidate_record": Path(path).relative_to(root).as_posix(),
                  "candidate_record_sha256": digest(Path(path).read_bytes()),
                  "machine_edge_sha256": canonical_sha(record["edge"]),
                  "machine_content_sha256": canonical_sha(record.get("content"))}
        if kind == "human_listening":
            review["head_only_exemption"] = True
        else:
            secondary = copy.deepcopy(record["asr"])
            secondary["model"] = "fixture-independent-secondary"
            secondary_path = root / "secondary.json"
            write_json(secondary_path, secondary)
            edit = record["content"]["edits"][0]
            review.update(secondary_asr_record=secondary_path.relative_to(root).as_posix(),
                          secondary_record_sha256=digest(secondary_path.read_bytes()),
                          substitution={"reference_index": edit["reference_index"],
                                        "reference": edit["reference"], "hypothesis": edit["hypothesis"]})
        reviews.append(review)
    trials = trials or []
    document = {"schema": AUTHORITY_SCHEMA, "plan_sha256": canonical_sha(plan),
                "source_script_sha256": plan["source_script_sha256"], "reviews": reviews,
                "trials": trials, "maximum_new_tts_calls": len(trials)}
    target = root / "authority.json"
    write_json(target, document)
    return load_authority(target, digest(target.read_bytes()), plan, root), document

def approved_entry(root, plan, path, record, authority, kind="human_listening"):
    entry = entry_fixture(root, plan, path, record)
    entry["human_review" if kind == "human_listening" else "orthography_review"] = authority.proof_for(entry, kind)
    return entry

def trial_fixture(plan, attempt=6, *, transform=None):
    return {"unit_id": 0, "attempt": attempt, "text": plan["units"][0]["text"],
            "maximum_new_tts_calls": 1, "reviewer": "synthetic test author",
            "review_text": "Explicit synthetic input-transform request, not audio approval.",
            "voice": FixtureBackend.identity["voice"], "instruction": "Read clearly.", "speed": 1.0,
            "transform": transform or {"kind": "terminal_pause", "suffix": "……"}}
