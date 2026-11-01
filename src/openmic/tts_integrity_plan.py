"""Deterministic sentence slices of a validated, unchanged stage plan.

Offsets are Unicode character offsets in the original source Speech, not
phoneme or word timestamps. Existing paired quotations remain one unit.
"""

import hashlib
import json
import math
import unicodedata

from openmic.tts_stage_plan import validate_plan as validate_stage_plan


SCHEMA = "openmic.integrity-plan/1"
PLAN_FIELDS = {"schema", "source_script_sha256", "spoken_text_sha256",
               "base_stage_plan_sha256", "source_tail_pause_seconds", "units"}
UNIT_FIELDS = {"id", "source_index", "start", "end", "text", "base_unit_index",
               "intent", "focus", "role", "beat", "boundary_before",
               "base_unit_start", "base_unit_end", "source_pause_before_seconds"}
INTENTS = {"setup", "audience_question", "self_question", "quoted_question",
           "contrast", "reveal", "aside", "button"}
ROLES = {"intro", "setup", "punchline", "callback"}
BASE_BOUNDARIES = {"flow", "breath", "turn", "reaction"}
ENDS = set("。！？；?!")
OPEN_QUOTES = {"“": "”", "‘": "’", "「": "」", "『": "』", "«": "»", "《": "》"}
CLOSE_QUOTES = set(OPEN_QUOTES.values()) | {'"', "'"}


def stage_plan_sha256(plan):
    return hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 10


def _hash(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _plain(text):
    return type(text) is str and bool(text.strip()) and not any(char in text for char in "[]<>") \
        and not any(unicodedata.category(char) in {"Cc", "Cf"} and char not in "\t\r\n" for char in text)


def _quote_depths(text):
    stack, depths = [], [0]
    for index, char in enumerate(text):
        apostrophe = char == "'" and index > 0 and index + 1 < len(text) \
            and text[index-1].isascii() and text[index-1].isalnum() \
            and text[index+1].isascii() and text[index+1].isalnum()
        if not apostrophe:
            if char in OPEN_QUOTES:
                stack.append(OPEN_QUOTES[char])
            elif char in {'"', "'"}:
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    stack.append(char)
            elif char in CLOSE_QUOTES:
                _require(bool(stack) and stack[-1] == char, "Unmatched closing quotation in a base unit")
                stack.pop()
        depths.append(len(stack))
    _require(not stack, "Unclosed quotation in a base unit; refuse an ambiguous sentence split")
    return depths


def sentence_slices(text):
    """Return exact [start,end) slices, including punctuation and whitespace.

    Delimiters are Chinese 。！？； and ASCII ?!. ASCII periods are deliberately
    not delimiters, so decimal numbers and abbreviations are never split here.
    A closing quote after a quoted sentence can finish the enclosing unit.
    """
    _require(_plain(text), "Sentence source must be nonempty plain text")
    depths = _quote_depths(text)
    cuts = []
    for position in range(1, len(text)):
        if depths[position] != 0 or text[position].isspace() or text[position] in ENDS | CLOSE_QUOTES:
            continue
        left = text[:position].rstrip()
        while left and left[-1] in CLOSE_QUOTES - {"》"}:
            left = left[:-1].rstrip()
        if left and left[-1] in ENDS:
            cuts.append(position)
    positions = [0, *cuts, len(text)]
    return [(start, end) for start, end in zip(positions, positions[1:])]


def _source_pauses(stage_plan):
    before, pending, speech_index = {}, 0.0, 0
    for event in stage_plan["source_events"]:
        if event["kind"] == "pause":
            pending = max(pending, float(event["seconds"]))
        else:
            before[speech_index] = pending
            speech_index += 1
            pending = 0.0
    return before, pending


def build_integrity_plan(script, stage_plan):
    report = validate_stage_plan(stage_plan, script)
    _require(report["passed"], "Invalid original stage plan: " + "; ".join(report["errors"]))
    pauses, tail = _source_pauses(stage_plan)
    units, seen_sources = [], set()
    for base in stage_plan["units"]:
        slices = sentence_slices(base["text"])
        focus_start = base["text"].find(base["focus"]) if base["focus"] is not None else -1
        for part_index, (start, end) in enumerate(slices):
            text = base["text"][start:end]
            focus = base["focus"]
            if focus is not None and not (start <= focus_start and focus_start + len(focus) <= end
                                           and text.find(focus) == text.rfind(focus)):
                focus = None
            first_source_unit = base["source_index"] not in seen_sources
            units.append({"id": len(units), "source_index": base["source_index"],
                          "start": base["start"] + start, "end": base["start"] + end, "text": text,
                          "base_unit_index": base["index"], "intent": base["intent"], "focus": focus,
                          "role": base["role"], "beat": base["beat"],
                          "boundary_before": base["boundary_before"] if part_index == 0 else "sentence",
                          "base_unit_start": part_index == 0, "base_unit_end": part_index == len(slices)-1,
                          "source_pause_before_seconds": pauses.get(base["source_index"], 0.0) if first_source_unit else 0.0})
            seen_sources.add(base["source_index"])
    plan = {"schema": SCHEMA, "source_script_sha256": stage_plan["source_script_sha256"],
            "spoken_text_sha256": stage_plan["spoken_text_sha256"],
            "base_stage_plan_sha256": stage_plan_sha256(stage_plan),
            "source_tail_pause_seconds": tail, "units": units}
    report = validate_execution_integrity_plan(plan)
    _require(report["passed"], "; ".join(report["errors"]))
    return plan


def _validate_execution(plan):
    _require(type(plan) is dict and set(plan) == PLAN_FIELDS, "Integrity plan has missing or extra fields")
    _require(plan["schema"] == SCHEMA, "Unknown integrity schema")
    for key in ("source_script_sha256", "spoken_text_sha256", "base_stage_plan_sha256"):
        _require(_hash(plan[key]), key + " must be a lowercase SHA256")
    _require(_number(plan["source_tail_pause_seconds"]), "Invalid source tail pause")
    units = plan["units"]
    _require(type(units) is list and 1 <= len(units) <= 10000, "Integrity units must be a bounded nonempty list")
    previous, base_focuses, groups = None, {}, {}
    for index, unit in enumerate(units):
        _require(type(unit) is dict and set(unit) == UNIT_FIELDS, "Unit has missing or extra fields")
        for key in ("id", "source_index", "start", "end", "base_unit_index", "beat"):
            _require(type(unit[key]) is int and unit[key] >= 0, key + " must be a nonnegative integer, not bool")
        _require(unit["id"] == index and _plain(unit["text"]), "Unit id or text is invalid")
        groups.setdefault(unit["base_unit_index"], []).append(unit)
        _require(unit["end"] > unit["start"] and unit["end"] - unit["start"] == len(unit["text"]), "Unit offsets do not match text")
        _require(unit["intent"] in INTENTS and unit["role"] in ROLES, "Unknown intent or role")
        _require(type(unit["base_unit_start"]) is bool and type(unit["base_unit_end"]) is bool, "Base boundary flags must be bool")
        _require(_number(unit["source_pause_before_seconds"]), "Invalid source pause")
        first_base = previous is None or unit["base_unit_index"] != previous["base_unit_index"]
        _require(unit["base_unit_start"] == first_base, "Incorrect base_unit_start flag")
        following = units[index+1] if index+1 < len(units) else None
        _require(following is None or type(following) is dict, "Following unit is not an object")
        last_base = following is None or following.get("base_unit_index") != unit["base_unit_index"]
        _require(unit["base_unit_end"] == last_base, "Incorrect base_unit_end flag")
        if previous is None:
            _require(unit["source_index"] == unit["base_unit_index"] == unit["start"] == 0, "First source/base/offset must start at zero")
        else:
            _require(unit["beat"] >= previous["beat"], "Source beat order decreases")
            _require(unit["base_unit_index"] in {previous["base_unit_index"], previous["base_unit_index"] + 1}, "Base unit order has a gap or duplicate group")
            if unit["source_index"] == previous["source_index"]:
                _require(unit["start"] == previous["end"] and unit["source_pause_before_seconds"] == 0, "Source slices overlap/gap or repeat a source pause")
            else:
                _require(unit["source_index"] == previous["source_index"] + 1 and unit["start"] == 0, "Source speech order has a gap")
                _require(first_base, "A base unit crosses source speeches")
            if not first_base:
                _require(all(unit[key] == previous[key] for key in ("source_index", "intent", "role", "beat")), "Sentence changes base annotations")
        _require(unit["boundary_before"] in BASE_BOUNDARIES if first_base else unit["boundary_before"] == "sentence", "Invalid inherited/sentence boundary")
        focus = unit["focus"]
        if focus is not None:
            _require(type(focus) is str and 1 <= len(focus) <= 10 and unit["text"].find(focus) >= 0
                     and unit["text"].find(focus) == unit["text"].rfind(focus)
                     and all(unicodedata.category(char)[0] in {"L", "N", "M"} for char in focus), "Invalid focus")
            base_focuses[unit["base_unit_index"]] = base_focuses.get(unit["base_unit_index"], 0) + 1
            _require(base_focuses[unit["base_unit_index"]] <= 1, "A base focus was duplicated across sentences")
        previous = unit
    for parts in groups.values():
        base_text = "".join(unit["text"] for unit in parts)
        expected_parts = [base_text[start:end] for start, end in sentence_slices(base_text)]
        _require([unit["text"] for unit in parts] == expected_parts, "Units do not follow deterministic sentence boundaries")
    spoken = "".join(unit["text"] for unit in units)
    _require(len(spoken) <= 30000, "Integrity spoken source exceeds its original limit")
    _require(hashlib.sha256(spoken.encode("utf-8")).hexdigest() == plan["spoken_text_sha256"], "Integrity units do not match spoken_text_sha256")


def validate_execution_integrity_plan(plan):
    """Validate executable structure without claiming original-source identity."""
    report = {"passed": False, "errors": [], "original_script_verified": False}
    try:
        _validate_execution(plan)
        report["passed"] = True
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        report["errors"].append(str(error))
    return report


def validate_integrity_plan(plan, script, stage_plan):
    report = validate_execution_integrity_plan(plan)
    if not report["passed"]:
        return report
    try:
        expected = build_integrity_plan(script, stage_plan)
        _require(plan == expected, "Integrity plan differs from deterministic original-source slices or annotations")
        report["original_script_verified"] = True
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        report["passed"] = False
        report["errors"].append(str(error))
    return report
