"""Bounded, source-preserving delivery annotations for the existing TTS plan.

This module never synthesizes audio. Rule planning is offline; request_plan
only requests annotations and never accepts replacement prose from a model.
"""

import copy
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict

from openmic.tts_plan import EMOTIONS, Pause, Speech, compile_script, plain_text


SCHEMA = "openmic.stage-plan/1"
INTENTS = ("setup", "audience_question", "self_question", "quoted_question",
           "contrast", "reveal", "aside", "button")
BOUNDARIES = ("flow", "breath", "turn", "reaction")
ROLES = ("intro", "setup", "punchline", "callback")
PLAN_FIELDS = {"schema", "source_script_sha256", "spoken_text_sha256", "source_events",
               "source_speeches", "units", "planner"}
SPEECH_FIELDS = {"index", "text", "marked_text", "role", "beat", "speed"}
UNIT_FIELDS = {"index", "source_index", "start", "end", "text", "intent", "focus",
               "boundary_before", "role", "beat"}
_HAN = re.compile(r"[\u3400-\u9fff]")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ENDS = set("。？！；.!?;")
_OPEN_QUOTES = {"“": "”", "‘": "’", "「": "」", "『": "』", "«": "»"}
_CLOSE_QUOTES = set(_OPEN_QUOTES.values()) | {'"', "'"}
_QUESTION = set("？?")


class StagePlanError(ValueError):
    """A rejected plan, with safe planner records for the caller to persist."""

    def __init__(self, message, *, planner=None, errors=None):
        super().__init__(message)
        self.errors = list(errors or [message])
        self.planner = copy.deepcopy(planner or {"type": "rules", "status": "failed",
                                               "annotations_supplied": False})


def _check(condition, message):
    if not condition:
        raise StagePlanError(message)


def _object(value, fields, label):
    _check(type(value) is dict, label + " must be an object")
    _check(set(value) == set(fields), label + " has missing or extra fields")


def _integer(value, label, low=0):
    _check(type(value) is int and value >= low, label + " must be an integer, not a boolean")


def _number(value, low, high, label):
    _check(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
           label + " must be finite and within the allowed range")


def _safe_plain(text, label, *, empty=False):
    _check(type(text) is str and (empty or bool(text.strip())), label + " must be text")
    _check(not any(char in text for char in "[]<>"), label + " contains a control token or markup")
    _check(not any((unicodedata.category(char) in {"Cc", "Cf"} and char not in "\t\r\n")
                   for char in text), label + " contains hidden control characters")


def _marked_parts(marked):
    _check(type(marked) is str, "marked_text must be text")
    plain, emphases, current, active = [], [], [], False
    for token in re.split(r"(</?strong>)", marked):
        if token == "<strong>":
            _check(not active, "nested strong markup is not allowed")
            active, current = True, []
        elif token == "</strong>":
            _check(active, "strong closing markup has no opening")
            emphases.append("".join(current))
            active = False
        else:
            _safe_plain(token, "marked_text prose", empty=True)
            plain.append(token)
            if active:
                current.append(token)
    _check(not active, "strong markup is unclosed")
    return "".join(plain), emphases


def _quote_depths(text):
    stack, depths, valid = [], [0], True
    for index, char in enumerate(text):
        apostrophe = (char == "'" and index > 0 and index + 1 < len(text)
                      and text[index - 1].isascii() and text[index - 1].isalnum()
                      and text[index + 1].isascii() and text[index + 1].isalnum())
        if not apostrophe:
            if char in _OPEN_QUOTES:
                stack.append(_OPEN_QUOTES[char])
            elif char in {'"', "'"}:
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    stack.append(char)
            elif char in _CLOSE_QUOTES:
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    valid = False
        depths.append(len(stack))
    return depths, valid and not stack


def _legal_split(text, position):
    if not 0 < position < len(text):
        return False
    depths, balanced = _quote_depths(text)
    if not balanced or depths[position] != 0:
        return False
    left = text[:position].rstrip()
    while left and left[-1] in _CLOSE_QUOTES:
        left = left[:-1].rstrip()
    if not left or left[-1] not in _ENDS:
        return False
    right = text[position:].lstrip()
    if left[-1] == "." and len(left) > 1 and right and left[-2].isdigit() and right[0].isdigit():
        return False
    return True


def _question_features(text):
    depths, balanced = _quote_depths(text)
    outside = balanced and any(char in _QUESTION and depths[index] == 0
                               for index, char in enumerate(text))
    quoted = any(char in _QUESTION and depths[index] > 0 for index, char in enumerate(text))
    audience = outside and any(
        all(depths[index] == 0 for index in range(match.start(), match.end()))
        for match in re.finditer("你们|大家|各位", text)
    )
    return bool(audience), bool(outside), bool(quoted)


def _valid_focus(focus, text):
    return (type(focus) is str and 1 <= len(focus) <= 10
            and text.find(focus) >= 0 and text.find(focus) == text.rfind(focus)
            and all(unicodedata.category(char)[0] in {"L", "N", "M"} for char in focus))


def _source(script):
    _check(type(script) is str, "script must be a Unicode string")
    events = compile_script(script)
    source_events, speeches = [], []
    for event in events:
        source_events.append({**asdict(event), "kind": "speech" if isinstance(event, Speech) else "pause"})
        if isinstance(event, Speech):
            speeches.append({"index": len(speeches), "text": plain_text(event.text),
                             "marked_text": event.text, "role": event.role,
                             "beat": event.beat, "speed": event.speed})
    return {"schema": SCHEMA,
            "source_script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
            "spoken_text_sha256": hashlib.sha256("".join(item["text"] for item in speeches).encode("utf-8")).hexdigest(),
            "source_events": source_events, "source_speeches": speeches}


def _rule_annotations(speeches):
    segments = []
    for speech in speeches:
        text, role = speech["text"], speech["role"]
        audience, outside, quoted = _question_features(text)
        if role in {"punchline", "callback"}:
            intent = "reveal" if role == "punchline" else "button"
        elif audience:
            intent = "audience_question"
        elif outside:
            intent = "self_question"
        elif quoted:
            intent = "quoted_question"
        elif re.search("不是|而是|但是|结果|没想到|原来|却", text):
            intent = "contrast"
        elif text.startswith(("说实话", "其实", "真的")):
            intent = "aside"
        else:
            intent = "setup"
        _, prior_focus = _marked_parts(speech["marked_text"])
        focus = next((phrase for phrase in prior_focus if _valid_focus(phrase, text)), None)
        segments.append({"source_index": speech["index"], "parts": [
            {"text": text, "intent": intent, "focus": focus, "boundary_before": "flow"}]})
    budget = math.ceil(0.65 * len(segments))
    candidates = [index for index, segment in enumerate(segments) if segment["parts"][0]["focus"] is not None]
    candidates.sort(key=lambda index: (speeches[index]["role"] not in {"punchline", "callback"}, index))
    for index in candidates[budget:]:
        segments[index]["parts"][0]["focus"] = None
    return {"segments": segments}


def _units(annotations, speeches):
    _object(annotations, {"segments"}, "annotations")
    segments = annotations["segments"]
    _check(type(segments) is list and len(segments) == len(speeches), "segments must cover every source speech once")
    units = []
    for index, (segment, speech) in enumerate(zip(segments, speeches)):
        _object(segment, {"source_index", "parts"}, "segment")
        _integer(segment["source_index"], "source_index")
        _check(segment["source_index"] == index, "segments are duplicated, omitted or out of order")
        parts = segment["parts"]
        _check(type(parts) is list and 1 <= len(parts) <= 2, "each source speech requires one or two parts")
        position = 0
        for part in parts:
            _object(part, {"text", "intent", "focus", "boundary_before"}, "part")
            _safe_plain(part["text"], "part text")
            end = position + len(part["text"])
            _check(speech["text"][position:end] == part["text"], "part text rewrites source words or punctuation")
            # The execution text comes from the immutable source, not from
            # model-supplied text, even after the exact-equality check.
            units.append({"index": len(units), "source_index": index,
                          "start": position, "end": end, "text": speech["text"][position:end],
                          "intent": part["intent"], "focus": part["focus"],
                          "boundary_before": part["boundary_before"],
                          "role": speech["role"], "beat": speech["beat"]})
            position = end
        _check(position == len(speech["text"]), "parts omit source text")
    return units


def _validate_execution(plan):
    _object(plan, PLAN_FIELDS, "plan")
    _check(plan["schema"] == SCHEMA, "unknown plan schema")
    for key in ("source_script_sha256", "spoken_text_sha256"):
        _check(type(plan[key]) is str and _HASH.fullmatch(plan[key]) is not None, key + " must be a lowercase SHA256")
    events, speeches, units = plan["source_events"], plan["source_speeches"], plan["units"]
    _check(type(events) is list and 1 <= len(events) <= 10000, "source_events must be a bounded nonempty list")
    _check(type(speeches) is list and 1 <= len(speeches) <= 300, "source_speeches must be a bounded nonempty list")
    expected_speeches = []
    previous_beat = -1
    for event in events:
        _check(type(event) is dict and event.get("kind") in {"speech", "pause"}, "unknown source event")
        if event["kind"] == "pause":
            _object(event, {"kind", "seconds", "reason"}, "pause event")
            _number(event["seconds"], 0, 10, "source pause")
            _check(event["reason"] in {"explicit", "automatic"}, "unknown source pause reason")
            continue
        _object(event, {"kind", "text", "role", "emotion", "speed", "emphasis", "beat"}, "speech event")
        text, _ = _marked_parts(event["text"])
        _safe_plain(text, "source speech text")
        _check(event["role"] in ROLES and event["emotion"] in EMOTIONS, "unknown source role or emotion")
        _number(event["speed"], 0.75, 1.5, "source speed")
        _integer(event["beat"], "source beat")
        _check(event["beat"] >= previous_beat, "source beat order is invalid")
        previous_beat = event["beat"]
        _check(type(event["emphasis"]) is bool and event["emphasis"] == ("<strong>" in event["text"]), "source emphasis metadata is inconsistent")
        expected_speeches.append({"index": len(expected_speeches), "text": text, "marked_text": event["text"],
                                  "role": event["role"], "beat": event["beat"], "speed": event["speed"]})
    for index, speech in enumerate(speeches):
        _object(speech, SPEECH_FIELDS, "source speech")
        _integer(speech["index"], "source speech index")
        _integer(speech["beat"], "source speech beat")
        _number(speech["speed"], 0.75, 1.5, "source speech speed")
        _safe_plain(speech["text"], "source speech text")
        _check(speech["index"] == index, "source speech indexes are duplicated or omitted")
    _check(speeches == expected_speeches, "source_speeches do not match source_events")
    _check(sum(len(speech["text"]) for speech in speeches) <= 30000, "spoken source exceeds the original script limit")
    spoken_hash = hashlib.sha256("".join(speech["text"] for speech in speeches).encode("utf-8")).hexdigest()
    _check(plan["spoken_text_sha256"] == spoken_hash, "spoken text SHA256 does not match source speeches")
    _check(type(units) is list and len(speeches) <= len(units) <= math.ceil(1.5 * len(speeches)), "unit count exceeds the bounded split budget")
    grouped = [[] for _ in speeches]
    previous_source = -1
    focus_count = 0
    for index, unit in enumerate(units):
        _object(unit, UNIT_FIELDS, "unit")
        for key in ("index", "source_index", "start", "end", "beat"):
            _integer(unit[key], "unit " + key)
        _check(unit["index"] == index, "unit indexes are duplicated, omitted or out of order")
        source_index = unit["source_index"]
        _check(source_index < len(speeches) and source_index >= previous_source, "cross-speech or reordered unit")
        previous_source = source_index
        source = speeches[source_index]
        _check(0 <= unit["start"] < unit["end"] <= len(source["text"]), "unit source offsets are invalid")
        _safe_plain(unit["text"], "unit text")
        _check(unit["text"] == source["text"][unit["start"]:unit["end"]], "unit text differs from its source slice")
        _check(unit["role"] == source["role"] and unit["beat"] == source["beat"], "unit changes source role or beat")
        _check(unit["intent"] in INTENTS and unit["boundary_before"] in BOUNDARIES, "unknown intent or boundary")
        if unit["intent"] == "audience_question":
            _check(_question_features(unit["text"])[0], "audience_question requires unquoted audience address and question")
        elif unit["intent"] == "self_question":
            _check(_question_features(unit["text"])[1], "self_question requires an existing unquoted question")
        elif unit["intent"] == "quoted_question":
            _check(any(char in _QUESTION for char in unit["text"]), "quoted_question requires an existing question")
        if unit["focus"] is not None:
            _check(_valid_focus(unit["focus"], unit["text"]), "focus must be one unique source phrase of 1..10 characters without punctuation or controls")
            focus_count += 1
        grouped[source_index].append(unit)
    _check(focus_count <= math.ceil(0.65 * len(units)), "too many units contain focus")
    for speech, parts in zip(speeches, grouped):
        _check(1 <= len(parts) <= 2, "a source speech was omitted or split too many times")
        _check(parts[0]["start"] == 0 and parts[-1]["end"] == len(speech["text"]), "units do not cover the full source speech")
        _check("".join(part["text"] for part in parts) == speech["text"], "units overlap, omit or rewrite source text")
        if len(parts) == 2:
            _check(parts[0]["end"] == parts[1]["start"], "split parts overlap or leave a gap")
            _check(_legal_split(speech["text"], parts[0]["end"]), "split must follow sentence punctuation outside paired quotes")
            _check(all(len(_HAN.findall(part["text"])) >= 14 for part in parts), "split parts must each contain at least 14 Han characters")
    planner = plan["planner"]
    _check(type(planner) is dict and planner.get("type") in {"rules", "llm"}, "unknown planner type")
    _check(planner.get("status") == "validated", "planner has not validated this plan")
    if planner["type"] == "rules":
        _object(planner, {"type", "status", "annotations_supplied"}, "rule planner")
        _check(type(planner["annotations_supplied"]) is bool, "annotations_supplied must be boolean")
    else:
        _object(planner, {"type", "status", "model", "attempts"}, "LLM planner")
        _check(type(planner["model"]) is str and bool(planner["model"].strip()), "planner model is missing")
        attempts = planner["attempts"]
        _check(type(attempts) is list and 1 <= len(attempts) <= 2, "planner attempt budget is invalid")
        for attempt in attempts:
            _object(attempt, {"raw_response", "errors"}, "planner attempt")
            _check(type(attempt["raw_response"]) is str and type(attempt["errors"]) is list
                   and all(type(error) is str for error in attempt["errors"]), "planner attempt record is invalid")
        _check(not attempts[-1]["errors"], "last planner attempt failed validation")


def validate_execution_plan(plan):
    """Check executable structure; the original script hash is not authenticated."""
    report = {"passed": False, "errors": [], "original_script_verified": False,
              "scope": "internal source/event/unit consistency; original script not supplied"}
    try:
        _validate_execution(plan)
        report["passed"] = True
    except (StagePlanError, TypeError, ValueError, KeyError, OverflowError) as exc:
        report["errors"].append(str(exc))
    return report


def validate_plan(plan, script):
    """Additionally rebuild source events and hashes from the original script."""
    report = validate_execution_plan(plan)
    if not report["passed"]:
        return report
    report["scope"] = "original script plus internal source/event/unit consistency"
    try:
        expected = _source(script)
        for key, value in expected.items():
            _check(plan[key] == value, key + " does not match the original script")
        report["original_script_verified"] = True
    except (StagePlanError, TypeError, ValueError, KeyError, OverflowError) as exc:
        report["passed"] = False
        report["errors"].append(str(exc))
    return report


def build_plan(script, annotations=None):
    """Build a no-network plan, optionally validating supplied annotations."""
    planner = {"type": "rules", "status": "validated", "annotations_supplied": annotations is not None}
    try:
        plan = _source(script)
        supplied = _rule_annotations(plan["source_speeches"]) if annotations is None else annotations
        plan["units"] = _units(supplied, plan["source_speeches"])
        plan["planner"] = planner
        report = validate_plan(plan, script)
        if not report["passed"]:
            raise StagePlanError("; ".join(report["errors"]), errors=report["errors"])
        return plan
    except (StagePlanError, TypeError, ValueError, KeyError, OverflowError) as exc:
        planner["status"] = "failed"
        raise StagePlanError(str(exc), planner=planner, errors=getattr(exc, "errors", None)) from None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _check(key not in result, "duplicate JSON field: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise StagePlanError("Non-finite JSON number: " + value)


def _scrub(value, config):
    text = str(value)
    key = getattr(config, "api_key", None)
    if isinstance(key, str) and key:
        text = text.replace(key, "[REDACTED]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", text)


def request_plan(script, config, client=None, max_attempts=2):
    """Request bounded annotations, with at most one format-repair request.

    A failed call raises StagePlanError with .planner records. There is no
    automatic fallback to a successful rule plan and no TTS request here.
    """
    model = getattr(config, "model", "")
    planner = {"type": "llm", "status": "failed", "model": _scrub(model, config) if type(model) is str else "", "attempts": []}
    if type(max_attempts) is not int or not 1 <= max_attempts <= 2:
        raise StagePlanError("max_attempts must be 1 or 2", planner=planner)
    if type(model) is not str or not model.strip():
        raise StagePlanError("planner model must be nonempty text", planner=planner)
    try:
        source = _source(script)
    except (TypeError, ValueError) as exc:
        raise StagePlanError(_scrub(exc, config), planner=planner) from None
    system = (
        "你是同一位中文单口演员的表演规划员，只标注已有正文，绝不重写、不增删字词或标点，"
        "不加观众称呼、反问句、笑声或新情节。source_speeches是完整上下文；parts.text必须"
        "来自每项text，不抄marked_text中的标签。不要模仿角色或更换音色。"
        "只返回JSON对象，唯一顶层字段segments；覆盖全部source_index且保持从0递增，不能漏项或重复。"
        "每项只有source_index、parts。每个原Speech用1或2个part，短揭底不拆；"
        "全文part数最多ceil(1.5*Speech数)。仅在句末。？！；或其闭引号之后且成对引号外切分，"
        "两个part各至少14个汉字；不能逗号处或词中切分。parts.text逐字拼接必须与原text完全相同。"
        "每个part恰有text、intent、focus、boundary_before四个字段。intent只允许："
        + ", ".join(INTENTS)
        + "。audience_question仅用于正文已有你们/大家/各位且有引号外问号的句子，"
        "引用人物问话用quoted_question，不把所有问号都当问观众。reveal是改变理解的信息，"
        "button是原有收尾，不据此增加结构标签。focus为原文唯一出现的1至10字符短语或null，"
        "只选一个关键焦点，不能含标点、空白、标签或特殊控制符；最多ceil(.65*part总数)个part有focus。"
        "boundary_before仅flow/breath/turn/reaction；优先flow，转折前短停用turn，揭底后反应用reaction。"
        "不要输出速度、音量、自由指令、解释或正文以外的其他字段。"
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"source_speeches": source["source_speeches"]}, ensure_ascii=False)}]
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.api_key, base_url=getattr(config, "base_url", None),
                            timeout=getattr(config, "timeout_seconds", 120), max_retries=0)
        except Exception as exc:
            planner["attempts"].append({"raw_response": "", "errors": ["Planner client unavailable: " + type(exc).__name__]})
            raise StagePlanError("Planner client unavailable", planner=planner) from None
    for attempt_number in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=config.model, messages=messages, response_format={"type": "json_object"},
                temperature=0.2, max_tokens=6000,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            status = status if type(status) is int and 100 <= status <= 599 else None
            error = "Planner request failed: " + type(exc).__name__ + ("; HTTP " + str(status) if status else "")
            planner["attempts"].append({"raw_response": "", "errors": [error]})
            raise StagePlanError(error, planner=planner) from None
        try:
            raw = response.choices[0].message.content
            _check(type(raw) is str, "planner response content is not text")
        except (AttributeError, IndexError, TypeError, StagePlanError) as exc:
            raw = ""
            initial_errors = ["Invalid planner response shape: " + type(exc).__name__]
        else:
            initial_errors = []
        record = {"raw_response": _scrub(raw, config), "errors": initial_errors}
        planner["attempts"].append(record)
        if not initial_errors:
            try:
                annotations = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
                plan = build_plan(script, annotations)
                planner["status"] = "validated"
                plan["planner"] = copy.deepcopy(planner)
                report = validate_plan(plan, script)
                _check(report["passed"], "; ".join(report["errors"]))
                return plan
            except (StagePlanError, TypeError, ValueError, KeyError, OverflowError) as exc:
                planner["status"] = "failed"
                record["errors"] = [_scrub(error, config) for error in getattr(exc, "errors", [str(exc)])]
        if attempt_number + 1 < max_attempts:
            messages.extend([{"role": "assistant", "content": raw},
                             {"role": "user", "content": "验证失败：" + "; ".join(record["errors"])
                              + "。仅修复标注JSON，逐字保留所有source文本，重返完整segments，不输出解释。"}])
    raise StagePlanError("Planner annotations failed validation; no speech requested", planner=planner,
                         errors=planner["attempts"][-1]["errors"])
