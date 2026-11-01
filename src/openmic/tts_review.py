"""Fail-closed checks before paid TTS; does not grade whether a joke is funny."""

import math
import re

from openmic.tts import extract_spoken_script
from openmic.tts_plan import Speech, compile_script, plain_text


SCORE_LABELS = ("HUMOR_SCORE", "CULTURE_SCORE", "STRUCTURE_SCORE")


def normalized_spoken_text(text):
    """Ignore formatting/punctuation, but preserve all words and their order."""
    body = extract_spoken_script(text)
    body = re.sub(r"\[FILLER=([^\]]+)\]", r"\1", body, flags=re.IGNORECASE)
    body = re.sub(r"\[[^\]\n]*\]", "", body)
    body = re.sub(r"<[^>]+>", "", body)
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", body)).lower()


def parse_quality(content):
    """Require all three finite scores and exactly one final approval line."""
    clean = content.replace("**", "").replace("__", "").replace("`", "")
    scores, errors = {}, []
    for label in SCORE_LABELS:
        values = re.findall(
            rf"\b{label}\b\s*(?:[（(][^\n)）]*[)）])?\s*[:：|]\s*(-?\d+(?:\.\d+)?)",
            clean, flags=re.IGNORECASE,
        )
        if len(values) != 1:
            errors.append(f"质检必须恰好给出一个 {label} 分数")
            continue
        score = float(values[0])
        if not math.isfinite(score) or not 0 <= score <= 10:
            errors.append(f"{label} 必须是0–10的有限数字")
            continue
        scores[label] = score
    decisions = re.findall(r"(?im)^\s*DECISION\s*:\s*(APPROVED|REVISION_REQUIRED)\s*$", clean)
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    final = re.fullmatch(r"DECISION\s*:\s*(APPROVED|REVISION_REQUIRED)", lines[-1], flags=re.IGNORECASE) if lines else None
    explicit_approval = len(decisions) == 1 and final is not None and final.group(1).upper() == "APPROVED"
    if not explicit_approval:
        errors.append("质检必须只有一个明确决策，且最后一行是 DECISION: APPROVED")
    mean = sum(scores.values()) / len(scores) if len(scores) == 3 else None
    if mean is not None and mean < 7:
        errors.append("三项质检平均分低于7.0")
    return {"quality_scores": scores, "quality_mean": mean,
            "approved": explicit_approval and len(scores) == 3 and mean >= 7,
            "errors": errors}


def review_script(script, trace, *, min_characters=850, max_characters=1100, strict_length=False):
    """Review the final coach script against its final writer/QC trace.

    The lower length bound prevents accidentally synthesizing a tiny sample as
    a course deliverable. Upper length is a warning by default: actual 180–300s
    must still be measured, and no character count establishes the duration.
    """
    errors, warnings = [], []
    result = {"passed": False, "errors": errors, "warnings": warnings,
              "chinese_characters": 0, "speech_segments": 0, "punchlines": 0,
              "callbacks": 0, "writer_coach_text_match": False,
              "quality_scores": {}, "quality_mean": None, "approved": False}
    try:
        events = compile_script(script)
    except (ValueError, TypeError) as exc:
        errors.append("表演协议解析失败：" + str(exc))
        return result
    spoken = [event for event in events if isinstance(event, Speech)]
    characters = len(re.findall(r"[\u4e00-\u9fff]", "".join(plain_text(event.text) for event in spoken)))
    # A beat split across lines or emotion changes still counts once.
    punches = len({event.beat for event in spoken if event.role == "punchline"})
    callbacks = len({event.beat for event in spoken if event.role == "callback"})
    result.update(chinese_characters=characters, speech_segments=len(spoken),
                  punchlines=punches, callbacks=callbacks)
    if characters < min_characters:
        errors.append(f"正文仅{characters}个汉字，少于本轮最低{min_characters}字")
    if characters > max_characters:
        message = f"正文{characters}个汉字超过建议上限{max_characters}字，须按实际音频时长判断"
        (errors if strict_length else warnings).append(message)
    if not 12 <= len(spoken) <= 18:
        warnings.append(f"语音段数为{len(spoken)}，建议12–18段；必须试听分段连接")
    if punches < 4:
        errors.append(f"包袱只有{punches}处，至少需要4处")
    if callbacks < 1:
        errors.append("缺少CALLBACK回调")
    messages = trace.get("messages", []) if isinstance(trace, dict) else []
    final = {}
    for message in messages:
        if isinstance(message, dict):
            final[message.get("agent")] = str(message.get("content", ""))
    writer, coach, qc = (final.get(name, "") for name in ("JokeWriter", "PerformanceCoach", "QualityController"))
    if not writer:
        errors.append("缺少最近一版JokeWriter原稿")
    if not coach:
        errors.append("缺少最近一版PerformanceCoach输出")
    same = bool(writer and coach) and normalized_spoken_text(writer) == normalized_spoken_text(coach)
    result["writer_coach_text_match"] = same
    if not same:
        errors.append("Coach必须保留最近Writer稿件的全部字词及顺序")
    if coach and normalized_spoken_text(script) != normalized_spoken_text(coach):
        errors.append("待合成稿件与最近Coach输出的正文不同")
    if not qc:
        errors.append("缺少QualityController输出")
    quality = parse_quality(qc)
    result.update({key: quality[key] for key in ("quality_scores", "quality_mean", "approved")})
    errors.extend(quality["errors"])
    if isinstance(trace, dict) and trace.get("approved") is False:
        errors.append("工作流trace明确记录质检未通过")
    result["passed"] = not errors
    result["duration_note"] = "字数仅是生成门槛，最终必须实测180–300秒，不能用静音补足"
    result["human_review_pending"] = True
    return result
