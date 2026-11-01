"""Compile the coach's markers before the old adapter discards them.

One role marker starts one beat. Emotion/speed persist within that beat.
An unclosed legacy [EMPHASIS] ends at a newline or the next control marker.
Explicit pauses override automatic pauses at the same boundary, including 0.
"""

import math
import re
from dataclasses import dataclass
from typing import List, Union

from openmic.tts import extract_spoken_script


EMOTIONS = {
    "neutral": "自然平静", "happy": "高兴", "sad": "悲伤",
    "angry": "轻微生气", "surprised": "惊讶", "calm": "冷静",
    "serious": "严肃", "self_deprecating": "带着自嘲的平静",
}
ALIASES = {
    "开心": "happy", "高兴": "happy", "悲伤": "sad", "伤心": "sad",
    "生气": "angry", "愤怒": "angry", "惊讶": "surprised",
    "冷静": "calm", "严肃": "serious", "自然": "neutral",
    "平静": "neutral", "自嘲": "self_deprecating",
}
ROLE_SPEED = {"intro": 1.0, "setup": 1.05, "punchline": 0.95, "callback": 0.95}


@dataclass(frozen=True)
class Speech:
    text: str
    role: str = "setup"
    emotion: str = "neutral"
    speed: float = 1.0
    emphasis: bool = False
    beat: int = 0


@dataclass(frozen=True)
class Pause:
    seconds: float
    reason: str = "explicit"


Event = Union[Speech, Pause]


def plain_text(text: str) -> str:
    return text.replace("<strong>", "").replace("</strong>", "")


def _number(value: str, low: float, high: float, label: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{label} 必须是数字：{value}") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{label} 必须在 {low}–{high} 之间：{value}")
    return number


def compile_script(script: str, *, add_fillers: bool = False) -> List[Event]:
    """Parse strict controls, then add nonduplicated 0.8/2.0 s pauses.

    Filler insertion is opt-in and restricted to every third long setup beat.
    All experimental conditions must use the SAME compiled plan.
    """
    source = extract_spoken_script(script).replace("\r\n", "\n").strip()
    source = re.sub(r"(?m)^\s*(?:```\w*|```)\s*$", "", source)
    source = re.sub(r"(?m)^\s*(?:#{1,6}\s+|>\s*)", "", source)
    source = source.replace("**", "").replace("`", "")
    if not source or len(source) > 30000:
        raise ValueError("脚本不能为空，且须少于 30000 个字符")
    # No provider control tokens / HTML may enter through spoken prose.
    if "<" in source or ">" in source:
        raise ValueError("正文含未知尖括号标签，请仅使用方括号协议；外层可用 <SCRIPT>")
    tokens = re.split(r"(\[[^\]\n]*\]|\n)", source)
    raw: List[Event] = []
    role, emotion, speed, beat = "setup", "neutral", ROLE_SPEED["setup"], 0
    pieces: List[str] = []
    emph = False

    def close_emphasis() -> None:
        nonlocal emph
        if emph:
            pieces.append("</strong>")
            emph = False

    def flush() -> None:
        close_emphasis()
        text = "".join(pieces).strip()
        pieces.clear()
        if plain_text(text).strip():
            if "[" in text or "]" in text:
                raise ValueError("存在未闭合的方括号标记")
            raw.append(Speech(text, role, emotion, speed, "<strong>" in text, beat))

    roles = {"INTRO": "intro", "开场": "intro", "SETUP": "setup",
             "PUNCHLINE": "punchline", "CALLBACK": "callback"}
    for token in tokens:
        if not token:
            continue
        if token == "\n":
            flush()
            continue
        if not token.startswith("["):
            pieces.append(token)
            continue
        tag = token[1:-1].strip()
        name, sep, value = tag.partition("=")
        name = name.strip().upper()
        value = value.strip()
        if name == "/EMPHASIS" and not sep:
            if not emph:
                raise ValueError("[/EMPHASIS] 缺少对应的 [EMPHASIS]")
            close_emphasis()
        elif name == "EMPHASIS" and not sep:
            close_emphasis()
            pieces.append("<strong>")
            emph = True
        elif name in roles and not sep:
            flush()
            role, emotion = roles[name], "neutral"
            speed = ROLE_SPEED[role]
            beat += 1
        elif name == "PAUSE" and sep:
            flush()
            raw.append(Pause(_number(value, 0, 10, "PAUSE")))
        elif name == "EMOTION" and sep:
            flush()
            emotion = ALIASES.get(value, value.lower())
            if emotion not in EMOTIONS:
                raise ValueError(f"未知 EMOTION={value}；支持 {', '.join(EMOTIONS)}")
        elif name == "SPEED" and sep:
            flush()
            speed = _number(value, 0.75, 1.5, "SPEED")
        elif name == "FILLER" and sep:
            if value not in {"呃", "那个", "嗯"}:
                raise ValueError("FILLER 仅支持 呃 / 那个 / 嗯")
            pieces.append(value + "，")
        else:
            raise ValueError(f"未知标记 [{tag}]；请修正，避免让 TTS 朗读标记")
    flush()
    if not any(isinstance(e, Speech) for e in raw):
        raise ValueError("脚本没有可朗读正文")
    # Boundaries are indexed before raw[i]. Several automatic requests at a
    # shared boundary use max(), not sum(); an explicit Pause always wins.
    boundaries = {}
    for i, event in enumerate(raw):
        if not isinstance(event, Speech) or event.role not in {"punchline", "callback"}:
            continue
        previous = next((e for e in reversed(raw[:i]) if isinstance(e, Speech)), None)
        following = next((e for e in raw[i + 1:] if isinstance(e, Speech)), None)
        if previous is None or previous.beat != event.beat:
            if i == 0 or not isinstance(raw[i - 1], Pause):
                boundaries[i] = max(boundaries.get(i, 0), 0.8)
        if following is None or following.beat != event.beat:
            if i + 1 == len(raw) or not isinstance(raw[i + 1], Pause):
                boundaries[i + 1] = max(boundaries.get(i + 1, 0), 2.0)
    result: List[Event] = []
    setup_count = 0
    seen_beats = set()
    for i in range(len(raw) + 1):
        if i in boundaries:
            result.append(Pause(boundaries[i], "automatic"))
        if i == len(raw):
            break
        event = raw[i]
        if isinstance(event, Speech) and event.beat not in seen_beats:
            seen_beats.add(event.beat)
            if event.role == "setup":
                setup_count += 1
                already_filled = re.match(r"^\s*(?:那个|呃|嗯)(?:[，,、\s]|$)", plain_text(event.text))
                if add_fillers and setup_count % 3 == 0 and len(plain_text(event.text)) >= 24 and not already_filled:
                    event = Speech("那个，" + event.text, event.role, event.emotion,
                                   event.speed, event.emphasis, event.beat)
        result.append(event)
    if sum(isinstance(e, Speech) for e in result) > 300:
        raise ValueError("分段超过 300，请合并过短的换行或标记")
    return result
