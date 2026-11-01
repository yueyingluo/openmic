"""Offline sentence-level ASR fidelity screen, never a listening guarantee."""

import math
import re
import unicodedata


NORMALIZATION_VERSION = "openmic.integrity-normalization/1"
MAX_ALIGNMENT_CELLS = 2_000_000
_DIGITS = {char: str(index) for index, char in enumerate("零一二三四五六七八九")}
_DIGITS.update({"〇": "0", "○": "0", "两": "2"})
_SMALL = {"十": 10, "百": 100, "千": 1000}
_LARGE = {"万": 10000, "亿": 100000000}
_NUMERAL = re.compile(r"[0-9]+(?:\.[0-9]+)?|[零〇○一二两三四五六七八九十百千万亿]+(?:[点.][零〇○一二两三四五六七八九]+)?")


def _chinese_integer(text):
    if not text or text[0] not in set(_DIGITS) | {"十"}:
        # 万一 / 亿万 are lexical/ambiguous forms, not explicit integers.
        return None
    if all(char in _DIGITS for char in text):
        # Preserve leading zeros; do not turn an ambiguous 两三 into 23.
        if len(text) > 1 and not any(char in "零〇○" for char in text):
            return None
        return "".join(_DIGITS[char] for char in text)
    if re.search(r"[一二两三四五六七八九]{2}", text):
        return None
    total, section, number, previous_small = 0, 0, 0, 10000
    for char in text:
        if char in _DIGITS:
            number = int(_DIGITS[char])
        elif char in _SMALL:
            unit = _SMALL[char]
            if unit >= previous_small:
                return None
            section += (number or 1) * unit
            number, previous_small = 0, unit
        elif char in _LARGE:
            section += number
            unit = _LARGE[char]
            total = (total + section) * unit if unit == 100000000 else total + section * unit
            section, number, previous_small = 0, 0, 10000
        else:
            return None
    return str(total + section + number)


def _number_token(match):
    token = match.group(0)
    if token[0].isascii() and token[0].isdigit():
        # Keep the decimal distinction after punctuation cleaning: 2.5 != 25.
        return token.replace(".", "点")
    integer, separator, fraction = token.replace(".", "点").partition("点")
    value = _chinese_integer(integer)
    if value is None:
        return token
    if separator:
        return value + "点" + "".join(_DIGITS[char] for char in fraction)
    return value


def normalize_content(text, *, numbers=True):
    """NFKC, explicit numeral equivalents, then Unicode letters/numbers only.

    Numeral folding is syntactic and documented; it does not rewrite the source
    or equate homophones, abbreviations, missing words or arbitrary paraphrases.
    """
    if type(text) is not str:
        raise TypeError("Content text must be a string")
    text = unicodedata.normalize("NFKC", text)
    if numbers:
        text = _NUMERAL.sub(_number_token, text)
    return "".join(char for char in text if unicodedata.category(char)[0] in {"L", "N"}).lower()


def _distance(left, right):
    previous = list(range(len(right)+1))
    for i, first in enumerate(left, 1):
        current = [i]
        for j, second in enumerate(right, 1):
            current.append(min(previous[j]+1, current[-1]+1, previous[j-1]+(first != second)))
        previous = current
    return previous[-1]


def _alignment(reference, hypothesis):
    """Minimum character edits; positions refer to normalized text, not audio."""
    n, m = len(reference), len(hypothesis)
    if (n+1)*(m+1) > MAX_ALIGNMENT_CELLS:
        raise ValueError("Content alignment exceeds the bounded sentence budget")
    directions = [bytearray(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        directions[i][0] = 1  # delete reference character
    for j in range(1, m+1):
        directions[0][j] = 2  # insert hypothesis character
    previous = list(range(m+1))
    for i in range(1, n+1):
        current = [i]
        for j in range(1, m+1):
            costs = (previous[j-1] + (reference[i-1] != hypothesis[j-1]),
                     previous[j]+1, current[j-1]+1)
            choice = min(range(3), key=lambda value: costs[value])
            directions[i][j] = choice
            current.append(costs[choice])
        previous = current
    operations, i, j = [], n, m
    while i or j:
        direction = directions[i][j]
        if direction == 0:
            operations.append({"op": "equal" if reference[i-1] == hypothesis[j-1] else "substitute",
                               "reference_index": i-1, "hypothesis_index": j-1,
                               "reference": reference[i-1], "hypothesis": hypothesis[j-1]})
            i, j = i-1, j-1
        elif direction == 1:
            operations.append({"op": "delete", "reference_index": i-1, "hypothesis_index": j,
                               "reference": reference[i-1], "hypothesis": ""})
            i -= 1
        else:
            operations.append({"op": "insert", "reference_index": i, "hypothesis_index": j-1,
                               "reference": "", "hypothesis": hypothesis[j-1]})
            j -= 1
    operations.reverse()
    return previous[-1], operations


def _edge_check(operations, start, end):
    affected = [row for row in operations if start <= row["reference_index"] < end and row["op"] != "insert"]
    deletions = sum(row["op"] == "delete" for row in affected)
    substitutions = sum(row["op"] == "substitute" for row in affected)
    return {"passed": deletions == 0 and substitutions <= 1, "reference_start": start, "reference_end": end,
            "deletions": deletions, "substitutions": substitutions,
            "positions_scope": "normalized reference character offsets; not audio timestamps"}


def assess_content(expected, transcript):
    """Screen one physical sentence clip with explicit, conservative limits.

    raw_cer uses NFKC/punctuation cleaning without numeral equivalence. cer uses
    the explicit numeral fold. Neither metric proves every phoneme is audible.
    """
    raw_reference = normalize_content(expected, numbers=False)
    raw_hypothesis = normalize_content(transcript, numbers=False)
    reference, hypothesis = normalize_content(expected), normalize_content(transcript)
    if max((len(raw_reference)+1)*(len(raw_hypothesis)+1), (len(reference)+1)*(len(hypothesis)+1)) > MAX_ALIGNMENT_CELLS:
        raise ValueError("Content alignment exceeds the bounded sentence budget")
    distance, operations = _alignment(reference, hypothesis)
    n = len(reference)
    deletions = sum(row["op"] == "delete" for row in operations)
    insertions = sum(row["op"] == "insert" for row in operations)
    substitutions = sum(row["op"] == "substitute" for row in operations)
    longest, run = 0, 0
    for row in operations:
        if row["op"] == "delete":
            run += 1
            longest = max(longest, run)
        elif row["op"] != "insert":
            run = 0
    k = min(4, n)
    head, tail = _edge_check(operations, 0, k), _edge_check(operations, n-k, n)
    cer, deletion_limit = distance/max(1, n), math.floor(.03*n)
    reasons = []
    if not reference:
        reasons.append("Reference has no normalized spoken characters")
    if not hypothesis:
        reasons.append("ASR transcript has no normalized spoken characters")
    if n <= 8:
        if reference != hypothesis:
            reasons.append("Short sentences (<=8 normalized characters) require an exact normalized match")
    else:
        if cer > .12:
            reasons.append("Sentence CER exceeds 12%")
        if deletions > deletion_limit:
            reasons.append("Sentence deletions exceed floor(3% of reference length)")
        if longest > 1:
            reasons.append("A consecutive source deletion run exceeds one character")
        if not head["passed"]:
            reasons.append("Sentence head has deletion or more than one substitution")
        if not tail["passed"]:
            reasons.append("Sentence tail has deletion or more than one substitution")
    return {"passed": not reasons, "raw_cer": _distance(raw_reference, raw_hypothesis)/max(1, len(raw_reference)),
            "cer": cer, "deletions": deletions, "insertions": insertions, "substitutions": substitutions,
            "longest_deletion_run": longest, "head_check": head, "tail_check": tail,
            "head_passed": head["passed"], "tail_passed": tail["passed"],
            "reference_normalized": reference, "hypothesis_normalized": hypothesis,
            "raw_reference_normalized": raw_reference, "raw_hypothesis_normalized": raw_hypothesis,
            "reference_length": n, "hypothesis_length": len(hypothesis), "deletion_limit": deletion_limit,
            "short_sentence": n <= 8, "errors": reasons, "reasons": reasons,
            "edits": [row for row in operations if row["op"] != "equal"],
            "normalization": {"version": NORMALIZATION_VERSION, "unicode": "NFKC",
                              "punctuation": "Keep Unicode letters/numbers; lowercase Latin letters",
                              "numbers": "Explicit Chinese integer/decimal tokens and Arabic digits, e.g. 五=5, 二十=20, 双十一=双11; preserve leading zeros and decimal distinctions",
                              "raw_cer_scope": "NFKC plus punctuation cleaning; no numeral folding",
                              "uncertainty": "No homophone or paraphrase correction; ambiguous numeral strings remain unchanged"},
            "scope": "ASR source-word screen only; not a listening guarantee, word timing, MOS or stage-quality score"}
