"""Shared local-only I/O. Importing this module performs no network/model calls."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESIGN = ROOT.parent


def sha256(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    if isinstance(value, str):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')


def finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def unique_by(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        if key in result:
            raise ValueError(f'Duplicate key {key}')
        result[key] = row
    return result
