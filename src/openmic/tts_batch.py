"""Generate and render the five fixed topics without shell-specific loops."""

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from openmic.tts_cli import _generate
from openmic.tts_backends import SiliconFlowBackend, ToneBackend
from openmic.tts_plan import compile_script
from openmic.tts_render import CONDITIONS, render, save_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    gen = subs.add_parser("generate")
    gen.add_argument("--topics", type=Path, default=Path("examples/jyb_topics.json"))
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--backend", choices=["mock", "autogen"], default="mock")
    synth = subs.add_parser("synth")
    synth.add_argument("--scripts", type=Path, required=True)
    synth.add_argument("--out", type=Path, required=True)
    synth.add_argument("--backend", choices=["tone", "siliconflow"], default="tone")
    synth.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    synth.add_argument("--cache", type=Path, default=Path("artifacts/jyb/cache"))
    synth.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "generate":
            topics = json.loads(args.topics.read_text(encoding="utf-8-sig"))
            seen = set()
            for topic in topics:
                name = topic["id"]
                if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in seen:
                    raise ValueError("主题 id 必须唯一，且只能使用字母、数字、下划线、连字符")
                seen.add(name)
            for topic in topics:
                _generate(SimpleNamespace(topic=topic["topic"], style=topic["style"],
                                          duration=topic["duration_minutes"], audience=topic["audience"],
                                          backend=args.backend, out=args.out / (topic["id"] + ".txt")))
            return
        paths = sorted(args.scripts.glob("*.txt"))
        if not paths:
            raise ValueError("scripts 目录下没有 .txt 稿件")
        # Validate every topic before starting paid speech generation.
        plans = [(path, compile_script(path.read_text(encoding="utf-8-sig"))) for path in paths]
        backend = ToneBackend() if args.backend == "tone" else SiliconFlowBackend()
        if backend.synthetic:
            print("仅生成测试音，不是中文语音或课程成品。")
        summaries = []
        for path, plan in plans:
            topic_dir = args.out / path.stem
            topic_dir.mkdir(parents=True, exist_ok=True)
            (topic_dir / "script.txt").write_text(path.read_text(encoding="utf-8-sig"), encoding="utf-8")
            (topic_dir / "plan.json").write_text(json.dumps([asdict(e) for e in plan], ensure_ascii=False, indent=2), encoding="utf-8")
            for condition in args.conditions:
                target = topic_dir / condition
                target.mkdir(parents=True, exist_ok=True)
                (target / "metrics.json").unlink(missing_ok=True)
                print(f"{path.stem}/{condition} 开始", flush=True)
                result = render(plan, backend, condition=condition,
                                cache_dir=None if args.no_cache else args.cache)
                save_result(result, target)
                summaries.append({"topic": path.stem, **result.metrics})
                print(f"完成：{result.metrics['duration_seconds']:.2f}s；3–5分钟={result.metrics['in_target_duration']}")
        (args.out / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    except (KeyError, ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
