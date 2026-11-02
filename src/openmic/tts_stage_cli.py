"""Stage delivery entry point; original CFunSet generation and TTS stay intact."""
import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

from openmic.tts_backends import SiliconFlowBackend, ToneBackend
from openmic.tts_render import save_result
from openmic.tts_review import review_script
from openmic.tts_stage_plan import build_plan, request_plan, validate_plan
from openmic.tts_stage_render import STAGE_INSTRUCTION, render_stage

PROFILES = ("stage_prompt", "stage_course", "stage_club")


def planner_timeout(value):
    try:
        seconds = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("planner timeout must be an integer, 30–600 seconds") from None
    if not 30 <= seconds <= 600:
        raise argparse.ArgumentTypeError("planner timeout must be 30–600 seconds")
    return seconds


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def source_review(script, trace_path, *, manual=False):
    if trace_path is None:
        if not manual:
            raise ValueError("请提供 --trace 核查CFunSet来源和正文，手工稿须显式 --manual-script")
        return {"passed": True, "manual_script": True, "human_review_pending": True,
                "note": "Explicit manually supplied script; no model approval claimed"}
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    report = review_script(script, trace, min_characters=1, min_punchlines=0, min_callbacks=0)
    report["cfunset_example_ids"] = trace.get("cfunset_example_ids", [])
    report["source_policy"] = trace.get("source_policy")
    report["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if not report["passed"]:
        raise ValueError("原稿交接未通过，未请求TTS：" + "；".join(report["errors"]))
    return report


def parser():
    main = argparse.ArgumentParser(description="固定音色的舞台口吻与内容驱动的表演对照")
    subs = main.add_subparsers(dest="command", required=True)
    for name in ("plan", "synth"):
        sub = subs.add_parser(name)
        sub.add_argument("--script", type=Path, required=True)
        sub.add_argument("--trace", type=Path)
        sub.add_argument("--manual-script", action="store_true")
        sub.add_argument("--out", type=Path, required=True)
        if name == "plan":
            sub.add_argument("--planner", choices=["rules", "llm"], default="rules")
            sub.add_argument("--planner-timeout", type=planner_timeout, default=240,
                             help="single LLM request timeout in seconds, 30–600 (default: 240)")
        else:
            sub.add_argument("--plan", type=Path, required=True)
            sub.add_argument("--backend", choices=["tone", "siliconflow"], default="tone")
            sub.add_argument("--profiles", choices=PROFILES, nargs="+", default=list(PROFILES))
            sub.add_argument("--cache", type=Path, default=Path("artifacts/jyb/stage/cache"))
            sub.add_argument("--no-cache", action="store_true")
            sub.add_argument("--voice")
    return main


def main(argv=None):
    args = parser().parse_args(argv)
    script = args.script.read_text(encoding="utf-8-sig")
    review = source_review(script, args.trace, manual=args.manual_script)
    if args.command == "plan":
        try:
            if args.planner == "llm":
                from dataclasses import replace
                from openmic.autogen_v02 import AutoGenConfig
                config = replace(AutoGenConfig.from_env(), timeout_seconds=args.planner_timeout)
                plan = request_plan(script, config)
            else:
                plan = build_plan(script)
            report = validate_plan(plan, script)
            if not report["passed"]:
                raise ValueError("；".join(report["errors"]))
        except Exception as exc:
            json_write(args.out.with_suffix(".failed.json"), {"error_type": type(exc).__name__,
                       "planner": getattr(exc, "planner", None), "source_review": review,
                       "tts_requested": False})
            raise
        json_write(args.out, plan)
        json_write(args.out.with_suffix(".source_review.json"), review)
        print(f"表演计划通过：{len(plan['source_speeches'])}原段 → {len(plan['units'])}执行段；规划器={args.planner}")
        return
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    report = validate_plan(plan, script)
    if not report["passed"]:
        raise ValueError("；".join(report["errors"]))
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / ".stage-synthesis.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "source_script": str(args.script)}, handle)
    try:
        for name, text in (("script.txt", script), ("stage_plan.json", json.dumps(plan, ensure_ascii=False, indent=2))):
            path = args.out / name
            if path.exists() and path.read_text(encoding="utf-8") != text:
                raise ValueError("已有结果的稿件/计划不同，请使用新输出目录")
            path.write_text(text, encoding="utf-8")
        json_write(args.out / "source_review.json", review)
        backend = ToneBackend() if args.backend == "tone" else SiliconFlowBackend(voice=args.voice)
        for profile in args.profiles:
            target = args.out / profile
            existing = target / "metrics.json"
            if existing.is_file():
                metrics = json.loads(existing.read_text(encoding="utf-8"))
                if metrics.get("execution_plan_sha256") != fingerprint or metrics.get("backend") != backend.identity:
                    raise ValueError("已有结果属于不同计划或后端；拒绝覆盖")
                if metrics.get("actual_instruction") != STAGE_INSTRUCTION:
                    raise ValueError("已有结果使用不同合成指令；请使用新目录，不覆盖此前试听")
                if not (target / "audio.wav").is_file() or not (target / "timeline.json").is_file():
                    raise ValueError("已有结果不完整，请检查后改用新输出目录")
                print("保留已完成结果：" + profile, flush=True)
                continue
            print("正在合成：" + profile, flush=True)
            run_id = uuid.uuid4().hex[:12]
            counter = 0

            def chunk(audio, row):
                nonlocal counter
                counter += 1
                folder = target / "chunks" / run_id
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{counter:04d}.wav").write_bytes(audio)

            result = render_stage(plan, backend, profile=profile,
                                  cache_dir=None if args.no_cache else args.cache, on_chunk=chunk)
            result.metrics.update(execution_plan_sha256=fingerprint, source_script_sha256=plan["source_script_sha256"],
                                  chunk_directory="chunks/" + run_id)
            save_result(result, target)
            print(f"{profile}：{result.metrics['duration_seconds']:.2f}秒；请求={result.metrics['api_calls']}，缓存={result.metrics['cache_hits']}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        print("舞台执行失败：" + str(error), file=sys.stderr)
        raise SystemExit(1)
