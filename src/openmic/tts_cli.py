"""TTS performance commands, integrated into the OpenMic package."""

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from openmic.tts_backends import SiliconFlowBackend, ToneBackend
from openmic.tts_plan import compile_script
from openmic.tts_render import CONDITIONS, render, save_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="jyb: OpenMic 表演控制实验（无需训练）")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="不调用 API，先检查脚本协议")
    plan.add_argument("--script", type=Path, required=True)
    plan.add_argument("--out", type=Path, default=Path("artifacts/jyb/plan.json"))
    plan.add_argument("--fillers", action="store_true")
    synth = commands.add_parser("synth", help="同一计划渲染多条件；真实后端会调用付费 API")
    synth.add_argument("--script", type=Path, required=True)
    synth.add_argument("--out", type=Path, required=True)
    synth.add_argument("--backend", choices=["tone", "siliconflow"], default="tone")
    synth.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                       default=["baseline", "timing", "emotion", "performance"])
    synth.add_argument("--cache", type=Path, default=Path("artifacts/jyb/cache"))
    synth.add_argument("--no-cache", action="store_true", help="冷启动性能测试；每段重新请求")
    synth.add_argument("--fillers", action="store_true", help="在所有条件中统一插入少量语气词")
    synth.add_argument("--voice", help="完整模型音色ID；不填则读取TTS_VOICE")
    generate = commands.add_parser("generate", help="复用原五Agent，只生成稿件及轨迹")
    generate.add_argument("--topic", required=True)
    generate.add_argument("--style", choices=["observational", "self_deprecating", "roast"], default="roast")
    generate.add_argument("--duration", type=int, choices=[3, 4, 5], default=3)
    generate.add_argument("--audience", default="大学生")
    generate.add_argument("--backend", choices=["mock", "autogen"], default="mock")
    generate.add_argument("--content-mode", choices=["baseline", "cfunset"], default="baseline")
    generate.add_argument("--cfunset-index", type=Path, help="复用团队已准备的 CFunSet 索引")
    generate.add_argument("--context", default="", help="保持团队生成模块的用户事实约束")
    generate.add_argument("--source-mode", choices=["original", "adaptation"], default=None)
    generate.add_argument("--adaptation-seed", type=int, default=None)
    generate.add_argument("--out", type=Path, required=True)
    return parser


def _generate(args: argparse.Namespace) -> None:
    from openmic.autogen_v02 import AutoGenConfig
    from openmic.tts_workflow import JybWorkflow
    from openmic.models import ProjectRequest
    from openmic.tts import extract_spoken_script
    from openmic.workflow import OpenMicWorkflow
    content_mode = getattr(args, "content_mode", "baseline")
    if args.backend == "mock" and content_mode == "cfunset":
        raise ValueError("CFunSet 参考筛选需要真实 AutoGen；Mock 不冒充检索增强生成")
    request = ProjectRequest(args.topic, args.style, args.duration, args.audience, getattr(args, "context", ""))
    config = AutoGenConfig.from_env() if args.backend == "autogen" else None
    workflow = JybWorkflow(config, content_mode=content_mode,
                           cfunset_index=getattr(args, "cfunset_index", None),
                           source_mode=getattr(args, "source_mode", None),
                           adaptation_seed=getattr(args, "adaptation_seed", None)) if config else OpenMicWorkflow()
    started = time.perf_counter()
    result = workflow.run(request, on_message=lambda item: print(item.agent + " 已完成", flush=True))
    messages = result.messages
    wanted = "PerformanceCoach" if args.backend == "autogen" else "JokeWriter"
    script = next((item.content for item in reversed(messages) if item.agent == wanted), "")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("<SCRIPT>\n" + extract_spoken_script(script) + "\n</SCRIPT>\n", encoding="utf-8")
    trace = {**asdict(result), "request": asdict(request), "approved": result.approved, "score": result.score,
             "generation_seconds": time.perf_counter() - started, "backend": args.backend,
             "revision_count": result.revision_count, "content_mode": content_mode,
             "model": config.model if config else None,
             "messages": [asdict(item) for item in messages]}
    args.out.with_suffix(".trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    # New CFunSet drafts follow the team's natural-length/optional-callback
    # policy. Keep all fidelity/QC checks, without expanding the spoken story.
    from openmic.tts_review import review_script
    limits = {"min_characters": 1, "min_punchlines": 0, "min_callbacks": 0} if content_mode == "cfunset" else {}
    review = review_script(args.out.read_text(encoding="utf-8"), trace, **limits)
    review["content_mode"] = content_mode
    args.out.with_suffix(".review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"独立协议/正文/质检核查：passed={review['passed']}")
    print(f"稿件已保存：{args.out}；需预览并检查标记与实际时长。approved={result.approved}")
    if not result.approved:
        print("质检未通过；请修订，勿作为合格样例交付。")
    try:
        compile_script(args.out.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"稿件已保留，但协议需要手动修正：{exc}")


def main(argv=None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["integrity"]:
        from openmic.tts_integrity_cli import main as integrity_main
        return integrity_main(arguments[1:])
    if arguments[:1] == ["stage"]:
        from openmic.tts_stage_cli import main as stage_main
        return stage_main(arguments[1:])
    args = build_parser().parse_args(arguments)
    try:
        if args.command == "generate":
            _generate(args)
            return
        script = args.script.read_text(encoding="utf-8-sig")
        events = compile_script(script, add_fillers=args.fillers)
        plan_json = json.dumps([asdict(e) for e in events], ensure_ascii=False, indent=2)
        if args.command == "plan":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(plan_json, encoding="utf-8")
            print(f"协议通过：{len(events)} 个事件，计划保存至 {args.out}")
            return
        # Validate the whole script before constructing any paid backend.
        backend = ToneBackend() if args.backend == "tone" else SiliconFlowBackend(voice=args.voice)
        if backend.synthetic:
            print("tone 是合成测试音，不含中文语音，不能作音质、MOS或实时率成绩。")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "script.txt").write_text(script, encoding="utf-8")
        (args.out / "plan.json").write_text(plan_json, encoding="utf-8")
        for condition in args.conditions:
            target = args.out / condition
            # Remove a prior successful result manifest before a new attempt;
            # a failed rerun must not leave old audio looking like new results.
            target.mkdir(parents=True, exist_ok=True)
            (target / "metrics.json").unlink(missing_ok=True)
            chunk_index = 0
            run_id = uuid.uuid4().hex[:12]

            def save_chunk(audio: bytes, row: dict) -> None:
                nonlocal chunk_index
                chunk_index += 1
                chunks = target / "chunks" / run_id
                chunks.mkdir(parents=True, exist_ok=True)
                (chunks / f"{chunk_index:04d}.wav").write_bytes(audio)
                print(f"{condition}: 片段 {chunk_index} {row['kind']} {row['duration_seconds']:.2f}s", flush=True)

            result = render(events, backend, condition=condition,
                            cache_dir=None if args.no_cache else args.cache, on_chunk=save_chunk)
            result.metrics["chunk_directory"] = "chunks/" + run_id
            save_result(result, target)
            print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
            if not result.metrics["in_target_duration"]:
                print("该音频未达到3–5分钟；短示例只用于功能演示，正式稿件应按实测时长改写。")
    except (ValueError, OSError, RuntimeError) as exc:
        print("错误：" + str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
