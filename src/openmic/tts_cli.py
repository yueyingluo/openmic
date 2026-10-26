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
    generate.add_argument("--out", type=Path, required=True)
    return parser


def _generate(args: argparse.Namespace) -> None:
    from openmic.autogen_v02 import AutoGenConfig
    from openmic.tts_workflow import JybWorkflow
    from openmic.models import ProjectRequest
    from openmic.tts import extract_spoken_script
    from openmic.workflow import OpenMicWorkflow
    request = ProjectRequest(args.topic, args.style, args.duration, args.audience)
    workflow = JybWorkflow(AutoGenConfig.from_env()) if args.backend == "autogen" else OpenMicWorkflow()
    started = time.perf_counter()
    result = workflow.run(request, on_message=lambda item: print(item.agent + " 已完成", flush=True))
    messages = result.messages
    wanted = "PerformanceCoach" if args.backend == "autogen" else "JokeWriter"
    script = next((item.content for item in reversed(messages) if item.agent == wanted), "")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("<SCRIPT>\n" + extract_spoken_script(script) + "\n</SCRIPT>\n", encoding="utf-8")
    trace = {"request": asdict(request), "approved": result.approved, "score": result.score,
             "generation_seconds": time.perf_counter() - started, "backend": args.backend,
             "revision_count": result.revision_count,
             "messages": [asdict(item) for item in messages]}
    args.out.with_suffix(".trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"稿件已保存：{args.out}；需预览并检查标记与实际时长。approved={result.approved}")
    if not result.approved:
        print("质检未通过；请修订，勿作为合格样例交付。")
    try:
        compile_script(args.out.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"稿件已保留，但协议需要手动修正：{exc}")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
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
