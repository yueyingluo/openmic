import argparse

from openmic.autogen_v02 import AutoGenConfig, AutoGenV02Workflow
from openmic.models import ProjectRequest
from openmic.workflow import OpenMicWorkflow


STYLE_ALIASES = {
    "观察": "observational",
    "自嘲": "self_deprecating",
    "吐槽": "roast",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Chinese stand-up comedy with the OpenMic workflow"
    )
    parser.add_argument("--topic", required=True, help="生活主题")
    parser.add_argument(
        "--style",
        default="observational",
        choices=["observational", "self_deprecating", "roast", "观察", "自嘲", "吐槽"],
    )
    parser.add_argument("--duration", type=int, default=3, choices=[3, 4, 5])
    parser.add_argument("--audience", default="大学生")
    parser.add_argument(
        "--backend",
        choices=["mock", "autogen"],
        default="mock",
        help="mock 无需 Key；autogen 使用课程要求的 AutoGen 0.2 GroupChat",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    request = ProjectRequest(
        topic=args.topic,
        style=STYLE_ALIASES.get(args.style, args.style),
        duration_minutes=args.duration,
        audience=args.audience,
    )
    if args.backend == "autogen":
        result = AutoGenV02Workflow(AutoGenConfig.from_env()).run(request)
    else:
        result = OpenMicWorkflow().run(request)

    for message in result.messages:
        print(f"\n[{message.agent}]\n{message.content}")
    print(
        f"\nResult: approved={result.approved}, score={result.score:.1f}, "
        f"revisions={result.revision_count}"
    )


if __name__ == "__main__":
    main()
