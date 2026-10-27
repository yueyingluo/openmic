"""Run one bounded stage inside an allocated single-GPU Slurm job."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Requires an existing Slurm allocation")
    probe = "import torch,json;assert torch.cuda.device_count()==1;print(json.dumps({'uuid':str(torch.cuda.get_device_properties(0).uuid),'name':torch.cuda.get_device_name(0)}))"
    device = json.loads(
        subprocess.check_output([sys.executable, "-c", probe], text=True)
    )
    uuid = device["uuid"]
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=uuid,
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    command = [uuid if value == "{GPU_UUID}" else value for value in command]
    if not command:
        raise ValueError("No stage command")
    receipt = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "node": os.environ.get("SLURMD_NODENAME"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "original_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuid": uuid,
        "gpu_name": device["name"],
        "seconds_cap": args.seconds,
        "command": command,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as target:
        target.write(json.dumps(receipt, indent=2) + "\n")
    # The probe has exited before the stage's own idle-GPU assertion.
    result = subprocess.run(
        ["timeout", "--signal=TERM", "--kill-after=15s", str(args.seconds), *command],
        env=env,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
