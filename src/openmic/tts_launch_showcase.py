"""Start the read-only showcase locally, without a visible helper console."""

import argparse
import ctypes
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "openmic" / "tts_showcase_app.py"


def _identity(pid):
    """Process creation identity prevents trusting a stale/reused process ID."""
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        created, exited, kernel_time, user_time = [wintypes.FILETIME() for _ in range(4)]
        try:
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return None
            return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    try:
        # After the parenthesized command, /proc fields restart at state (3).
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, IndexError):
        return None


def _healthy(url):
    try:
        with urllib.request.urlopen(url + "/_stcore/health", timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _occupied(port):
    with socket.socket() as probe:
        probe.settimeout(0.3)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    folder = ROOT / "artifacts" / "jyb" / "execution"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "showcase_server.json"
    log_path = folder / "showcase_server.log"
    state = {}
    try:
        state = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    ours = (state.get("app") == str(APP) and state.get("url") == url
            and state.get("identity") is not None
            and _identity(state.get("pid", -1)) == state.get("identity"))
    if _occupied(args.port):
        if not ours or not _healthy(url):
            print(f"Port {args.port} is occupied by another service; no process was stopped.")
            return 1
    else:
        command = [sys.executable, "-m", "streamlit", "run", str(APP),
                   "--server.address", "127.0.0.1", "--server.port", str(args.port),
                   "--server.headless", "true", "--browser.gatherUsageStats", "false"]
        with log_path.open("ab") as output:
            process = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                       start_new_session=os.name != "nt")
        state = {"app": str(APP), "url": url, "pid": process.pid,
                 "identity": _identity(process.pid), "log": str(log_path)}
        manifest.write_text(json.dumps(state, indent=2), encoding="utf-8")
        for _ in range(80):
            if _healthy(url):
                break
            if process.poll() is not None:
                print("Showcase exited during startup. See " + str(log_path))
                return 1
            time.sleep(0.25)
        else:
            print("Showcase is still starting. See " + str(log_path))
            return 1
    print("Showcase ready: " + url)
    if not args.no_browser:
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
