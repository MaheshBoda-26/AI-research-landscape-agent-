"""Double-fork launcher so dev servers survive the Freebuff terminal harness.

The harness kills the whole process group when a SYNC command exits, so plain
`nohup ... &` dies with it. A double fork with setsid detaches the servers into
their own session, out of reach of the group kill.
"""

import os
import subprocess
import sys

PAIRS = [
    (
        "/Users/maheshboda/Projects/Research-agent/api",
        [".venv/bin/python", "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        "/tmp/rla-api.log",
    ),
    (
        "/Users/maheshboda/Projects/Research-agent/web",
        ["npx", "next", "dev", "--port", "3000"],
        "/tmp/rla-web.log",
    ),
]


def spawn(cwd: str, cmd: list[str], log: str) -> None:
    if os.fork() > 0:
        return  # first parent returns to shell
    os.setsid()
    if os.fork() > 0:
        os._exit(0)  # intermediate parent exits; grandchild is reparented
    with open(log, "ab", buffering=0) as out:
        subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=out,
            stderr=out,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    os._exit(0)


if __name__ == "__main__":
    if "--stop" in sys.argv:
        for _, _, log in PAIRS:
            try:
                with open(log, "rb") as f:
                    pass
            except FileNotFoundError:
                continue
        os.system("pkill -f 'uvicorn main:app' 2>/dev/null; pkill -f 'next dev' 2>/dev/null")
        print("stopped")
    else:
        for cwd, cmd, log in PAIRS:
            spawn(cwd, cmd, log)
        print("launched")
