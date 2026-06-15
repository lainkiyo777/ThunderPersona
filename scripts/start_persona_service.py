from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    python = root / ".venv-wechat-cli" / "Scripts" / "python.exe"
    server = root / "persona_app" / "server.py"
    out_path = root / ".persona-app.out.log"
    err_path = root / ".persona-app.err.log"

    if not python.is_file():
        print(f"Python not found: {python}", file=sys.stderr)
        return 1
    if not server.is_file():
        print(f"Server not found: {server}", file=sys.stderr)
        return 1

    stdout = out_path.open("ab", buffering=0)
    stderr = err_path.open("ab", buffering=0)
    args = [str(python), str(server), *sys.argv[1:]]

    popen_kwargs = {
        "cwd": str(root),
        "stdin": subprocess.DEVNULL,
        "stdout": stdout,
        "stderr": stderr,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(args, **popen_kwargs)
    (root / ".persona-app.pid").write_text(str(process.pid), encoding="utf-8")
    print(f"Persona app started. pid={process.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
