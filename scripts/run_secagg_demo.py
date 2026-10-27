"""Launch Flower's Ray-based local simulation from Windows through WSL."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    app_path = root / "federated" / "secagg_demo"
    wsl_root = subprocess.check_output(["wsl.exe", "-e", "wslpath", "-a", str(root)], text=True).strip()
    wsl_app = f"{wsl_root}/federated/secagg_demo"
    command = (
        "tmp=$(mktemp -d /tmp/safesight-secagg.XXXXXX) && "
        "trap 'case \"$tmp\" in /tmp/safesight-secagg.*) rm -rf -- \"$tmp\" ;; esac' EXIT && "
        f"tar -C {shlex.quote(wsl_app)} --exclude=.venv -cf - . | tar -C \"$tmp\" -xf - && "
        "cd \"$tmp\" && "
        f"export SAFESIGHT_PROJECT_ROOT={shlex.quote(wsl_root)} && "
        "uv run --project . flwr run . --stream"
    )
    return subprocess.call(["wsl.exe", "-e", "bash", "-lc", command], cwd=app_path)


if __name__ == "__main__":
    raise SystemExit(main())
