from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def save_run_snapshot(config_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, out / "config.yaml")

    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        (out / "git_commit.txt").write_text(git_hash)
    except Exception:
        (out / "git_commit.txt").write_text("unavailable")

    try:
        result = subprocess.run(
            ["uv", "pip", "freeze"],
            capture_output=True,
            text=True,
        )
        (out / "requirements.txt").write_text(result.stdout)
    except Exception:
        pass
