"""
Build the backend executable for desktop packaging (PyInstaller).

Run from repo root:
  python scripts/build_backend.py

Outputs:
  backend/dist/ink-ember-backend.exe (Windows)
  backend/dist/ink-ember-backend      (macOS/Linux)
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def add_data_arg(src: Path, dest: str) -> str:
    """
    PyInstaller --add-data uses a platform-specific separator:
      Windows:  SRC;DEST
      *nix:     SRC:DEST
    """
    sep = ";" if platform.system() == "Windows" else ":"
    return f"{src}{sep}{dest}"


def main() -> None:
    name = "ink-ember-backend"
    exe_name = f"{name}.exe" if platform.system() == "Windows" else name

    # Optional: allow auto-install for convenience, but keep it opt-in.
    if os.getenv("INK_EMBER_AUTO_INSTALL_PYINSTALLER", "").lower() in {"1", "true", "yes"}:
        run(["python", "-m", "pip", "install", "--upgrade", "pip"], cwd=ROOT)
        run(["python", "-m", "pip", "install", "--upgrade", "pyinstaller"], cwd=ROOT)

    # Clean previous builds
    for p in [BACKEND / "build", BACKEND / "dist", BACKEND / f"{name}.spec"]:
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    # Collect data folders/files if they exist
    add_data: list[str] = []
    candidates: list[tuple[Path, str]] = [
        (BACKEND / ".env", "."),
        (BACKEND / "data", "data"),
        (BACKEND / "static", "static"),
        (BACKEND / "templates", "templates"),
    ]
    for src, dest in candidates:
        if src.exists():
            add_data.extend(["--add-data", add_data_arg(src, dest)])

    # Hidden imports that commonly fix FastAPI/Uvicorn/PyInstaller misses
    hidden_imports = [
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets.auto",
    ]

    cmd = [
        "python",
        "-m",
        "PyInstaller",
        "--name",
        name,
        "--onefile",
        "--noconfirm",
        "--clean",
        # If you want a console window, remove this:
        "--console",
        *add_data,
    ]

    for hi in hidden_imports:
        cmd.extend(["--hidden-import", hi])

    # Entry point (relative to backend/)
    cmd.append("run_backend.py")

    run(cmd, cwd=BACKEND)

    produced = BACKEND / "dist" / exe_name
    if not produced.exists():
        raise SystemExit(f"Build failed; missing {produced}")

    print(f"OK: {produced}")


if __name__ == "__main__":
    main()
