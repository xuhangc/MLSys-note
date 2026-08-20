"""One-command reproducibility entry point for the FlashAttention study note."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(CODE_DIR))

from flashattention_tutorial import verify_forward_equivalence, verify_gradients


def main() -> None:
    print("[1/2] Checking forward and backward equivalence ...")
    verify_forward_equivalence()
    verify_gradients()
    print("\n[2/2] Creating formula-driven figures ...")
    subprocess.run([sys.executable, str(SCRIPTS_DIR / "make_visualizations.py")], check=True)
    print(f"\nDone. Read {ROOT / 'FLASHATTENTION_STUDY_NOTE.md'}")


if __name__ == "__main__":
    main()
