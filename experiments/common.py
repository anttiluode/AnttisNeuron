from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable


def emit(result: dict, out: str | None = None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")


def main(run: Callable[[int], dict], default_seed: int = 17) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    emit(run(seed=args.seed), args.out)
