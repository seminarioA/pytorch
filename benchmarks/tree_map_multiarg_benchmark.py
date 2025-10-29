"""Microbenchmark for torch.utils._pytree.tree_map with many arguments."""
from __future__ import annotations

import importlib.util
import pathlib
import statistics
import os
import sys
import time
import types
from typing import Any, Callable

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

_torch_stub = types.ModuleType("torch")
_torch_stub.__path__ = [str(_REPO_ROOT / "torch")]
sys.modules.setdefault("torch", _torch_stub)

_torch_utils_stub = types.ModuleType("torch.utils")
_torch_utils_stub.__path__ = [str(_REPO_ROOT / "torch" / "utils")]
sys.modules.setdefault("torch.utils", _torch_utils_stub)

_torch_version_stub = types.ModuleType("torch.version")
_torch_version_stub.__version__ = "0.0.0"
sys.modules.setdefault("torch.version", _torch_version_stub)

_PYTREE_PATH = pathlib.Path(
    os.environ.get("PYTREE_PATH_OVERRIDE", _REPO_ROOT / "torch" / "utils" / "_pytree.py")
)
_spec = importlib.util.spec_from_file_location("torch.utils._pytree", _PYTREE_PATH)
assert _spec and _spec.loader
_pytree = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pytree)

tree_map = _pytree.tree_map


def _build_tree(size: int) -> list[dict[str, Any]]:
    return [
        {
            "coords": (float(i), float(i + 1), float(i + 2)),
            "values": [i, i + 1, i + 2, i + 3],
            "meta": {"index": i, "valid": (i % 2) == 0},
        }
        for i in range(size)
    ]


def _make_args(size: int, num_rests: int) -> tuple[Any, ...]:
    base = _build_tree(size)
    rest = []
    for shift in range(1, num_rests + 1):
        rest.append(
            [
                {
                    "coords": (float(i + shift), float(i + shift + 1), float(i + shift + 2)),
                    "values": [i + shift, i + shift + 1, i + shift + 2, i + shift + 3],
                    "meta": {"index": i + shift, "valid": (i + shift) % 2 == 0},
                }
                for i in range(size)
            ]
        )
    return (base, *rest)


def _run_once(fn: Callable[[], Any]) -> float:
    start = time.perf_counter()
    fn()
    end = time.perf_counter()
    return end - start


def _time_tree_map(size: int, repetitions: int, num_rests: int) -> list[float]:
    args = _make_args(size, num_rests)
    fn = lambda: tree_map(lambda *xs: xs[0], *args)  # noqa: E731
    return [_run_once(fn) for _ in range(repetitions)]


def main() -> None:
    timings = _time_tree_map(size=512, repetitions=25, num_rests=16)
    print("min", min(timings))
    print("median", statistics.median(timings))
    print("max", max(timings))


if __name__ == "__main__":
    main()
