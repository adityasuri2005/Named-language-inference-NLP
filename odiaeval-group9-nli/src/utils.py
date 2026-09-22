"""Shared helpers: reproducible seeding, config loading, environment capture."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/indicbertv2_odia.yaml") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_all_seeds(seed: int) -> None:
    """Seed every RNG we touch, and ask cuDNN for deterministic kernels.

    Bit-exact reproducibility across different GPUs is not achievable, but this
    makes repeated runs on the same hardware match.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash of a data file, so the grader can confirm we used the same bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def capture_environment() -> Dict[str, Any]:
    """Everything needed to explain a score that does not reproduce exactly."""
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        env["torch"] = None
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        env["transformers"] = None
    try:
        import datasets

        env["datasets"] = datasets.__version__
    except ImportError:
        env["datasets"] = None
    return env


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    print(f"[write] {path}")
    return path


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
