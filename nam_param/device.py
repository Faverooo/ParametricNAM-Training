"""Device selection: cuda / rocm (via torch.cuda) → directml → cpu."""

from __future__ import annotations

from typing import Optional

import torch


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """
    :param requested: 'auto' | 'cuda' | 'cpu' | 'dml' | None (=auto)
    """
    req = (requested or "auto").lower()
    if req in ("auto",):
        if torch.cuda.is_available():
            return torch.device("cuda")
        dml = _try_directml()
        if dml is not None:
            return dml
        return torch.device("cpu")
    if req in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if req in ("dml", "directml"):
        dml = _try_directml()
        if dml is None:
            raise RuntimeError(
                "DirectML requested but torch_directml is not importable. "
                "On Windows+AMD install a torch-directml version compatible with your "
                "Python/PyTorch in a dedicated environment. See README_IT.md."
            )
        return dml
    if req == "cpu":
        return torch.device("cpu")
    # allow bare device strings
    return torch.device(req)


def _try_directml():
    try:
        import torch_directml  # type: ignore
        return torch_directml.device()
    except Exception:
        return None


def device_label(device: torch.device) -> str:
    s = str(device)
    if "privateuseone" in s or "dml" in s.lower():
        return f"directml ({s})"
    return s
