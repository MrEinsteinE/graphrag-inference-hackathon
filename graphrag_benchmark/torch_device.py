"""Pick torch device for local models (sentence-transformers, BERTScore)."""

from __future__ import annotations


def resolve_torch_device(preference: str = "auto") -> str:
    """
    preference: auto | cuda | mps | cpu
    Returns a string suitable for SentenceTransformer(..., device=...).
    """
    import torch

    p = (preference or "auto").strip().lower()
    if p == "cpu":
        return "cpu"
    if p == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if p == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bert_score_device(preference: str = "auto") -> str:
    """BERTScore is most reliable on cuda/cpu; fall back to cpu for mps."""
    d = resolve_torch_device(preference)
    return "cuda" if d == "cuda" else "cpu"
