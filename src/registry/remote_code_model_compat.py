"""
Repair missing ``PreTrainedModel`` caches on some ``trust_remote_code`` checkpoints.

Recent ``transformers`` sets ``_named_pretrained_submodules`` at the end of
``PreTrainedModel.post_init`` (see ``modeling_utils.py``). Remote-code models
that override ``post_init`` without ``super().post_init()`` never run that tail,
so later code (PEFT, Trainer internals, etc.) can raise:

``AttributeError: ... has no attribute '_named_pretrained_submodules'``

We only apply the same list comprehension as upstream—never re-run
``init_weights()`` or the full ``post_init`` (that would risk clobbering
weights loaded from disk).
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_pretrained_submodule_cache"]


def ensure_pretrained_submodule_cache(model: Any) -> Any:
    """Populate ``_named_pretrained_submodules`` if absent (same logic as ``PreTrainedModel.post_init``)."""
    from transformers.modeling_utils import PreTrainedModel

    if not isinstance(model, PreTrainedModel):
        return model
    if hasattr(model, "_named_pretrained_submodules"):
        return model
    model._named_pretrained_submodules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, PreTrainedModel)
    ]
    return model
