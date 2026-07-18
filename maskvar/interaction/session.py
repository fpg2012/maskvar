"""Session helpers for interactive segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Click = tuple[int, int, str]


@dataclass
class Session:
    """Store one sample interaction context."""

    sample: dict[str, Any]
    state: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)


def create_session(
    image: Any,
    meta: dict[str, Any] | None = None,
    prev_output: Any | None = None,
    clicks: list[Click] | None = None,
) -> Session:
    """Create a session with minimal external fields."""

    return Session(
        sample={
            "image": image,
            "meta": {} if meta is None else meta,
        },
        state={
            "clicks": [] if clicks is None else list(clicks),
            "prev_output": prev_output,
            "cur_logits": None,
            "step_idx": 0,
        },
        model={},
    )
