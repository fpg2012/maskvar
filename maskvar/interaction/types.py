"""Shared interaction types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Click = tuple[int | None, int | None, str | None]

CLK_POSITIVE = "positive"
CLK_NEGATIVE = "negative"


@dataclass
class ClickMaps:
    """Store one sampling debug payload."""

    false_negative: Any
    false_positive: Any
    negative_distance: Any
    positive_distance: Any
    candidate_mask: Any
    negative_max: float
    positive_max: float
    threshold: float
    mode: str | None
