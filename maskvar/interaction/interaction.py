"""Unified interaction loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .sampler import ClickSampler
from .session import Click, Session


class ModelProtocol:
    """Model protocol for interaction."""

    def prepare(self, sessions: list[Session]) -> list[Session]:
        """Prepare model-side context for sessions."""

    def forward(self, sessions: list[Session]) -> list[Session]:
        """Write current logits back to sessions."""


@dataclass
class Interaction:
    """Run the shared interaction process."""

    model: ModelProtocol
    sampler: ClickSampler

    def prepare(self, sessions: list[Session]) -> list[Session]:
        """Initialize model-side context for sessions."""

        return self.model.prepare(sessions)

    def step(
        self,
        sessions: list[Session],
        gts: list[Any],
        **kwargs: Any,
    ) -> list[Session]:
        """Run one interaction step for each session."""

        if len(sessions) != len(gts):
            raise ValueError("The number of sessions and gts must match.")

        prev_output_type = kwargs.get("prev_output_type", "logits")
        mask_threshold = kwargs.get("mask_threshold", 0.0)
        sampler_kwargs = kwargs.get("sampler_kwargs", {})

        sessions = self.model.forward(sessions)
        for session, gt in zip(sessions, gts):
            state = session.state
            cur_logits = state["cur_logits"]
            next_click = self.sampler.sample_click(
                gt=gt,
                prev_output=cur_logits,
                clicks=state["clicks"],
                **sampler_kwargs,
            )
            if next_click[2] is not None:
                state["clicks"].append(next_click)
            state["prev_output"] = build_prev_output(
                cur_logits=cur_logits,
                prev_output_type=prev_output_type,
                mask_threshold=mask_threshold,
            )
            state["step_idx"] += 1
        return sessions

    def run(
        self,
        sessions: list[Session],
        gts: list[Any],
        num_steps: int,
        **kwargs: Any,
    ) -> list[Session]:
        """Run a fixed number of interaction steps."""

        if num_steps < 0:
            raise ValueError("num_steps must be non-negative.")

        for _ in range(num_steps):
            sessions = self.step(sessions, gts, **kwargs)
        return sessions


def build_prev_output(
    cur_logits: Any,
    prev_output_type: str,
    mask_threshold: float = 0.0,
) -> Any:
    """Build the next prev_output from current logits."""

    if prev_output_type == "logits":
        return cur_logits
    if prev_output_type == "mask":
        return (cur_logits > mask_threshold).to(cur_logits)
    raise ValueError(f"Unsupported prev_output_type: {prev_output_type}")
