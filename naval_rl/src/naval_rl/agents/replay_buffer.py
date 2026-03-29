"""
Replay buffers for off-policy RL training.

ReplayBuffer     — standard circular experience buffer
RareEventBuffer  — thin wrapper that oversamples rare transitions (e.g. missile engagements)
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
import torch


Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class ReplayBuffer:
    """Standard circular replay buffer."""

    def __init__(self, capacity: int = 200_000) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self._buf.append((
            np.asarray(state,      dtype=np.float32),
            np.asarray(action,     dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int) -> Batch:
        if len(self._buf) < batch_size:
            # Sample with replacement when buffer is small
            idxs = np.random.randint(0, len(self._buf), size=batch_size)
            batch = [self._buf[i] for i in idxs]
        else:
            idxs = np.random.choice(len(self._buf), size=batch_size, replace=False)
            batch = [self._buf[i] for i in idxs]

        s, a, r, ns, d = zip(*batch)
        return (
            torch.as_tensor(np.array(s),               dtype=torch.float32),
            torch.as_tensor(np.array(a),               dtype=torch.float32),
            torch.as_tensor(np.array(r, dtype=np.float32)).unsqueeze(1),
            torch.as_tensor(np.array(ns),              dtype=torch.float32),
            torch.as_tensor(np.array(d, dtype=np.float32)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self._buf)

    def is_ready(self, batch_size: int) -> bool:
        return len(self._buf) >= batch_size


class MixedReplayBuffer:
    """
    Combines a main replay buffer with a rare-event buffer.

    Rare transitions (e.g. missile engagements) are stored in both buffers;
    during sampling, a fraction *rare_ratio* of each batch is drawn from the
    rare buffer to correct for their low base-rate.

    Parameters
    ----------
    main_capacity  : capacity of the main buffer
    rare_capacity  : capacity of the rare-event buffer
    rare_ratio     : fraction of each batch drawn from the rare buffer
    """

    def __init__(
        self,
        main_capacity: int = 200_000,
        rare_capacity: int = 10_000,
        rare_ratio: float = 0.1,
    ) -> None:
        self.main = ReplayBuffer(main_capacity)
        self.rare = ReplayBuffer(rare_capacity)
        self.rare_ratio = rare_ratio

    def add(self, state, action, reward, next_state, done,
            rare: bool = False) -> None:
        self.main.add(state, action, reward, next_state, done)
        if rare:
            self.rare.add(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> Batch:
        k_rare = min(
            int(batch_size * self.rare_ratio),
            len(self.rare),
        )
        k_main = batch_size - k_rare

        main_batch = self.main.sample(k_main)

        if k_rare > 0:
            rare_batch = self.rare.sample(k_rare)
            combined = tuple(
                torch.cat([m, r], dim=0)
                for m, r in zip(main_batch, rare_batch)
            )
        else:
            combined = main_batch

        # Shuffle to prevent order bias
        perm = torch.randperm(batch_size)
        return tuple(t[perm] for t in combined)   # type: ignore[return-value]

    def is_ready(self, batch_size: int) -> bool:
        return self.main.is_ready(batch_size)

    def __len__(self) -> int:
        return len(self.main)
