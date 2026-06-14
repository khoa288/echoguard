"""Fixed, non-contextual, and linear contextual replay policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .arms import ARM_REGISTRY


class Policy:
    def reset(self) -> None:
        raise NotImplementedError

    def select_action(self, context: np.ndarray) -> str:
        raise NotImplementedError

    def observe(self, context: np.ndarray, action: str, reward: float) -> None:
        pass


@dataclass
class FixedPolicy(Policy):
    action: str

    def __post_init__(self) -> None:
        if self.action not in ARM_REGISTRY:
            raise ValueError(f"Unknown action: {self.action}")

    def reset(self) -> None:
        pass

    def select_action(self, context: np.ndarray) -> str:
        return self.action


class EpsilonGreedyPolicy(Policy):
    def __init__(self, actions: Sequence[str] = tuple(ARM_REGISTRY), epsilon: float = 0.1, seed: int = 0):
        self.actions = tuple(actions)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.counts = {action: 0 for action in self.actions}
        self.values = {action: 0.0 for action in self.actions}

    def select_action(self, context: np.ndarray) -> str:
        unseen = [action for action in self.actions if self.counts[action] == 0]
        if unseen:
            return unseen[0]
        if self.rng.random() < self.epsilon:
            return str(self.rng.choice(self.actions))
        return max(self.actions, key=lambda action: (self.values[action], -self.actions.index(action)))

    def observe(self, context: np.ndarray, action: str, reward: float) -> None:
        self.counts[action] += 1
        count = self.counts[action]
        self.values[action] += (reward - self.values[action]) / count


class LinUCBPolicy(Policy):
    def __init__(
        self, n_features: int, actions: Sequence[str] = tuple(ARM_REGISTRY),
        alpha: float = 1.0, ridge: float = 1.0,
    ):
        self.actions = tuple(actions)
        self.n_features = int(n_features)
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.reset()

    def reset(self) -> None:
        self.a = {action: np.eye(self.n_features) * self.ridge for action in self.actions}
        self.b = {action: np.zeros(self.n_features) for action in self.actions}

    def select_action(self, context: np.ndarray) -> str:
        x = np.asarray(context, dtype=float).reshape(self.n_features)
        scores = {}
        for action in self.actions:
            inverse = np.linalg.inv(self.a[action])
            theta = inverse @ self.b[action]
            scores[action] = float(theta @ x + self.alpha * np.sqrt(x @ inverse @ x))
        return max(self.actions, key=lambda action: (scores[action], -self.actions.index(action)))

    def observe(self, context: np.ndarray, action: str, reward: float) -> None:
        x = np.asarray(context, dtype=float).reshape(self.n_features)
        self.a[action] += np.outer(x, x)
        self.b[action] += reward * x
