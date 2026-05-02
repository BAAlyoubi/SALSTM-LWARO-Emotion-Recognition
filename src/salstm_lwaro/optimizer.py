from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


Objective = Callable[[dict[str, float]], float]


@dataclass
class OptimizationResult:
    best_params: dict[str, float]
    best_score: float
    history: list[float]


class LWAROOptimizer:
    """Levy-flight weighted Artificial Rabbits Optimization.

    The optimizer minimizes an objective over continuous bounds. Integer-valued
    parameters can be rounded by the caller when evaluating the objective.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float] | list[float]],
        population_size: int = 10,
        iterations: int = 20,
        levy_beta: float = 1.5,
        seed: int | None = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        self.bounds = {key: (float(value[0]), float(value[1])) for key, value in bounds.items()}
        self.population_size = population_size
        self.iterations = iterations
        self.levy_beta = levy_beta
        self.rng = np.random.default_rng(seed)
        self.keys = list(self.bounds)
        self.lower = np.array([self.bounds[key][0] for key in self.keys], dtype=np.float64)
        self.upper = np.array([self.bounds[key][1] for key in self.keys], dtype=np.float64)

    def optimize(self, objective: Objective) -> OptimizationResult:
        population = self.rng.uniform(self.lower, self.upper, size=(self.population_size, len(self.keys)))
        scores = np.array([objective(self._decode(row)) for row in population], dtype=np.float64)
        best_idx = int(np.argmin(scores))
        best = population[best_idx].copy()
        best_score = float(scores[best_idx])
        history = [best_score]

        for iteration in range(1, self.iterations + 1):
            energy = 2.0 * (1.0 - iteration / max(self.iterations, 1))
            weights = self._fitness_weights(scores)
            center = np.average(population, axis=0, weights=weights)

            for idx in range(self.population_size):
                candidate = self._move(population, idx, best, center, energy)
                candidate = np.clip(candidate, self.lower, self.upper)
                candidate_score = float(objective(self._decode(candidate)))
                if candidate_score <= scores[idx]:
                    population[idx] = candidate
                    scores[idx] = candidate_score
                    if candidate_score < best_score:
                        best = candidate.copy()
                        best_score = candidate_score

            history.append(best_score)

        return OptimizationResult(self._decode(best), best_score, history)

    def _move(self, population: np.ndarray, idx: int, best: np.ndarray, center: np.ndarray, energy: float) -> np.ndarray:
        current = population[idx]
        peer_idx = int(self.rng.integers(0, self.population_size - 1))
        if peer_idx >= idx:
            peer_idx += 1
        peer = population[peer_idx]
        levy = self._levy_flight(len(self.keys))

        if self.rng.random() < 0.5:
            step = self.rng.random(len(self.keys)) * (peer - current) + levy * (best - current)
        else:
            step = self.rng.normal(size=len(self.keys)) * (center - current) + levy * (best - peer)
        return current + energy * step

    def _fitness_weights(self, scores: np.ndarray) -> np.ndarray:
        shifted = scores - scores.min()
        weights = 1.0 / (shifted + 1e-12)
        return weights / weights.sum()

    def _levy_flight(self, size: int) -> np.ndarray:
        beta = self.levy_beta
        sigma_u = (
            math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = self.rng.normal(0, sigma_u, size=size)
        v = self.rng.normal(0, 1, size=size)
        return 0.01 * u / (np.abs(v) ** (1 / beta) + 1e-12)

    def _decode(self, vector: np.ndarray) -> dict[str, float]:
        return {key: float(value) for key, value in zip(self.keys, vector)}
