# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import math
import random
from typing import Literal

from gepa.core.state import GEPAState
from gepa.gepa_utils import idxmax, select_program_candidate_from_pareto_front
from gepa.proposer.reflective_mutation.base import CandidateSelector


class ParetoCandidateSelector(CandidateSelector):
    def __init__(self, rng: random.Random | None):
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        return select_program_candidate_from_pareto_front(
            state.get_pareto_front_mapping(),
            state.per_program_tracked_scores,
            self.rng,
        )


class CurrentBestCandidateSelector(CandidateSelector):
    def __init__(self):
        pass

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        return idxmax(state.program_full_scores_val_set)


class EpsilonGreedyCandidateSelector(CandidateSelector):
    def __init__(self, epsilon: float, rng: random.Random | None):
        assert 0.0 <= epsilon <= 1.0
        self.epsilon = epsilon
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, len(state.program_candidates) - 1)
        else:
            return idxmax(state.program_full_scores_val_set)


class BanditCandidateSelector(CandidateSelector):
    """
    Multi-armed bandit candidate selector using UCB algorithm with distribution sharpening.

    Features:
    - Exploration-exploitation balance (UCB)
    - Distribution sharpening (exponentiation)

    Adapted for GEPA:
    - Uses GEPAState.program_full_scores_val_set for scores
    - Compatible with existing Pareto selection logic
    """

    def __init__(
        self,
        c: float = 2.0,                    # UCB exploration parameter
        alpha: float = 1.5,                # Distribution sharpness parameter
        rng: random.Random | None = None
    ):
        self.c = c
        self.alpha = alpha

        # Selection count tracker
        self.selection_counts = {}

        self.rng = rng or random.Random(0)

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)

        current_iteration = state.i + 1
        scores = state.program_full_scores_val_set

        ucb_scores = []
        for idx in range(len(scores)):
            current_score = scores[idx]

            # Initialize selection count
            if idx not in self.selection_counts:
                self.selection_counts[idx] = 0

            # Compute UCB score
            count = self.selection_counts[idx]

            if count < 1:
                ucb_score = float('inf')  # Underexplored program
            else:
                # UCB1 formula: score + confidence interval
                confidence_bonus = self.c * math.sqrt(math.log(current_iteration + 1) / count)
                ucb_score = current_score + confidence_bonus

            ucb_scores.append(ucb_score)

        # Update selection count
        selected_idx = self._select_from_ucb_scores(ucb_scores, len(scores))
        self.selection_counts[selected_idx] += 1

        return selected_idx

    def _select_from_ucb_scores(self, ucb_scores: list[float], num_candidates: int) -> int:
        """Select a candidate from UCB scores."""
        # Handle infinite values to avoid downstream computation issues
        max_finite_score = max((s for s in ucb_scores if s != float('inf')), default=0)
        ucb_scores = [min(s, max_finite_score * 1000) if s == float('inf') else s for s in ucb_scores]

        # Exponentiate to sharpen the distribution (better programs are more likely to be selected)
        if self.alpha != 1.0:
            # Shift to avoid negative values
            min_score = min(ucb_scores) if ucb_scores else 0
            shifted_scores = [s - min_score + 1e-6 for s in ucb_scores]  # Ensure positivity
            ucb_scores = [s ** self.alpha for s in shifted_scores]

        # Normalize and sample
        total = sum(ucb_scores)
        if total == 0 or not all(math.isfinite(s) for s in ucb_scores):
            # If all scores are zero or contain non-finite values, select uniformly at random
            selected_idx = self.rng.randint(0, num_candidates - 1)
        else:
            probs = [s / total for s in ucb_scores]
            selected_idx = self.rng.choices(range(num_candidates), weights=probs)[0]

        return selected_idx

    def get_selection_stats(self) -> dict:
        """Return selection statistics for debugging and monitoring."""
        return {
            "selection_counts": self.selection_counts.copy(),
        }
