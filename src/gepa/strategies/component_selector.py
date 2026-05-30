# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
import math
import os
import random
import re
from typing import Any

from gepa.core.adapter import Trajectory
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import ReflectionComponentSelector


class RoundRobinReflectionComponentSelector(ReflectionComponentSelector):
    """Legacy round-robin selector (kept for compatibility)"""
    def __call__(
        self,
        state: GEPAState,
        trajectories: list[Trajectory],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        pid = state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx]
        state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] = (pid + 1) % len(
            state.list_of_named_predictors
        )
        name = state.list_of_named_predictors[pid]
        return [name]


class AllReflectionComponentSelector(ReflectionComponentSelector):
    """Legacy all-component selector (kept for compatibility)"""
    def __call__(
        self,
        state: GEPAState,
        trajectories: list[Trajectory],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        return list(candidate.keys())


class CorrelationBasedReflectionComponentSelector(ReflectionComponentSelector):
    """
    Component selector based on API correlation and system_prompt periodic updates.

    Prioritizes API modules with low correlation to the most recently updated module,
    and periodically updates system_prompt.
    """

    # Mapping from API module number to API name
    API_NAME_MAP = {
        "API_information1": "NCBI",
        "API_information2": "UniProt",
        "API_information3": "Reactome",
        "API_information4": "KEGG",
        "API_information5": "Ensembl",
        "API_information6": "Cellosaurus",
        "API_information7": "CCLE",
        "API_information8": "DepMap",
        "API_information9": "PubChem",
        "API_information10": "DrugBank",
    }

    def __init__(self, system_prompt_update_interval: int = 10):
        """
        Initialize the correlation-based component selector.

        Args:
            system_prompt_update_interval: How often to force a system_prompt update (default: every 10 iterations)
        """
        self.api_similarities = self._load_api_similarities()
        self.similarity_matrix = self._build_similarity_matrix()
        self.system_prompt_update_interval = system_prompt_update_interval

    def _load_api_similarities(self) -> dict[str, float]:
        """Load API similarities from Graph_API/output/api_similarities.json"""
        # First, try loading from the project root's Graph_API/output
        similarities_path = os.path.join("Graph_API", "output", "api_similarities.json")

        if not os.path.exists(similarities_path):
            # If not found, try a relative path from the current file location
            current_dir = os.path.dirname(__file__)
            similarities_path = os.path.join(current_dir, "..", "..", "..", "Graph_API", "output", "api_similarities.json")

        if not os.path.exists(similarities_path):
            raise FileNotFoundError(
                f"API similarities file not found. Please ensure the file exists at: "
                f"Graph_API/output/api_similarities.json"
            )

        with open(similarities_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def _build_similarity_matrix(self) -> dict[str, dict[str, float]]:
        """Build a similarity matrix from the loaded similarities data"""
        matrix = {}

        for pair_key, similarity in self.api_similarities.items():
            if " <-> " not in pair_key:
                continue

            api1, api2 = pair_key.split(" <-> ")

            if api1 not in matrix:
                matrix[api1] = {}
            if api2 not in matrix:
                matrix[api2] = {}

            matrix[api1][api2] = similarity
            matrix[api2][api1] = similarity

        return matrix

    def _get_correlation_weight(self, api1: str, api2: str) -> float:
        """
        Get correlation weight between two APIs using a Logistic Decay function.

        Uses the sigmoid function to convert similarity into a weight:
        - Low similarity (< center) -> high weight (near 1.0), encouraging selection
        - High similarity (> center) -> low weight (near 0.0), avoiding selection

        Formula: weight = 1 / (1 + e^(k * (similarity - center)))

        Args:
            api1: API module name (e.g., "API_information1" or "NCBI")
            api2: API module name (e.g., "API_information2" or "UniProt")

        Returns:
            Weight value in [0, 1]; lower similarity gives higher weight.
        """
        # 1. Map API module number to actual API name
        api1_name = self.API_NAME_MAP.get(api1, api1)
        api2_name = self.API_NAME_MAP.get(api2, api2)

        # 2. Retrieve similarity
        if api1_name == api2_name:
            similarity = 1.0  # Self-similarity is highest
        else:
            # Look up from the similarity matrix; default to medium similarity 0.5 if not found
            similarity = self.similarity_matrix.get(api1_name, {}).get(api2_name, 0.5)

        # 3. Logistic Decay parameters
        # k: controls curve steepness; higher k gives sharper discrimination
        #    - k=10: gentle discrimination, smooth transition
        #    - k=15: aggressive discrimination, steep transition (current setting)
        #    - k=20: extreme discrimination, nearly a step function
        k = 10

        # center: the similarity "watershed"; values below this get high weight
        #    - 0.3: prefers uncorrelated APIs, heavily penalizes similarity > 0.3
        #    - 0.4: balanced (current setting)
        #    - 0.5: conservative, allows some similarity
        center = 0.6

        # 4. Compute weight
        try:
            # When similarity < center, exponent is negative, denominator shrinks, weight approaches 1
            # When similarity > center, exponent is positive, denominator grows, weight approaches 0
            weight = 1.0 / (1.0 + math.exp(k * (similarity - center)))
        except OverflowError:
            # Guard against math overflow in extreme cases
            weight = 0.0 if similarity > center else 1.0

        return weight

    def _select_next_api_module(self, candidate_predictors: list[str], last_predictor: str) -> str:
        """Select the next API module based on correlation weights"""
        if not candidate_predictors:
            raise ValueError("No predictors available")

        # Only consider API modules (not system_prompt)
        api_predictors = [p for p in candidate_predictors if p.startswith("API_information")]
        if not api_predictors:
            # If no API modules available, return the first available predictor
            return candidate_predictors[0]

        weights = []
        for predictor in api_predictors:
            weight = self._get_correlation_weight(last_predictor, predictor)
            weights.append((predictor, weight))

        if not weights:
            return random.choice(api_predictors)

        predictors, weight_values = zip(*weights)
        total_weight = sum(weight_values)

        if total_weight == 0:
            return random.choice(predictors)

        # Weighted random selection based on correlation
        r = random.uniform(0, total_weight)
        cumulative = 0
        for predictor, weight in zip(predictors, weight_values):
            cumulative += weight
            if r <= cumulative:
                return predictor

        return predictors[-1]

    def __call__(
        self,
        state: GEPAState,
        trajectories: list[Trajectory],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        available_predictors = list(candidate.keys())

        # Check if it's time to update system_prompt
        current_iteration = state.i + 1
        should_update_system_prompt = (current_iteration % self.system_prompt_update_interval == 1)

        if should_update_system_prompt and "system_prompt" in available_predictors:
            # Force update system_prompt
            system_prompt_idx = state.list_of_named_predictors.index("system_prompt")
            state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] = system_prompt_idx
            return ["system_prompt"]

        # Get the last updated predictor for this candidate
        last_predictor_idx = state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx]

        # Determine the last predictor name
        if last_predictor_idx < len(state.list_of_named_predictors):
            last_predictor_name = state.list_of_named_predictors[last_predictor_idx]
        else:
            last_predictor_name = None

        # If no previous predictor or last was system_prompt, start with a random API
        if last_predictor_name is None or last_predictor_name == "system_prompt":
            api_predictors = [p for p in available_predictors if p.startswith("API_information")]
            if api_predictors:
                selected = random.choice(api_predictors)
            else:
                selected = available_predictors[0] if available_predictors else "system_prompt"
        else:
            # Select next API based on correlation with the last updated predictor
            selected = self._select_next_api_module(available_predictors, last_predictor_name)

        # Update the state to track which predictor was selected
        if selected in state.list_of_named_predictors:
            selected_idx = state.list_of_named_predictors.index(selected)
            state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] = selected_idx

        return [selected]

