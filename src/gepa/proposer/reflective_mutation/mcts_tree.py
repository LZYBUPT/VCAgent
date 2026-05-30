# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
MCTS Tree Selection Mechanism

Uses Monte Carlo Tree Search (MCTS) with the PUCT algorithm to select candidate
solutions and modules to mutate. Each node in the tree represents a Pareto candidate.

Core features:
1. PUCT algorithm for node and module selection
2. API correlation weights as prior terms
3. First iteration forces system_prompt, producing a single branch
4. Forced system_prompt update every 4 generations (produces a single child)
5. When the mutation is ineffective, no child node is created but statistics are updated
6. 0/0 is defined as 0 to avoid infinite exploration of unvisited nodes
"""

import math
from typing import Dict, List, Tuple, Optional


class MCTSNode:
    """MCTS tree node representing a candidate solution."""

    def __init__(
        self,
        candidate: dict[str, str],
        parent: Optional['MCTSNode'] = None,
        module_type: str = "root"
    ):
        """
        Initialize an MCTS node.

        Args:
            candidate: The candidate solution (contains system_prompt and API_information modules)
            parent: Parent node
            module_type: The module type used to mutate from the parent to this node
        """
        self.candidate = candidate
        self.parent = parent
        self.module_type = module_type  # Which module was mutated to produce this node from its parent
        self.children: List[MCTSNode] = []

        # MCTS statistics
        self.visit_count = 0
        self.success_count = 0  # Number of successful mutations (high-quality offspring)

        # Per-module statistics (used to select which module to mutate next)
        # Note: only includes API modules, not system_prompt.
        # system_prompt is updated via forced periodic updates and does not compete in MCTS.
        self.module_stats: Dict[str, Dict[str, int]] = {}
        for key in candidate.keys():
            if key.startswith("API_information"):  # Only API modules
                self.module_stats[key] = {
                    "visit_count": 0,
                    "success_count": 0
                }

        # Virtual children: record failed attempts that may be "promoted" if they later succeed.
        # {module_type: {'visit_count': X, 'success_count': Y, 'candidate': last_attempt_candidate}}
        self.virtual_children: Dict[str, Dict] = {}

        # Generation calculation
        if parent is None:
            self.generation = 0
        else:
            self.generation = parent.generation + 1
            parent.children.append(self)

        # Flag indicating whether the last forced system_prompt update failed
        self.system_prompt_update_failed = False

        # Validation set score (used for computing descendants mean driver term)
        self.valset_score: Optional[float] = None  # This node's score on the validation set
    
    def count_descendants(self) -> int:
        """
        Recursively count all descendant nodes (including children, grandchildren,
        great-grandchildren, and virtual nodes).

        Returns:
            Total number of descendant nodes.
        """
        count = len(self.children)  # Direct children count
        count += len(self.virtual_children)  # Virtual children count
        for child in self.children:
            count += child.count_descendants()  # Recursively count each child's descendants
        return count

    def compute_descendants_valset_mean(self) -> Tuple[float, int]:
        """
        Compute the mean validation set score of this node and all its descendants
        (children, grandchildren, etc.).

        Returns:
            (mean, number_of_valid_nodes)
            - If there are no valid scores, returns (0.0, 0).
        """
        scores = []

        # Add the current node's score
        if self.valset_score is not None:
            scores.append(self.valset_score)

        # Recursively collect scores from all children and their descendants
        for child in self.children:
            self._collect_valset_scores(child, scores)
        
        if len(scores) == 0:
            return 0.0, 0
        
        return sum(scores) / len(scores), len(scores)
    
    def _collect_valset_scores(self, node: 'MCTSNode', scores: List[float]) -> None:
        """
        Recursively collect validation set scores from a node and all its descendants.

        Args:
            node: The node from which to collect scores
            scores: List of scores (mutated in-place)
        """
        if node.valset_score is not None:
            scores.append(node.valset_score)
        
        for child in node.children:
            self._collect_valset_scores(child, scores)
    
    def puct_score(self, c: float, prior: float = 1.0, lambda_depth: float = 0.5, k_children: float = 3.0) -> float:
        """
        Compute PUCT score with sigmoid cliff-gate exploration control and descendants mean driver.

        PUCT = Q + U_final + V_descendants (when count_gate < 1e-4)

        Where:
        - Q: exploitation term = success_count / visit_count
        - U_raw: raw exploration term = c * prior * sqrt(N_parent) / (1 + N)
        - depth_scale: depth control = 1.0 - lambda_depth / (depth + 1)
        - count_gate: sigmoid gate = 1 / (1 + exp(k * (num_children - threshold)))
        - U_final: final exploration term = U_raw * depth_scale * count_gate
        - V_descendants: descendants mean driver (only when count_gate < 1e-4)

        Sigmoid gate effect:
        - First 15 descendants: normal exploration (gate ~ 1.0)
        - Starting from the 16th descendant: cliff drop (gate rapidly approaches 0)

        Descendants mean driver:
        - When exploration is suppressed (count_gate < 1e-4), use the mean of all descendants'
          validation set scores
        - This avoids over-exploration while still guiding search toward high-quality regions

        Args:
            c: exploration constant
            prior: prior probability (from API correlation weights)
            lambda_depth: depth control coefficient (default 0.5)
            k_children: sigmoid steepness (default 3.0; larger values produce steeper drops)

        Returns:
            PUCT score
        """
        # Q value: exploitation term
        if self.visit_count == 0:
            q_value = 0.0
        else:
            q_value = self.success_count / self.visit_count

        # --- 1. Raw U-Score ---
        if self.parent is None:
            parent_visits = 1
        else:
            parent_visits = self.parent.visit_count

        u_score_raw = c * prior * math.sqrt(parent_visits) / (1 + self.visit_count)

        # --- 2. Sigmoid Gate (Count Gate) ---
        # Based on total descendants (including children, grandchildren, virtual nodes, etc.)
        num_children = self.count_descendants()

        # Sigmoid parameters
        threshold = 15  # Threshold: normal exploration for first 15 iterations, then cliff drop
        k_steepness = k_children  # Steepness (uses the passed-in parameter)

        # Sigmoid formula: gate ~ 1.0 for first 15 nodes; cliff drop starting from the 16th
        try:
            exponent = k_steepness * (num_children - threshold)
            if exponent > 100:  # Prevent exp overflow
                count_gate = 0.0
            else:
                count_gate = 1.0 / (1.0 + math.exp(exponent))
        except OverflowError:
            count_gate = 0.0

        # --- 3. Final exploration term ---
        final_u_score = u_score_raw * count_gate

        # --- 4. Descendants mean driver (only when exploration is suppressed) ---
        v_descendants = 0.0
        if count_gate < 1e-4:
            # Compute mean validation set score across this node and all its descendants
            descendants_mean, num_valid = self.compute_descendants_valset_mean()
            if num_valid > 0:
                # Use normalized mean as the driver term
                # Assumes validation scores are in [0, 1]; adjust if needed
                v_descendants = descendants_mean
                # Optional: add weight coefficient to balance importance of different terms
                # v_descendants = 0.5 * descendants_mean  # e.g. weight of 0.5
        
        return q_value + final_u_score + v_descendants


class MCTSTree:
    """MCTS tree for managing the tree structure of candidate solutions."""

    def __init__(self, root_candidate: dict[str, str], c: float = 1.414,
                 parent_module_decay: float = 0.7,
                 lambda_depth: float = 0.5,
                 k_children: float = 3.0):
        """
        Initialize the MCTS tree.

        Args:
            root_candidate: Root node candidate (initial candidate)
            c: PUCT exploration constant (default sqrt(2))
            parent_module_decay: Weight decay factor for modules already used by ancestors (default 0.7)
            lambda_depth: Depth regularization coefficient; controls how quickly exploration
                decreases for deeper nodes (default 0.5)
            k_children: Child node smoothness parameter; controls how strongly the number of
                children affects exploration (default 3.0)
        """
        self.root = MCTSNode(root_candidate, parent=None, module_type="root")
        self.c = c
        self.parent_module_decay = parent_module_decay  # Decay factor for API modules used by ancestors
        self.lambda_depth = lambda_depth  # Depth regularization coefficient
        self.k_children = k_children  # Child node smoothness parameter

        # Maintain a mapping from candidate id to node (for fast lookup)
        self.nodes_map: Dict[int, MCTSNode] = {id(root_candidate): self.root}

        # Iteration counter (used to determine whether it is the first iteration)
        self.iteration_count = 0
    
    def select_with_puct(
        self,
        component_selector,
        valid_candidates=None
    ) -> Tuple[MCTSNode, str]:
        """
        Use the PUCT algorithm to select a node and a module to mutate.

        Process:
        1. Start traversal from the root node
        2. At each level, select the child with the highest PUCT score
        3. Once at the selected node, choose the module based on module_stats
        4. Force selection of system_prompt module on the first iteration
        5. Force system_prompt update every 4 generations

        Args:
            component_selector: Component selector object for obtaining API correlations
            valid_candidates: Current set of valid candidates for filtering (optional)

        Returns:
            (selected_node, module_to_update)
        """
        self.iteration_count += 1
        print(f"[MCTS DEBUG] Iteration {self.iteration_count}: Starting PUCT selection")
        
        # First iteration: force selection of root node and system_prompt module
        if self.iteration_count == 1:
            print(f"[MCTS DEBUG] First iteration: forcing system_prompt selection")
            return self.root, "system_prompt"

        # Start traversal from the root
        current = self.root

        # If a valid candidates list is provided, only select among still-valid candidates
        if valid_candidates is not None:
            valid_ids = set(id(c) for c in valid_candidates)
            if id(current.candidate) not in valid_ids:
                current = self.root

        print(f"[MCTS DEBUG] Starting traversal from gen {current.generation}")

        # MCTS selection: at each node, compute PUCT scores for all possible actions
        max_depth = 50  # Limit max traversal depth to avoid infinite loops
        traversal_depth = 0

        while traversal_depth < max_depth:
            print(f"[MCTS DEBUG] Depth {traversal_depth}, current node gen {current.generation}, children: {len(current.children)}")

            # Special case: if current node has exactly one child produced by a system_prompt mutation, pass through directly
            if len(current.children) == 1 and current.children[0].module_type == "system_prompt":
                print(f"[MCTS DEBUG] Only one child (system_prompt), passing through directly")
                current = current.children[0]
                traversal_depth += 1
                continue

            # Check whether to force a system_prompt update (every 4 generations)
            if should_force_system_prompt_update(current):
                print(f"[MCTS DEBUG] Forcing system_prompt update at generation {current.generation}")
                selected_node = current
                module_to_update = "system_prompt"
                return selected_node, module_to_update

            # Gather all possible actions at current node (existing children + unexplored modules)
            # First compute the parent (current) sigmoid gate; all children and unexplored modules share this gate value
            current_descendants = current.count_descendants()
            threshold = 15
            k_steepness = self.k_children
            try:
                exponent = k_steepness * (current_descendants - threshold)
                current_count_gate = 0.0 if exponent > 100 else 1.0 / (1.0 + math.exp(exponent))
            except OverflowError:
                current_count_gate = 0.0
            
            # 1. Actions corresponding to existing children
            child_actions = []  # [(child_node, module_type, puct_score)]
            for child in current.children:
                # Skip invalid candidates if a valid candidates list is provided
                if valid_candidates is not None and id(child.candidate) not in valid_ids:
                    print(f"[MCTS DEBUG] Skipping invalid child gen {child.generation}")
                    continue

                # Get the module type corresponding to this child
                module_type = child.module_type

                # [KEY] Compute prior from the parent's (current) perspective, not the child's.
                # This avoids the "self vs. self" low-weight problem.
                # Weight decay should only apply during "select next module to expand", not during "select existing child".
                prior = self._get_prior_weight(component_selector, current, module_type)

                # Manually compute PUCT score (use parent's gate value, not child's own)
                # Q value
                if child.visit_count == 0:
                    q_value = 0.0
                else:
                    q_value = child.success_count / child.visit_count

                # U value: raw exploration term
                parent_visits = current.visit_count
                u_score_raw = self.c * prior * math.sqrt(parent_visits) / (1 + child.visit_count)

                # Apply parent's sigmoid gate (all sibling children share it)
                u_score_final = u_score_raw * current_count_gate

                # Descendants mean driver (only when exploration is suppressed)
                v_descendants = 0.0
                if current_count_gate < 1e-4:
                    descendants_mean, num_valid = child.compute_descendants_valset_mean()
                    if num_valid > 0:
                        v_descendants = descendants_mean

                # Total PUCT score
                score = q_value + u_score_final + v_descendants
                
                child_actions.append((child, module_type, score))
                
                # Verbose output of child statistics
                if current_count_gate < 1e-4 and v_descendants > 0:
                    print(f"[MCTS DEBUG] Child gen {child.generation} (module {module_type}) PUCT score: {score:.3f}")
                    print(f"           Q={q_value:.3f} ({child.success_count}/{child.visit_count}), U={u_score_final:.3f} (raw_U={u_score_raw:.3f}, gate={current_count_gate:.6f}), V_desc={v_descendants:.3f} (exploration suppressed, using descendants mean)")
                else:
                    print(f"[MCTS DEBUG] Child gen {child.generation} (module {module_type}) PUCT score: {score:.3f}")
                    print(f"           Q={q_value:.3f} ({child.success_count}/{child.visit_count}), U={u_score_final:.3f} (raw_U={u_score_raw:.3f}, gate={current_count_gate:.3f}, parent_descendants={current_descendants})")

            # 2. Actions for unexplored/failed modules (all modules without children)
            unexplored_actions = []  # [(None, module_type, puct_score)]

            # Get all modules without children (both never-tried and tried-but-failed)
            explored_module_types = set(child.module_type for child in current.children)
            unexplored_modules = [m for m in current.module_stats.keys() if m not in explored_module_types]

            # Get all API modules used in the ancestor chain
            ancestor_modules = self._get_ancestor_modules(current)

            # Step 1: collect prior weights for all modules (before normalization)
            module_priors = {}  # {module_name: prior_weight}
            for module in unexplored_modules:
                # Get prior weight (API correlation)
                if hasattr(component_selector, 'api_similarities') and module.startswith("API_information"):
                    prior = self._get_prior_weight(component_selector, current, module)
                else:
                    prior = 1.0

                # [Decay 2] Ancestor module decay: if the module was used in the ancestor chain, reduce its weight
                if module in ancestor_modules:
                    prior *= self.parent_module_decay
                    print(f"[MCTS DEBUG] Module {module} was used in ancestor chain, applying decay {self.parent_module_decay}")

                module_priors[module] = prior

            # # Step 2: normalize prior weights (make sum = 1)
            # total_prior = sum(module_priors.values())
            # if total_prior > 0:
            #     normalized_priors = {module: prior / total_prior for module, prior in module_priors.items()}
            # else:
            #     normalized_priors = {module: 1.0 / len(module_priors) for module in module_priors}
            normalized_priors = module_priors

            # Step 3: use (optionally) normalized priors to compute PUCT scores (applying parent's sigmoid gate)
            # Parent's gate has already been computed above (current_count_gate, current_descendants)

            for module in unexplored_modules:
                # Get statistics for this module
                stats = current.module_stats[module]
                visit_count = stats["visit_count"]
                success_count = stats["success_count"]

                # Q value (0/0 = 0)
                if visit_count == 0:
                    q_value = 0.0
                else:
                    q_value = success_count / visit_count

                # Use (optionally) normalized prior
                prior = normalized_priors[module]

                # U value: raw exploration bonus (same formula as puct_score)
                u_score_raw = self.c * prior * math.sqrt(current.visit_count + 1) / (1 + visit_count)

                # Apply sigmoid gate (based on parent's descendant count)
                u_score_final = u_score_raw * current_count_gate

                # Total PUCT score
                score = q_value + u_score_final
                unexplored_actions.append((None, module, score))

                # Distinguish fully unexplored vs tried-but-failed modules
                if visit_count == 0:
                    status = "Unexplored"
                else:
                    status = f"Failed (tried {visit_count} times)"
                print(f"[MCTS DEBUG] {status} module {module} PUCT score: {score:.3f} (Q={q_value:.3f}, U={u_score_final:.3f}, raw_U={u_score_raw:.3f}, gate={current_count_gate:.3f}, parent_descendants={current_descendants})")

            # 3. Merge all actions and select the best
            all_actions = child_actions + unexplored_actions
            
            if not all_actions:
                # No available actions; stop at current node and select any module
                print(f"[MCTS DEBUG] No available actions, stopping at gen {current.generation}")
                selected_node = current
                module_to_update = self._select_fallback_module(current)
                return selected_node, module_to_update

            # Select the action with the highest PUCT score
            # If multiple actions tie, pick one randomly
            max_score = max(action[2] for action in all_actions)
            best_actions = [action for action in all_actions if action[2] == max_score]

            import random
            best_action = random.choice(best_actions)
            best_node, best_module, best_score = best_action

            print(f"[MCTS DEBUG] Best action: module {best_module} with score {best_score:.3f} (from {len(best_actions)} tied actions)")

            # If best action is an existing child, continue traversing down
            if best_node is not None:
                print(f"[MCTS DEBUG] Moving to existing child gen {best_node.generation}")
                current = best_node
                traversal_depth += 1
            else:
                # If best action is an unexplored module, stop at current node and expand that module
                print(f"[MCTS DEBUG] Stopping at gen {current.generation} to expand module {best_module}")
                selected_node = current
                module_to_update = best_module
                return selected_node, module_to_update

        # Reached max depth; stop at current node (fallback)
        print(f"[MCTS DEBUG] Reached max depth, stopping at gen {current.generation}")
        selected_node = current
        module_to_update = self._select_module_for_expansion(selected_node, component_selector)
        
        return selected_node, module_to_update
    
    def _get_prior_weight(self, component_selector, node: MCTSNode, module_type: str) -> float:
        """
        Get the prior weight (from API correlation).

        Args:
            component_selector: Component selector
            node: The node
            module_type: Module type

        Returns:
            Prior weight
        """
        if hasattr(component_selector, 'api_similarities') and hasattr(component_selector, '_get_correlation_weight'):
            # For API modules, use correlation weight as prior
            if module_type.startswith("API_information"):
                # Get the last updated module of this node as the basis for correlation computation
                last_updated = self._get_last_updated_module(node)

                # If no previous API module (first API selection), all APIs compete fairly.
                # Return a uniform prior (summing to 1).
                if last_updated is None:
                    # Count all API modules
                    num_apis = len([m for m in node.module_stats.keys() if m.startswith("API_information")])
                    return 1.0 / num_apis if num_apis > 0 else 1.0

                # Compute weight based on correlation with the previous API module
                prior = component_selector._get_correlation_weight(last_updated, module_type)
                return prior
            else:
                return 1.0  # system_prompt uses default weight
        else:
            return 1.0  # Default weight
    
    def _get_last_updated_module(self, node: MCTSNode) -> Optional[str]:
        """
        Get the last updated API module of the node (for API correlation computation).

        If the current node was produced via system_prompt mutation, traverse upward
        to find the nearest API module. This ensures API module selection is based on
        meaningful correlation.

        Args:
            node: The node

        Returns:
            Name of the last updated API module, or None if none exists.
        """
        current = node

        # Traverse up to find the nearest API module
        while current is not None:
            if current.module_type.startswith("API_information"):
                return current.module_type
            current = current.parent

        # If no API module found all the way up to the root, return None.
        # This indicates the first API module selection; all APIs should compete fairly.
        return None

    def _get_unexplored_modules(self, node: MCTSNode) -> List[str]:
        """
        Get modules that have not been explored for this node.

        A module is "unexplored" iff its visit_count == 0.
        If visit_count > 0 but no child exists, it was tried and failed, and is NOT
        considered "unexplored".

        Args:
            node: The node

        Returns:
            List of unexplored module names.
        """
        unexplored = []

        for module_name in node.module_stats.keys():
            stats = node.module_stats[module_name]
            # Only modules with visit_count == 0 are truly unexplored
            if stats['visit_count'] == 0:
                unexplored.append(module_name)

        return unexplored

    def _get_ancestor_modules(self, node: MCTSNode) -> set:
        """
        Get all API modules used in the ancestor chain.

        Traverse upward to the root, collecting the module_type of all ancestor nodes.

        Args:
            node: The current node

        Returns:
            Set of API module names used in the ancestor chain.
        """
        ancestor_modules = set()
        current = node

        while current is not None:
            # Only collect API modules (skip root and system_prompt)
            if current.module_type.startswith("API_information"):
                ancestor_modules.add(current.module_type)
            current = current.parent

        return ancestor_modules

    def _select_fallback_module(self, node: MCTSNode) -> str:
        """
        Select a fallback module when no actions are available (prefer API modules,
        fall back to system_prompt).

        Args:
            node: The node

        Returns:
            Module name
        """
        # Prefer the first API module
        for module in node.module_stats.keys():
            if module.startswith("API_information"):
                return module

        # If no API module, fall back to system_prompt
        if "system_prompt" in node.module_stats:
            return "system_prompt"

        # If even system_prompt is missing, pick the first available module
        if node.module_stats:
            return next(iter(node.module_stats.keys()))

        # If no modules at all, raise an error
        raise ValueError("No modules available for fallback")
    
    def _select_module_for_expansion(self, node: MCTSNode, component_selector) -> str:
        """
        Select a module to expand for the given node.

        Computes PUCT scores for each module based on module_stats and selects the
        module with the highest score. Note: only API modules are considered, not
        system_prompt.

        Args:
            node: The node
            component_selector: Component selector

        Returns:
            Name of the module to mutate
        """
        available_modules = []
        module_scores = {}

        # Get all possible modules (API modules only)
        candidate = node.candidate
        for key in candidate.keys():
            if key.startswith("API_information"):  # Only API modules
                available_modules.append(key)

        # Step 1: collect prior weights for all modules
        module_priors = {}
        for module in available_modules:
            # Get prior weight (API correlation)
            if hasattr(component_selector, 'api_similarities') and module.startswith("API_information"):
                prior = self._get_prior_weight(component_selector, node, module)
            else:
                prior = 1.0
            module_priors[module] = prior

        # Step 2: normalize prior weights
        total_prior = sum(module_priors.values())
        if total_prior > 0:
            normalized_priors = {module: prior / total_prior for module, prior in module_priors.items()}
        else:
            normalized_priors = {module: 1.0 / len(module_priors) for module in module_priors}

        # Step 3: compute PUCT scores using normalized priors
        for module in available_modules:
            # Get statistics for this module
            stats = node.module_stats[module]
            visit_count = stats["visit_count"]
            success_count = stats["success_count"]

            # Q value (special rule: 0/0 = 0)
            if visit_count == 0:
                q_value = 0.0
            else:
                q_value = success_count / visit_count

            # Use normalized prior
            prior = normalized_priors[module]

            # U value: exploration bonus
            exploration = self.c * prior * math.sqrt(node.visit_count + 1) / (1 + visit_count)

            # Total PUCT score
            module_scores[module] = q_value + exploration

        # Select the module with the highest score
        if module_scores:
            max_score = max(module_scores.values())
            # Collect all modules tied at the max score
            top_modules = [module for module, score in module_scores.items() if score == max_score]

            # If multiple modules are tied, pick one randomly
            import random
            return random.choice(top_modules)
        else:
            # If no modules are available, try picking from available_modules
            if available_modules:
                return available_modules[0]
            else:
                # Edge case: no modules at all, fall back to system_prompt
                if "system_prompt" in node.module_stats:
                    return "system_prompt"
                # If even system_prompt is missing, raise error
                raise ValueError("No modules available for expansion")
    
    def add_virtual_child(
        self,
        parent: MCTSNode,
        candidate: dict[str, str],
        module_type: str
    ):
        """
        Record a virtual child node (a failed attempt).

        Args:
            parent: Parent node
            candidate: The attempted candidate solution
            module_type: The module type used for the mutation
        """
        if module_type not in parent.virtual_children:
            parent.virtual_children[module_type] = {
                'candidate': candidate
            }
        else:
            # Update the last attempted candidate
            parent.virtual_children[module_type]['candidate'] = candidate

        print(f"[MCTS DEBUG] Added virtual child for module {module_type} at gen {parent.generation}")

    def add_child(
        self,
        parent: MCTSNode,
        candidate: dict[str, str],
        module_type: str,
        inherit_virtual: bool = True
    ) -> MCTSNode:
        """
        Add a child node to the tree.

        If the module previously had a virtual child (failure record), the statistical
        information is inherited and the virtual child is "promoted" to a real child.

        Args:
            parent: Parent node
            candidate: New candidate solution
            module_type: Module type used for the mutation
            inherit_virtual: Whether to inherit statistics from a virtual child

        Returns:
            The newly created child node
        """
        child = MCTSNode(candidate, parent=parent, module_type=module_type)

        # If there is a virtual child record, inherit its statistics (previous failure records)
        if inherit_virtual and module_type in parent.virtual_children:
            # Retrieve historical stats from module_stats (includes all previous failed attempts)
            if module_type in parent.module_stats:
                stats = parent.module_stats[module_type]
                # Note: visit_count and success_count here are already accumulated values.
                # We don't need to manually set them because subsequent backpropagation will update them.
                # But we should note that this node was promoted from a virtual child.
                print(f"[MCTS DEBUG] Converting virtual child to real child for {module_type}")
                print(f"[MCTS DEBUG] Previous attempts: {stats['visit_count']} visits, {stats['success_count']} successes")

            # Remove the virtual child record (it has been promoted)
            del parent.virtual_children[module_type]

        self.nodes_map[id(candidate)] = child
        print(f"[MCTS DEBUG] Added child node gen {child.generation} to tree (module: {module_type})")
        return child

    def backpropagate(self, node: MCTSNode, success: bool):
        """
        Backpropagate to update statistics (mutation was effective, a child node was created).

        Args:
            node: The child node
            success: Whether the mutation was successful (True means effective)
        """
        current = node
        while current is not None:
            current.visit_count += 1
            if success:
                current.success_count += 1

            # Update the parent's module_stats
            if current.parent is not None:
                module_type = current.module_type
                if module_type in current.parent.module_stats:
                    current.parent.module_stats[module_type]["visit_count"] += 1
                    if success:
                        current.parent.module_stats[module_type]["success_count"] += 1

            current = current.parent

        print(f"[MCTS DEBUG] Backpropagated {'success' if success else 'failure'} from gen {node.generation}")

    def backpropagate_virtual(
        self,
        parent: MCTSNode,
        module_type: str,
        success: bool
    ):
        """
        Virtual backpropagation (mutation was ineffective, no child node created,
        but statistics are still updated).

        This is a key innovation: even without creating a child node, the parent's
        statistics are updated so that the MCTS algorithm can learn which module
        mutations are ineffective.

        Args:
            parent: Parent node
            module_type: Module type that was attempted for mutation
            success: Whether the mutation was successful (False means ineffective)
        """
        # Update the parent's module_stats
        if module_type in parent.module_stats:
            parent.module_stats[module_type]["visit_count"] += 1
            if success:
                parent.module_stats[module_type]["success_count"] += 1

        # Propagate upward to ancestor nodes
        current = parent
        while current is not None:
            current.visit_count += 1
            if success:
                current.success_count += 1
            current = current.parent

        print(f"[MCTS DEBUG] Virtual backpropagation {'success' if success else 'failure'} for module {module_type} at gen {parent.generation}")


def should_force_system_prompt_update(node: MCTSNode) -> bool:
    """
    Determine whether to force a system_prompt update.

    Rules:
    1. First iteration (root -> generation 1): forced update (handled in select_with_puct)
    2. Subsequently, forced update every 4 generations: generation 5, 9, 13, ...
    3. If the last system_prompt update failed, skip this forced update

    Args:
        node: The current node

    Returns:
        Whether to force a system_prompt update
    """
    # If the last system_prompt update failed, do not force it
    if node.system_prompt_update_failed:
        return False

    # The next generation
    next_generation = node.generation + 1

    # Every 4 generations: gen 5, 9, 13, ...
    # i.e., (generation - 1) % 4 == 0 and generation > 1
    if next_generation > 1 and (next_generation - 1) % 4 == 0:
        return True

    return False
