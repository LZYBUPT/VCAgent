# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
import traceback
import os
from collections.abc import Sequence
from typing import Generic

from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import FrontierType, GEPAState, ValsetEvaluation, initialize_gepa_state
from gepa.core.checkpoint import CheckpointManager, create_checkpoint_state
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import LoggerProtocol
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.logging.mcts_visualizer import MCTSVisualizer
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.mcts_tree import MCTSTree, MCTSNode
from gepa.proposer.reflective_mutation.reflective_mutation import (
    ReflectiveMutationProposer,
)
from gepa.strategies.eval_policy import EvaluationPolicy, FullEvaluationPolicy
from gepa.utils import StopperProtocol

# Import tqdm for progress bar functionality
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class GEPAEngine(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """Orchestrates the optimization loop using pluggable candidate proposers."""

    def __init__(
        self,
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        run_dir: str | None,
        valset: list[DataInst] | DataLoader[DataId, DataInst] | None,
        seed_candidate: dict[str, str],
        # Controls
        perfect_score: float,
        seed: int,
        # Strategies and helpers
        reflective_proposer: ReflectiveMutationProposer,
        merge_proposer: MergeProposer | None,
        frontier_type: FrontierType,
        # Logging
        logger: LoggerProtocol,
        experiment_tracker: ExperimentTracker,
        # Optional parameters
        track_best_outputs: bool = False,
        display_progress_bar: bool = False,
        raise_on_exception: bool = True,
        use_cloudpickle: bool = False,
        # Budget and Stop Condition
        stop_callback: StopperProtocol | None = None,
        val_evaluation_policy: EvaluationPolicy[DataId, DataInst] | None = None,
        # Checkpoint parameters
        enable_checkpoints: bool = True,
        checkpoint_every: int = 5,
        keep_last_checkpoints: int = 3,
    ):
        self.logger = logger
        self.run_dir = run_dir

        # Graceful stopping mechanism
        self._stop_requested = False

        # Set up stopping mechanism
        self.stop_callback = stop_callback
        self.adapter = adapter

        def evaluator(
            batch: list[DataInst], program: dict[str, str]
        ) -> tuple[list[RolloutOutput], list[float], Sequence[dict[str, float]] | None]:
            eval_result = adapter.evaluate(batch, program, capture_traces=False)
            return eval_result.outputs, eval_result.scores, eval_result.objective_scores

        self.evaluator = evaluator

        self.valset = ensure_loader(valset) if valset is not None else None
        self.seed_candidate = seed_candidate

        self.perfect_score = perfect_score
        self.seed = seed
        self.experiment_tracker = experiment_tracker

        self.reflective_proposer = reflective_proposer
        self.merge_proposer = merge_proposer
        self.frontier_type: FrontierType = frontier_type

        # Merge scheduling flags (mirroring previous behavior)
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        self.track_best_outputs = track_best_outputs
        self.display_progress_bar = display_progress_bar
        self.use_cloudpickle = use_cloudpickle

        self.raise_on_exception = raise_on_exception
        self.val_evaluation_policy: EvaluationPolicy[DataId, DataInst] = (
            val_evaluation_policy if val_evaluation_policy is not None else FullEvaluationPolicy()
        )
        
        # Initialize visualizer for real-time monitoring. Always initialize regardless of MCTS usage.
        if run_dir:
            output_dir = os.path.join(run_dir, "monitoring")
            self.mcts_visualizer = MCTSVisualizer(output_dir=output_dir)
            has_mcts = hasattr(self.reflective_proposer, 'mcts_tree') and self.reflective_proposer.mcts_tree is not None
            mode = "MCTS" if has_mcts else "non-MCTS"
            self.logger.log(f"[Visualizer] Initialized visualizer in {mode} mode: {output_dir}")
        else:
            self.mcts_visualizer = None
            self.logger.log("[Visualizer] Visualizer disabled (no run_dir specified)")

        # Initialize checkpoint manager for resume-from-checkpoint support
        self.enable_checkpoints = enable_checkpoints
        if self.enable_checkpoints and run_dir is not None:
            checkpoint_dir = os.path.join(run_dir, "checkpoints")
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                save_every=checkpoint_every,
                keep_last_n=keep_last_checkpoints,
                auto_resume=True
            )
            self.checkpoint_manager.print_checkpoint_status()
        else:
            self.checkpoint_manager = None
            if self.enable_checkpoints:
                self.logger.log("[Checkpoint] Warning: Checkpoints enabled but run_dir is None, checkpoints disabled")

    def _evaluate_on_valset(
        self,
        program: dict[str, str],
        state: GEPAState[RolloutOutput, DataId],
    ) -> ValsetEvaluation[RolloutOutput, DataId]:
        valset = self.valset
        assert valset is not None

        val_ids = self.val_evaluation_policy.get_eval_batch(valset, state)
        batch = valset.fetch(val_ids)
        outputs, scores, objective_scores = self.evaluator(batch, program)
        assert len(outputs) == len(val_ids), "Eval outputs should match length of selected validation indices"
        assert len(scores) == len(val_ids), "Eval scores should match length of selected validation indices"

        outputs_by_val_idx = dict(zip(val_ids, outputs, strict=False))
        scores_by_val_idx = dict(zip(val_ids, scores, strict=False))
        objective_by_val_idx = (
            dict(zip(val_ids, objective_scores, strict=False)) if objective_scores is not None else None
        )
        return ValsetEvaluation(
            outputs_by_val_id=outputs_by_val_idx,
            scores_by_val_id=scores_by_val_idx,
            objective_scores_by_val_id=objective_by_val_idx,
        )
    
    def _run_full_eval_and_add(
        self,
        new_program: dict[str, str],
        state: GEPAState[RolloutOutput, DataId],
        parent_program_idx: list[int],
        parent_node: MCTSNode = None,
        module_type: str = None,
    ) -> tuple[int, int]:
        num_metric_calls_by_discovery = state.total_num_evals

        valset_evaluation = self._evaluate_on_valset(new_program, state)

        state.num_full_ds_evals += 1
        state.total_num_evals += len(valset_evaluation.scores_by_val_id)

        new_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_evaluation=valset_evaluation,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery,
        )
        state.full_program_trace[-1]["new_program_idx"] = new_program_idx
        state.full_program_trace[-1]["evaluated_val_indices"] = sorted(valset_evaluation.scores_by_val_id.keys())

        valset_score = self.val_evaluation_policy.get_valset_score(new_program_idx, state)

        linear_pareto_front_program_idx = self.val_evaluation_policy.get_best_program(state)
        if new_program_idx == linear_pareto_front_program_idx:
            self.logger.log(f"Iteration {state.i + 1}: Found a better program on the valset with score {valset_score}.")

        valset = self.valset
        assert valset is not None

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=state,
            new_program_idx=new_program_idx,
            valset_evaluation=valset_evaluation,
            objective_scores=state.prog_candidate_objective_scores[new_program_idx],
            experiment_tracker=self.experiment_tracker,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
            valset_size=len(valset),
            val_evaluation_policy=self.val_evaluation_policy,
        )

        # MCTS backpropagation: if the program passed the subsample check, create a real child node and mark as success.
        # Previously created virtual nodes (from failures) will be converted with inherited statistics.
        if hasattr(self.reflective_proposer, 'mcts_tree') and self.reflective_proposer.mcts_tree is not None and module_type is not None:
            actual_parent_node = parent_node if parent_node is not None else self.reflective_proposer.mcts_tree.root

            # Use the actual object stored in the candidates list, not the local copy
            actual_new_program = state.program_candidates[new_program_idx]
            child_node = self.reflective_proposer.mcts_tree.add_child(
                parent=actual_parent_node,
                candidate=actual_new_program,
                module_type=module_type,
                inherit_virtual=True
            )

            # Store valset score for descendant mean calculation
            child_node.valset_score = valset_score

            self.reflective_proposer.mcts_tree.backpropagate(child_node, success=True)

            # Check if a virtual node was converted to a real node
            if module_type in actual_parent_node.module_stats:
                stats = actual_parent_node.module_stats[module_type]
                if stats['visit_count'] > 1:
                    self.logger.log(f"[MCTS] Virtual node converted to real node! Module {module_type} at gen {actual_parent_node.generation}")
                    self.logger.log(f"[MCTS] Inherited history: {stats['visit_count']-1} previous failures, now success")
            


        return new_program_idx, linear_pareto_front_program_idx

    def _get_parent_score(self, parent_node: MCTSNode, state: GEPAState) -> float:
        """Get the evaluation score of a parent node."""
        parent_candidate = parent_node.candidate
        for idx, candidate in enumerate(state.program_candidates):
            if candidate is parent_candidate:
                return state.program_full_scores_val_set[idx]
        return 0.0

    def run(self) -> GEPAState[RolloutOutput, DataId]:
        # Check tqdm availability if progress bar is enabled
        progress_bar = None
        if self.display_progress_bar:
            if tqdm is None:
                raise ImportError("tqdm must be installed when display_progress_bar is enabled")

            # Check if stop_callback contains MaxMetricCallsStopper
            total_calls: int | None = None
            stop_cb = self.stop_callback
            if stop_cb is not None:
                max_calls_attr = getattr(stop_cb, "max_metric_calls", None)
                if isinstance(max_calls_attr, int):
                    # Direct MaxMetricCallsStopper
                    total_calls = max_calls_attr
                else:
                    stoppers = getattr(stop_cb, "stoppers", None)
                    if stoppers is not None:
                        # CompositeStopper - iterate to find MaxMetricCallsStopper
                        for stopper in stoppers:
                            stopper_max = getattr(stopper, "max_metric_calls", None)
                            if isinstance(stopper_max, int):
                                total_calls = stopper_max
                                break

            if total_calls is not None:
                progress_bar = tqdm(total=total_calls, desc="GEPA Optimization", unit="rollouts")
            else:
                progress_bar = tqdm(desc="GEPA Optimization", unit="rollouts")
            progress_bar.update(0)

        # Prepare valset
        valset = self.valset
        if valset is None:
            raise ValueError("valset must be provided to GEPAEngine.run()")

        def valset_evaluator(
            program: dict[str, str],
        ) -> ValsetEvaluation[RolloutOutput, DataId]:
            all_ids = list(valset.all_ids())
            outputs, scores, objective_scores = self.evaluator(valset.fetch(all_ids), program)
            outputs_dict = dict(zip(all_ids, outputs, strict=False))
            scores_dict = dict(zip(all_ids, scores, strict=False))
            objective_scores_dict = (
                dict(zip(all_ids, objective_scores, strict=False)) if objective_scores is not None else None
            )
            return ValsetEvaluation(
                outputs_by_val_id=outputs_dict,
                scores_by_val_id=scores_dict,
                objective_scores_by_val_id=objective_scores_dict,
            )
        # The initialization mirrors a genetic algorithm: define Pareto frontier structures before the evolutionary loop
        # Attempt to resume from a checkpoint
        checkpoint_data = None
        if self.checkpoint_manager is not None:
            checkpoint_data = self.checkpoint_manager.load_checkpoint()

        # Initialize state
        if checkpoint_data is not None:
            # Resume from checkpoint
            self.logger.log("\n" + "="*60)
            self.logger.log("RESUMING FROM CHECKPOINT")
            self.logger.log("="*60)

            state = checkpoint_data["state"]["state"]

            # Restore MCTS tree
            if hasattr(self.reflective_proposer, 'mcts_tree') and 'mcts_tree' in checkpoint_data["state"]:
                self.reflective_proposer.mcts_tree = checkpoint_data["state"]["mcts_tree"]
                self.logger.log("Restored MCTS tree")

            # Restore merge proposer state
            if self.merge_proposer is not None and 'merge_proposer_state' in checkpoint_data["state"]:
                merge_state = checkpoint_data["state"]["merge_proposer_state"]
                self.merge_proposer.last_iter_found_new_program = merge_state.get("last_iter_found_new_program", False)
                self.merge_proposer.merges_due = merge_state.get("merges_due", 0)
                self.logger.log("Restored merge proposer state")

            self.logger.log(f"Resumed from iteration {state.i}")
            self.logger.log(f"Total evals so far: {state.total_num_evals}")
            self.logger.log(f"Number of candidates: {len(state.program_candidates)}")
            self.logger.log("="*60 + "\n")
        else:
            # Start from scratch
            state = initialize_gepa_state(
                run_dir=self.run_dir,
                logger=self.logger,
                seed_candidate=self.seed_candidate,
                valset_evaluator=valset_evaluator,
                track_best_outputs=self.track_best_outputs,
                frontier_type=self.frontier_type,
            )

        # Log base program score
        base_val_avg, base_val_coverage = state.get_program_average_val_subset(0)
        self.experiment_tracker.log_metrics(
            {
                "base_program_full_valset_score": base_val_avg,
                "base_program_val_coverage": base_val_coverage,
                "iteration": state.i + 1,
            },
            step=state.i + 1,
        )
        self.logger.log(
            f"Iteration {state.i + 1}: Base program full valset score: {base_val_avg} "
            f"over {base_val_coverage} / {len(valset)} examples"
        )

        # Merge scheduling
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        # Main loop
        last_pbar_val = 0
        while not self._should_stop(state):
            if self.display_progress_bar and progress_bar is not None:
                delta = state.total_num_evals - last_pbar_val
                progress_bar.update(delta)
                last_pbar_val = state.total_num_evals

            assert state.is_consistent()
            try:
                state.save(self.run_dir, use_cloudpickle=self.use_cloudpickle)
                state.i += 1
                state.full_program_trace.append({"i": state.i})

                # 1) Attempt merge first if scheduled and last iter found a new program
                if self.merge_proposer is not None and self.merge_proposer.use_merge:
                    if self.merge_proposer.merges_due > 0 and self.merge_proposer.last_iter_found_new_program:
                        proposal = self.merge_proposer.propose(state)
                        self.merge_proposer.last_iter_found_new_program = False  # old behavior

                        if proposal is not None and proposal.tag == "merge":
                            parent_sums = proposal.subsample_scores_before or [
                                float("-inf"),
                                float("-inf"),
                            ]
                            new_sum = sum(proposal.subsample_scores_after or [])

                            if new_sum >= max(parent_sums):
                                # ACCEPTED: consume one merge attempt and record it
                                self._run_full_eval_and_add(
                                    new_program=proposal.candidate,
                                    state=state,
                                    parent_program_idx=proposal.parent_program_ids,
                                    parent_node=None,  # merge does not use MCTS
                                    module_type=None,
                                )
                                self.merge_proposer.merges_due -= 1
                                self.merge_proposer.total_merges_tested += 1
                                continue  # skip reflective this iteration
                            else:
                                # REJECTED: do NOT consume merges_due or total_merges_tested
                                self.logger.log(
                                    f"Iteration {state.i + 1}: New program subsample score {new_sum} "
                                    f"is worse than both parents {parent_sums}, skipping merge"
                                )
                                # Skip reflective this iteration (old behavior)
                                continue

                    self.merge_proposer.last_iter_found_new_program = False

                # 2) Reflective mutation proposer
                proposal = self.reflective_proposer.propose(state)
                if proposal is None:
                    self.logger.log(f"Iteration {state.i + 1}: Reflective mutation did not propose a new candidate")
                    continue

                # Acceptance: require strict improvement on subsample
                old_sum = sum(proposal.subsample_scores_before or [])
                new_sum = sum(proposal.subsample_scores_after or [])

                # Extract parent_node and modules from proposal metadata before evaluating scores
                parent_node = proposal.metadata.get('parent_node', None)
                modules = proposal.metadata.get('modules', None)
                module_type = modules[0] if modules and len(modules) > 0 else None

                # Threshold mechanism: new candidate must exceed original score by minibatch_size * 0.03
                minibatch_size = len(proposal.subsample_indices) if proposal.subsample_indices else 0
                improvement_margin = minibatch_size * 0.03
                threshold_score = old_sum + improvement_margin
                if new_sum <= threshold_score:
                    self.logger.log(
                        f"Iteration {state.i + 1}: New subsample score {new_sum} does not exceed threshold {threshold_score:.4f} "
                        f"(old_score {old_sum} + minibatch_size {minibatch_size} x 0.03 = +{improvement_margin:.4f}), skipping"
                    )

                    # MCTS mode: on failure, create a virtual child node that records the failure
                    # without occupying actual child space. If the same module succeeds later,
                    # the virtual node will be converted to a real node, inheriting prior statistics.
                    if hasattr(self.reflective_proposer, 'mcts_tree') and self.reflective_proposer.mcts_tree is not None and module_type is not None:
                        actual_parent_node = parent_node if parent_node is not None else self.reflective_proposer.mcts_tree.root

                        # API module failure: create virtual child node and backpropagate failure
                        if module_type != "system_prompt":
                            self.reflective_proposer.mcts_tree.add_virtual_child(
                                parent=actual_parent_node,
                                candidate=proposal.candidate,
                                module_type=module_type
                            )
                            # Backpropagate failure info to update parent and ancestor statistics
                            self.reflective_proposer.mcts_tree.backpropagate_virtual(
                                parent=actual_parent_node,
                                module_type=module_type,
                                success=False
                            )
                            self.logger.log(f"[MCTS] Created virtual child for {module_type} at gen {actual_parent_node.generation} (not added to candidates)")
                            self.logger.log(f"[MCTS] If this module succeeds later, virtual node will convert to real node with inherited history")

                        # system_prompt failure: mark to prevent further forced updates
                        if module_type == "system_prompt":
                            actual_parent_node.system_prompt_update_failed = True
                            self.logger.log(f"[MCTS] system_prompt update failed at generation {actual_parent_node.generation + 1}")
                            self.logger.log(f"[MCTS] Marked system_prompt_update_failed=True, will not force update again")

                    continue
                else:
                    self.logger.log(
                        f"Iteration {state.i + 1}: New subsample score {new_sum} exceeds threshold {threshold_score:.4f} "
                        f"(old_score {old_sum} + minibatch_size {minibatch_size} x 0.03 = +{improvement_margin:.4f}). "
                        f"Continue to full eval and add to candidate pool."
                    )

                # Accept: full eval + add

                # In MCTS mode, ensure correct modules are set for the first iteration
                if hasattr(self.reflective_proposer, 'mcts_tree') and self.reflective_proposer.mcts_tree is not None:
                    if modules is None or len(modules) == 0:
                        modules = ["system_prompt"]

                # For MCTS mode modules is a list; we only need the first module for engine logic
                module_type = modules[0] if modules and len(modules) > 0 else None
                self._run_full_eval_and_add(
                    new_program=proposal.candidate,
                    state=state,
                    parent_program_idx=proposal.parent_program_ids,
                    parent_node=parent_node,
                    module_type=module_type,
                )

                # Schedule merge attempts like original behavior
                if self.merge_proposer is not None:
                    self.merge_proposer.last_iter_found_new_program = True
                    if self.merge_proposer.total_merges_tested < self.merge_proposer.max_merge_invocations:
                        self.merge_proposer.merges_due += 1

            except Exception as e:
                self.logger.log(f"Iteration {state.i + 1}: Exception during optimization: {e}")
                self.logger.log(traceback.format_exc())
                if self.raise_on_exception:
                    raise e
                else:
                    continue
            finally:
                # After each iteration (success or failure), update monitoring and visualizations
                self._update_monitoring(state)

                # Periodically save checkpoints
                if self.checkpoint_manager is not None:
                    if self.checkpoint_manager.should_save(state.i):
                        try:
                            self.logger.log(f"[Checkpoint] Starting checkpoint save at iteration {state.i}...")

                            # Prepare state to save
                            checkpoint_state = {
                                "state": state,
                                "mcts_tree": self.reflective_proposer.mcts_tree if hasattr(self.reflective_proposer, 'mcts_tree') else None,
                                "merge_proposer_state": {
                                    "last_iter_found_new_program": self.merge_proposer.last_iter_found_new_program,
                                    "merges_due": self.merge_proposer.merges_due,
                                } if self.merge_proposer is not None else None,
                            }

                            # Prepare metadata
                            best_program_idx = self.val_evaluation_policy.get_best_program(state)
                            best_val_score = self.val_evaluation_policy.get_valset_score(best_program_idx, state)
                            metadata = {
                                "iteration": state.i,
                                "total_evals": state.total_num_evals,
                                "num_candidates": len(state.program_candidates),
                                "best_program_idx": best_program_idx,
                                "best_val_score": best_val_score,
                            }

                            # Save checkpoint
                            saved_path = self.checkpoint_manager.save_checkpoint(
                                iteration=state.i,
                                state=checkpoint_state,
                                metadata=metadata
                            )
                            self.logger.log(f"[Checkpoint] Checkpoint saved successfully to: {saved_path}")

                        except Exception as e:
                            import traceback
                            error_msg = traceback.format_exc()
                            self.logger.log(f"[Checkpoint] ERROR: Failed to save checkpoint at iteration {state.i}")
                            self.logger.log(f"[Checkpoint] Error details:\n{error_msg}")
                    else:
                        # Log a hint about next save interval
                        next_save = ((state.i // self.checkpoint_manager.save_every) + 1) * self.checkpoint_manager.save_every
                        if state.i % 10 == 0:
                            self.logger.log(f"[Checkpoint] Current iteration: {state.i}, next checkpoint at iteration: {next_save}")
                else:
                    if state.i == 1:
                        self.logger.log("[Checkpoint] WARNING: Checkpoint manager is None - checkpoints are DISABLED!")
                        self.logger.log("[Checkpoint] Make sure enable_checkpoints=True and run_dir is set!")

        # Close progress bar if it exists
        if self.display_progress_bar and progress_bar is not None:
            progress_bar.close()

        state.save(self.run_dir)
        return state

    def _should_stop(self, state: GEPAState[RolloutOutput, DataId]) -> bool:
        """Check if the optimization should stop."""
        if self._stop_requested:
            return True
        if self.stop_callback and self.stop_callback(state):
            return True
        return False

    def request_stop(self) -> None:
        """Manually request the optimization to stop gracefully."""
        self.logger.log("Stop requested manually. Initiating graceful shutdown...")
        self._stop_requested = True
    
    def _update_monitoring(self, state: GEPAState[RolloutOutput, DataId]):
        """
        Update monitoring data and visualizations (called after each iteration).

        Supports two modes:
        1. With MCTS tree: draw MCTS tree, Pareto curve, best valset score curve
        2. Without MCTS tree: draw API descendant graph and best Pareto graph
        """
        if self.mcts_visualizer is None:
            return

        # Get current best program info
        best_program_idx = self.val_evaluation_policy.get_best_program(state)
        best_valset_score = self.val_evaluation_policy.get_valset_score(best_program_idx, state)

        # Calculate Pareto front aggregate score
        pareto_front_aggregate_score = self._calculate_pareto_aggregate(state)

        # Get new program info (if a new program was added this iteration)
        new_program_idx = None
        new_program_valset_score = None
        if len(state.program_candidates) > 0:
            new_program_idx = len(state.program_candidates) - 1
            if new_program_idx < len(state.program_full_scores_val_set):
                new_program_valset_score = state.program_full_scores_val_set[new_program_idx]

        self.mcts_visualizer.log_iteration(
            iteration=state.i + 1,
            total_metric_calls=state.total_num_evals,
            best_valset_score=best_valset_score,
            best_program_idx=best_program_idx,
            pareto_front_aggregate_score=pareto_front_aggregate_score,
            num_candidates=len(state.program_candidates),
            new_program_idx=new_program_idx,
            new_program_valset_score=new_program_valset_score,
        )
        
        has_mcts_tree = (hasattr(self.reflective_proposer, 'mcts_tree') and
                        self.reflective_proposer.mcts_tree is not None)

        # Update visualizations every 5 iterations
        if (state.i + 1) % 5 == 0:
            try:
                if has_mcts_tree:
                    self.logger.log(f"[MCTS] Updating MCTS visualizations at iteration {state.i + 1}...")

                    self.mcts_visualizer.plot_pareto_score_curve()
                    self.logger.log(f"[MCTS] Pareto score curve updated")

                    self.mcts_visualizer.plot_best_valset_score_curve()
                    self.logger.log(f"[MCTS] Best valset score curve updated")

                    # Visualize MCTS tree
                    mcts_tree = self.reflective_proposer.mcts_tree
                    num_nodes = self._count_tree_nodes(mcts_tree.root) if hasattr(mcts_tree, 'root') else 0
                    self.logger.log(f"[MCTS] Visualizing MCTS tree with {num_nodes} nodes...")

                    self.mcts_visualizer.visualize_mcts_tree(
                        mcts_tree,
                        state
                    )
                    self.logger.log(f"[MCTS] MCTS tree visualization saved")

                else:
                    self.logger.log(f"[Visualization] Updating visualizations (non-MCTS mode) at iteration {state.i + 1}...")

                    self.mcts_visualizer.plot_best_valset_score_curve()
                    self.logger.log(f"[Visualization] Best valset score curve updated")

                    if hasattr(self.mcts_visualizer, 'plot_candidate_evolution'):
                        self.mcts_visualizer.plot_candidate_evolution(state)
                        self.logger.log(f"[Visualization] Candidate evolution tree saved")
                    else:
                        self.logger.log(f"[Visualization] plot_candidate_evolution not available")

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                mode_str = "MCTS" if has_mcts_tree else "non-MCTS"
                self.logger.log(f"[{mode_str}] ERROR during visualization: {e}")
                self.logger.log(f"[{mode_str}] Error details:\n{error_msg}")
    
    def _count_tree_nodes(self, node) -> int:
        """Recursively count the number of nodes in the MCTS tree."""
        if node is None:
            return 0
        count = 1
        if hasattr(node, 'children'):
            for child in node.children:
                count += self._count_tree_nodes(child)
        return count
    
    def _calculate_pareto_aggregate(self, state: GEPAState[RolloutOutput, DataId]) -> float:
        """Calculate the aggregate score of the Pareto front."""
        if state.frontier_type == "instance":
            pareto_scores = list(state.pareto_front_valset.values())
            if pareto_scores:
                return sum(pareto_scores) / len(pareto_scores)
            return 0.0
        elif state.frontier_type == "objective":
            objective_scores = list(state.objective_pareto_front.values())
            if objective_scores:
                return sum(objective_scores) / len(objective_scores)
            return 0.0
        else:
            return 0.0