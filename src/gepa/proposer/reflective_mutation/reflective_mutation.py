# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections.abc import Mapping, Sequence
from typing import Any, Tuple, Union

from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate
from gepa.proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from gepa.proposer.reflective_mutation.mcts_tree import MCTSTree, MCTSNode, should_force_system_prompt_update
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from gepa.strategies.deduplication_proposal import DeduplicationInstructionProposalSignature,RelevancePruningInstructionProposalSignature


# API initialization templates - used when an API module is selected for the first time
# Each template contains placeholders that will be replaced with actual API data
# The template itself can be optimized by LLM during the mutation process
# Templates are in English to be consistent with the assistant's training data
API_INITIALIZATION_TEMPLATES = {
    "API_information1": "{NCBI}: {symbol} is a gene with NCBI Gene ID {gene_id}, full name {description}, from {source}. The gene is located on chromosome {chromosome}, with map location {map_location}. Functional annotation: {summary}. Other common aliases include: {aliases_summary}.",
    
    "API_information2": "{UniProt}: The protein corresponding to {gene_symbol} has UniProt protein ID {protein_id}, protein name {protein_name}. Main function: {function}. Subcellular localization: {subcellular_locations_summary}. Associated keywords include {keywords_summary}.",
    
    "API_information3": "{Reactome}: {gene_symbol} is a gene whose pathway annotation comes from {source} database. {gene_symbol} participates in the following biological pathways: {reactome_pathways_summary}. (total {total_pathways} pathways).",
    
    "API_information4": "{KEGG}: {gene_name} is a gene with KEGG gene ID {kegg_gene_id}. Pathways involved: {pathways_summary} (total {total_pathways} pathways). Associated diseases: {diseases_summary}. Known drugs targeting this gene: {drug_targets_summary}.",
    
    "API_information5": "{Ensembl}: {gene_symbol} is a gene with Ensembl gene ID {ensembl_id}, full name {gene_full_name}, gene type {gene_type}. Located at {location}, strand {strand}. Annotated transcripts include: {transcripts_summary}. Database identifiers: NCBI Gene ID {NCBI_Gene_ID}, HGNC ID {HGNC_ID}.",
    
    "API_information6": "{Cellosaurus}: Cell line {cell_line_name} has Cellosaurus accession {accession_id}, category {category}, species {species}. Derived from {tissue_origin} tissue, associated disease {disease}, cell type {cell_type}. Donor sex {sex}, age {age}, population {population}. Doubling time {doubling_time}. Key mutations: {key_mutations}.",
    
    "API_information7": "{CCLE}: Cell line {cell_line_name} corresponds to {ccle_name} in CCLE, from {source} database. Primary disease: {primary_disease}, lineage: {lineage}, subtype: {subtype}, expression profile {expression_profile_available}.",
    
    "API_information8": "{DepMap}: Cell line {cell_line_name} has DepMap ID {depmap_id}, CCLE name {ccle_name}, from {source}. Tissue of origin: {tissue}, lineage: {lineage}, disease: {disease}, growth pattern: {growth_pattern}.",
    
    "API_information9": "{PubChem}: {drug_name} is a compound with PubChem CID {pubchem_cid}, molecular formula {molecular_formula}, molecular weight {molecular_weight}. IUPAC name: {iupac_name}, common names: {common_names}. Physicochemical properties: LogP {logp}, H-bond donors {h_bond_donor}, H-bond acceptors {h_bond_acceptor}, canonical SMILES {canonical_smiles}.",

    "API_information10": "{DrugBank}: {name} is a drug identified by DrugBank ID {drugbank_id}. Description: {description_drug}. Regulatory groups: {groups}. Mechanism of action: {mechanism_of_action}. Pharmacodynamics: {pharmacodynamics}. CAS number: {cas_number}."
}


class ReflectiveMutationProposer(ProposeNewCandidate[DataId]):
    """
    Implements current reflective mutation flow:
    - Select candidate via selector
    - Select minibatch via sampler
    - capture_traces_and_eval -> trajectories, subsample_scores
    - skip if all scores==perfect and skip_perfect_score
    - reflection + mutate -> new candidate
    - evaluate new candidate on same minibatch -> new_subsample_scores
    - Return proposal if improved; else None
    """

    def __init__(
        self,
        logger: Any,
        trainset: list[DataInst] | DataLoader[DataId, DataInst],
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        candidate_selector: Union[CandidateSelector, None],
        module_selector: ReflectionComponentSelector,
        batch_sampler: BatchSampler[DataId, DataInst],
        perfect_score: float,
        skip_perfect_score: bool,
        experiment_tracker: Any,
        mcts_tree: MCTSTree,  # MCTS tree instance (may be None for traditional mode)
        reflection_lm: LanguageModel | None = None,
        reflection_prompt_template: str | None = None,
    ):
        self.logger = logger
        self.trainset = ensure_loader(trainset)
        self.adapter = adapter
        self.candidate_selector = candidate_selector  # None in MCTS mode
        self.module_selector = module_selector
        self.batch_sampler = batch_sampler
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.experiment_tracker = experiment_tracker
        self.mcts_tree = mcts_tree  # None in traditional mode
        self.reflection_lm = reflection_lm

        InstructionProposalSignature.validate_prompt_template(reflection_prompt_template)
        self.reflection_prompt_template = reflection_prompt_template
    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        Propose new texts for components using instruction proposal.
        
        For API modules:
        - If uninitialized: Return initialization template directly (no LLM mutation)
        - If already has API info: Use InstructionProposalSignature to mutate
        
        For system_prompt and other modules: Use InstructionProposalSignature normally
        
        Args:
            candidate: Current candidate
            reflective_dataset: Dataset for reflection
            components_to_update: List of components to update
        """
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update)

        if self.reflection_lm is None:
            raise ValueError("reflection_lm must be provided when adapter.propose_new_texts is None.")
        
        new_texts: dict[str, str] = {}
        
        for name in components_to_update:
            # Gracefully handle cases where a selected component has no data in reflective_dataset
            if name not in reflective_dataset or not reflective_dataset.get(name):
                self.logger.log(
                    f"Component '{name}' is not in reflective dataset. Skipping."
                )
                continue

            base_instruction = candidate[name]
            
            # Handle API modules
            if name.startswith("API_information"):
                # Check if API module is uninitialized
                is_uninitialized = (
                    base_instruction.strip() == "The information is unfavorable for the current question and will not be provided."
                    or base_instruction.strip() == ""
                )
                
                if is_uninitialized:
                    # For uninitialized API: Three-step pipeline (Deduplication → Relevance Pruning → Final Mutation)
                    self.logger.log(f"Component '{name}' is uninitialized. Starting 3-step initialization pipeline...")
                    template = API_INITIALIZATION_TEMPLATES.get(name, "")
                    
                    # Build API context
                    other_api_context = self._build_other_api_context(candidate, name)
                    dataset_with_feedback = reflective_dataset[name]
                    
                    # Step 1: Deduplication - Remove information redundant with other APIs
                    self.logger.log(f"  Step 1/3: Deduplicating template for '{name}' (removing cross-API redundancy)...")
                    deduplicated_template = template
                    if other_api_context:
                        deduplicated_template = DeduplicationInstructionProposalSignature.run(
                            lm=self.reflection_lm,
                            input_dict={
                                "current_instruction_doc": template,
                                "other_api_context": other_api_context,
                                "prompt_template": None,  # Use default deduplication template
                            },
                        )["new_instruction"]
                        self.logger.log(f"  Step 1/3 complete: Deduplicated template length = {len(deduplicated_template)} chars")
                    else:
                        self.logger.log(f"  Step 1/3 skipped: No other API context available for '{name}'")
                    
                    # Step 2: Relevance Pruning - Remove fields irrelevant to user questions
                    self.logger.log(f"  Step 2/3: Pruning irrelevant fields for '{name}' (based on question examples)...")
                    pruned_template = RelevancePruningInstructionProposalSignature.run(
                        lm=self.reflection_lm,
                        input_dict={
                            "current_instruction_doc": deduplicated_template,
                            "dataset_with_feedback": dataset_with_feedback,
                            "prompt_template": None,  # Use default pruning template
                        },
                    )["new_instruction"]
                    self.logger.log(f"  Step 2/3 complete: Pruned template length = {len(pruned_template)} chars")

                    # Step 3: Final Mutation - Optimize instruction with feedback
                    self.logger.log(f"  Step 3/3: Final mutation for '{name}' (optimizing with feedback)...")
                    new_texts[name] = InstructionProposalSignature.run(
                        lm=self.reflection_lm,
                        input_dict={
                            "current_instruction_doc": pruned_template,  # Use pruned template as base
                            "dataset_with_feedback": dataset_with_feedback,
                            "prompt_template": self.reflection_prompt_template,
                            "other_api_context": other_api_context,
                            "component_name": name,
                        },
                    )["new_instruction"]
                    self.logger.log(f"  Step 3/3 complete: Final instruction length = {len(new_texts[name])} chars")
                    self.logger.log(f"✅ Component '{name}' initialized successfully (3-step pipeline complete)")
                else:
                    # For initialized API: Use two-step mutation (deduplication + mutation)
                    self.logger.log(f"Component '{name}' already has API info.")
                    other_api_context = self._build_other_api_context(candidate, name)
                    dataset_with_feedback = reflective_dataset[name]
                    
                    # Step 1: Deduplication - Remove redundant information from current instruction
                    self.logger.log(f"  Step 1: Deduplicating existing instruction for '{name}' based on other API context...")
                    deduplicated_instruction = base_instruction
                    if other_api_context:
                        deduplicated_instruction = DeduplicationInstructionProposalSignature.run(
                            lm=self.reflection_lm,
                            input_dict={
                                "current_instruction_doc": base_instruction,
                                "other_api_context": other_api_context,
                                "prompt_template": None,  # Use default deduplication template
                            },
                        )["new_instruction"]
                        self.logger.log(f"  Deduplication complete for '{name}'.")
                    else:
                        self.logger.log(f"  No other API context available, skipping deduplication for '{name}'.")
                    
                    # Step 2: Mutation - Optimize with feedback data
                    self.logger.log(f"  Step 2: Mutating deduplicated instruction for '{name}' with feedback data...")
                    new_texts[name] = InstructionProposalSignature.run(
                        lm=self.reflection_lm,
                        input_dict={
                            "current_instruction_doc": deduplicated_instruction,  # Use deduplicated instruction as base
                            "dataset_with_feedback": dataset_with_feedback,
                            "prompt_template": self.reflection_prompt_template,  # Will be set in run_with_preprocessing
                            "other_api_context": other_api_context,
                            "component_name": name,
                        },
                    )["new_instruction"]
                    self.logger.log(f"Component '{name}' mutated (with deduplication).")
            else:
                # For system_prompt and other modules: Normal LLM mutation with API context
                # Build context from all enabled API modules
                other_api_context = self._build_other_api_context(candidate, name)
                dataset_with_feedback = reflective_dataset[name]
                new_texts[name] = InstructionProposalSignature.run(
                    lm=self.reflection_lm,
                    input_dict={
                        "current_instruction_doc": base_instruction,
                        "dataset_with_feedback": dataset_with_feedback,
                        "prompt_template": self.reflection_prompt_template,
                        "other_api_context": other_api_context,  # Now includes API context
                        "component_name": name,
                    },
                )["new_instruction"]
        
        return new_texts
    
    def _build_other_api_context(self, candidate: dict[str, str], current_module: str) -> str:
        """
        Build context string containing information about other modules.
        
        For API modules: Shows other enabled APIs (to avoid redundancy)
        For system_prompt: Shows all enabled APIs (to be aware of available information)
        
        Args:
            candidate: Current candidate with all module information
            current_module: The module currently being updated (e.g., "API_information1", "system_prompt")
            
        Returns:
            Formatted string describing other modules
        """
        other_apis = []
        
        # Determine if we're updating an API module or system_prompt
        is_api_module = current_module.startswith("API_information")
        
        for key in sorted(candidate.keys()):
            if key.startswith("API_information"):
                # For API modules: exclude the current one
                # For system_prompt: include all APIs
                if is_api_module and key == current_module:
                    continue
                
                api_text = candidate[key].strip()
                # Only include enabled APIs (not the unfavorable message)
                if api_text and api_text != "The information is unfavorable for the current question and will not be provided.":
                    other_apis.append(f"- {key}: {api_text}")
        
        if not other_apis:
            return ""
        
        # Different context messages for API vs system_prompt
        if is_api_module:
            context = "**OTHER AVAILABLE APIs** (for reference - avoid redundancy with these):\n"
        else:
            context = "**AVAILABLE API MODULES** (you can reference these in your instructions):\n"
        
        context += "\n".join(other_apis)
        context += "\n"

        return context

    def _select_node_and_module(self, state: GEPAState, valid_candidates=None) -> Tuple[MCTSNode, str]:
        """Core logic for selecting node and module in MCTS mode."""
        candidate_node, module_to_update = self.mcts_tree.select_with_puct(
            component_selector=self.module_selector,
            valid_candidates=valid_candidates
        )

        # Check whether to force a system_prompt update (accounting for prior failures)
        if should_force_system_prompt_update(candidate_node):
            module_to_update = "system_prompt"

        return candidate_node, module_to_update

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        """Core method driving prompt mutation, supporting both MCTS and traditional mode."""
        i = state.i + 1

        if self.mcts_tree is not None:
            # MCTS mode
            return self._propose_mcts_mode(state, i)
        else:
            # Traditional mode
            return self._propose_traditional_mode(state, i)

    def _propose_mcts_mode(self, state: GEPAState, i: int) -> CandidateProposal | None:
        """Propose logic for MCTS mode."""
        # 1) Use MCTS to select node and module
        selected_node, module_to_update = self._select_node_and_module(state, valid_candidates=state.program_candidates)
        curr_prog = selected_node.candidate

        # Find the corresponding program_id (for compatibility)
        curr_prog_id = None

        # First, try reference comparison (fastest)
        for idx, candidate in enumerate(state.program_candidates):
            if candidate is curr_prog:
                curr_prog_id = idx
                break

        # If reference comparison fails, try content comparison
        if curr_prog_id is None:
            for idx, candidate in enumerate(state.program_candidates):
                if candidate == curr_prog:
                    curr_prog_id = idx
                    break

        # If still not found, log details and skip
        if curr_prog_id is None:
            self.logger.log(f"Iteration {i}: Could not find program ID for selected candidate.")
            self.logger.log(f"  Selected candidate ID: {id(curr_prog)}")
            self.logger.log(f"  Candidates in pool: {len(state.program_candidates)}")
            self.logger.log(f"  Candidate IDs in pool: {[id(c) for c in state.program_candidates[:5]]}...")  # Show first 5 only
            self.logger.log(f"  MCTS tree nodes: {len(self.mcts_tree.nodes_map)}")
            return None

        state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
        self.logger.log(
            f"Iteration {i}: MCTS selected program {curr_prog_id} (generation {selected_node.generation}) score: {state.program_full_scores_val_set[curr_prog_id]}, modules: {module_to_update}"
        )

        self.experiment_tracker.log_metrics({"iteration": i, "selected_program_candidate": curr_prog_id}, step=i)

        # 2) Sample minibatch from training set
        subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        minibatch = self.trainset.fetch(subsample_ids)

        # 3) Evaluate current program with traces
        eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
        state.total_num_evals += len(subsample_ids)
        state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores

        if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
            self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
            return None

        if self.skip_perfect_score and all(s >= self.perfect_score for s in eval_curr.scores):
            self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
            return None

        self.experiment_tracker.log_metrics({"subsample_score": sum(eval_curr.scores)}, step=i)

        # 4) Use the MCTS-selected module (converted to a list)
        predictor_names_to_update = [module_to_update]
        self.logger.log(f"Iteration {i}: MCTS selected module to update: {predictor_names_to_update}")

        # 5) Build reflective dataset and propose new texts
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(curr_prog, eval_curr, predictor_names_to_update)
            new_texts = self.propose_new_texts(
                curr_prog,
                reflective_dataset,
                predictor_names_to_update
            )
            for pname, text in new_texts.items():
                self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text[:200]}...")
            self.experiment_tracker.log_metrics(
                {f"new_instruction_{pname}": text for pname, text in new_texts.items()}, step=i
            )
        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            import traceback
            self.logger.log(traceback.format_exc())

            # Don't return None - instead return current candidate unchanged
            self.logger.log(f"Iteration {i}: Returning current candidate unchanged, will try next module")
            return CandidateProposal(
                candidate=curr_prog,  # Return unchanged
                parent_program_ids=[curr_prog_id],
                subsample_indices=subsample_ids,
                subsample_scores_before=eval_curr.scores,
                subsample_scores_after=eval_curr.scores,  # Same scores
                tag="reflective_mutation_exception",
            )

        # 6) Generate new candidate and evaluate it
        final_candidate = curr_prog.copy()
        final_eval_scores = None

        for pname in predictor_names_to_update:
            if pname.startswith("API_information"):
                # API module
                self.logger.log(f"Iteration {i}: Processing API module '{pname}'")

                # Get current API value from the program
                current_api_value = curr_prog[pname]

                # Check if API was originally uninitialized
                is_uninitialized = (
                    current_api_value.strip() == "The information is unfavorable for the current question and will not be provided."
                    or current_api_value.strip() == ""
                )

                if is_uninitialized:
                    # Case 1: API was uninitialized, now has template
                    # Simply add the template API (engine will compare with original)
                    self.logger.log(f"  - Case: Uninitialized API. Adding template API")

                    if pname not in new_texts:
                        self.logger.log(f"  - No mutation generated for '{pname}', skipping")
                        continue

                    template_api = new_texts[pname]
                    final_candidate[pname] = template_api
                    self.logger.log(f"  - Added template API to candidate")

                else:
                    # Case 2: API already had information, now mutated
                    # Here we need to compare TWO directions: (mutated API) vs (no API)
                    # Choose the better one as the new candidate
                    # engine.py will then compare this new candidate with the original
                    self.logger.log(f"  - Case: API already has info. Comparing mutated API vs no API")

                    if pname not in new_texts:
                        self.logger.log(f"  - No mutation generated for '{pname}', skipping")
                        continue

                    mutated_api = new_texts[pname]

                    # Direction A: Mutated API
                    candidate_mutated = final_candidate.copy()
                    candidate_mutated[pname] = mutated_api

                    eval_mutated = self.adapter.evaluate(minibatch, candidate_mutated, capture_traces=False)
                    state.total_num_evals += len(subsample_ids)
                    score_mutated = sum(eval_mutated.scores)

                    # Direction B: No API
                    candidate_no_api = final_candidate.copy()
                    candidate_no_api[pname] = "The information is unfavorable for the current question and will not be provided."

                    eval_no_api = self.adapter.evaluate(minibatch, candidate_no_api, capture_traces=False)
                    state.total_num_evals += len(subsample_ids)
                    score_no_api = sum(eval_no_api.scores)

                    self.logger.log(f"  - Mutated API score: {score_mutated}")
                    self.logger.log(f"  - No API score: {score_no_api}")

                    # Select better direction as new candidate
                    if score_mutated >= score_no_api:
                        final_candidate[pname] = mutated_api
                        final_eval_scores = eval_mutated.scores
                        self.logger.log(f"  - Selected: Mutated API (better than no API)")
                        self.experiment_tracker.log_metrics({
                            f"{pname}_direction": "mutated_api",
                            f"{pname}_score_diff": score_mutated - score_no_api
                        }, step=i)
                    else:
                        final_candidate[pname] = "The information is unfavorable for the current question and will not be provided."
                        final_eval_scores = eval_no_api.scores
                        self.logger.log(f"  - Selected: No API (better than mutated)")
                        self.experiment_tracker.log_metrics({
                            f"{pname}_direction": "no_api",
                            f"{pname}_score_diff": score_no_api - score_mutated
                        }, step=i)

            elif pname in new_texts:
                # Non-API module (e.g., system_prompt): Simple mutation
                # Just apply the mutation, engine will compare with original
                mutated_text = new_texts[pname]
                final_candidate[pname] = mutated_text
                self.logger.log(f"Iteration {i}: Applied mutation to non-API module '{pname}'")

        # Evaluate the final candidate if not already evaluated
        if final_eval_scores is None:
            eval_final = self.adapter.evaluate(minibatch, final_candidate, capture_traces=False)
            state.total_num_evals += len(subsample_ids)
            final_eval_scores = eval_final.scores

        new_sum = sum(final_eval_scores)
        old_sum = sum(eval_curr.scores)
        self.logger.log(f"Iteration {i}: New candidate score: {new_sum}, Original score: {old_sum}")
        self.experiment_tracker.log_metrics({"new_candidate_score": new_sum, "original_score": old_sum}, step=i)

        # Return proposal with new candidate and scores
        # engine.py will compare and decide whether to accept
        proposal = CandidateProposal(
            candidate=final_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=subsample_ids,
            subsample_scores_before=eval_curr.scores,
            subsample_scores_after=final_eval_scores,
            tag="reflective_mutation_mcts",
            metadata={
                "parent_node": selected_node,
                "modules": predictor_names_to_update
            }
        )
        return proposal

    def _propose_traditional_mode(self, state: GEPAState, i: int) -> CandidateProposal | None:
        """Propose logic for traditional (round-robin) mode."""
        # 1) Select candidate to mutate
        curr_prog_id = self.candidate_selector.select_candidate_idx(state)
        curr_prog = state.program_candidates[curr_prog_id]
        state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
        self.logger.log(
            f"Iteration {i}: Selected program {curr_prog_id} score: {state.program_full_scores_val_set[curr_prog_id]}"
        )

        self.experiment_tracker.log_metrics({"iteration": i, "selected_program_candidate": curr_prog_id}, step=i)

        # 2) Sample minibatch from training set
        subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        minibatch = self.trainset.fetch(subsample_ids)

        # 3) Evaluate current program with traces
        eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
        state.total_num_evals += len(subsample_ids)
        state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores

        if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
            self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
            return None

        if self.skip_perfect_score and all(s >= self.perfect_score for s in eval_curr.scores):
            self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
            return None

        self.experiment_tracker.log_metrics({"subsample_score": sum(eval_curr.scores)}, step=i)

        # 4) Select module(s) to update (round-robin)
        predictor_names_to_update = self.module_selector(
            state, eval_curr.trajectories, eval_curr.scores, curr_prog_id, curr_prog
        )

        self.logger.log(f"Iteration {i}: Selected module(s) to update: {predictor_names_to_update}")

        # 5) Build reflective dataset and propose new texts
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(curr_prog, eval_curr, predictor_names_to_update)
            new_texts = self.propose_new_texts(
                curr_prog,
                reflective_dataset,
                predictor_names_to_update
            )
            for pname, text in new_texts.items():
                self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text[:200]}...")
            self.experiment_tracker.log_metrics(
                {f"new_instruction_{pname}": text for pname, text in new_texts.items()}, step=i
            )
        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            import traceback
            self.logger.log(traceback.format_exc())

            # Don't return None - instead return current candidate unchanged
            self.logger.log(f"Iteration {i}: Returning current candidate unchanged, will try next module")
            return CandidateProposal(
                candidate=curr_prog,  # Return unchanged
                parent_program_ids=[curr_prog_id],
                subsample_indices=subsample_ids,
                subsample_scores_before=eval_curr.scores,
                subsample_scores_after=eval_curr.scores,  # Same scores
                tag="reflective_mutation_exception",
            )

        # 6) Generate new candidate and evaluate it
        # Note: engine.py will compare new vs old and decide whether to accept
        # Here we only generate the new candidate and return its score
        final_candidate = curr_prog.copy()
        final_eval_scores = None

        for pname in predictor_names_to_update:
            if pname.startswith("API_information"):
                # API module
                self.logger.log(f"Iteration {i}: Processing API module '{pname}'")

                # Get current API value from the program
                current_api_value = curr_prog[pname]

                # Check if API was originally uninitialized
                is_uninitialized = (
                    current_api_value.strip() == "The information is unfavorable for the current question and will not be provided."
                    or current_api_value.strip() == ""
                )

                if is_uninitialized:
                # Case 1: API was uninitialized, now has template
                # Simply add the template API (engine will compare with original)
                    self.logger.log(f"  - Case: Uninitialized API. Adding template API")

                    if pname not in new_texts:
                        self.logger.log(f"  - No mutation generated for '{pname}', skipping")
                        continue

                    template_api = new_texts[pname]
                    final_candidate[pname] = template_api
                    self.logger.log(f"  - Added template API to candidate")

                else:
                    # Case 2: API already had information, now mutated
                    # Here we need to compare TWO directions: (mutated API) vs (no API)
                    # Choose the better one as the new candidate
                    # engine.py will then compare this new candidate with the original
                    self.logger.log(f"  - Case: API already has info. Comparing mutated API vs no API")

                    if pname not in new_texts:
                        self.logger.log(f"  - No mutation generated for '{pname}', skipping")
                        continue

                    mutated_api = new_texts[pname]

                    # Direction A: Mutated API
                    candidate_mutated = final_candidate.copy()
                    candidate_mutated[pname] = mutated_api

                    eval_mutated = self.adapter.evaluate(minibatch, candidate_mutated, capture_traces=False)
                    state.total_num_evals += len(subsample_ids)
                    score_mutated = sum(eval_mutated.scores)

                    # Direction B: No API
                    candidate_no_api = final_candidate.copy()
                    candidate_no_api[pname] = "The information is unfavorable for the current question and will not be provided."

                    eval_no_api = self.adapter.evaluate(minibatch, candidate_no_api, capture_traces=False)
                    state.total_num_evals += len(subsample_ids)
                    score_no_api = sum(eval_no_api.scores)

                    self.logger.log(f"  - Mutated API score: {score_mutated}")
                    self.logger.log(f"  - No API score: {score_no_api}")

                    # Select better direction as new candidate
                    if score_mutated >= score_no_api:
                        final_candidate[pname] = mutated_api
                        final_eval_scores = eval_mutated.scores
                        self.logger.log(f"  - Selected: Mutated API (better than no API)")
                        self.experiment_tracker.log_metrics({
                            f"{pname}_direction": "mutated_api",
                            f"{pname}_score_diff": score_mutated - score_no_api
                        }, step=i)
                    else:
                        final_candidate[pname] = "The information is unfavorable for the current question and will not be provided."
                        final_eval_scores = eval_no_api.scores
                        self.logger.log(f"  - Selected: No API (better than mutated)")
                        self.experiment_tracker.log_metrics({
                            f"{pname}_direction": "no_api",
                            f"{pname}_score_diff": score_no_api - score_mutated
                        }, step=i)

            elif pname in new_texts:
                # Non-API module (e.g., system_prompt): Simple mutation
                # Just apply the mutation, engine will compare with original
                mutated_text = new_texts[pname]
                final_candidate[pname] = mutated_text
                self.logger.log(f"Iteration {i}: Applied mutation to non-API module '{pname}'")

        # Evaluate the final candidate if not already evaluated
        if final_eval_scores is None:
            eval_final = self.adapter.evaluate(minibatch, final_candidate, capture_traces=False)
            state.total_num_evals += len(subsample_ids)
            final_eval_scores = eval_final.scores

        new_sum = sum(final_eval_scores)
        old_sum = sum(eval_curr.scores)
        self.logger.log(f"Iteration {i}: New candidate score: {new_sum}, Original score: {old_sum}")
        self.experiment_tracker.log_metrics({"new_candidate_score": new_sum, "original_score": old_sum}, step=i)

        # Return proposal with new candidate and scores
        # engine.py will compare and decide whether to accept
        return CandidateProposal(
            candidate=final_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=subsample_ids,
            subsample_scores_before=eval_curr.scores,
            subsample_scores_after=final_eval_scores,
            tag="reflective_mutation_roundrobin",
        )