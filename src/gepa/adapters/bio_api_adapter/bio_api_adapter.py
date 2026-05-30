# Copyright (c) 2025 - Custom Bio API Adapter
# Adapter for biological API integration with GEPA optimization

import re
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol, TypedDict, cast

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

class BioAPIDataInst(TypedDict):
    input: str
    additional_context: dict[str, str]
    answer: str


class EvaluationResult(NamedTuple):
    score: float
    feedback: str
    objective_scores: dict[str, float] | None = None


class BioAPITrajectory(TypedDict):
    data: BioAPIDataInst
    full_assistant_response: str
    feedback: str


class BioAPIRolloutOutput(TypedDict):
    full_assistant_response: str


BioAPIReflectiveRecord = TypedDict(
    "BioAPIReflectiveRecord",
    {
        "Inputs": str,
        "Generated Outputs": str,
        "Feedback": str,
    },
)


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatCompletionCallable(Protocol):
    """Protocol for chat completion callables (duck typing for custom model wrappers)."""

    def __call__(self, messages: Sequence[ChatMessage]) -> str: ...


class Evaluator(Protocol):
    def __call__(self, data: BioAPIDataInst, response: str) -> EvaluationResult:
        """
        Evaluates a response and returns a score, feedback, and optional objective scores.
        """
        ...


class ContainsAnswerEvaluator:
    """Default evaluator that checks if the expected answer is contained in the response."""

    def __init__(self, failure_score: float = 0.0):
        self.failure_score = failure_score

    def __call__(self, data: BioAPIDataInst, response: str) -> EvaluationResult:
        is_correct = data["answer"] in response
        score = 1.0 if is_correct else self.failure_score

        if is_correct:
            feedback = f"The generated response is correct. The response include the correct answer '{data['answer']}'"
        else:
            additional_context_str = "\n".join(f"{k}: {v}" for k, v in data["additional_context"].items())
            feedback = (
                f"The generated response is incorrect. The correct answer is '{data['answer']}'. "
                "Ensure that the correct answer is included in the response exactly as it is."
            )
            if additional_context_str:
                feedback += f" Here is some additional context that might be helpful:\n{additional_context_str}"

        return EvaluationResult(score=score, feedback=feedback, objective_scores=None)


class BioAPIAdapter(GEPAAdapter[BioAPIDataInst, BioAPITrajectory, BioAPIRolloutOutput]):
    """
    Custom adapter that integrates biological API calls during evaluation.
    
    During training/mutation: APIs remain as text placeholders
    During testing/evaluation: API placeholders are replaced with actual API calls
    """
    
    # Mapping from API names to their placeholder tokens
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
    
    def __init__(
        self,
        model: str | ChatCompletionCallable,
        evaluator: Evaluator | None = None,
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] | None = None,
        api_client=None,  # API client from API_use.py
        enable_api_calls: bool = False,  # Set to True during testing, False during training
    ):
        if isinstance(model, str):
            import litellm
            self.litellm = litellm
        self.model = model
        self.evaluator = evaluator or ContainsAnswerEvaluator()
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs or {}
        self.api_client = api_client
        self.enable_api_calls = enable_api_calls
    def _extract_entity_from_input(self, user_input: str) -> dict[str, str]:
        """
        Extract gene, cell line, or drug name from user input.

        Data format is fixed as:
        "Does a drug perturbation of [drug] in [cell_line] cells cause ... expression of [gene] ..."

        Returns:
            Dictionary containing keys such as 'gene', 'cell_line', 'drug'
        """
        entities = {}
        
        # Extract drug name between "of " and " in "
        # Format: "of Goserelin (acetate) in" or "of Vortioxetine in"
        drug_match = re.search(r'of\s+(.+?)\s+in\s+', user_input)
        if drug_match:
            drug_name = drug_match.group(1).strip()
            # Remove parenthetical content, e.g. "(acetate)"
            drug_name = re.sub(r'\s*\([^)]*\)', '', drug_name).strip()
            entities['drug'] = drug_name

        # Extract cell line name between " in " and " cells"
        # Format: "in Hs 766T cells" or "in HepG2 cells"
        cell_line_match = re.search(r'in\s+(.+?)\s+cells', user_input)
        if cell_line_match:
            entities['cell_line'] = cell_line_match.group(1).strip()

        # Extract gene symbol between "expression of " and the next space or question mark
        # Format: "expression of ENSG00000254707?" or "expression of CP to"
        gene_match = re.search(r'expression of\s+([^\s?]+)', user_input)
        if gene_match:
            entities['gene'] = gene_match.group(1).strip()
        
        return entities

    def _extract_template_from_text(self, text: str, api_placeholder: str) -> str:
        """
        Extract the template from text containing the API placeholder.
        The entire text IS the template now (no "Template:" prefix anymore).
        
        Args:
            text: The full text containing the API placeholder and template
            api_placeholder: The API placeholder (e.g., "{NCBI}")
            
        Returns:
            The template string (which is the entire text)
        """
        return text
    
    def _format_api_data(self, api_data: dict, template: str, api_identifier: str) -> str:
        """
        Format API data using the provided template.
        Intelligently detects placeholders actually present in the template and only
        substitutes those that have corresponding values in api_data.
        
        Args:
            api_data: Dictionary containing API data
            template: Template string with placeholders
            api_identifier: API identifier like '{NCBI}', '{UniProt}', etc.
            
        Returns:
            Formatted string
        """
        if not template:
            # If no template is provided, return a simple dict description
            return str(api_data)

        try:
            # Step 1: Remove the API identifier (e.g. {NCBI}) since it is not a data field
            result = template.replace(api_identifier, '')

            # Step 2: Extract all placeholders from the template
            placeholder_pattern = r'\{([^}]+)\}'
            placeholders = re.findall(placeholder_pattern, result)

            # Step 3: Build a dictionary containing only fields actually used in the template
            available_data = {}
            for placeholder in placeholders:
                if placeholder in api_data:
                    available_data[placeholder] = api_data[placeholder]
                else:
                    # Placeholder not found in API data; keep it as-is
                    available_data[placeholder] = f"{{{placeholder}}}"

            # Step 4: Format the template using the filtered data dictionary
            result = result.format(**available_data)

            return result

        except KeyError as e:
            # Should not reach here in theory, since all placeholders have been handled
            print(f"Warning: Unexpected template key {e} not found in API data")
            return template.replace(api_identifier, '')
        except Exception as e:
            print(f"Warning: Error formatting template: {e}")
            return str(api_data)
    
    def _replace_api_placeholders(self, text: str, user_input: str) -> str:
        """
        Replace API placeholders like {NCBI}, {UniProt}, etc. with actual API call results if enabled.
        Otherwise, keep the text as is.
        
        New approach: API_use.py returns dict, we extract template from text and format the dict.
        """
        if not self.enable_api_calls or not self.api_client:
            return text
        # Extract entities from user input
        entities = self._extract_entity_from_input(user_input)
        gene_symbol = entities.get('gene')
        cell_line = entities.get('cell_line')
        drug_name = entities.get('drug')
        
        result_text = text
        
        # Replace gene-related API placeholders
        if gene_symbol:
            # NCBI
            if '{NCBI}' in result_text:
                try:
                    api_data = self.api_client.get_ncbi_gene_info(gene_symbol)
                    if api_data:
                        template = self._extract_template_from_text(result_text, '{NCBI}')
                        formatted_data = self._format_api_data(api_data, template, '{NCBI}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling NCBI API: {e}")
            
            # UniProt
            if '{UniProt}' in result_text:
                try:
                    api_data = self.api_client.get_uniprot_info(gene_symbol)
                    if api_data:
                        template = self._extract_template_from_text(result_text, '{UniProt}')
                        formatted_data = self._format_api_data(api_data, template, '{UniProt}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling UniProt API: {e}")
            
            # Reactome
            if '{Reactome}' in result_text:
                try:
                    api_data = self.api_client.get_reactome_pathways(gene_symbol)
                    if api_data:
                        template = self._extract_template_from_text(result_text, '{Reactome}')
                        formatted_data = self._format_api_data(api_data, template, '{Reactome}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling Reactome API: {e}")
            
            # KEGG
            if '{KEGG}' in result_text:
                try:
                    api_data = self.api_client.get_kegg_info(gene_symbol)
                    if api_data:
                        template = self._extract_template_from_text(result_text, '{KEGG}')
                        formatted_data = self._format_api_data(api_data, template, '{KEGG}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling KEGG API: {e}")
            
            # Ensembl
            if '{Ensembl}' in result_text:
                try:
                    api_data = self.api_client.get_ensembl_info(gene_symbol)
                    if api_data:
                        template = self._extract_template_from_text(result_text, '{Ensembl}')
                        formatted_data = self._format_api_data(api_data, template, '{Ensembl}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling Ensembl API: {e}")
        
        # Replace cell line-related API placeholders
        if cell_line:
            # Cellosaurus
            if '{Cellosaurus}' in result_text:
                try:
                    api_data = self.api_client.get_cellosaurus_info(cell_line)
                    if api_data:
                        template = self._extract_template_from_text(result_text, '{Cellosaurus}')
                        formatted_data = self._format_api_data(api_data, template, '{Cellosaurus}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling Cellosaurus API: {e}")
            
            # CCLE
            if '{CCLE}' in result_text:
                try:
                    api_data = self.api_client.get_ccle_info(cell_line)
                    if api_data:
                        template = self._extract_template_from_text(result_text, '{CCLE}')
                        formatted_data = self._format_api_data(api_data, template, '{CCLE}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling CCLE API: {e}")
            
            # DepMap
            if '{DepMap}' in result_text:
                try:
                    api_data = self.api_client.get_depmap_info(cell_line)
                    if api_data:
                        template = self._extract_template_from_text(result_text, '{DepMap}')
                        formatted_data = self._format_api_data(api_data, template, '{DepMap}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling DepMap API: {e}")
        
        # Replace drug-related API placeholders
        if drug_name:
            # PubChem
            if '{PubChem}' in result_text:
                try:
                    api_data = self.api_client.get_pubchem_info(drug_name)
                    if api_data:
                        template = self._extract_template_from_text(result_text, '{PubChem}')
                        formatted_data = self._format_api_data(api_data, template, '{PubChem}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling PubChem API: {e}")
            
            # DrugBank
            if '{DrugBank}' in result_text:
                try:
                    api_data = self.api_client.get_drugbank_info(drug_name)
                    if api_data:
                        template = self._extract_template_from_text(result_text, '{DrugBank}')
                        formatted_data = self._format_api_data(api_data, template, '{DrugBank}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling DrugBank API: {e}")
        
        return result_text

    def evaluate(
        self,
        batch: list[BioAPIDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[BioAPITrajectory, BioAPIRolloutOutput]:
        outputs: list[BioAPIRolloutOutput] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float] | None] = []
        trajectories: list[BioAPITrajectory] | None = [] if capture_traces else None

        # Construct system prompt from all components
        system_parts = []
        system_parts.append(candidate.get("system_prompt", ""))
        
        # Add API information (only include if it's not marked as "unfavorable")
        excluded_messages = [
            "The information is unfavorable for the current question and will not be provided.",
        ]
        # Build prompt from API information; only non-default (non-exclusion) messages are included
        for i in range(1, 11):  # Supports 10 APIs (API_information1 through API_information10)
            api_key = f"API_information{i}"
            if api_key in candidate:
                api_text = candidate[api_key].strip()
                # Only include API info if it's not an exclusion message
                if api_text and api_text not in excluded_messages:
                    system_parts.append(api_text)
        
        system_content = "\n\n".join(system_parts)

        litellm_requests = []

        for data in batch:
            user_content = f"{data['input']}"
            
            # Replace API placeholders if enabled
            processed_system_content = self._replace_api_placeholders(system_content, user_content)

            messages: list[ChatMessage] = [
                {"role": "system", "content": processed_system_content},
                {"role": "user", "content": user_content},
            ]

            litellm_requests.append(messages)

        if isinstance(self.model, str):
            responses = [
                resp.choices[0].message.content.strip() if resp.choices[0].message.content else "[FILTERED]"
                for resp in self.litellm.batch_completion(
                    model=self.model,
                    messages=litellm_requests,
                    max_workers=self.max_litellm_workers,
                    **self.litellm_batch_completion_kwargs,
                )
            ]
        else:
            responses = [self.model(messages) for messages in litellm_requests]

        for data, assistant_response in zip(batch, responses, strict=True):
            eval_result = self.evaluator(data, assistant_response)
            score = eval_result.score
            feedback = eval_result.feedback
            obj_scores = eval_result.objective_scores
            
            output: BioAPIRolloutOutput = {"full_assistant_response": assistant_response}

            outputs.append(output)
            scores.append(score)
            objective_scores.append(obj_scores)

            if trajectories is not None:
                trajectories.append(
                    {
                        "data": data,
                        "full_assistant_response": assistant_response,
                        "feedback": feedback,
                    }
                )

        objective_scores_arg: list[dict[str, float]] | None = None
        if objective_scores:
            all_none = all(x is None for x in objective_scores)
            all_not_none = all(x is not None for x in objective_scores)
            if not (all_none or all_not_none):
                raise ValueError("Objective scores must either be all None or all not None.")
            if all_not_none:
                objective_scores_arg = cast(list[dict[str, float]], objective_scores)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores_arg,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[BioAPITrajectory, BioAPIRolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Build reflective dataset for each component to update.
        """
        ret_d: dict[str, list[BioAPIReflectiveRecord]] = {}

        trajectories = eval_batch.trajectories
        assert trajectories is not None, "Trajectories are required to build a reflective dataset."

        for comp in components_to_update:
            items: list[BioAPIReflectiveRecord] = []
            
            for traj in trajectories:
                d: BioAPIReflectiveRecord = {
                    "Inputs": traj["data"]["input"],
                    "Generated Outputs": traj["full_assistant_response"],
                    "Feedback": traj["feedback"],
                }
                items.append(d)
            
            ret_d[comp] = items

        if len(ret_d) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d
