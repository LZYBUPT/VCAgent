# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from gepa.proposer.reflective_mutation.base import Signature


class InstructionProposalSignature(Signature):
    # Template for system_prompt modules
    system_prompt_template = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```
<api_context_block>
Subsequent updates will be posted here.

</api_context_block>

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Please note: Do not repeat the original content of task inputs and response.Do not disclose specific factual details from the examples; retain only generalisable strategies.

Provide the new instructions within ``` blocks."""

    # Template for API_information modules
    api_information_template = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Your task is to coordinate the relationship between {} and {}, thereby adapting the current instructions to the aforementioned task examples.

Keep the output format requirements in mind: The new instruction should *exactly* copy the API-calling string(s) from <curr_instructions> without any changes, deletions, or replacements. 

Please provide the answer directly without any leading text.

You may refer to this format, Example of the instruction before optimization:
```
{GO}: {symbol} is a gene with GO ID {go_id}, full name {description}, from {source}. Located on chromosome {chromosome}, map position {map_location}. Functional annotation: {summary}. Aliases: {aliases_summary}."
```

After optimisation:
```
{GO}: {symbol} is a gene identified by GO ID {go_id}, with the full name {description}, sourced from {source}. This gene is located on chromosome {chromosome}, with its map position at {map_location}. Its functional annotation includes its biological role, molecular function, cellular component, and associated pathways, as detailed in summary. This provides us with in-depth information about the gene's role in the cell, its potential regulatory mechanisms, and its interactions with other genes. When analyzing the effects of 9-ING-41 on Hs 766T cells, the functional details provided in summary will help us understand how the gene's expression changes under drug perturbation. For example, if the gene is involved in regulating cell growth or response mechanisms, the impact of 9-ING-41 on these pathways could directly or indirectly lead to the upregulation or downregulation of symbol. Therefore, understanding the common aliases mentioned in aliases_summary and their corresponding expression forms will further help determine how this gene's function changes across different cellular environments.
```
Provide the new instructions within ``` blocks."""


    input_keys: ClassVar[list[str]] = ["current_instruction_doc", "dataset_with_feedback", "prompt_template", "other_api_context", "component_name"]
    output_keys: ClassVar[list[str]] = ["new_instruction"]
    @classmethod
    def validate_prompt_template(cls, prompt_template: str | None) -> None:
        if prompt_template is None:
            return
        # Check for required placeholders
        required_placeholders = ("<curr_instructions>", "<inputs_outputs_feedback>")
        missing_placeholders = [
            placeholder
            for placeholder in required_placeholders
            if placeholder not in prompt_template
        ]
        if missing_placeholders:
            raise ValueError(
                f"Missing placeholder(s) in prompt template: {', '.join(missing_placeholders)}"
            )

    @classmethod
    def prompt_renderer(cls, input_dict: Mapping[str, Any]) -> str:
        current_instruction = input_dict.get("current_instruction_doc")
        if not isinstance(current_instruction, str):
            raise TypeError("current_instruction_doc must be a string")

        dataset = input_dict.get("dataset_with_feedback")
        if not isinstance(dataset, Sequence) or isinstance(dataset, (str, bytes)):
            raise TypeError("dataset_with_feedback must be a sequence of records")
        def format_samples(samples):# Render dataset samples as Markdown (input data is in JSON format and must be converted)
            def render_value(value, level=3):
                # level controls markdown header depth (###, ####, etc.)
                if isinstance(value, dict):
                    s = ""
                    for k, v in value.items():
                        s += f"{'#' * level} {k}\n"
                        s += render_value(v, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                elif isinstance(value, list | tuple):
                    s = ""
                    for i, item in enumerate(value):
                        s += f"{'#' * level} Item {i + 1}\n"
                        s += render_value(item, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                else:
                    return f"{str(value).strip()}\n\n"

            def convert_sample_to_markdown(sample, examplenum):
                s = f"# Example {examplenum}\n"
                for key, val in sample.items():
                    s += f"## {key}\n"
                    s += render_value(val, level=3)
                return s

            return "\n\n".join(convert_sample_to_markdown(sample, i + 1) for i, sample in enumerate(samples))

        prompt_template = input_dict.get("prompt_template")
        other_api_context = input_dict.get("other_api_context", "")
        component_name = input_dict.get("component_name", "")
        
        # Select appropriate template based on component type
        if prompt_template is None:
            # Determine module type from component_name
            if component_name.startswith("API_information"):
                # Use API-specific template for API modules
                prompt_template = cls.api_information_template
            else:
                # Use system prompt template for other modules (like system_prompt)
                prompt_template = cls.system_prompt_template

        cls.validate_prompt_template(prompt_template)

        prompt = prompt_template.replace("<curr_instructions>", current_instruction)
        # Only use first 10 samples to reduce token consumption
        dataset_subset = dataset[:5] if len(dataset) > 5 else dataset
        prompt = prompt.replace("<inputs_outputs_feedback>", format_samples(dataset_subset))
        
        # Handle the entire api_context_block
        if "<api_context_block>" in prompt_template:
            if other_api_context:
                # Keep the block and replace the placeholder with actual context
                api_context_section = """
### System Environment Context
The following information describes other APIs/Components available in the system. 
Use this context ONLY to understand the system's capabilities and boundaries. 
**CRITICAL:** Do NOT hardcode specific data from this context into the new instruction, as API data is dynamic.
```
""" + other_api_context + """
```
"""
                # Replace the entire block with the filled-in section
                prompt = re.sub(
                    r'<api_context_block>.*?</api_context_block>',
                    api_context_section,
                    prompt,
                    flags=re.DOTALL
                )
            else:
                # Remove the entire block if no context provided
                prompt = re.sub(
                    r'<api_context_block>.*?</api_context_block>',
                    '',
                    prompt,
                    flags=re.DOTALL
                )

        return prompt

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:# Extracts the code block content from the LM output and returns it as new_instruction
        def extract_instruction_text() -> str:
            # Find the first and last backtick positions (if any)
            start = lm_out.find("```") + 3
            end = lm_out.rfind("```")

            # Handle if the first and last backticks are the same or overlap
            if start >= end:
                # Handle incomplete blocks
                stripped = lm_out.strip()
                if stripped.startswith("```"):
                    # Remove opening ``` and optional language specifier
                    match = re.match(r"^```\S*\n?", lm_out)
                    if match:
                        return lm_out[match.end() :].strip()
                elif stripped.endswith("```"):
                    # Remove closing ```
                    return stripped[:-3].strip()
                return stripped

            # Skip optional language specifier
            content = lm_out[start:end]
            match = re.match(r"^\S*\n", content)
            if match:
                content = content[match.end() :]

            return content.strip()

        return {"new_instruction": extract_instruction_text()}