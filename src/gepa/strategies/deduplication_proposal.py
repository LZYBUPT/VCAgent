import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from gepa.proposer.reflective_mutation.base import Signature

class DeduplicationInstructionProposalSignature(Signature):
    """
    Deduplication module: removes redundant information from the current API template
    based on the System Context.
    """

    # Deduplication prompt template
    deduplication_template = """You are a System Architect tasked with optimizing API documentation by removing redundant information.

I have a specific API template that generates descriptions. However, some information in this template overlaps with other existing APIs in the system. 
Your goal is to **remove** any information from the Target Template that is already better covered or explicitly defined in the "System Environment Context".

### System Environment Context (The "Reference")
The following APIs already exist. Information found here should NOT be repeated in the Target Template unless necessary for linking.
<api_context_block>
(No context provided)
</api_context_block>

### Target Template (The "Input" to be cleaned)
```
<curr_instructions>
```

### Constraints & Formatting Rules (CRITICAL)
1. **Immutable Anchor:** You MUST preserve the starting tag exactly (e.g., `{GO}:` or `{Drug}:`).
2. **Placeholder Integrity:** - You **MAY** delete a placeholder if the sentence is redundant.
   - You **MUST NOT** rename any placeholder.
   - You **MUST NOT** change the wrapping style (keep `{}` brackets).
3. **Redundancy Removal:** If the Context says "API_A provides gene location", and the Target Template contains "Located on chromosome {chr}", REMOVE that sentence.
4. **NO-OP Clause (Silence Rule):** **If no redundancies are found, simply output the original Target Template code block.**
   - **DO NOT** write sentences like "No changes made."
   - **DO NOT** explain why you preserved it.
   - **DO NOT** mention "NO-OP Clause" in your output.
   - Just output the code block.

### Example of Valid/Invalid Changes

**Input Target:** `{GO}: {symbol} is a gene. Location: {loc}.`

**Scenario A: Context has LocationAPI**
**✅ Valid Output:** `{GO}: {symbol} is a gene.`

**Scenario B: Context has NO LocationAPI (No overlap)**
**✅ Valid Output:** `{GO}: {symbol} is a gene. Location: {loc}.`
**❌ Invalid Output:** `{GO}: {symbol} is a gene. Location: {loc}. \n(Note: No overlap found.)`

### Your Task
Refine the Target Template. 
Output **ONLY** the final template string within ``` blocks.
**STRICTLY NO EXPLANATION OR CONVERSATIONAL FILLER.**

New Template:
```"""

    input_keys: ClassVar[list[str]] = ["current_instruction_doc", "other_api_context", "prompt_template"]
    output_keys: ClassVar[list[str]] = ["new_instruction"]

    @classmethod
    def prompt_renderer(cls, input_dict: Mapping[str, Any]) -> str:
        current_instruction = input_dict.get("current_instruction_doc")
        other_api_context = input_dict.get("other_api_context", "")
        # Allow user to provide a custom template; fall back to the default deduplication template
        prompt_template = input_dict.get("prompt_template") or cls.deduplication_template

        if not isinstance(current_instruction, str):
            raise TypeError("current_instruction_doc must be a string")

        # 1. Substitute the core instruction
        prompt = prompt_template.replace("<curr_instructions>", current_instruction)

        # 2. Handle the Context Block placeholder
        api_context_section = ""
        if other_api_context:
            api_context_section = f"""
### System Environment Context
The following information describes other APIs/Components available in the system.
Use this context to identify REDUNDANT information in the Target Template.
{other_api_context}

"""

        # Use regex for smart replacement of the context block
        if "<api_context_block>" in prompt:
            replacement = api_context_section if other_api_context else "No other API context provided. No deduplication based on context possible."
            prompt = re.sub(
                r'<api_context_block>.*?</api_context_block>',
                replacement,
                prompt,
                flags=re.DOTALL
            )

        return prompt

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        def extract_instruction_text() -> str:
            start = lm_out.find("```") + 3
            end = lm_out.rfind("```")
            if start >= end:
                stripped = lm_out.strip()
                if stripped.startswith("```"): return stripped.strip("`").strip()
                return stripped
            content = lm_out[start:end]
            # Strip optional language specifier (e.g., ```python)
            match = re.match(r"^\S*\n", content)
            if match:
                content = content[match.end() :]
            return content.strip()

        return {"new_instruction": extract_instruction_text()}
    




class RelevancePruningInstructionProposalSignature(Signature):
    """
    Relevance pruning module: removes fields in the API template that are irrelevant
    to the task, based on the provided input-output-feedback examples.
    """

    # Pruning prompt template based on question relevance
    pruning_template = """You are a System Architect tasked with optimizing API documentation.

Your goal is to prune **redundant metadata** from the Target Template while strictly preserving **core knowledge**.

Since API field names vary (e.g., `{summary}` vs `{desc}` vs `{comment}`), you must judge relevance based on the **Semantic Role** of the information in the sentence.

### 🧠 Semantic Judgment Logic (The "Brain")
Analyze each sentence/placeholder in the Target Template and classify it into one of these categories:

1.  **IDENTITY, DEFINITION & CONTEXT (🛡️ PROTECT AT ALL COSTS):**
    * **Identity:** Names, IDs, symbols, synonyms (`{symbol}`, `{id}`, `{aliases}`).
    * **Definition:** Description, summary, full name (`{description}`, `{summary}`).
    * **Biological Context (CRITICAL):** **Pathways**, **Functions**, **Disease Associations**, **Subcellular Locations**.
        * *Example:* `{reactome_pathways}`, `{go_terms}`, `{kegg_pathways}`.
    * *Action:* **KEEP IT**. Even if the user only asks "What is X?", knowing its pathways is part of "What it is".

2.  **CORE ATTRIBUTES (🛡️ KEEP UNLESS PROVEN USELESS):**
    * Does this field provide standard attributes? (e.g., location, molecular weight, pathway).
    * *Action:* **KEEP IT**, unless the User Questions specifically focus on a completely different aspect (e.g., only asking for IDs).

3.  **METADATA & LOGGING (✂️ PRUNE IF UNUSED):**
    * Is this administrative info? (e.g., data source names, version numbers, creation dates, author names, obscure database flags).
    * *Action:* **REMOVE** if the User Questions never ask about metadata.

### Usage Examples (User Intent)

```
<inputs_outputs_feedback>
```

### Target Template (The "Input" to be cleaned)
```
<curr_instructions>
```

### Constraints & Formatting Rules (CRITICAL)
1.  **Immutable Anchor:** You MUST preserve the starting tag exactly (e.g., `{GO}:` or `{Drug}:`).
2.  **Placeholder Integrity:** - You **MUST NOT** rename any placeholder.
    - You **MUST NOT** change the wrapping style (keep `{}` brackets).
3.  **Sentence Integrity:** If you remove a placeholder, **remove the surrounding words** that depend on it to keep the text grammatical.
    - *Bad:* `The gene is located at .` (Deleted `{loc}` but left the sentence hanging).
    - *Good:* `(Sentence removed)` (Deleted the whole irrelevant part).
4.  **Conservative Default:** If you are not 100% sure a field is "Metadata/Noise", **KEEP IT**.

### Example of Semantic Judgment

**Input:** `{Item}: {name} (ID: {uuid}). Created by {author} on {date}. Description: {narrative}.`

**Scenario: User asks "What is {name} and what does it do?"**
* `{name}` -> Identity -> **KEEP**
* `{uuid}` -> Identity -> **KEEP** (Good for reference)
* `{narrative}` -> Definition -> **KEEP** (Crucial for "what does it do")
* `{author}`, `{date}` -> Metadata -> **REMOVE** (User didn't ask about history)

**✅ Valid Output:** `{Item}: {name} (ID: {uuid}). Description: {narrative}.`

### Your Task
Refine the Target Template based on the Semantic Judgment.
Output **ONLY** the final template string within ``` blocks.
**STRICTLY NO EXPLANATION.**

New Template:
```"""

    input_keys: ClassVar[list[str]] = ["current_instruction_doc", "dataset_with_feedback", "prompt_template"]
    output_keys: ClassVar[list[str]] = ["new_instruction"]

    @classmethod
    def prompt_renderer(cls, input_dict: Mapping[str, Any]) -> str:
        current_instruction = input_dict.get("current_instruction_doc")
        dataset = input_dict.get("dataset_with_feedback")
        prompt_template = input_dict.get("prompt_template") or cls.pruning_template

        if not isinstance(current_instruction, str):
            raise TypeError("current_instruction_doc must be a string")
        if not isinstance(dataset, Sequence):
            raise TypeError("dataset_with_feedback must be a sequence")

        # --- Internal helper: format samples as markdown ---
        def format_samples(samples):
            def render_value(value, level=3):
                if isinstance(value, dict):
                    s = ""
                    for k, v in value.items():
                        s += f"{'#' * level} {k}\n"
                        s += render_value(v, min(level + 1, 6))
                    return s + ("\n" if not value else "")
                elif isinstance(value, list | tuple):
                    s = ""
                    for i, item in enumerate(value):
                        s += f"{'#' * level} Item {i + 1}\n"
                        s += render_value(item, min(level + 1, 6))
                    return s + ("\n" if not value else "")
                else:
                    return f"{str(value).strip()}\n\n"

            def convert_sample_to_markdown(sample, examplenum):
                s = f"# Example {examplenum}\n"
                for key, val in sample.items():
                    s += f"## {key}\n"
                    s += render_value(val, level=3)
                return s

            return "\n\n".join(convert_sample_to_markdown(sample, i + 1) for i, sample in enumerate(samples))
        # ------------------------------------------------

        # 1. Substitute the Target Template
        prompt = prompt_template.replace("<curr_instructions>", current_instruction)

        # 2. Substitute the Examples
        # Use only the first 5 samples to avoid token overflow
        dataset_subset = dataset[:5] if len(dataset) > 5 else dataset
        formatted_examples = format_samples(dataset_subset)
        prompt = prompt.replace("<inputs_outputs_feedback>", formatted_examples)

        return prompt

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        def extract_instruction_text() -> str:
            start = lm_out.find("```") + 3
            end = lm_out.rfind("```")
            
            if start >= end: # Handle case of a single ``` or no backticks at all
                stripped = lm_out.strip()
                if stripped.startswith("```"): return stripped.strip("`").strip()
                return stripped

            content = lm_out[start:end]
            # Strip optional language specifier (e.g., ```python)
            match = re.match(r"^\S*\n", content)
            if match:
                content = content[match.end() :]
            return content.strip()

        return {"new_instruction": extract_instruction_text()}