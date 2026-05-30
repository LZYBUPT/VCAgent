"""GEPA's key advantage is that any component can be replaced — any module can be externally specified."""
import gepa
import json
import random
from typing import List, Dict, Tuple
import os
import sys
from gepa.core.state import GEPAState

# Add API_use to path for importing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from API_use import api_client
from gepa.proposer.reflective_mutation.mcts_tree import MCTSTree
# Import custom adapter and selector
from gepa.adapters.bio_api_adapter import BioAPIAdapter
from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector,CorrelationBasedReflectionComponentSelector

from gepa.utils.stop_condition import PeriodicSaveCallback, MaxIterationsStopper, CompositeStopper

# ✅ only keep one base url
# os.environ.pop("OPENAI_API_BASE", None)
# os.environ["OPENAI_BASE_URL"] = 
# os.environ["OPENAI_API_KEY"] = 
# os.environ["HF_TOKEN"] = 
# ✅ ensure api key exists (you must set it externally)
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is missing!"
# Option 2: Direct OpenAI (uncomment if suyu.io doesn't work)
# os.environ["OPENAI_API_KEY"] = "your-openai-key-here"
# os.environ.pop("BASE_URL", None)  # Remove base_url for direct OpenAI

# Option 3: Other providers (examples)
# For DeepSeek: os.environ["OPENAI_API_KEY"] = "your-key"; os.environ["BASE_URL"] = "https://api.deepseek.com/v1"
# For Together AI: os.environ["OPENAI_API_KEY"] = "your-key"; os.environ["BASE_URL"] = "https://api.together.xyz/v1"
def load_psa_scatac_dataset(data_path: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load PSA SCATAC dataset and split into train/val/test sets."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle the data
    random.seed(55)
    random.shuffle(data)

    # Split into train/val/test (80/10/10)
    n_total = len(data)
    n_train = int(0.8 * n_total)
    n_val = int(0.2 * n_total)###########

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    return train_data, val_data, test_data

def convert_to_gepa_format(data: List[Dict]) -> List[Dict]:
    """Convert PSA SCATAC data to GEPA format."""
    gepa_data = []

    for item in data:
        messages = item['messages']
        user_msg = None
        assistant_msg = None

        for msg in messages:
            if msg['role'] == 'user':
                user_msg = msg['content']
            elif msg['role'] == 'assistant':
                assistant_msg = msg['content']

        if user_msg and assistant_msg:
            gepa_item = {
                'input': user_msg,
                'answer': assistant_msg,
                'additional_context': {}  # Required by GEPA's default adapter
            }
            gepa_data.append(gepa_item)

    return gepa_data

# Load PSA SCATAC dataset
_ROOT = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(_ROOT, "data", "C32", "DIR_train.json")
train_data, val_data, test_data = load_psa_scatac_dataset(data_path)
output_path = os.path.join(_ROOT, "output", "iteration100_ger", "C32_DIR.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
# Convert to GEPA format
trainset = convert_to_gepa_format(train_data)
valset = convert_to_gepa_format(val_data)

# Use the system prompt from the dataset as seed prompt
# Each API_information starts empty (not helpful message)
seed_prompt = {
    "system_prompt": "You are an expert who knows a lot about single cell biology and genomics and will help me solve a series of tasks related to single cell data analysis.",
    "API_information1": "The information is unfavorable for the current question and will not be provided.",
    "API_information2": "The information is unfavorable for the current question and will not be provided.",
    "API_information3": "The information is unfavorable for the current question and will not be provided.",
    "API_information4": "The information is unfavorable for the current question and will not be provided.",
    "API_information5": "The information is unfavorable for the current question and will not be provided.",
    "API_information6": "The information is unfavorable for the current question and will not be provided.",
    "API_information7": "The information is unfavorable for the current question and will not be provided.",
    "API_information8": "The information is unfavorable for the current question and will not be provided.",
    "API_information9": "The information is unfavorable for the current question and will not be provided.",
    "API_information10": "The information is unfavorable for the current question and will not be provided."
}

print(f"Loaded dataset: {len(trainset)} train samples, {len(valset)} val samples")

MAX_ITERATIONS = 100
MAX_METRIC_CALLS = 200000
SAVE_INTERVAL = 3

print(f"\n{'='*60}")
print(f"Run Configuration")
print(f"{'='*60}")
print(f"Max iterations: {MAX_ITERATIONS}")
print(f"Max metric calls: {MAX_METRIC_CALLS}")
print(f"Save interval: every {SAVE_INTERVAL} rounds")
print(f"{'='*60}\n")

# Create custom adapter with API integration
# During training: enable_api_calls=False (APIs remain as text)
# During final testing: set enable_api_calls=True (APIs will be called)
custom_adapter = BioAPIAdapter(
    model="openai/gpt-4o",
    api_client=api_client,
    enable_api_calls=True,  # Set to False during optimization, True during final testing
)

# Create round-robin module selector (selects modules in sequence)
custom_module_selector = CorrelationBasedReflectionComponentSelector()

# Create stopping conditions
periodic_save_callback = PeriodicSaveCallback(
    save_interval=SAVE_INTERVAL, 
    seed_prompt=seed_prompt,
    output_path=output_path
)
max_iterations_stopper = MaxIterationsStopper(max_iterations=MAX_ITERATIONS)

# Combine stopping conditions: stop when ANY condition is met
composite_stopper = CompositeStopper(
    periodic_save_callback,
    max_iterations_stopper,
    mode="any"
)

mcts_tree = MCTSTree(root_candidate=seed_prompt)
# Let's run GEPA optimization process with custom adapter and selector
gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    adapter=custom_adapter,  # Use custom adapter
    module_selector=custom_module_selector,  # Use GPT-based API selector
    max_metric_calls=MAX_METRIC_CALLS,
    stop_callbacks=composite_stopper,
    reflection_lm="openai/gpt-4o",
    use_merge=False,
    reflection_minibatch_size = 200,
    mcts_tree=mcts_tree,
    run_dir=os.path.join(_ROOT, "output", "iteration100_ger", "C32_DIR"),
    enable_checkpoints=True,
    checkpoint_every=3,
    keep_last_checkpoints=5,
    display_progress_bar=True,
)
optimized_prompt1 = gepa_result.best_candidate['system_prompt']
optimized_prompt2 = gepa_result.best_candidate['API_information1']
optimized_prompt3 = gepa_result.best_candidate['API_information2']
optimized_prompt4 = gepa_result.best_candidate['API_information3']
optimized_prompt5 = gepa_result.best_candidate['API_information4']
optimized_prompt6 = gepa_result.best_candidate['API_information5']
optimized_prompt7 = gepa_result.best_candidate['API_information6']
optimized_prompt8 = gepa_result.best_candidate['API_information7']
optimized_prompt9 = gepa_result.best_candidate['API_information8']
optimized_prompt10 = gepa_result.best_candidate['API_information9']
optimized_prompt11 = gepa_result.best_candidate['API_information10']
print("GEPA Optimized Prompt:", optimized_prompt1)
# Save all the optimized information
optimized_prompts = {
    "optimized_system_prompt": optimized_prompt1,
    "optimized_API_information1": optimized_prompt2,
    "optimized_API_information2": optimized_prompt3,
    "optimized_API_information3": optimized_prompt4,
    "optimized_API_information4": optimized_prompt5,
    "optimized_API_information5": optimized_prompt6,
    "optimized_API_information6": optimized_prompt7,
    "optimized_API_information7": optimized_prompt8,
    "optimized_API_information8": optimized_prompt9,
    "optimized_API_information9": optimized_prompt10,
    "optimized_API_information10": optimized_prompt11
}

# Combine the original and optimized prompts
output_data = {
    "original_system_prompt": seed_prompt["system_prompt"],
    "original_API_information1": seed_prompt["API_information1"],
    "original_API_information2": seed_prompt["API_information2"],
    "original_API_information3": seed_prompt["API_information3"],
    "original_API_information4": seed_prompt["API_information4"],
    "original_API_information5": seed_prompt["API_information5"],
    "original_API_information6": seed_prompt["API_information6"],
    "original_API_information7": seed_prompt["API_information7"],
    "original_API_information8": seed_prompt["API_information8"],
    "original_API_information9": seed_prompt["API_information9"],
    "original_API_information10": seed_prompt["API_information10"],
    
    **optimized_prompts,  # Include all optimized prompts here
    
    "dataset": "psa_scatac",
    "optimization_method": "gepa",
    "task_lm": "openai/gpt-4o-mini",  # Adapter model used
    "reflection_lm": "openai/gpt-4o",
    "max_metric_calls": 1000
}

# Save the optimized prompts to a JSON file
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Optimized prompt saved to: {output_path}")
