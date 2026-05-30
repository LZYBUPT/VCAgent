"""
Evaluate the test set using a GPT model.
Reads the optimized prompt template from the output folder and test data from the data folder,
then evaluates accuracy.
"""

import json
import os
import sys
import re
from typing import List, Dict
from collections import Counter
import litellm
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import API client
from API_use import api_client

# Configure API keys
# os.environ.pop("OPENAI_API_BASE", None)
# os.environ["OPENAI_BASE_URL"] = 
# os.environ["OPENAI_API_KEY"] = 
# Configuration
GPT_MODEL = "openai/deepseek-v3"  # can be changed to "openai/gpt-4o-mini" to reduce cost
ENABLE_API_CALLS = True  # Enable/disable API calls

class GPTEvaluator:
    """GPT model evaluator"""
    
    def __init__(self, model: str, enable_api_calls: bool = True):
        self.model = model
        self.api_client = api_client
        self.enable_api_calls = enable_api_calls
    
    def _extract_entity_from_input(self, user_input: str) -> dict:
        """
        Extract gene, cell line, or drug name from the user input.

        The data format is fixed as:
        "Does a drug perturbation of [drug] in [cell_line] cells cause ... expression of [gene] ..."
        """
        entities = {}

        # Extract drug name between "of " and " in "
        drug_match = re.search(r'of\s+(.+?)\s+in\s+', user_input)
        if drug_match:
            drug_name = drug_match.group(1).strip()
            # Remove any content in parentheses
            drug_name = re.sub(r'\s*\([^)]*\)', '', drug_name).strip()
            entities['drug'] = drug_name

        # Extract cell line name between " in " and " cells"
        cell_line_match = re.search(r'in\s+(.+?)\s+cells', user_input)
        if cell_line_match:
            entities['cell_line'] = cell_line_match.group(1).strip()

        # Extract gene symbol between "expression of " and the next space or question mark
        gene_match = re.search(r'expression of\s+([^\s?]+)', user_input)
        if gene_match:
            entities['gene'] = gene_match.group(1).strip()

        return entities
    
    def _format_api_data(self, api_data: dict, template: str, api_identifier: str) -> str:
        """Format API data by intelligently detecting and replacing placeholders in the template"""
        if not template:
            return str(api_data)

        try:
            # Remove the API identifier tag from the template
            result = template.replace(api_identifier, '')

            # Extract all placeholders used in the template
            placeholder_pattern = r'\{([^}]+)\}'
            placeholders = re.findall(placeholder_pattern, result)

            # Build a dict containing only the fields actually used in the template
            available_data = {}
            for placeholder in placeholders:
                if placeholder in api_data:
                    available_data[placeholder] = api_data[placeholder]
                else:
                    available_data[placeholder] = f"{{{placeholder}}}"

            # Format the template using the filtered data dict
            result = result.format(**available_data)
            return result
            
        except Exception as e:
            print(f"Warning: Error formatting template: {e}")
            return str(api_data)
    
    def _replace_api_placeholders(self, text: str, user_input: str) -> str:
        """Replace API placeholders with actual API call results"""
        if not self.enable_api_calls or not self.api_client:
            return text

        # Extract entities from user input
        entities = self._extract_entity_from_input(user_input)
        gene_symbol = entities.get('gene')
        cell_line = entities.get('cell_line')
        drug_name = entities.get('drug')
        
        result_text = text
        
        # Replace gene-related APIs
        if gene_symbol:
            if '{NCBI}' in result_text:
                try:
                    api_data = self.api_client.get_ncbi_gene_info(gene_symbol)
                    if api_data:
                        formatted_data = self._format_api_data(api_data, result_text, '{NCBI}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling NCBI API: {e}")
            
            if '{UniProt}' in result_text:
                try:
                    api_data = self.api_client.get_uniprot_info(gene_symbol)
                    if api_data:
                        formatted_data = self._format_api_data(api_data, result_text, '{UniProt}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling UniProt API: {e}")
            
            if '{Reactome}' in result_text:
                try:
                    api_data = self.api_client.get_reactome_pathways(gene_symbol)
                    if api_data:
                        formatted_data = self._format_api_data(api_data, result_text, '{Reactome}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling Reactome API: {e}")
            
            if '{KEGG}' in result_text:
                try:
                    api_data = self.api_client.get_kegg_info(gene_symbol)
                    if api_data:
                        formatted_data = self._format_api_data(api_data, result_text, '{KEGG}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling KEGG API: {e}")
            
            if '{Ensembl}' in result_text:
                try:
                    api_data = self.api_client.get_ensembl_info(gene_symbol)
                    if api_data:
                        formatted_data = self._format_api_data(api_data, result_text, '{Ensembl}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling Ensembl API: {e}")
        
        # Replace cell-line-related APIs
        if cell_line:
            if '{Cellosaurus}' in result_text:
                try:
                    api_data = self.api_client.get_cellosaurus_info(cell_line)
                    if api_data:
                        formatted_data = self._format_api_data(api_data, result_text, '{Cellosaurus}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling Cellosaurus API: {e}")
            
            if '{CCLE}' in result_text:
                try:
                    api_data = self.api_client.get_ccle_info(cell_line)
                    if api_data:
                        formatted_data = self._format_api_data(api_data, result_text, '{CCLE}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling CCLE API: {e}")
            
            if '{DepMap}' in result_text:
                try:
                    api_data = self.api_client.get_depmap_info(cell_line)
                    if api_data:
                        formatted_data = self._format_api_data(api_data, result_text, '{DepMap}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling DepMap API: {e}")
        
        # Replace drug-related APIs
        if drug_name:
            if '{PubChem}' in result_text:
                try:
                    api_data = self.api_client.get_pubchem_info(drug_name)
                    if api_data:
                        formatted_data = self._format_api_data(api_data, result_text, '{PubChem}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling PubChem API: {e}")
            
            if '{DrugBank}' in result_text:
                try:
                    api_data = self.api_client.get_drugbank_info(drug_name)
                    if api_data:
                        formatted_data = self._format_api_data(api_data, result_text, '{DrugBank}')
                        result_text = formatted_data
                except Exception as e:
                    print(f"Error calling DrugBank API: {e}")
        
        return result_text
    
    def build_prompt(self, candidate: dict, user_input: str) -> str:
        """Build the complete system prompt"""
        system_parts = []
        system_parts.append(candidate.get("optimized_system_prompt", ""))

        # Include API information (excluding unfavorable messages)
        excluded_messages = [
            "The information is unfavorable for the current question and will not be provided.",
        ]
        
        for i in range(1, 11):
            api_key = f"optimized_API_information{i}"
            if api_key in candidate:
                api_text = candidate[api_key].strip()
                if api_text and api_text not in excluded_messages:
                    system_parts.append(api_text)
        
        system_content = "\n\n".join(system_parts)
        
        processed_system_content = self._replace_api_placeholders(system_content, user_input)
        
        return processed_system_content
    
    def predict(self, candidate: dict, user_input: str) -> str:
        """Generate a prediction using the GPT model"""
        system_prompt = self.build_prompt(candidate, user_input)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                # Use default temperature (consistent with training in main code)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""
    
    def check_answer(self, expected_answer: str, response: str) -> bool:
        """
        Check whether the response contains the expected answer (consistent with the ContainsAnswerEvaluator logic in the main code).

        Args:
            expected_answer: Expected answer (e.g., "Up", "Down", "Yes", "No")
            response: Model's response

        Returns:
            True if the response contains the expected answer
        """
        return expected_answer in response


def load_optimized_prompt(prompt_file: str) -> dict:
    """Load the optimized prompt"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_test_data(test_file: str) -> List[Dict]:
    """Load test data"""
    with open(test_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_metrics(results: List[Dict]) -> dict:
    """
    Compute evaluation metrics: Precision, Recall, F1-score (per class).
    """
    # Collect all expected and predicted labels
    y_true = []
    y_pred = []

    for r in results:
        expected = r['expected']
        response = r['response']

        # Try to extract the predicted answer from the response
        predicted = None
        for answer_type in ['Up', 'Down', 'Yes', 'No']:
            if answer_type in response:
                predicted = answer_type
                break
        
        if predicted is None:
            predicted = "Unknown"
        
        y_true.append(expected)
        y_pred.append(predicted)
    
    # Get all unique class labels
    all_labels = sorted(set(y_true + y_pred))

    class_metrics = {}
    for label in all_labels:
        if label == "Unknown":
            continue
            
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == label)
        }
    
    # Compute macro average
    if class_metrics:
        macro_precision = sum(m['precision'] for m in class_metrics.values()) / len(class_metrics)
        macro_recall = sum(m['recall'] for m in class_metrics.values()) / len(class_metrics)
        macro_f1 = sum(m['f1'] for m in class_metrics.values()) / len(class_metrics)
    else:
        macro_precision = macro_recall = macro_f1 = 0
    
    # Compute weighted average
    total_support = sum(m['support'] for m in class_metrics.values())
    if total_support > 0:
        weighted_precision = sum(m['precision'] * m['support'] for m in class_metrics.values()) / total_support
        weighted_recall = sum(m['recall'] * m['support'] for m in class_metrics.values()) / total_support
        weighted_f1 = sum(m['f1'] * m['support'] for m in class_metrics.values()) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0
    
    return {
        "class_metrics": class_metrics,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1
        }
    }


def load_previous_progress(temp_output_file: str) -> tuple:
    """
    Load previous evaluation progress.

    Returns:
        (completed results list, processed count, correct count) or (None, 0, 0)
    """
    if not os.path.exists(temp_output_file):
        return None, 0, 0
    
    try:
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        if saved_data.get('status') == 'completed':
            # Already completed in a previous run, no need to resume
            return None, 0, 0
        
        results = saved_data.get('results', [])
        processed = saved_data.get('processed', 0)
        correct = saved_data.get('correct', 0)
        
        print(f"\n📂 Found previous progress: {processed} items processed, {correct} correct")
        user_input = input("Continue previous evaluation? (y/n): ").strip().lower()
        
        if user_input == 'y':
            return results, processed, correct
        else:
            print("Restarting evaluation...")
            return None, 0, 0
            
    except Exception as e:
        print(f"⚠️  Failed to load progress: {e}")
        return None, 0, 0


def evaluate_testset(evaluator: GPTEvaluator, candidate: dict, test_data: List[Dict], output_file: str, save_interval: int = 100) -> dict:
    """
    Evaluate the test set (supports auto-saving and resumable evaluation).

    Args:
        evaluator: The evaluator instance
        candidate: The optimized prompt template
        test_data: Test data list
        output_file: Path for the output results file
        save_interval: Save progress every N items (default: 100)
    """
    total = len(test_data)
    
    # Temporary file for incremental saves
    temp_output_file = output_file.replace('.json', '_temp.json')

    # Try to load previous progress
    previous_results, start_idx, correct = load_previous_progress(temp_output_file)

    if previous_results is not None:
        results = previous_results
        print(f"✅ Resuming from item {start_idx + 1}...")
    else:
        results = []
        start_idx = 0
        correct = 0

    print(f"\nStarting evaluation, {total} test items in total...")
    print(f"💾 Auto-save: every {save_interval} items to {temp_output_file}")

    # Progress bar starting from start_idx
    for idx in tqdm(range(start_idx, total), initial=start_idx, total=total, desc="Evaluation progress"):
        item = test_data[idx]
        messages = item['messages']
        user_msg = None
        expected_answer = None

        for msg in messages:
            if msg['role'] == 'user':
                user_msg = msg['content']
            elif msg['role'] == 'assistant':
                expected_answer = msg['content']

        if not user_msg or not expected_answer:
            continue

        response = evaluator.predict(candidate, user_msg)

        # Check correctness (using the same logic as the main code)
        is_correct = evaluator.check_answer(expected_answer, response)
        if is_correct:
            correct += 1

        results.append({
            "user_input": user_msg,
            "expected": expected_answer,
            "response": response,
            "correct": is_correct
        })

        current_processed = idx + 1

        # Save progress incrementally
        if current_processed % save_interval == 0 or current_processed == total:
            accuracy = correct / current_processed if current_processed > 0 else 0
            metrics = calculate_metrics(results)

            temp_results = {
                "status": "in_progress" if current_processed < total else "completed",
                "processed": current_processed,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "metrics": metrics,
                "results": results
            }

            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(temp_results, f, indent=2, ensure_ascii=False)

            print(f"\n💾 Progress saved: {current_processed}/{total} ({accuracy:.2%})")

    accuracy = correct / total if total > 0 else 0

    # Compute detailed metrics
    metrics = calculate_metrics(results)

    # Clean up temporary file
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)
        print(f"🗑️  Temporary file removed: {temp_output_file}")
    
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "metrics": metrics,
        "results": results
    }


def main():
    """Main function"""
    print("\n" + "="*80)
    print("🔬 Evaluating test set with GPT model")
    print("="*80)

    # Configure file paths
    prompt_file = r".\gepa\output\with_API\Hs 766T_DIR.json"  # Optimized prompt file
    test_file = r".\data\Hs 766T\DIR_test.json"  # Test data file
    output_file = r".\gepa\output\with_API\Hs 766T_DIR_ours_result.json"  # Result output file
    save_interval = 100  # Save every 100 items (adjustable)

    print(f"\n📁 Configuration:")
    print(f"  - Model: {GPT_MODEL}")
    print(f"  - Prompt file: {prompt_file}")
    print(f"  - Test data: {test_file}")
    print(f"  - API calls: {'Enabled' if ENABLE_API_CALLS else 'Disabled'}")
    print(f"  - Output: {output_file}")
    print(f"  - Save interval: every {save_interval} items")

    # Load data
    print(f"\n📥 Loading data...")
    candidate = load_optimized_prompt(prompt_file)
    test_data = load_test_data(test_file)
    print(f"  ✅ Loaded successfully: {len(test_data)} test items")

    # Create evaluator
    evaluator = GPTEvaluator(model=GPT_MODEL, enable_api_calls=ENABLE_API_CALLS)

    # Run evaluation
    eval_results = evaluate_testset(evaluator, candidate, test_data, output_file, save_interval)

    # Print results
    print("\n" + "="*80)
    print("📊 Evaluation Results")
    print("="*80)
    print(f"  Total samples: {eval_results['total']}")
    print(f"  Correct: {eval_results['correct']}")
    print(f"  Accuracy: {eval_results['accuracy']:.2%}")
    print("\n" + "-"*80)
    print("📈 Detailed Metrics")
    print("-"*80)

    # Show per-class metrics
    metrics = eval_results['metrics']
    print("\nPer-class metrics:")
    for label, m in metrics['class_metrics'].items():
        print(f"  {label}:")
        print(f"    - Precision: {m['precision']:.4f}")
        print(f"    - Recall:    {m['recall']:.4f}")
        print(f"    - F1-score:  {m['f1']:.4f}")
        print(f"    - Support:   {m['support']}")

    # Show average metrics
    print("\nMacro Average:")
    print(f"  - Precision: {metrics['macro_avg']['precision']:.4f}")
    print(f"  - Recall:    {metrics['macro_avg']['recall']:.4f}")
    print(f"  - F1-score:  {metrics['macro_avg']['f1']:.4f}")

    print("\nWeighted Average:")
    print(f"  - Precision: {metrics['weighted_avg']['precision']:.4f}")
    print(f"  - Recall:    {metrics['weighted_avg']['recall']:.4f}")
    print(f"  - F1-score:  {metrics['weighted_avg']['f1']:.4f}")
    print("="*80)

    # Save results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {output_file}")

    # Show some error samples
    errors = [r for r in eval_results['results'] if not r['correct']]
    if errors:
        print(f"\n❌ Error samples (first 5):")
        for i, error in enumerate(errors[:5], 1):
            print(f"\n  {i}. Input: {error['user_input'][:80]}...")
            print(f"     Expected answer: {error['expected']}")
            print(f"     Model response: {error['response'][:150]}...")
            print(f"     ⚠️  Response does not contain expected answer '{error['expected']}'")


if __name__ == "__main__":
    main()
