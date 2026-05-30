"""
Evaluate the test set using a GPT model (without API).
Reads the optimized prompt template from the output folder and test data from the data folder,
then evaluates accuracy. Uses only optimized_system_prompt, without any API information.
"""

import json
import os
import sys
from typing import List, Dict
import litellm
from tqdm import tqdm

# # Configure API keys
# os.environ.pop("OPENAI_API_BASE", None)
# os.environ["OPENAI_BASE_URL"] = 
# os.environ["OPENAI_API_KEY"] = 

# Configuration - available models:
# "openai/gpt-5" - GPT-5 (if available)
# "openai/o1-preview" - O1 preview
# "openai/o1-mini" - O1 mini
# "openai/gpt-4o" - GPT-4o
# "openai/gpt-4o-mini" - GPT-4o mini (cheaper)
GPT_MODEL = "openai/deepseek-v3"

class GPTEvaluator:
    """GPT model evaluator - reads system and user content directly from test data"""
    
    def __init__(self, model: str):
        self.model = model
    
    def predict(self, system_content: str, user_content: str) -> str:
        """Generate a prediction using the GPT model"""
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
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


def evaluate_testset(evaluator: GPTEvaluator, test_data: List[Dict], output_file: str, save_interval: int = 100) -> dict:
    """
    Evaluate the test set (supports auto-saving and resumable evaluation).

    Args:
        evaluator: The evaluator instance
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

        system_content = None
        user_content = None
        expected_answer = None

        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            elif msg['role'] == 'user':
                user_content = msg['content']
            elif msg['role'] == 'assistant':
                expected_answer = msg['content']

        if not system_content or not user_content or not expected_answer:
            continue

        response = evaluator.predict(system_content, user_content)

        # Check correctness
        is_correct = evaluator.check_answer(expected_answer, response)
        if is_correct:
            correct += 1

        results.append({
            "system_content": system_content,
            "user_input": user_content,
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
    print(f"🔬 Baseline evaluation - using {GPT_MODEL}")
    print("="*80)

    # Configuration: only input and output paths needed
    test_file = "data/HOP62/DE_test.json"  # Test data file
    output_file = "results/baseline/HOP62_DE_baseline_result.json"  # Result output file
    save_interval = 100  # Save every 100 items (adjustable)

    print(f"\n📁 Configuration:")
    print(f"  - Model: {GPT_MODEL}")
    print(f"  - Test data: {test_file}")
    print(f"  - Mode: directly use original system + user content from test data")
    print(f"  - Output: {output_file}")
    print(f"  - Save interval: every {save_interval} items")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load data
    print(f"\n📥 Loading data...")
    test_data = load_test_data(test_file)
    print(f"  ✅ Loaded successfully: {len(test_data)} test items")

    # Create evaluator
    evaluator = GPTEvaluator(model=GPT_MODEL)

    # Run evaluation
    eval_results = evaluate_testset(evaluator, test_data, output_file, save_interval)

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
