import json
import os
import re

def extract_answer(model_output: str, domain: str) -> str:
    """Extracts the final answer from the model output."""
    if domain == "crass":
        # For CRASS, we expect a single letter answer A, B, C, or D.
        # We can look for the last standalone letter A-D
        matches = re.findall(r'\b([A-D])\b', model_output.upper())
        if matches:
            return matches[-1]
        
        # Fallback: just look for the first occurrence of A, B, C, D in the response near the end
        for char in reversed(model_output.upper()):
            if char in ['A', 'B', 'C', 'D']:
                return char
        return "UNKNOWN"
        
    elif domain == "cruxeval":
        # For CRUXEval, we typically want the exact python literal
        # e.g., '15', 'True', '"hello"'
        # Given it's generative, let's grab the last line or what seems to be the answer
        lines = model_output.strip().split('\n')
        # Filter empty lines
        lines = [l for l in lines if l.strip()]
        if lines:
            ans = lines[-1].strip()
            # Often they format like "Answer: 15" or just "15"
            if "Answer:" in ans:
                ans = ans.split("Answer:")[-1].strip()
            return ans
        return "UNKNOWN"
    
    return "UNKNOWN"

def compute_accuracy(results: list) -> float:
    """Computes overall accuracy from a list of result dicts."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.get("correct", False))
    return correct / len(results)

def save_results(results_data: dict, filepath: str):
    """Saves the results dictionary to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)

def load_results(filepath: str) -> dict:
    """Loads results dictionary from JSON."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def compare_results(baseline: dict, pipeline: dict) -> dict:
    """Compares baseline and pipeline results, returning delta dict."""
    base_acc = baseline.get("accuracy", 0.0)
    pipe_acc = pipeline.get("accuracy", 0.0)
    
    return {
        "baseline_accuracy": base_acc,
        "pipeline_accuracy": pipe_acc,
        "delta": pipe_acc - base_acc
    }
