import os
import sys
import pandas as pd
from datasets import load_dataset
from typing import List, Dict, Any

# root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_crass(sample_size: int = None) -> List[Dict[str, Any]]:
    """Loads and standardizes the CRASS dataset.
    
    Returns a list of dicts with:
    - id: str
    - question: str (the counterfactual question string)
    - context: dict with premise and counterfactual strings
    - choices: dict with A, B, C, D text options
    - ground_truth: str (the correct letter)
    - domain: 'nl'
    """
    csv_path = os.path.join(config.DATA_DIR, "crass", "CRASS_FTM_main_data_set.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CRASS dataset not found at {csv_path}. Run download step.")
        
    df = pd.read_csv(csv_path, sep=';')
    
    # We only take rows which have defined premises and questions
    df = df.dropna(subset=['Premise', 'QCC', 'CorrectAnswer'])
    df = df.fillna('')
    
    standardized = []
    
    for idx, row in df.iterrows():
        choices_list = [
            str(row.get('CorrectAnswer', '')).strip(),
            str(row.get('Answer1', '')).strip(),
            str(row.get('Answer2', '')).strip(),
            str(row.get('PossibleAnswer3', '')).strip()
        ]
        
        # Filter out empty answers
        choices_list = [c for c in choices_list if c]
        if not choices_list: continue
        
        original_correct = choices_list[0]
        
        import random
        shuffled_choices = choices_list[:]
        random.seed(idx)
        random.shuffle(shuffled_choices)
        
        letters = ['A', 'B', 'C', 'D']
        choices_dict = {}
        gt = 'A'
        
        for i, c_str in enumerate(shuffled_choices):
            if c_str == original_correct:
                gt = letters[i]
            choices_dict[letters[i]] = c_str
            
        # Fill remaining choices if any
        for i in range(len(shuffled_choices), 4):
            choices_dict[letters[i]] = "N/A"
        
        standardized.append({
            'id': f"crass_{idx}",
            'question': str(row['QCC']),
            'context': {
                'premise': str(row['Premise']),
                'counterfactual': str(row['QCC']) # We'll just map QCC here
            },
            'choices': choices_dict,
            'ground_truth': gt,
            'domain': 'crass'
        })
        
    if sample_size and sample_size < len(standardized):
        import random
        # Seed for reproducibility in development
        random.seed(42)
        standardized = random.sample(standardized, sample_size)
        
    return standardized


def load_cruxeval(sample_size: int = None) -> List[Dict[str, Any]]:
    """Loads and standardizes the CRUXEval dataset (Output prediction).
    
    Returns a list of dicts with:
    - id: str
    - question: str (the code + input, requesting output)
    - context: dict with code function and input value
    - choices: None
    - ground_truth: str (the literal output)
    - domain: 'code'
    """
    # Load cruxeval from huggingface datasets
    ds = load_dataset("cruxeval-org/cruxeval")
    
    # Normally we evaluate on 'test' split or 'train' for baselines.
    # We will use the test split.
    test_ds = ds['test']
    
    standardized = []
    
    for i, item in enumerate(test_ds):
        # item has: 'code', 'input', 'output' (CRUXEval-O task)
        code = item['code']
        input_val = item['input']
        target_output = item['output']
        
        q_str = f"Given this code:\n{code}\nIf input is {input_val}, what is the output?"
        
        standardized.append({
            'id': f"crux_{item.get('id', i)}",
            'question': q_str,
            'context': {
                'code': code,
                'input': input_val
            },
            'choices': None,
            'ground_truth': target_output,
            'domain': 'cruxeval'
        })
        
    if sample_size and sample_size < len(standardized):
        import random
        random.seed(42)
        standardized = random.sample(standardized, sample_size)
        
    return standardized


def load_data(domain: str, sample_size: int = None) -> List[Dict[str, Any]]:
    """Main entrypoint to load data by domain."""
    if domain == "crass":
        return load_crass(sample_size)
    elif domain == "cruxeval":
        return load_cruxeval(sample_size)
    else:
        raise ValueError(f"Unknown domain {domain}. Expected 'crass' or 'cruxeval'.")
