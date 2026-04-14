import os
import sys
import time
import json
import argparse
from datetime import datetime
from tqdm import tqdm

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.load_data import load_data
from models.loader import load_model, unload_model
import evaluation.metrics as metrics

def get_prompt_template(domain: str, condition: str) -> str:
    """Loads the appropriate prompt template."""
    template_path = os.path.join(config.PROMPTS_DIR, f"{domain}_{condition}.txt")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

def run_condition(model, tokenizer, data, condition: str, domain: str, model_name: str, out_file: str) -> dict:
    # Get the prompt template
    template_str = get_prompt_template(domain, condition)
    
    results = []
    
    print(f"Running {model_name} on {domain} ({condition}) - {len(data)} examples")
    for item in tqdm(data):
        start_time = time.time()
        
        # Build prompt
        if domain == "crass":
            prompt = template_str.format(
                premise=item['context']['premise'],
                counterfactual=item['context']['counterfactual'],
                question=item['question'],
                choice_a=item['choices']['A'],
                choice_b=item['choices']['B'],
                choice_c=item['choices']['C'],
                choice_d=item['choices']['D']
            )
        else: # cruxeval
            prompt = template_str.format(
                code=item['context']['code'],
                input_val=item['context']['input']
            )
            
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        
        prompt_str = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            # Attempt 0 is deterministic (Greedy). Subsequent attempts use temperature sampling to force variant paths.
            use_sampling = (attempt > 0)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=use_sampling,
                temperature=0.7 if use_sampling else None,
                top_p=0.9 if use_sampling else None,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Only take the newly generated tokens
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            model_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            predicted = metrics.extract_answer(model_output, domain)
            
            # If the extraction script successfully parsed the logic, break the retry loop early!
            if predicted != "UNKNOWN":
                break
        
        gt = item['ground_truth']
        
        if domain == "cruxeval":
            # Loose matching for strings / code literals by stripping quotes symmetrically
            clean_gt = str(gt).strip("'\"").strip()
            clean_pred = predicted.strip("'\"").strip()
            correct = (clean_pred == clean_gt)
        else:
            correct = (predicted == gt)
        
        latency = (time.time() - start_time) * 1000
        
        results.append({
            "id": item["id"],
            "predicted": predicted,
            "ground_truth": gt,
            "correct": correct,
            "latency_ms": latency,
            "model_output": model_output
        })
        
        # CHECKPOINTING: Save state every 20 iterations to prevent catastrophic data loss
        if len(results) % 20 == 0:
            checkpoint_data = {
                "model": model_name,
                "domain": domain,
                "condition": condition,
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "metrics": metrics.compute_metrics(results)
            }
            metrics.save_results(checkpoint_data, out_file)
        
    final_metrics = metrics.compute_metrics(results)
    
    return {
        "model": model_name,
        "domain": domain,
        "condition": condition,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "metrics": final_metrics
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(config.MODELS.keys()))
    parser.add_argument("--domain", type=str, required=True, choices=config.BENCHMARKS)
    parser.add_argument("--sample", type=int, default=config.SAMPLE_SIZE_DEV)
    parser.add_argument("--condition", type=str, default="all", choices=["zero_shot", "cot", "all"])
    args = parser.parse_args()
    
    data = load_data(args.domain, sample_size=args.sample)
    
    model, tokenizer = load_model(args.model, quantize=(args.model == "qwen25_7b"))
    
    conditions_to_run = ["zero_shot", "cot"] if args.condition == "all" else [args.condition]
    
    for condition in conditions_to_run:
        run_folder = f"{args.model}_base"
        out_dir = os.path.join(config.RESULTS_DIR, run_folder)
        os.makedirs(out_dir, exist_ok=True)
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_file = os.path.join(out_dir, f"{args.domain}_{condition}_{timestamp_str}.json")
        
        # Execute run condition with localized out_file parameter for checkpointing
        res_data = run_condition(model, tokenizer, data, condition, args.domain, args.model, out_file)
        
        metrics.save_results(res_data, out_file)
        print(f"Results finalized to {out_file}. Final Metrics: {res_data['metrics']}")
        
    unload_model(model, tokenizer)
    print("Done.")
