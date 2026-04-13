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

def run_condition(model, tokenizer, data, condition: str, domain: str, model_name: str) -> dict:
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
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Only take the newly generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        model_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        predicted = metrics.extract_answer(model_output, domain)
        gt = item['ground_truth']
        
        # Simple exact match for code, or string equality for letters
        if domain == "cruxeval":
            # Very loose matching for strings / code literals
            correct = (predicted == str(gt) or predicted == repr(gt) or str(gt) in predicted)
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
        
    accuracy = metrics.compute_accuracy(results)
    
    return {
        "model": model_name,
        "domain": domain,
        "condition": condition,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(config.MODELS.keys()))
    parser.add_argument("--domain", type=str, required=True, choices=config.BENCHMARKS)
    parser.add_argument("--sample", type=int, default=config.SAMPLE_SIZE_DEV)
    args = parser.parse_args()
    
    # Run both zero_shot and cot for the baselines script
    data = load_data(args.domain, sample_size=args.sample)
    
    model, tokenizer = load_model(args.model, quantize=(args.model == "qwen25_7b"))
    
    for condition in ["zero_shot", "cot"]:
        res_data = run_condition(model, tokenizer, data, condition, args.domain, args.model)
        out_file = os.path.join(config.RESULTS_DIR, f"{args.model}_{args.domain}_{condition}.json")
        metrics.save_results(res_data, out_file)
        print(f"Results saved to {out_file}. Accuracy: {res_data['accuracy']:.2f}")
        
    unload_model(model, tokenizer)
    print("Done.")
