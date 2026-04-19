import os
import sys
import time
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.load_data import load_data
from models.loader import load_model, unload_model
import evaluation.metrics as metrics

def get_prompt_template(domain: str, condition: str) -> str:
    template_path = os.path.join(config.PROMPTS_DIR, f"{domain}_{condition}.txt")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def _batch_llm_call(model, tokenizer, prompts: List[str], max_new_tokens: int = 512) -> List[str]:
    if not prompts: return []
    valid_prompts = [p for p in prompts if p is not None]
    if not valid_prompts: return [None] * len(prompts)
    
    inputs = tokenizer(valid_prompts, return_tensors="pt", padding=True).to(model.device)
    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    results = []
    it = iter(decoded)
    for p in prompts:
        results.append(next(it) if p is not None else None)
    return results

def run_condition(
    model,
    tokenizer,
    data,
    condition: str,
    domain: str,
    model_name: str,
    out_file: str,
    batch_size: int = 8,
    ablation_config: dict = None,
) -> dict:
    from causal_slm.pipeline.agentic_loop import (
        AgentState, build_batch_confidence_prompts, build_batch_hypothesis_prompts,
        build_batch_update_prompts, _parse_confidence_scores, _apply_graph_updates,
        _sandbox_cruxeval, _sandbox_crass, CONFIDENCE_THRESHOLD, MAX_ITERATIONS,
        _build_crass_hypothesis_prompt, _build_cruxeval_hypothesis_prompt, _llm_call,
    )
    
    template_str = get_prompt_template(domain, condition)
    ablation_config = ablation_config or {}
    results = []
    
    print(f"Running BATCHED AGENTIC pipeline {model_name} on {domain} ({condition}) - {len(data)} examples")
    data = [item for item in data]
    for i in tqdm(range(0, len(data), batch_size)):
        batch_items = data[i : i + batch_size]
        start_time = time.time()
        
        # 1. Initialize Agent States and traces for the batch
        states = [AgentState(item, ablation_config) for item in batch_items]
        # Per-item trace for observability and later analysis
        traces = [
            {
                "item_id": item["id"],
                "initial_edges": [(e[0], e[2].get("relation","?"), e[1]) for e in s.edges],
                "iterations": [],
            }
            for item, s in zip(batch_items, states)
        ]
        
        # 2. STEP A: Batch Confidence Scoring
        conf_prompts = build_batch_confidence_prompts(states, condition)
        conf_max_tokens = 512 if condition == "cot" else 256
        conf_responses = _batch_llm_call(model, tokenizer, conf_prompts, max_new_tokens=conf_max_tokens)
        
        for s, trace, resp in zip(states, traces, conf_responses):
            s.scores = _parse_confidence_scores(resp, len(s.edges))
            # Record initial confidence scores in the trace
            trace["initial_scores"] = [
                {"edge": (e[0], e[2].get("relation","?"), e[1]), "score": sc}
                for e, sc in zip(s.edges, s.scores)
            ]
            trace["loop_triggered"] = any(sc < CONFIDENCE_THRESHOLD for sc in s.scores)
            
        # 3. AGENTIC LOOP (Synchronized)
        from causal_slm.pipeline.intervention import InterventionEngine
        engine = InterventionEngine()
        
        for iteration in range(MAX_ITERATIONS):
            # Check who is active
            active_states = [s for s in states if not s.finished and any(sc < CONFIDENCE_THRESHOLD for sc in s.scores)]
            if not active_states: break
            
            # STEP B+C: Sequential Hypothesis Generation + Sandbox
            # NOTE: hypothesis prompts vary greatly in length per item, causing
            # batch padding to collapse attention and produce empty strings.
            # We process only the active items one at a time.
            hyp_responses = [None] * len(states)
            observations  = [None] * len(states)
            for idx, s in enumerate(states):
                low_idx = next((i for i, sc in enumerate(s.scores) if sc < CONFIDENCE_THRESHOLD), None)
                s.active_low_idx = low_idx
                if low_idx is None or s.finished:
                    continue
                edge = s.edges[low_idx]
                src, tgt, edge_data = edge[0], edge[1], edge[2]
                rel = edge_data.get("relation", "relates_to")
                
                # Generate the hypothesis test (sequential single call)
                if s.domain == "cruxeval":
                    hyp_prompt = _build_cruxeval_hypothesis_prompt(
                        s.parsed_query.original_state.get("code", ""), src, tgt, rel)
                else:
                    hyp_prompt = _build_crass_hypothesis_prompt(
                        s.parsed_query.original_state.get("premise", ""), src, tgt, rel)
                
                test_resp = _llm_call(model, tokenizer, hyp_prompt, max_new_tokens=64).strip()
                hyp_responses[idx] = test_resp
                s.active_test_line = test_resp
                
                # Run the sandbox
                if s.domain == "cruxeval":
                    observations[idx] = _sandbox_cruxeval(
                        s.parsed_query.original_state.get("code", ""), test_resp, engine)
                else:
                    observations[idx] = _sandbox_crass(test_resp, model, tokenizer)

                    
            # STEP D: Batch Graph Updates
            update_prompts = build_batch_update_prompts(states, observations)
            update_responses = _batch_llm_call(model, tokenizer, update_prompts, max_new_tokens=256)
            
            for s, trace, hyp_resp, obs, upd_resp in zip(states, traces, hyp_responses, observations, update_responses):
                # Record what happened in this iteration
                iter_log = {
                    "iteration": iteration,
                    "low_confidence_edge_idx": s.active_low_idx,
                    "low_confidence_edge": (
                        (s.edges[s.active_low_idx][0], s.edges[s.active_low_idx][2].get("relation","?"), s.edges[s.active_low_idx][1])
                        if s.active_low_idx is not None and s.active_low_idx < len(s.edges) else None
                    ),
                    "hypothesis_test": hyp_resp,
                    "sandbox_observation": obs,
                }
                if upd_resp:
                    s.edges, s.scores = _apply_graph_updates(s.edges, s.scores, upd_resp)
                iter_log["edges_after_update"] = [
                    {"edge": (e[0], e[2].get("relation","?"), e[1]), "score": sc}
                    for e, sc in zip(s.edges, s.scores)
                ]
                trace["iterations"].append(iter_log)
                s.iteration += 1
                if s.iteration >= MAX_ITERATIONS: s.finished = True
                
        # 4. Finalize Contexts and Inject into Final Prompts
        final_prompts = []
        for s in states:
            causal_context = s.get_final_context()
            item = s.item
            if domain == "crass":
                base_prompt = template_str.format(
                    premise=item["context"]["premise"],
                    counterfactual=item["context"]["counterfactual"],
                    question=item["question"],
                    choice_a=item["choices"]["A"],
                    choice_b=item["choices"]["B"],
                    choice_c=item["choices"]["C"],
                    choice_d=item["choices"]["D"],
                )
                prompt = base_prompt.replace("Question:", f"{causal_context}\n\nQuestion:")
            else: # cruxeval
                base_prompt = template_str.format(
                    code=item["context"]["code"],
                    input_val=item["context"]["input"],
                )
                prompt = base_prompt.replace("If the input is", f"{causal_context}\n\nIf the input is")
                
            messages = [{"role": "user", "content": prompt}]
            final_prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
            
        # 5. Final Batch Answer Generation
        final_responses = _batch_llm_call(model, tokenizer, final_prompts, max_new_tokens=1024)
        
        latency_per_item = ((time.time() - start_time) * 1000) / len(batch_items)
        
        for idx, item in enumerate(batch_items):
            model_output = final_responses[idx]
            predicted = metrics.extract_answer(model_output, domain)
            gt = item["ground_truth"]
            
            if domain == "cruxeval":
                correct = str(predicted).strip("'\"") == str(gt).strip("'\"")
            else:
                correct = predicted == gt
            
            # Finalize the trace with the outcome
            trace = traces[idx]
            trace["final_edges"] = [
                {"edge": (e[0], e[2].get("relation","?"), e[1]), "score": sc}
                for e, sc in zip(states[idx].edges, states[idx].scores)
            ]
            trace["edges_pruned"] = len(trace["initial_edges"]) - len(states[idx].edges)
            trace["predicted"] = predicted
            trace["correct"] = correct
                
            results.append({
                "id": item["id"],
                "predicted": predicted,
                "ground_truth": gt,
                "correct": correct,
                "latency_ms": latency_per_item,
                "model_output": model_output,
                "injected_prompt": final_prompts[idx],
                "agentic_trace": trace,   # Full step-by-step cognitive log
            })
            
        # Checkpointing
        if len(results) % 20 < batch_size:
            checkpoint_data = {
                "model": model_name,
                "domain": domain,
                "condition": condition,
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "metrics": metrics.compute_metrics(results),
                "ablation": ablation_config,
                "pipeline": "agentic",
            }
            metrics.save_results(checkpoint_data, out_file)
            
    final_metrics = metrics.compute_metrics(results)
    return {
        "model": model_name, "domain": domain, "condition": condition,
        "timestamp": datetime.now().isoformat(), "results": results,
        "metrics": final_metrics, "ablation": ablation_config, "pipeline": "agentic"
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Causal Discovery Loop Evaluator")
    parser.add_argument("--model",     type=str, required=True, choices=list(config.MODELS.keys()))
    parser.add_argument("--domain",    type=str, required=True, choices=config.BENCHMARKS)
    parser.add_argument("--sample",    type=int, default=config.SAMPLE_SIZE_DEV)
    parser.add_argument("--batch_size",type=int, default=8, help="Batch size for synchronized agentic inference.")
    parser.add_argument("--condition", type=str, default="all", choices=["zero_shot", "cot", "all"])

    # Ablation flags — identical surface API to run_causal.py
    parser.add_argument("--no_interventions", action="store_true")
    parser.add_argument("--no_relationships",  action="store_true")
    parser.add_argument("--no_attributes",     action="store_true")

    args = parser.parse_args()

    ablation_config = {
        "disable_interventions": args.no_interventions,
        "disable_relationships":  args.no_relationships,
        "disable_attributes":     args.no_attributes,
    }

    data = load_data(args.domain, sample_size=args.sample)
    model, tokenizer = load_model(args.model, quantize=(args.model == "qwen25_7b"))

    conditions_to_run = ["zero_shot", "cot"] if args.condition == "all" else [args.condition]

    for condition in conditions_to_run:
        ablation_label = ""
        if args.no_interventions: ablation_label += "_no_int"
        if args.no_relationships:  ablation_label += "_no_rel"
        if args.no_attributes:     ablation_label += "_no_attr"

        run_folder = f"{args.model}_agentic{ablation_label}"
        out_dir    = os.path.join(config.RESULTS_DIR, run_folder)
        os.makedirs(out_dir, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(out_dir, f"{args.domain}_{condition}_{timestamp_str}.json")

        res_data = run_condition(
            model, tokenizer, data, condition, args.domain,
            args.model, out_file, args.batch_size, ablation_config
        )
        metrics.save_results(res_data, out_file)
        print(f"Results saved to {out_file}. Final Metrics: {res_data['metrics']}")

    unload_model(model, tokenizer)
    print("Done.")
