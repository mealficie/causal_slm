import ast
import io
import sys
import json
import contextlib
import traceback
import torch
from typing import Any, Dict, List, Optional, Tuple

from causal_slm.pipeline.parser import parse_query, ParsedQuery
from causal_slm.pipeline.graph_builder import CausalGraph
from causal_slm.pipeline.intervention import InterventionEngine
from causal_slm.pipeline.regenerator import CausalRegenerator

CONFIDENCE_THRESHOLD = 0.85
MAX_ITERATIONS       = 2


def _llm_call(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:

    messages = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

def _build_confidence_prompt(
    domain: str,
    parsed_query: ParsedQuery,
    graph_summary: dict,
    condition: str = "zero_shot",
) -> str:
    edges = graph_summary.get("edges", [])
    nodes = graph_summary.get("nodes", [])

    edges_str = "\n".join(
        f"  [{i}] {e[0]} --[{e[2].get('relation', '?')}]--> {e[1]}"
        for i, e in enumerate(edges)
    ) or "  (no edges extracted)"

    nodes_str = ", ".join(n[0] for n in nodes) or "(none)"

    if domain == "cruxeval":
        code_snippet = parsed_query.original_state.get("code", "")
        rubric = (
            "Classify each edge as PLAUSIBLE or SUSPECT:\n"
            "  PLAUSIBLE : The relationship directly follows from the code (e.g., direct variable assignment or function argument).\n"
            "  SUSPECT   : The relationship is ambiguous or cannot be confirmed from reading the code.\n"
        )
        context_block = f"CODE SNIPPET:\n```python\n{code_snippet}\n```\n"
    else:  # crass
        premise = parsed_query.original_state.get("premise", "")
        rubric = (
            "Classify each edge as PLAUSIBLE or SUSPECT:\n"
            "  PLAUSIBLE : The action is physically possible and consistent with the real world.\n"
            "  SUSPECT   : The action is physically impossible or absurd.\n"
        )
        context_block = f"PREMISE: \"{premise}\"\n"

    if condition == "cot":
        response_instruction = (
            "For each edge, give one sentence explaining your classification, then classify it.\n"
            "Respond ONLY with a JSON list in this exact format:\n"
            '[{"edge_idx": 0, "label": "SUSPECT", "reason": "branches cannot produce sound"}]\n'
            "If there are no edges, respond with: []\n"
            "JSON:"
        )
    else:
        response_instruction = (
            "Respond ONLY with a compact JSON list. No explanation outside the JSON.\n"
            '[{"edge_idx": 0, "label": "PLAUSIBLE"}]\n'
            "If there are no edges, respond with: []\n"
            "JSON:"
        )

    prompt = (
        f"You are a physical plausibility checker for causal graph edges.\n\n"
        f"{context_block}\n"
        f"EXTRACTED CAUSAL EDGES:\n{edges_str}\n\n"
        f"{rubric}\n"
        f"{response_instruction}"
    )
    return prompt


def _parse_confidence_scores(response: str, num_edges: int) -> List[float]:
    """
    Parse PLAUSIBLE/SUSPECT labels into scores.
    PLAUSIBLE → 1.0 (above threshold, no sandbox needed)
    SUSPECT   → 0.0 (below threshold, triggers sandbox)
    Fallback  → 0.0 (conservative: test anything we cannot parse)
    """
    try:
        clean = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        start = clean.find("[")
        end   = clean.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON list found")
        data = json.loads(clean[start:end])
        scores = [0.0] * num_edges  # conservative default: everything is SUSPECT until proven PLAUSIBLE
        for item in data:
            idx = item.get("edge_idx", -1)
            if 0 <= idx < num_edges:
                label = str(item.get("label", "SUSPECT")).strip().upper()
                scores[idx] = 1.0 if label == "PLAUSIBLE" else 0.0
        return scores
    except Exception:
        return [0.0] * num_edges  # conservative fallback: trigger sandbox for all


def _build_cruxeval_hypothesis_prompt(code: str, source: str, target: str, relation: str) -> str:
    return (
        f"You are a Python execution analyst.\n\n"
        f"CODE:\n```python\n{code}\n```\n\n"
        f"WEAK CAUSAL EDGE: `{source}` --[{relation}]--> `{target}`\n\n"
        f"Task: Write ONE executable Python line (e.g. a print statement or a simple mutation) "
        f"that directly tests whether `{source}` actually influences `{target}` in the code above.\n"
        f"The line must be self-contained and safe to exec().\n"
        f"Respond with ONLY the single Python line, nothing else."
    )


def _build_crass_hypothesis_prompt(premise: str, source: str, target: str, relation: str) -> str:
    return (
        f"Answer with only 'Yes' or 'No'.\n\n"
        f"Is it physically possible for a {source} to {relation} a {target}?\n\n"
        f"Answer:"
    )


def _sandbox_cruxeval(code: str, test_line: str, intervention_engine: InterventionEngine) -> str:
    observation = ""
    combined_code = f"{code}\n\n# Hypothesis test injected by Agentic Loop\n{test_line}"
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(compile(combined_code, "<agentic_test>", "exec"), {})  # isolated namespace
        output = stdout_capture.getvalue().strip()
        observation = f"EXECUTION OUTPUT: {output!r}" if output else "EXECUTION OUTPUT: (no output)"
    except Exception as exc:
        observation = f"EXECUTION ERROR: {type(exc).__name__}: {exc}"
    return observation


def _sandbox_crass(premise_question: str, model, tokenizer) -> str:
    probe_prompt = (
        f"Answer the following question with only 'Yes' or 'No'.\n\n"
        f"Question: {premise_question}\n\nAnswer:"
    )
    response = _llm_call(model, tokenizer, probe_prompt, max_new_tokens=16)
    return f"SANDBOX RESPONSE: {response}"

#feed back to model
def _build_graph_update_prompt(
    domain: str,
    edges: List,
    scores: List[float],
    tested_edge_idx: int,
    test_line: str,
    observation: str,
) -> str:
    edges_str = "\n".join(
        f"  [{i}] {e[0]} --[{e[2].get('relation', '?')}]--> {e[1]}"
        for i, e in enumerate(edges)
    )
    return (
        f"You are a causal graph editor. Decide whether to KEEP or REMOVE edge [{tested_edge_idx}].\n\n"
        f"CAUSAL EDGES:\n{edges_str}\n\n"
        f"EVIDENCE:\n"
        f"  Question: {test_line}\n"
        f"  Answer:   {observation}\n\n"
        f"DECISION RULES:\n"
        f"  Answer is 'Yes' → the action is physically possible → KEEP the edge.\n"
        f"  Answer is 'No'  → the action is physically impossible → REMOVE the edge.\n\n"
        f"Respond ONLY with a JSON list for edge [{tested_edge_idx}]:\n"
        f'[{{"edge_idx": {tested_edge_idx}, "keep": true, "reason": "action is physically possible"}}]\n'
        f"JSON:"
    )


def _apply_graph_updates(edges: List, scores: List[float], response: str) -> Tuple[List, List[float]]:
    """Parse the graph update response and prune / rescore edges accordingly."""
    try:
        clean = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        start = clean.find("[")
        end   = clean.rfind("]") + 1
        data  = json.loads(clean[start:end])
        new_edges  = []
        new_scores = []
        update_map = {item["edge_idx"]: item for item in data if "edge_idx" in item}
        for i, (edge, score) in enumerate(zip(edges, scores)):
            upd = update_map.get(i)
            if upd and not upd.get("keep", True):
                continue  # Prune the edge
            # If keep=True (or not mentioned), mark as confirmed (0.95); otherwise keep existing score
            new_score = 0.95 if (upd and upd.get("keep", True)) else score
            new_edges.append(edge)
            new_scores.append(new_score)
        return new_edges, new_scores
    except Exception:
        return edges, scores  # Failsafe: keep original


class AgentState:
    """Maintains the per-item state for the batch orchestrator."""
    def __init__(self, item: dict, ablation_config: dict = None):
        self.item = item
        self.domain = item.get("domain", "crass")
        self.ablation_config = ablation_config or {}
        
        # Component 1: Parse
        self.parsed_query = parse_query(item, self.domain)
        # Apply ablation filters
        if self.ablation_config.get("disable_attributes"):
            self.parsed_query.interventions = [i for i in self.parsed_query.interventions if i["type"] not in ("property_addition", "entity_substitution")]
        if self.ablation_config.get("disable_relationships"):
            self.parsed_query.interventions = [i for i in self.parsed_query.interventions if i["type"] != "relationship_shift"]
            
        # Component 2 & 3: Build and Mutate
        # We run interventions immediately to get the "Proposed Graph"
        self.baseline_graph = CausalGraph(self.parsed_query)
        if self.ablation_config.get("disable_interventions"):
            self.active_graph = self.baseline_graph
        else:
            engine = InterventionEngine()
            self.active_graph = engine.apply_interventions(self.baseline_graph, self.parsed_query.interventions)
        
        self.graph_summary = self.active_graph.get_summary()
        self.edges = self.graph_summary.get("edges", [])
        self.scores = [0.0] * len(self.edges)
        
        # Loop Control
        self.iteration = 0
        self.finished = (len(self.edges) == 0)
        self.active_test_line = None
        self.active_low_idx = None
        
    def get_final_context(self) -> str:
        """Component 4: Finalize the result."""
        # Use the finalized summary from the agentic loop
        final_summary = self.active_graph.get_summary()
        final_summary["edges"] = self.edges  # Use the refined edge list
        
        regenerator = CausalRegenerator()
        return regenerator.generate_context(final_summary)

def build_batch_confidence_prompts(states: List[AgentState], condition: str) -> List[str]:
    return [_build_confidence_prompt(s.domain, s.parsed_query, s.graph_summary, condition) for s in states]

def build_batch_hypothesis_prompts(states: List[AgentState]) -> List[str]:
    prompts = []
    for s in states:
        low_idx = next((i for i, sc in enumerate(s.scores) if sc < CONFIDENCE_THRESHOLD), None)
        s.active_low_idx = low_idx
        if low_idx is None:
            prompts.append(None)
            continue
            
        edge = s.edges[low_idx]
        src, tgt, data = edge[0], edge[1], edge[2]
        rel = data.get("relation", "relates_to")
        
        if s.domain == "cruxeval":
            prompts.append(_build_cruxeval_hypothesis_prompt(s.parsed_query.original_state.get("code", ""), src, tgt, rel))
        else:
            prompts.append(_build_crass_hypothesis_prompt(s.parsed_query.original_state.get("premise", ""), src, tgt, rel))
    return prompts

def build_batch_update_prompts(states: List[AgentState], observations: List[str]) -> List[str]:
    prompts = []
    for s, obs in zip(states, observations):
        if s.active_low_idx is None:
            prompts.append(None)
            continue
        prompts.append(_build_graph_update_prompt(s.domain, s.edges, s.scores, s.active_low_idx, s.active_test_line, obs))
    return prompts

# Keep internal helpers visible for the new runner
__all__ = [
    'AgentState', 'build_batch_confidence_prompts', 'build_batch_hypothesis_prompts', 
    'build_batch_update_prompts', '_parse_confidence_scores', '_apply_graph_updates', 
    '_sandbox_cruxeval', '_sandbox_crass', 'CONFIDENCE_THRESHOLD', 'MAX_ITERATIONS'
]

