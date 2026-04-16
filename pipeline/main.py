from causal_slm.pipeline.parser import parse_query
from causal_slm.pipeline.graph_builder import CausalGraph
from causal_slm.pipeline.intervention import InterventionEngine
from causal_slm.pipeline.regenerator import CausalRegenerator

def run_causal_slm_pipeline(item: dict, ablation_config: dict = None) -> str:
    """
    Main execution loop chaining the 4 components.
    Returns the flattened Causal Logic Context to append to an SLM prompt.
    """
    ablation_config = ablation_config or {}
    domain = item.get("domain", "crass")
    
    # 1. Structure
    parsed_query = parse_query(item, domain)
    
    # Ablation: Disable components at the intervention level
    if ablation_config.get("disable_attributes"):
        parsed_query.interventions = [i for i in parsed_query.interventions if i["type"] not in ("property_addition", "entity_substitution")]
    if ablation_config.get("disable_relationships"):
        parsed_query.interventions = [i for i in parsed_query.interventions if i["type"] != "relationship_shift"]
    
    # 2. Baseline
    baseline_graph = CausalGraph(parsed_query)
    
    # 3. Intervene
    if ablation_config.get("disable_interventions"):
        # We skip applying any mutations and just summarize the baseline state
        mutated_graph = baseline_graph
    else:
        engine = InterventionEngine()
        mutated_graph = engine.apply_interventions(baseline_graph, parsed_query.interventions)
    
    # 4. Regenerate
    regenerator = CausalRegenerator()
    prompt_context = regenerator.generate_context(mutated_graph.get_summary())
    
    return prompt_context
