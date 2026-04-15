from causal_slm.pipeline.parser import parse_query
from causal_slm.pipeline.graph_builder import CausalGraph
from causal_slm.pipeline.intervention import InterventionEngine
from causal_slm.pipeline.regenerator import CausalRegenerator

def run_causal_slm_pipeline(crass_item: dict) -> str:
    """
    Main execution loop chaining the 4 components.
    Returns the flattened Causal Logic Context to append to an SLM prompt.
    """
    # 1. Structure
    parsed_query = parse_query(crass_item, "crass")
    
    # 2. Baseline
    baseline_graph = CausalGraph(parsed_query)
    
    # 3. Intervene
    engine = InterventionEngine()
    mutated_graph = engine.apply_interventions(baseline_graph, parsed_query.interventions)
    
    # 4. Regenerate
    regenerator = CausalRegenerator()
    prompt_context = regenerator.generate_context(mutated_graph.get_summary())
    
    return prompt_context
