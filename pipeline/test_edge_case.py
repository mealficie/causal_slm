import json
from .parser import parse_query
from .graph_builder import CausalGraph
from .intervention import InterventionEngine

engine = InterventionEngine()

# CRASS Edge Case Test
print("=== NLP MUTATION TEST (Chef Scenario) ===")
print("Original Premise: A chef cuts a ripe tomato with a sharp knife.")
print("Counterfactual: What if the chef cut an apple with a dull knife instead?")
crass_item = {
    "context": {"premise": "A chef cuts a ripe tomato with a sharp knife."},
    "question": "What if the chef cut an apple with a dull knife instead?",
    "choices": {"A": "Easy", "B": "Slow", "C": "Hard", "D": "Mess"}
}
parsed_crass = parse_query(crass_item, "crass")
crass_graph = CausalGraph(parsed_crass)
print("\n--- BASELINE GRAPH ---")
print(json.dumps(crass_graph.get_summary(), indent=2))

mutated_crass = engine.apply_interventions(crass_graph, parsed_crass.interventions)
print("\n--- MUTATED DAG (After Intervention Engine) ---")
print(json.dumps(mutated_crass.get_summary(), indent=2))

# CRUXEval Code Test
print("\n=== CODE MUTATION TEST (CRUXEval) ===")
print("Original Code:")
print("def f(s):\n    clean_s = s.strip()\n    return clean_s.upper()")
print("What if input is ' bye '?")
crux_item = {
    "context": {
        "code": "def f(s):\n    clean_s = s.strip()\n    return clean_s.upper()",
        "input": "' bye '"
    }
}
parsed_crux = parse_query(crux_item, "cruxeval")
crux_graph = CausalGraph(parsed_crux)

# Before the intervention Engine activates, we construct the graph strictly over the code.
# The Intervention Engine will override the parameter with the new input physically.
fake_interventions = [
    {"type": "parameter_override", "bindings": {"s": "' bye '"}}
]
mutated_crux = engine.apply_interventions(crux_graph, fake_interventions)
print("\n--- MUTATED CODE GRAPH ---")
print(json.dumps(mutated_crux.get_summary(), indent=2))
