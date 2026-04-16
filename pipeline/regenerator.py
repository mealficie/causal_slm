from typing import Dict, Any, List

class CausalRegenerator:
    """
    Component 4: The Flattening Layer.
    Translates mutated networkx dictionaries directly into logical context bullet points
    that a generic Small Language Model (SLM) can rigidly interpret.
    """
    
    def generate_context(self, graph_summary: Dict[str, Any]) -> str:
        """
        Main router mapping graph domains back to linguistic instructions.
        """
        if not graph_summary:
            return ""
            
        domain = graph_summary.get("domain", "crass")
        if domain == "crass":
            return self._flatten_nl_graph(graph_summary)
        elif domain == "cruxeval":
            return self._flatten_code_graph(graph_summary)
        else:
            return ""
            
    def _flatten_nl_graph(self, graph_summary: Dict[str, Any]) -> str:
        """
        Converts the NLP graph structure into bulleted facts.
        """
        bullets = []
        bullets.append("COUNTERFACTUAL EVENT STATE (Analyze these physical changes to determine the realistic outcome):")
        
        # 1. Evaluate Entity States
        for node in graph_summary.get("nodes", []):
            ent_name = node[0]
            attrs = node[1]
            if "adjective" in attrs and attrs["adjective"]:
                if "replaced_adjective" in attrs:
                    bullets.append(f"- Entity '{ent_name}' has property: {attrs['adjective']} (replaces {attrs['replaced_adjective']}).")
                else:
                    bullets.append(f"- Entity '{ent_name}' has property: {attrs['adjective']}.")
                
        # 2. Evaluate Executed Relationships
        for edge in graph_summary.get("edges", []):
            source = edge[0]
            target = edge[1]
            relation = edge[2].get("relation", "interacts with")
            bullets.append(f"- Active Event: '{source}' {relation} '{target}'")
            
        return "\n".join(bullets)
        
    def _flatten_code_graph(self, graph_summary: Dict[str, Any]) -> str:
        """
        Converts a code tree graph back into execution overrides and logic milestones.
        """
        bullets = []
        bullets.append("CAUSAL LOGIC STATE:")
        
        for node in graph_summary.get("nodes", []):
            var_name = node[0]
            attrs = node[1]
            
            # 1. Handle Overrides/Bindings
            if attrs.get("type") in ("parameter", "intervention_override") and "value" in attrs:
                val = attrs["value"]
                # Unpack the nested list-dictionary notation injected by CRUXEval parsers
                if isinstance(val, dict) and 0 in val and isinstance(val[0], list) and len(val[0]) > 0:
                    val = val[0][0]
                bullets.append(f"- State Initialization: variable '{var_name}' is set to {val}.")
                
            # 2. Handle Logic Milestones (structural branch/loop identification)
            elif attrs.get("type") == "constant" and ("branch_" in var_name or "loop_" in var_name):
                milestone_desc = attrs.get("value", "")
                bullets.append(f"- Logic Milestone: {milestone_desc}.")
                
        return "\n".join(bullets)
