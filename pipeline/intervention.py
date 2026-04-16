import networkx as nx
from typing import List, Dict, Any
from .graph_builder import CausalGraph
import copy

class InterventionEngine:
    def apply_interventions(self, baseline_graph: CausalGraph, interventions: List[Dict[str, Any]]) -> CausalGraph:
        if not interventions:
            return baseline_graph
            
        # Hard clone to ensure we NEVER mutate the baseline text
        new_graph = CausalGraph(parsed_query=self._mock_empty_query(baseline_graph.domain))
        new_graph.original_state = copy.deepcopy(baseline_graph.original_state)
        new_graph.all_entities = copy.deepcopy(baseline_graph.all_entities)
        new_graph.graph = baseline_graph.graph.copy()
        
        for action in interventions:
            t = action.get("type", "")
            
            # --- CRUXEval Execution ---
            if t == "parameter_override":
                self._execute_override(new_graph.graph, action)
                
            # --- CRASS Execution ---
            elif t == "entity_substitution":
                self._execute_topological_swap(new_graph.graph, action)
            elif t == "action_stopped":
                self._execute_severance(new_graph.graph, action)
            # Property updates handle state shifts rather than topological shifts
            elif t == "property_addition":
                self._execute_attribute_shift(new_graph.graph, action)
            elif t == "relationship_shift":
                self._execute_relationship_shift(new_graph.graph, action)

        return new_graph

    def _execute_override(self, G: nx.DiGraph, action: dict):
        """ Memory override for Coding Counterfactuals"""
        bindings = action.get("bindings", {})
        for target_node, new_value in bindings.items():
            if G.has_node(target_node):
                G.nodes[target_node]["value"] = new_value
            else:
                # if parsing missed an implicit default arg
                G.add_node(target_node, type="intervention_override", value=new_value)

    def _execute_topological_swap(self, G: nx.DiGraph, action: dict):
        """Topological Edges re-point to substituted entities"""
        target_entities = action.get("target_entity", [])
        deleted_entities = action.get("deleted_entities", [])
        
        # If there's an exact 1 substitution target if targets are more than 1
        if len(target_entities) == 1 and len(deleted_entities) > 0 and len(deleted_entities) <= 2:
            new_node = target_entities[0]
            
            # Find the deleted node that actually possesses physical dependencies to hijack!
            dead_node = None
            for d in deleted_entities:
                if G.has_node(d) and (G.in_degree(d) > 0 or G.out_degree(d) > 0):
                    dead_node = d
                    break
                    
            if dead_node:
                # 1. Map physical dependencies
                incoming_edges = list(G.in_edges(dead_node, data=True))
                outgoing_edges = list(G.out_edges(dead_node, data=True))
                attributes = dict(G.nodes[dead_node])
                
                # 2. Add New Node, adopting ONLY the foundational type
                new_attrs = {"type": attributes.get("type", "entity")}
                G.add_node(new_node, **new_attrs)
                
                # Map specific target attributes if specified
                if "attribute" in action and action["attribute"]:
                    G.nodes[new_node]["adjective"] = action["attribute"]
                
                # 3. Clone all dependencies natively onto the new Node
                for u, v, data in incoming_edges:
                    G.add_edge(u, new_node, **data)
                for u, v, data in outgoing_edges:
                    G.add_edge(new_node, v, **data)
                    
                # 4. Remove original physical node permanently
                G.remove_node(dead_node)
        else:
            # Handle generic node insertion without purging!
            for ent in target_entities:
                if not G.has_node(ent):
                    G.add_node(ent, type="entity")
                if "attribute" in action and action["attribute"]:
                    old_adj = G.nodes[ent].get("adjective")
                    G.nodes[ent]["adjective"] = action["attribute"]
                    if old_adj and old_adj != action["attribute"]:
                        G.nodes[ent]["replaced_adjective"] = old_adj
            
                    
    def _execute_severance(self, G: nx.DiGraph, action: dict):
        """Destroy edge bridges to emulate negative dependencies"""
        # If no specific target entity is provided, find the primary operational edge and slice it!
        if list(G.edges):
            first_edge = list(G.edges)[0]
            G.remove_edge(first_edge[0], first_edge[1])

    def _execute_attribute_shift(self, G: nx.DiGraph, action: dict):
        """ Adds adjectives or statuses locally"""
        for ent in action.get("target_entity", []):
            if G.has_node(ent):
                old_adj = G.nodes[ent].get("adjective")
                G.nodes[ent]["adjective"] = action.get("attribute", None)
                if old_adj and old_adj != G.nodes[ent]["adjective"]:
                    G.nodes[ent]["replaced_adjective"] = old_adj

    def _execute_relationship_shift(self, G: nx.DiGraph, action: dict):
        """ Injects new causal dependencies derived from the counterfactual question """
        source = action.get("source")
        target = action.get("target")
        relation = action.get("relation")
        
        if source and target and G.has_node(source) and G.has_node(target):
            # Resolve Paradox: If A acts on B, B can no longer act on A in the same event space
            if G.has_edge(target, source):
                G.remove_edge(target, source)
                
            # We add or update the relation between the source and target
            G.add_edge(source, target, relation=relation)

    # --- Utility Mock ---
    def _mock_empty_query(self, domain: str):
        class MockQuery:
            def __init__(self, d):
                self.domain = d
                self.original_state = {}
                self.all_entities = []
        return MockQuery(domain)
