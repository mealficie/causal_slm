import networkx as nx
from typing import Dict, Any, List
import spacy
import ast
from .parser import ParsedQuery

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

class DataFlowVisitor(ast.NodeVisitor):
    """AST Visitor to extract data flow edges between code variables and functional operations."""
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def _process_value_flow(self, value_node, target_node_id):
        """Helper to map right-hand-side computation into left-hand-side targets"""
        if isinstance(value_node, ast.Call):
            # Extract the raw operation name
            func_name = "call"
            if isinstance(value_node.func, ast.Name):
                func_name = value_node.func.id
            elif isinstance(value_node.func, ast.Attribute):
                func_name = value_node.func.attr
                
            op_node = f"operation_{func_name}"
            # Inject the procedural operation into the DAG
            self.graph.add_node(op_node, type="operation", operation=func_name)
            
            # 1. Variables flow INTO the operation
            for child in ast.walk(value_node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    self.graph.add_edge(child.id, op_node, type="data_flow")
                    
            # 1.5 Hardcoded Constants flow INTO the operation
            if hasattr(value_node, 'args'):
                for i, arg in enumerate(value_node.args):
                    if isinstance(arg, ast.Constant):
                        self.graph.add_edge(f"{func_name}_arg{i}", op_node, type="data_flow")
            if hasattr(value_node, 'keywords'):
                for kw in value_node.keywords:
                    if hasattr(kw, "arg") and kw.arg and isinstance(kw.value, ast.Constant):
                        self.graph.add_edge(kw.arg, op_node, type="data_flow")
                        
            # 2. Output flows FROM the operation into the Target variable
            self.graph.add_edge(op_node, target_node_id, type="data_flow")
        else:
            # Native data transfer (e.g. x = y)
            for child in ast.walk(value_node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    self.graph.add_edge(child.id, target_node_id, type="data_flow")

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._process_value_flow(node.value, target.id)
        self.generic_visit(node)

    def visit_Return(self, node):
        if node.value:
            self._process_value_flow(node.value, "return_output")
        self.generic_visit(node)

class CausalGraph:
    """
    A structural wrapper for the networkx DAG that encapsulates 
    the raw nodes and edges with operational metadata.
    """
    def __init__(self, parsed_query: ParsedQuery):
        self.domain = parsed_query.domain
        self.original_state = parsed_query.original_state
        self.all_entities = parsed_query.all_entities
        self.graph = nx.DiGraph()
        
        self.build_graph()

    def build_graph(self):
        """Routes to the correct mathematical builder based on domain."""
        if self.domain == "nl":
            self._build_nlp_graph()
        elif self.domain == "code":
            self._build_code_graph()
        else:
            raise ValueError(f"Unknown domain for graph building: {self.domain}")

    def _build_nlp_graph(self):
        """Constructs an entity-interaction state space for CRASS Natural Language."""
        doc = nlp(self.original_state.get("premise", ""))
        active_entities = self.original_state.get("active_entities", [])
        
        # 1. Map entities to their descriptive adjectives
        adj_map = {}
        for token in doc:
            ent_key = token.lemma_.lower() if token.lemma_.lower() in active_entities else token.text.lower()
            if ent_key in active_entities:
                adjs = [child.text for child in token.children if child.dep_ in ('amod', 'compound', 'nummod')]
                if adjs:
                    if ent_key not in adj_map:
                        adj_map[ent_key] = []
                    adj_map[ent_key].extend(adjs)

        # 2. Spawn baseline nodes with properties
        for ent in active_entities:
            attrs = {"type": "entity"}
            if ent in adj_map:
                attrs["adjective"] = " ".join(adj_map[ent])
            self.graph.add_node(ent, **attrs)

        # 3. Enhanced NLP topological mapping bridging prep dependencies
        for token in doc:
            if token.pos_ == "VERB":
                subj = None
                dobj = None
                pobjs = []
                
                for child in token.children:
                    child_ent = child.lemma_.lower() if child.lemma_.lower() in active_entities else child.text.lower()
                    if child.dep_ == "nsubj" and child_ent in active_entities:
                        subj = child_ent
                    elif child.dep_ == "dobj" and child_ent in active_entities:
                        dobj = child_ent
                    elif child.dep_ == "prep":
                        for grandchild in child.children:
                            gc_ent = grandchild.lemma_.lower() if grandchild.lemma_.lower() in active_entities else grandchild.text.lower()
                            if grandchild.dep_ == "pobj" and gc_ent in active_entities:
                                pobjs.append((gc_ent, child.text))
                
                # Link subject executing verb onto direct object
                if subj and dobj:
                    self.graph.add_edge(subj, dobj, relation=token.lemma_)
                    
                # Link direct object progressing onto prepositional targets (e.g. stone --on--> foot)
                if dobj and pobjs:
                    for pobj, prep in pobjs:
                        self.graph.add_edge(dobj, pobj, relation=prep)

    def _build_code_graph(self):
        """Constructs a data-flow dependency state space for CRUXEval Python code."""
        # 1. Seed Nodes
        parameters = self.original_state.get("parameters", [])
        local_vars = self.original_state.get("local_vars", [])
        constants = self.original_state.get("constants", {})
        
        for p in parameters:
            self.graph.add_node(p, type="parameter")
        for v in local_vars:
            self.graph.add_node(v, type="local_var")
        for k, v in constants.items():
            self.graph.add_node(k, type="constant", value=v)
            
        code = self.original_state.get("code", "")
        if not code:
            return
            
        # 2. Extract Data Flow using Python structural parsing
        try:
            tree = ast.parse(code)
            visitor = DataFlowVisitor(self.graph)
            visitor.visit(tree)
        except BaseException:
            pass # Failsafe if structural code is invalid
            
    def get_summary(self) -> dict:
        """Returns a mathematical footprint of the mapped topology."""
        return {
            "domain": self.domain,
            "nodes": list(self.graph.nodes(data=True)),
            "edges": list(self.graph.edges(data=True))
        }
