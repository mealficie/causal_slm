import networkx as nx
from typing import Dict, Any, List
import spacy
import ast
from .parser import ParsedQuery

# Load spaCy NLP model silently
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
        # Active entities defines the state space strictly BEFORE any counterfactuals occur!
        active_entities = self.original_state.get("active_entities", [])
        
        # 1. Add all temporal active entities to the DAG explicitly 
        for entity in active_entities:
            self.graph.add_node(entity, type="entity")
            
        premise = self.original_state.get("premise", "")
        if not premise or not nlp:
            return
            
        doc = nlp(premise)
        
        # 2. Reconstruct physical action edges bridging the nouns
        for token in doc:
            if token.pos_ == "VERB":
                # Find the subject doing the verb
                subject = None
                for child in token.lefts:
                    if child.dep_ == "nsubj":
                        subject = child.lemma_.lower()
                        break
                        
                # Find the object receiving the verb
                dobj = None
                for child in token.rights:
                    if child.dep_ in ["dobj", "pobj"]:
                        dobj = child.lemma_.lower()
                        break
                
                # If we have both, draw a directed edge mapping the causal action
                if subject and dobj:
                    if subject in active_entities and dobj in active_entities:
                        self.graph.add_edge(subject, dobj, relation=token.lemma_)

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
