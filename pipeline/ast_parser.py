import ast
from typing import Any

from .parser import ParsedQuery

class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.parameters = []
        self.local_vars = set()
        self.constants = {}
        
    def visit_FunctionDef(self, node):
        if not self.parameters:  # Only bind the top-level function signature
            self.parameters = [arg.arg for arg in node.args.args]
        self.generic_visit(node)
        
    def visit_Name(self, node):
        # Capture all internal state variables that aren't inputs
        if isinstance(node.ctx, (ast.Store, ast.Load)):
            if node.id not in self.parameters:
                self.local_vars.add(node.id)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        # Capture hardcoded keyword arguments mapped purely as constants
        for kw in node.keywords:
            if hasattr(kw, 'arg') and kw.arg and isinstance(kw.value, ast.Constant):
                self.constants[kw.arg] = kw.value.value
                
        # Capture generic positional constants
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Constant):
                func_name = "call"
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                self.constants[f"{func_name}_arg{i}"] = arg.value
                
        self.generic_visit(node)
        
    def visit_Assign(self, node):
        # Capture explicit static assignments inside the code scope
        if isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.constants[target.id] = node.value.value
        self.generic_visit(node)

def parse_code(code: str, input_val: Any) -> ParsedQuery:
    """Uses Python's AST module to dynamically map function parameters and state memory."""
    tree = ast.parse(code)
    
    visitor = CodeVisitor()
    visitor.visit(tree)
    
    builtins = {
        'print', 'len', 'range', 'int', 'str', 'float', 'list', 
        'dict', 'set', 'True', 'False', 'None', 'append', 'split',
        'join', 'replace', 'round', 'abs', 'pop', 'lower', 'upper',
        'islower', 'isupper', 'map', 'filter', 'sum', 'min', 'max'
    }
    
    local_vars = [v for v in visitor.local_vars if v not in builtins]
    
    #evaluate the Python string back into separated lists/tuples
    try:
        eval_val = ast.literal_eval(f"({input_val},)")
        parsed_new_value = list(eval_val)
    except BaseException:
        parsed_new_value = [input_val]
        
    # Safely associate variable names directly with their new assigned values
    variable_bindings = dict(zip(visitor.parameters, parsed_new_value))
    
    intervention = {
        "type": "parameter_override",
        "bindings": variable_bindings
    }
    
    original_state = {
        "code": code,
        "parameters": visitor.parameters,
        "local_vars": local_vars,
        "constants": visitor.constants
    }
    
    all_entities = list(visitor.parameters) + local_vars + list(visitor.constants.keys())
    
    return ParsedQuery(
        domain="code",
        original_state=original_state,
        interventions=[intervention],
        all_entities=all_entities
    )
