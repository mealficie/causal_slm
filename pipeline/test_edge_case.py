import sys
import os
import json
from dataclasses import asdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ast_parser import parse_code

def test_edge_case():
    code = '''def f(text, old, new):
    result = text.replace(old, new, 1)
    return result'''
    input_str = '"apple apple apple", "apple", "orange"'
    
    parsed = parse_code(code, input_str)
    
    print("\n--- Edge Case Test ---")
    print(json.dumps(parsed.__dict__, indent=2))

if __name__ == "__main__":
    test_edge_case()
