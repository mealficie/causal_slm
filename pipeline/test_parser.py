import sys
import os
import json
from dataclasses import asdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.parser import parse_query
from data.load_data import load_data

def test_parser():
    print("=== Testing CRASS (NL SpaCy Parser) ===")
    crass_data = load_data('crass', sample_size=3)
    for item in crass_data:
        parsed = parse_query(item, domain='crass')
        print(f"QUESTION: {item['question']}")
        print(f"PREMISE: {item['context']['premise']}")
        print("PARSED:", json.dumps(asdict(parsed), indent=2))
        print("-" * 50)
        
    print("\n=== Testing CRUXEval (Code AST Parser) ===")
    crux_data = load_data('cruxeval', sample_size=2)
    for item in crux_data:
        parsed = parse_query(item, domain='cruxeval')
        print(f"QUESTION: {item['question']}")
        print("PARSED:", json.dumps(asdict(parsed), indent=2))
        print("-" * 50)

if __name__ == "__main__":
    test_parser()
