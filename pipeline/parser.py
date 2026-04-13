from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ParsedQuery:
    domain: str
    original_state: Dict[str, Any]
    interventions: List[Dict[str, Any]]
    all_entities: List[str]

def parse_query(item: Dict[str, Any], domain: str, use_parser: bool = True) -> ParsedQuery:
    """Routes an item to the correct downstream parser."""
    if not use_parser:
        return ParsedQuery(domain=domain, original_state={"raw": item}, interventions=[], all_entities=[])
        
    if domain == "crass":
        from .spacy_parser import parse_nl
        return parse_nl(item['context']['premise'], item['question'])
    elif domain == "cruxeval":
        from .ast_parser import parse_code
        return parse_code(item['context']['code'], item['context']['input'])
    else:
        raise ValueError(f"Unknown domain {domain}")
