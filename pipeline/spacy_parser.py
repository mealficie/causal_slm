import spacy
from typing import Dict, Any

from .parser import ParsedQuery

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import logging
    logging.warning("en_core_web_sm not downloaded. It should be downloaded during initialization.")
    nlp = None

def build_coref_map(prem_doc, q_doc) -> dict:
    """A highly deterministic, lightweight coreference resolver mapping pronouns to the premise subject."""
    coref_map = {}
    # Find primary subjects in the premise to act as anchors (e.g. 'woman', 'man')
    prem_subjects = [chunk.root.text.lower() for chunk in prem_doc.noun_chunks if chunk.root.dep_ == 'nsubj']
    
    q_prons = [t.text.lower() for t in q_doc if t.pos_ == "PRON"]
    for pron in q_prons:
        # Standard third-person pronouns that substitute the main premise actor
        if pron in ["he", "she", "it", "they", "his", "her", "their"]:
            if prem_subjects:
                coref_map[pron] = prem_subjects[0]
    return coref_map

def extract_entities(doc, coref_map=None) -> set:
    """Extract root base nouns, shedding determiners, and actively resolving pronouns."""
    entities = set()
    coref_map = coref_map or {}
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == "PRON":
            pron = chunk.root.text.lower()
            if pron in coref_map:
                entities.add(coref_map[pron])
        else:
            entities.add(chunk.root.text.lower())
    return entities

def parse_nl(premise: str, question: str) -> ParsedQuery:
    """Feature Extraction: uses linguistics to classify counterfactual modifications."""
    if nlp is None:
        return ParsedQuery("nl", {"premise": premise}, [], [])
        
    prem_doc = nlp(premise)
    q_doc = nlp(question)
    
    coref_map = build_coref_map(prem_doc, q_doc)
    
    # 1. Global Entities List Union with active pronoun translation
    prem_entities = extract_entities(prem_doc)
    q_entities = extract_entities(q_doc, coref_map)
    all_entities = list(prem_entities.union(q_entities))
    
    new_nouns = q_entities - prem_entities
    dropped_nouns = prem_entities - q_entities
    
    original_state = {
        "premise": premise,
        "active_entities": list(prem_entities)
    }
    
    interventions = []
    
    # Sweep 1: Extract Negation Checks
    for token in q_doc:
        if token.dep_ == 'neg':
            interventions.append({
                "type": "action_stopped",
                "target_entity": [],
                "attribute": None,
                "relation_to_premise": "negated_execution",
                "origin": "existing_in_premise"
            })
            break
            
    # Sweep 2: Universal Substitution & Garbage Collection
    if new_nouns or dropped_nouns:
        sub_dict = {
            "type": "entity_substitution", 
            "target_entity": list(new_nouns),
            "attribute": None,
            "relation_to_premise": "modified_execution",
            "origin": "introduced_in_counterfactual" if new_nouns else "existing_in_premise"
        }
        if dropped_nouns:
            sub_dict["deleted_entities"] = list(dropped_nouns)
            for dead_noun in dropped_nouns:
                if dead_noun in all_entities:
                    all_entities.remove(dead_noun)
        interventions.append(sub_dict)
            
    # Sweep 3: Property Addition Checks
    for token in q_doc:
        if token.dep_ == 'amod':
            target = token.head.text.lower()
            is_new = target in new_nouns
            if is_new:
                # Merge into the existing substitution block!
                for inv in interventions:
                    if inv["type"] == "entity_substitution" and target in inv["target_entity"]:
                        inv["attribute"] = token.text.lower()
            else:
                # Spawn an independent surgical edit for the existing artifact
                interventions.append({
                    "type": "property_addition",
                    "target_entity": [target],
                    "attribute": token.text.lower(),
                    "relation_to_premise": "modified_execution",
                    "origin": "existing_in_premise"
                })
    
    return ParsedQuery(
        domain="nl",
        original_state=original_state,
        interventions=interventions,
        all_entities=all_entities
    )
