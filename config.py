import os

MODELS = {
    "gemma4_e2b": "google/gemma-4-E2B-it",
    "gemma4_e4b": "google/gemma-4-E4B-it",
    "qwen25_7b":  "Qwen/Qwen2.5-7B-Instruct",
}

BENCHMARKS = ["crass", "cruxeval"]
CONDITIONS  = ["zero_shot", "cot", "pipeline"]

SAMPLE_SIZE_DEV  = 50   
SAMPLE_SIZE_FULL = None  # Full benchmark (CRASS ~274, CRUXEval 800)

#kept relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

# TODO: remove before commiting
CACHE_DIR   = ""

# Ensure directories exist
for directory in [RESULTS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)
