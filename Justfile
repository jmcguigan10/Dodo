# Justfile for Dodo

set shell := ["bash", "-cu"]

python_bin := ".venv/bin/python3"
julia_bin := "julia"

# Default target: run the full preprocessing pipeline
default: preprocess

# Run preprocessing with optional custom config path:
#   just preprocess                     # uses config/pre_process.yaml
#   just preprocess config=other.yaml   # uses another config
preprocess config="config/pre_process.yaml":
    {{python_bin}} scripts/run_preprocess.py --config {{config}} --julia {{julia_bin}}
