# Justfile for Dodo

set shell := ["bash", "-cu"]

python_bin := ".venv/bin/python3"
julia_bin := "julia"

# Convenience alias
alias reprocess := preprocess

# Default target: run the full preprocessing pipeline
default: preprocess

# Run preprocessing with optional custom config path:
#   just preprocess                     # uses config/pre_process.yaml
#   just preprocess config=other.yaml   # uses another config
preprocess config="config/pre_process.yaml":
    JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-.julia_depot} {{python_bin}} scripts/run_preprocess.py --config {{config}} --julia {{julia_bin}}

# Train residual-on-Box3D model:
#   just train                          # uses config/model.yaml
#   just train config=other.yaml        # overrides config
train config="config/model.yaml":
    {{python_bin}} train.py --config {{config}}
