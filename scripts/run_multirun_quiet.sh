#!/bin/bash
# Run Hydra multirun with all output redirected to log files

# Configuration
CONFIG_NAME="${1:-ks}"  # Default to ks if not provided
SEEDS="${2:-100,200,300,400}"  # Default seeds if not provided

# Run with all output redirected
python ml4dynamics/trainers/train_jax.py \
    --config-name=$CONFIG_NAME \
    --multirun \
    sim.seed=$SEEDS \
    hydra/job_logging=file_only \
    hydra.job.chdir=true