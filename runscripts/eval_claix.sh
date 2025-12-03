#!/bin/zsh

# USAGE: ./eval_claix.sh EXPERIMENT CLUSTER SEARCH_SPACE RESULTS_DIR [PATH_FILTER]
# e.g.:  ./eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_hand_crafted CartPole

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 EXPERIMENT CLUSTER SEARCH_SPACE RESULTS_DIR [PATH_FILTER]"
    exit 1
fi

EXPERIMENT=$1
CLUSTER=$2
SEARCH_SPACE=$3
RESULTS_DIR=$4
PATH_FILTER=$5

echo "Starting evaluation submission..."

export HYDRA_FULL_ERROR=1

# Delegate to Python script for smart handling
python runscripts/submit_eval_jobs.py \
    --experiment "$EXPERIMENT" \
    --cluster "$CLUSTER" \
    --search_space "$SEARCH_SPACE" \
    --results_dir "$RESULTS_DIR" \
    --path_filter "$PATH_FILTER"