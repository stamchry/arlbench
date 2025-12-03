import argparse
import json
import os
import re
import subprocess
import sys
import csv
import numpy as np
from pathlib import Path

ARLBENCH_SCRIPT = "runscripts/run_arlbench.py"

# Columns in incumbent.csv that are metadata and NOT hyperparameters
IGNORED_COLUMNS = {
    "config_id",
    "performance",
    "budget",
    "budget_used",
    "total_wallclock_time",
    "total_optimization_time",
    "seed" 
}

def load_incumbent_from_csv(path):
    """Loads the last row from incumbent.csv."""
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise ValueError(f"No data found in {path}")
        return rows[-1] # Return the last incumbent (best found at end)

def run_single_evaluation(incumbent, experiment, cluster, test_seed):
    """Runs ARLBench for the incumbent on a SINGLE test seed."""
    
    # 1. Start with the BASE experiment config
    cmd = [
        "python",
        ARLBENCH_SCRIPT,
        f"experiments={experiment}",
        f"cluster={cluster}",
        f"autorl.seed={test_seed}",
    ]
    
    # 2. Process Incumbent Parameters
    for k, v in incumbent.items():
        if k in IGNORED_COLUMNS:
            continue
        
        # Skip empty values
        if v is None or v == "":
            continue

        # Hydra ++ syntax to add/override
        cmd.append(f"++{k}={v}")

    print(f"Running evaluation for seed {test_seed}...")
    
    try:
        # Run and capture output
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        
        # Parse output for objectives
        # FIXED REGEX: Matches "Returning objectives for Hypersweeper:"
        match = re.search(r"Returning objectives for Hypersweeper: (\{.*?\})", result, re.DOTALL)
        
        if match:
            obj_str = match.group(1).replace("'", '"').replace("nan", "NaN").replace("inf", "Infinity")
            objs = json.loads(obj_str)
            
            perf = objs.get("reward_mean", objs.get("performance", -np.inf))
            cost = objs.get("runtime", objs.get("cost", np.inf))
            
            return {"performance": perf, "cost": cost}
        else:
            print(f"Error: Could not parse output for seed {test_seed}")
            print("--- CAPTURED OUTPUT START ---")
            print(result)
            print("--- CAPTURED OUTPUT END ---")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running seed {test_seed}. Exit code: {e.returncode}")
        print("--- ERROR OUTPUT START ---")
        print(e.output)
        print("--- ERROR OUTPUT END ---")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent_path", required=True, help="Path to incumbent.csv")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--search_space", required=True)
    parser.add_argument("--opt_id", required=True, help="Unique identifier for this optimization run")
    parser.add_argument("--optimization_method", required=True, help="Name of the optimization method (e.g. smac_hand_crafted)")
    parser.add_argument("--test_seed", required=True, type=int)
    parser.add_argument("--cluster", default="local")
    parser.add_argument("--output_dir", default="results/evaluation")
    
    args = parser.parse_args()

    try:
        incumbent = load_incumbent_from_csv(args.incumbent_path)
    except Exception as e:
        print(f"Error loading incumbent: {e}")
        sys.exit(1)

    result = run_single_evaluation(incumbent, args.experiment, args.cluster, args.test_seed)

    if result:
        # Updated save path to include optimization method
        save_path = Path(args.output_dir) / args.experiment / args.search_space / args.optimization_method / args.opt_id
        save_path.mkdir(parents=True, exist_ok=True)
        
        file_name = save_path / f"seed_{args.test_seed}.json"
        
        output_data = {
            "experiment": args.experiment,
            "search_space": args.search_space,
            "optimization_method": args.optimization_method,
            "opt_id": args.opt_id,
            "test_seed": args.test_seed,
            "incumbent": incumbent,
            "result": result
        }
        
        with open(file_name, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved result to {file_name}")
    else:
        print("Evaluation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()