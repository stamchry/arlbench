import argparse
import os
import subprocess
from pathlib import Path

def submit_job(experiment, cluster, search_space, opt_id, incumbent_path, log_base, optimization_method, output_dir_base):
    """Generates and submits a SLURM array job, followed by an aggregation job."""
    
    job_name = f"eval_{search_space}_{opt_id}"
    log_dir = log_base / opt_id
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Define where results will be saved
    result_save_path = output_dir_base / experiment / search_space / optimization_method / opt_id
    
    # --- Determine Resources based on Cluster ---
    # Defaults for CPU (matching claix_cpu.yaml logic)
    partition = "c23ms"
    gres_directive = ""
    cpus_per_task = 4
    # Standard CPU modules
    module_loads = """module purge
module load GCCcore/12.2.0
module load Python/3.10.8"""

    # Overrides for GPU (matching claix_gpu_h100.yaml logic)
    if "gpu" in cluster:
        partition = "c23g"
        gres_directive = "#SBATCH --gres=gpu:1"
        cpus_per_task = 8  # Increased slightly for GPU data feeding
        # Add CUDA/cuDNN modules required for GPU runs
        module_loads = """module purge
module load GCCcore/12.2.0
module load Python/3.10.8
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0"""

    # --- 1. Create Evaluation Script ---
    submit_script_path = log_dir / "submit_eval.sh"
    
    script_content = f"""#!/bin/zsh
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu=2000M
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{gres_directive}
#SBATCH -t 00:30:00
#SBATCH --output {log_dir}/seed_%a.out
#SBATCH --error {log_dir}/seed_%a.err
#SBATCH --array 40-43

cd {os.getcwd()}

{module_loads}
source .venv/bin/activate

echo "Evaluating Optimization ID {opt_id} on Test Seed $SLURM_ARRAY_TASK_ID"

python runscripts/evaluate_incumbent_single_seed.py \\
    --incumbent_path "{incumbent_path}" \\
    --experiment {experiment} \\
    --search_space {search_space} \\
    --cluster {cluster} \\
    --opt_id "{opt_id}" \\
    --optimization_method "{optimization_method}" \\
    --output_dir "{output_dir_base}" \\
    --test_seed $SLURM_ARRAY_TASK_ID
"""

    with open(submit_script_path, "w") as f:
        f.write(script_content)
    
    # --- 2. Submit Evaluation Job ---
    # Capture the job ID to create a dependency
    result = subprocess.run(["sbatch", "--parsable", str(submit_script_path)], capture_output=True, text=True)
    job_id = result.stdout.strip()
    print(f"Submitted eval job {job_id} for ID: {opt_id} on partition {partition}")

    # --- 3. Create Aggregation Script ---
    agg_script_path = log_dir / "submit_agg.sh"
    
    agg_content = f"""#!/bin/zsh
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000M
#SBATCH --job-name=agg_{opt_id}
#SBATCH -t 00:05:00
#SBATCH --output {log_dir}/agg.out
#SBATCH --error {log_dir}/agg.err

cd {os.getcwd()}
module load GCCcore/12.2.0
module load Python/3.10.8
source .venv/bin/activate

echo "Aggregating results for {opt_id}..."
python runscripts/aggregate_results.py --folder "{result_save_path}"
"""
    with open(agg_script_path, "w") as f:
        f.write(agg_content)

    # --- 4. Submit Aggregation Job (Dependency) ---
    subprocess.run(["sbatch", f"--dependency=afterok:{job_id}", str(agg_script_path)])
    print(f"Submitted aggregation job (depends on {job_id})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--search_space", required=True)
    parser.add_argument("--results_dir", required=True, help="Base directory to search for configs")
    parser.add_argument("--path_filter", default=None, help="Substring that must be present in the path (e.g. CartPole)")
    
    args = parser.parse_args()
    
    base_path = Path(args.results_dir)
    optimization_method = base_path.name
    
    log_base = Path("logs/eval") / args.experiment / args.search_space / optimization_method
    output_dir_base = Path("results/evaluation")

    print(f"Searching for 'incumbent.csv' in {base_path}...")
    if args.path_filter:
        print(f"Filtering paths containing: '{args.path_filter}'")
    
    configs = list(base_path.rglob("incumbent.csv"))
    
    count = 0
    for config_path in configs:
        if args.search_space not in str(config_path):
            continue

        if args.path_filter and args.path_filter not in str(config_path):
            continue
            
        try:
            autorl_seed = config_path.parent.name
            smac_seed = config_path.parent.parent.name
            opt_id = f"{smac_seed}_{autorl_seed}"
        except:
            opt_id = f"unknown_{count}"

        print(f"Found incumbent: {config_path} -> ID: {opt_id}")
        submit_job(args.experiment, args.cluster, args.search_space, opt_id, config_path, log_base, optimization_method, output_dir_base)
        count += 1
        
    if count == 0:
        print("No matching configurations found.")
    else:
        print(f"Submitted {count} array jobs.")

if __name__ == "__main__":
    main()