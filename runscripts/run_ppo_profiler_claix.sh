#!/bin/zsh

# USAGE: ./runscripts/profile_ppo_job.sh EXPERIMENT CLUSTER [SEED]
# e.g.:  ./runscripts/profile_ppo_job.sh box2d_lunar_lander_ppo claix_gpu_h100 42

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 EXPERIMENT CLUSTER [SEED]"
    exit 1
fi

EXPERIMENT=$1
CLUSTER=$2
SEED=${3:-1} # Default seed is 1 if not provided

JOB_NAME="prof_${EXPERIMENT}"
DIRECTORY="results/profiling/${EXPERIMENT}"

# Create log directory
mkdir -p "$DIRECTORY/log"

# Create submission script
cat > "$DIRECTORY/submit_profile.sh" <<EOF
#!/bin/zsh

#SBATCH --cpus-per-task=4
#SBATCH --job-name=${JOB_NAME}
#SBATCH -t 00:14:00
#SBATCH --account=thes2105
#SBATCH --output $DIRECTORY/log/seed_${SEED}.out
#SBATCH --error $DIRECTORY/log/seed_${SEED}.err

# GPU settings if needed (simple check based on cluster name)
$(if [[ "$CLUSTER" == *"gpu"* ]]; then echo "#SBATCH --partition=c23g"; echo "#SBATCH --gres=gpu:1"; fi)

module purge
module load GCCcore/12.2.0
module load Python/3.10.8
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0
source /home/aq055502/projects/arlbench-smac-hyper/arlbench/.venv/bin/activate

echo "Starting Profiling Run for seed $SEED"

# Execute the profiler script
python runscripts/profile_ppo_running_time.py \\
    experiments=$EXPERIMENT \\
    cluster=$CLUSTER \\
    seed=$SEED \\
    autorl.seed=$SEED \\
    nas_config.hidden_size=128 \\
    nas_config.num_mlp_layers=2

EOF

echo "Generated submission script in $DIRECTORY/submit_profile.sh"
chmod +x "$DIRECTORY/submit_profile.sh"
sbatch "$DIRECTORY/submit_profile.sh"