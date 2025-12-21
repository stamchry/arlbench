#!/bin/zsh

# USAGE: ./run_default_claix.sh EXPERIMENT CLUSTER [SEED]
# e.g.:  ./run_default_claix.sh box2d_lunar_lander_ppo claix_gpu_h100 42

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 EXPERIMENT CLUSTER [SEED]"
    exit 1
fi

EXPERIMENT=$1
CLUSTER=$2
SEED=${3:-1} # Default seed is 1 if not provided

JOB_NAME="def_${EXPERIMENT}"
DIRECTORY="results/default/${EXPERIMENT}"

# Create log directory
mkdir -p "$DIRECTORY/log"

# Create submission script
cat > "$DIRECTORY/submit.sh" <<EOF
#!/bin/zsh

#SBATCH --cpus-per-task=8
#SBATCH --account=thes2105
#SBATCH --job-name=${JOB_NAME}
#SBATCH -t 01:00:00
#SBATCH --output $DIRECTORY/log/seed_${SEED}.out
#SBATCH --error $DIRECTORY/log/seed_${SEED}.err

# GPU settings if needed (simple check based on cluster name)
$(if [[ "$CLUSTER" == *"gpu"* ]]; then echo "#SBATCH --partition=c23g"; echo "#SBATCH --gres=gpu:1"; fi)


module purge
module load GCCcore/12.2.0
module load Python/3.10.8
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0
source .venv/bin/activate

echo "Starting Default Run for seed $SEED"

# Run without -m (single run)
python runscripts/run_arlbench.py \\
    experiments=$EXPERIMENT \\
    cluster=$CLUSTER \\
    +seed=$SEED \\
    autorl.seed=$SEED

EOF

echo "Generated submission script in $DIRECTORY/submit.sh"
chmod +x "$DIRECTORY/submit.sh"
sbatch "$DIRECTORY/submit.sh"
