#!/bin/zsh

# USAGE: ./smac_claix.sh EXPERIMENT      CLUSTER      SEARCH_SPACE      CONFIG_NAME      [RUNNING_TIME] [DEPENDENCY]
# e.g.:  ./smac_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid tune_smac_cost_aware_rf 120.0 afterok:12345

if [ "$#" -lt 4 ]; then
    echo "Illegal number of parameters. Usage: $0 EXPERIMENT CLUSTER SEARCH_SPACE CONFIG_NAME [RUNNING_TIME] [DEPENDENCY]"
    exit 1
fi

EXPERIMENT=$1
CLUSTER=$2
SEARCH_SPACE=$3
CONFIG_NAME=$4
RUNNING_TIME_ARG=""
DEPENDENCY_ARG=""

# Check if arg 5 is provided and doesn't look like a dependency (doesn't start with after)
if [ -n "$5" ] && [[ "$5" != after* ]]; then
    RUNNING_TIME_ARG="+default_hyperparameter.running_time=$5"
    # If 5 was running time, 6 might be dependency
    if [ -n "$6" ]; then
        DEPENDENCY_ARG="--dependency=$6"
    fi
elif [ -n "$5" ]; then
    # If 5 looks like a dependency, treat it as such (skipping running time)
    DEPENDENCY_ARG="--dependency=$5"
fi

JOB_NAME="smac_${EXPERIMENT}_${SEARCH_SPACE}_${CONFIG_NAME}"
DIRECTORY="smac/${EXPERIMENT}/${SEARCH_SPACE}/${CONFIG_NAME}"

# Create a dedicated directory for this specific experiment run
mkdir -p "$DIRECTORY/log"

# Use a 'here document' (cat <<EOF) for better readability and maintenance
cat > "$DIRECTORY/submit.sh" <<EOF
#!/bin/zsh

#SBATCH --cpus-per-task=4
#SBATCH --partition=c23ms
#SBATCH --account=thes2105
#SBATCH --job-name=${JOB_NAME}
#SBATCH -t 96:00:00
#SBATCH --mail-type fail,end
#SBATCH --mail-user stamatios.chrysanthidis@rwth-aachen.de
#SBATCH --output $DIRECTORY/log/%A.out
#SBATCH --error $DIRECTORY/log/%A.err
#SBATCH --array 42,43,44


module purge
module load GCCcore/12.2.0
module load Python/3.10.8
source /home/aq055502/projects/arlbench-smac-hyper/arlbench/.venv/bin/activate

echo "Starting SMAC optimization for seed \$SLURM_ARRAY_TASK_ID"

# Run the multi-node SMAC optimization
python runscripts/run_arlbench.py -m \\
    --config-name=$CONFIG_NAME \\
    experiments=$EXPERIMENT \\
    cluster=$CLUSTER \\
    search_space=$SEARCH_SPACE \\
    smac_seed=\$SLURM_ARRAY_TASK_ID \\
    $RUNNING_TIME_ARG \\
EOF

echo "Generated submission script in $DIRECTORY/submit.sh"
chmod +x "$DIRECTORY/submit.sh"

# Pass the dependency argument to sbatch
sbatch $DEPENDENCY_ARG --begin=now "$DIRECTORY/submit.sh"