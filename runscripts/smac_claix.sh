#!/bin/zsh

# USAGE: ./smac_claix.sh EXPERIMENT      CLUSTER      SEARCH_SPACE      CONFIG_NAME      [RUNNING_TIME]
# e.g.:  ./smac_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid tune_smac_cost_aware_rf 120.0

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Illegal number of parameters. Usage: $0 EXPERIMENT CLUSTER SEARCH_SPACE CONFIG_NAME [RUNNING_TIME]"
    exit 1
fi

EXPERIMENT=$1
CLUSTER=$2
SEARCH_SPACE=$3
CONFIG_NAME=$4
RUNNING_TIME_ARG=""

if [ -n "$5" ]; then
    RUNNING_TIME_ARG="+default_hyperparameter.running_time=$5"
fi

JOB_NAME="smac_${EXPERIMENT}_${SEARCH_SPACE}_${CONFIG_NAME}"
DIRECTORY="smac/${EXPERIMENT}/${SEARCH_SPACE}/${CONFIG_NAME}"

# Create a dedicated directory for this specific experiment run
mkdir -p "$DIRECTORY/log"

# Use a 'here document' (cat <<EOF) for better readability and maintenance
cat > "$DIRECTORY/submit.sh" <<EOF
#!/bin/zsh

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M
#SBATCH --account=thes2105
#SBATCH --job-name=${JOB_NAME}
#SBATCH -t 30:00:00
#SBATCH --mail-type fail,end
#SBATCH --mail-user stamatios.chrysanthidis@rwth-aachen.de
#SBATCH --output $DIRECTORY/log/%A.out
#SBATCH --error $DIRECTORY/log/%A.err
#SBATCH --array 1,3

# Change to the project directory
cd /home/aq055502/projects/arlbench-smac-hyper/arlbench

module purge
module load GCCcore/12.2.0
module load Python/3.10.8
source .venv/bin/activate

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
sbatch --begin=now "$DIRECTORY/submit.sh"