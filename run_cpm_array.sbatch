#!/bin/bash
#SBATCH --job-name=BSI_predictions
#SBATCH --output=/home/ajs332/BSI_modeling/results/cpm_bsi_%A_%a.out
#SBATCH --error=/home/ajs332/BSI_modeling/results/cpm_bsi_%A_%a.err
#SBATCH --array=0-49             # 50 BSI items (0-based indices)
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aj.simon@yale.edu

module purge
module load Python/3.10.8-GCCcore-12.2.0
source ~/BSI_modeling/bin/activate

echo "Starting job ${SLURM_ARRAY_TASK_ID}"

# ----- paths to your input files on McCleary -----
export AVERAGED_MATS_PATH=/home/ajs332/BSI_modeling/averaged_mats.mat
export BEHDATA_PATH=/home/ajs332/BSI_modeling/BSI_items.mat
export CONFOUNDS_PATH=/home/ajs332/BSI_modeling/confounds.mat

# where to dump outputs
OUTDIR=/home/ajs332/BSI_modeling/results
mkdir -p "$OUTDIR"

# this job's behavior index
export MEAS_INDEX=${SLURM_ARRAY_TASK_ID}

# unique output file per item
export OUTFILE=${OUTDIR}/items_test_predictions_item${SLURM_ARRAY_TASK_ID}.mat

# CPM hyperparams
export NPERMS=1000          # tweak as you like
export KFOLDS=10
export EDGE_THRESH=0.5
export RANDOM_STATE=42

# run the CPM script
python3 /home/ajs332/BSI_modeling/CPM_python.py