#!/bin/bash

# Change these!
#SBATCH --job-name=small_scale_tmars_5x
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=2
#SBATCH --nodelist=locus-1-29
#SBATCH --cpus-per-task=10
#SBATCH --time=3-12
#SBATCH --gpus=2
#SBATCH --mem=60GB
#SBATCH --mail-type=END
#SBATCH --mail-user=sachingo@andrew.cmu.edu
#SBATCH --output=slurm_logs/small_scale_tmars_5x.out
#SBATCH --error=slurm_logs/small_scale_tmars_5x.err
#SBATCH --requeue

# Example usage:
# sbatch slurm_train.sh
# Run using conda and make sure to have the conda env activated when running sbatch.


module load openmpi
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12803
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
echo go $COUNT_NODE
echo $HOSTNAMES

# Change these as needed!

# DATA_PATH="/project_data/datasets/datanet/small_scale/shards"
DATA_PATH="/project_data/datasets/datanet/small_scale_tmars/shards/"
SCALE="small_5x"
SEED=0
OUTPUT_DIR="/project_data2/projects/sachingo/datacomp_checkpoints/logs/"
NUM_CHECKPOINTS=5
EXP_NAME="smallscale_tmars_5x_$(date +%Y%m%d_%H%M%S)"
PRECISION="amp"  # We recommend using amp_bfloat16 if supported by your hardware.
if [ "$SCALE" == "xlarge" ]; then
    PRECISION="amp_bfloat16" # amp results in a significant performance drop at xlarge scale
fi

conda init bash
conda init
conda activate open_clip
# Change comment as needed
srun python train.py \
--scale ${SCALE} \
--data_dir ${DATA_PATH} \
--output_dir ${OUTPUT_DIR} \
--exp_name ${EXP_NAME} \
--precision ${PRECISION} \
--num_checkpoints ${NUM_CHECKPOINTS} \
--seed ${SEED} \
--report_to_wandb \
--accum_freq 1 \
--save_frequency 1
