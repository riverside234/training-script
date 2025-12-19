#!/bin/bash
#SBATCH --job-name=fl-pe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:h100:4
#SBATCH --time=0-00:30:00

# --- LOGGING ---------------------------------------------------------------
# %x = job name, %A = job ID, %a = array-task ID
#SBATCH --output=/scratch/user/u.yx314365/logs/%x_%A_%a.out
#SBATCH --open-mode=append              # append if file already exists
# --------------------------------------------------------------------------


set -e
export http_proxy=http://10.71.8.1:8080
export https_proxy=http://10.71.8.1:8080

module load CUDA/12.8.0
CUDA_ROOT=$(dirname "$(dirname "$(which nvcc)")")
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# quick debug print

echo "Using CUDA from: $CUDA_HOME"
which nvcc


module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"
conda activate xu2

#conda list

cd /scratch/user/u.yx314365/pp-train/

#ls
export PYTORCH_ALLOC_CONF=expandable_segments:True

srun deepspeed --num_gpus=4 train-dp-pp.py --deepspeed_config ds_config_0.json
