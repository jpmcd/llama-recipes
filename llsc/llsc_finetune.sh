#!/bin/bash
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --gpupower=250
#SBATCH -o /home/gridsan/%u/repos/llama-recipes/slurm/slurm-%j.out

cd ${HOME}/repos/llama-recipes
source env/activate.sh

OUTPUT_DIR="${HOME}/repos/llama-recipes/output"

python finetune.py --output_dir=${OUTPUT_DIR}