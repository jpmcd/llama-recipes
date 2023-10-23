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

python finetune.py --output_dir=${OUTPUT_DIR} \
    --run_validation \
    --model_name "/home/gridsan/JO30252/languagemodels/models/Llama-2-7b-hf-causal" \
    --quantization \
    --batch_size_training 16 \
    --val_batch_size 16 \
    --num_workers_dataloader 10 \
    --lr 1e-5