#!/bin/bash
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --gpupower=250
#SBATCH -o /home/gridsan/%u/repos/llama-recipes/slurm/slurm-%j.out

cd ${HOME}/repos/llama-recipes
source env/activate.sh

START_TIME=$(date +"%Y%m%d_%H%M%S")

OUTPUT_DIR="${HOME}/repos/llama-recipes/output/${SLURM_JOBID}_${START_TIME}"
MODEL_PATH="${HOME}/languagemodels/models/Llama-2-7b-hf-causal"

mkdir -p ${OUTPUT_DIR}

python finetune.py --output_dir=${OUTPUT_DIR} \
    --run_validation \
    --model_name ${MODEL_PATH} \
    --use_peft \
    --peft_method lora \
    --quantization \
    --num_epochs 1 \
    --max_training_data 0 \
    --max_validation_data 64 \
    --batch_size_training 16 \
    --val_batch_size 16 \
    --num_workers_dataloader 10 \
    --lr 1e-5