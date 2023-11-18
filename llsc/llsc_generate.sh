#!/bin/bash
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --gpupower=250
#SBATCH -o /home/gridsan/%u/repos/llama-recipes/slurm/slurm-%j.out

cd ${HOME}/repos/llama-recipes
source env/activate.sh

# MODEL_PATH="${HOME}/languagemodels/models/Llama-2-7b-hf-causal"
MODEL_PATH="${HOME}/languagemodels/models/Llama-2-7b-chat-hf"
FORM="${HOME}/repos/llama-recipes/llsc/form.txt"
DATA_PATH="${HOME}/repos/llama-recipes/llsc/inputs.json"
OUTPUT="${HOME}/repos/llama-recipes/llsc/output.txt"

python generate.py \
    --model_name ${MODEL_PATH} \
    --output ${OUTPUT} \
    --form ${FORM} \
    --data_path ${DATA_PATH} \
    --quantization \
    --batch_size_validation 8 \
    --num_workers_dataloader 10 \
    --max_length 512 \
    --skip_special_tokens
    # --peft_model "${HOME}/repos/llama-recipes/output/24094434_20231025_105250" \
