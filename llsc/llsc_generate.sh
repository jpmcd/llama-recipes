#!/bin/bash
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --gpupower=250
#SBATCH -o /home/gridsan/%u/repos/llama-recipes/slurm/slurm-%j.out

cd ${HOME}/repos/llama-recipes
source env/activate.sh

MODEL_PATH="${HOME}/languagemodels/models/Llama-2-7b-hf-causal"
FORM="${HOME}/repos/llama-recipes/llsc/form.txt"
DATA_PATH="${HOME}/repos/llama-recipes/llsc/inputs.json"
OUTPUT="${HOME}/repos/llama-recipes/llsc/output.txt"

python generate.py \
    --model_path ${MODEL_PATH} \
    --output ${OUTPUT} \
    --form ${FORM} \
    --data_path ${DATA_PATH} \
    --quantization \
    --max_length 512 \
    --skip_special_tokens