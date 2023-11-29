#!/bin/bash
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
##SBATCH --partition=test
##SBATCH --gres=gpu:ampere:1
#SBATCH --gres=gpu:volta:1
#SBATCH --gpupower=250
##SBATCH -o /home/gridsan/%u/repos/llama-recipes/slurm/slurm-%j.out
#SBATCH -o /home/gridsan/%u/repos/mimic/slurm-%j.out


cd ${HOME}/repos/llama-recipes
source env/activate.sh

START_TIME=$(date +"%Y%m%d_%H%M%S")

# OUTPUT_DIR="${HOME}/repos/llama-recipes/output/${SLURM_JOBID}_${START_TIME}"
MODEL_PATH="${HOME}/languagemodels/models/Llama-2-7b-hf-causal"
FORM="${HOME}/repos/mimic/form_r.txt"
CONTEXT="${HOME}/repos/mimic/context_r.txt"
DATA_PATH="${HOME}/repos/mimic/dataset_records.pkl"
OUTPUT_DIR="${HOME}/repos/mimic/output/${SLURM_JOBID}_${START_TIME}"

mkdir -p ${OUTPUT_DIR}

python finetune.py --output_dir=${OUTPUT_DIR} \
    --form ${FORM} \
    --context ${CONTEXT} \
    --data_path ${DATA_PATH} \
    --model_name ${MODEL_PATH} \
    --use_peft \
    --peft_method lora \
    --quantization \
    --num_epochs 1 \
    --max_training_data 12000 \
    --max_validation_data 400 \
    --keep_last \
    --batch_size_training 16 \
    --val_batch_size 8 \
    --num_workers_dataloader 10 \
    --max_length 1024 \
    --lr 1e-5
#    --run_validation \
#    --shuffle \