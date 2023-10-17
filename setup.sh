#!/bin/bash
module unload anaconda/2022a
module load cuda/11.6
module load anaconda/2023a-pytorch

ENVDIR="/home/gridsan/${USER}/repos/llama-recipes/env"
export PYTHONUSERBASE=$ENVDIR
export TMPDIR=/state/partition1/user/${USER}
mkdir -p $TMPDIR
pip install --user --no-cache-dir -r requirements.txt

# # To download a large model from Hugging Face hub...
# salloc --gres=gpu:volta:1 --cpus-per-task=20 --time=0:30:00 --qos=high  srun  --pty bash -i
# source repos/llama-recipes/env/activate.sh
# export HF_HOME=/home/gridsan/${USER}/.cache/huggingface/
# export TRANSFORMERS_OFFLINE=1

# from transformers import LlamaForCausalLM
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")
# model.save_pretrained("languagemodels/models/Llama-2-7b-chat-hf")
