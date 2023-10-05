#!/bin/bash
export PYTHONPATH=""
export PATH="${OLD_PATH}"
export PYTHONUSERBASE=""

module unload anaconda/2023a-pytorch
module load anaconda/2022a
