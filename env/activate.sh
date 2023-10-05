#!/bin/bash
# module load cuda/11.7
module load cuda/11.3
module unload anaconda/2022a
module load anaconda/2023a-pytorch

ENVDIR="/home/gridsan/${USER}/repos/llama-recipes/env"
LIBDIR="${ENVDIR}/lib/python3.8/site-packages"
# LOCALDIR="${HOME}/.local/lib/python3.8/site-packages/"
LOCALDIR=""
# check if python user base is set to envdir first
if [ -z $PYTHONUSERBASE ]; then
    export PYTHONPATH="${LIBDIR}:${LOCALDIR}"
    export OLD_PATH=$PATH
    export PATH="${ENVDIR}/bin:$PATH"
    export PYTHONUSERBASE=$ENVDIR
fi
