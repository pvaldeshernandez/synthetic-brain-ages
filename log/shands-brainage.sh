#!/bin/bash
#SBATCH --job-name=shands-brainage
#SBATCH --partition=gpu
#SBATCH --gpus=a100:2
#SBATCH --account=cruzalmeida
#SBATCH --qos=cruzalmeida
# Outputs ----------------------------------
#SBATCH --output=log/shands-brainage-%j-%A-%a.out
#SBATCH --error=log/shands-brainage-%j-%A-%a.err
# ------------------------------------------
pwd; hostname; date

#==============Shell script==============#
#Load the software needed
ml conda
conda activate /orange/cruzalmeida/pvaldeshernandez/projects/envs/shands-brainage_env


cmd="python /orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/shands_brainage.py"

echo Commandline: $cmd
    eval $cmd
exitcode=$?


date
