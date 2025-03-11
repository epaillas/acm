#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi_g
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus 4
#SBATCH -q regular
#SBATCH -t 12:00:00

python /global/u1/e/epaillas/code/acm/projects/emc/training/optimize_sunbird.py