#!/bin/bash
#SBATCH --job-name=hod_mocks
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=desi
#SBATCH --output="jobs/hods_%a.out"
#SBATCH --error="jobs/hods_error_%a.out"

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# run parallelised mock generation on a single node
srun -n 1 -c 64 python3 generate_hods.py --chunk 0 22 &
srun -n 1 -c 64 python3 generate_hods.py --chunk 22 43 &
srun -n 1 -c 64 python3 generate_hods.py --chunk 43 64 &
srun -n 1 -c 64 python3 generate_hods.py --chunk 64 85 &
wait
