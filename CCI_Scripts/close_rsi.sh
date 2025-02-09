#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --cpus-per-task=8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
venv/bin/python3 main.py 5 close_normalized,rsi_normalized