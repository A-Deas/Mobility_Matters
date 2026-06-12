#!/bin/bash
#SBATCH -A MED128              # your project allocation
#SBATCH -J repast_run          # job name
#SBATCH -p gpu_acmhs           # Section's DGX box, but could use "batch" instead
#SBATCH -N 1                   # 1 node
#SBATCH --gres=gpu:0           # Explicitly request 0 GPUs
#SBATCH --cpus-per-task=1      # This sets the number of CPUs per MPI process
#SBATCH -t 20:00:00            # wall time
#SBATCH --mem=0                # let the job use all available memory on the node (probably the default, but just being safe)
#SBATCH -o /ccsopen/home/p5d/Utah_ABM/Logs/debug_output.log  # redirect output log
#SBATCH -e /ccsopen/home/p5d/Utah_ABM/Logs/debug_error.log   # redirect error log
#SBATCH --open-mode=truncate

# Load modules
module purge
module load DefApps gcc/12.2.0 openmpi/4.0.4
source /sw/baseline/miniconda/3.11/anaconda-base/etc/profile.d/conda.sh
conda activate /ccsopen/home/p5d/.conda/envs/utah_abm

# Run your program
srun -n 8 python simulation.py parameters.yaml

