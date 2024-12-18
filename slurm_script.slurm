#!/bin/bash
#SBATCH -A a08tra01
#SBATCH -p boost_usr_prod
#SBATCH --time 00:03:00     # format: HH:MM:SS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=open_vit_test
#SBATCH --output=res.out
#SBATCH --error=err.out

srun ./run_cpp_nsys.sh --profile