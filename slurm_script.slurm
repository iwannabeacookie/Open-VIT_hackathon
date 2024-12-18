#!/bin/bash
#SBATCH -A a08tra01
#SBATCH -p boost_usr_prod
#SBATCH --time 00:03:00     # format: HH:MM:SS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=open_vit_test

srun ./run_cpp_nsys.sh --profile

# srun -N 1 -p boost_usr_prod -A tra24_hckunitn  --reservation=s_tra_hckunitn -t 00:03:00 --gres=gpu:1 run_acc_nsys.sh --profile