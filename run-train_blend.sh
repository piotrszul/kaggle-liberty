#!/bin/bash

#SBATCH --job-name="train_blend"
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=131072

module load R/3.1.3

R --vanilla  < train_blend.R 
