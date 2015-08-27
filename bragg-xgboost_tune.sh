#!/bin/bash

#SBATCH --job-name="compare"
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=131072

module load R/3.0.2
module load gcc/4.9.2

R --vanilla --args 2  < xgboost_tune.R


