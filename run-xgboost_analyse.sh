#!/bin/bash

#SBATCH --job-name="analyze"
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=131072

module load R/3.2.1
module load gcc/4.9.2

R --vanilla --args 2  < xgboost_analyse.R


