#!/bin/bash

#SBATCH --job-name="search"
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=131072

module load R/3.2.1
module load gcc/4.9.2

R --vanilla --args $SLURM_ARRAY_TASK_ID  < xgboost_train.R


