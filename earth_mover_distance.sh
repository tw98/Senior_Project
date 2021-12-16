#!/bin/bash

##SBATCH --partition=psych_day
#SBATCH --job-name=emd
#SBATCH --output=results/emd.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tom.wallenstein@yale.edu
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --mem=80G

module load miniconda
conda activate tjl

python3 earth_mover_distance.py