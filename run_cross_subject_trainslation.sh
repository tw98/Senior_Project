#!/bin/bash

#SBATCH --partition=psych_day
#SBATCH --job-name=manifold_AE_cross_subject
#SBATCH --output=results/manifold_AE_cross_subject
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tom.wallenstein@yale.edu
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --mem=50G

module load miniconda
conda activate tjl

ROI=early_visual
HIDDIM=64
ZDIM=20
HALF=1
N_PT=10
N_TR=3599

for ROI in early_visual early_visual_neurosynth aud_early
do
    for PT in 1 2 3 4 5 6 9 10 14 15
    do
        python3 cross-subject-translation.py --ROI=$ROI --hiddim=$HIDDIM --zdim=$ZDIM \
        --input_subj=$PT --n_TR=$N_TR --TR_range=$HALF --n_pt=$N_PT
    done
done