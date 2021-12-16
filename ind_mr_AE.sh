#!/bin/bash

##SBATCH --partition=psych_day
#SBATCH --job-name=forrest_ind_mrAE
#SBATCH --output=results/sher_ind_mrAE.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tom.wallenstein@yale.edu
#SBATCH --time=23:59:00
#SBATCH -c 8
#SBATCH --mem=80G


module load miniconda
conda activate tjl

BS=64
NEPOCHS=4000
SAVEFREQ=1000
PTNUM=10 # only 16 patients now.
VOLSURF=MNI152_3mm_data #  for surface data or MNI152_3mm_data for volumetric data
HIDDIM=64
ZDIM=20
EMBED=PHATE
LAMMANI=0.1
LAM=0 #
TR=3599

for ROI in late_visual_neurosynth #early_visual early_visual_neurosynth aud_early pmc_nn
do
  for pt in 1 2 3 4 5 6 9 10 14 15
  do
    for HALF in 1 2
    do
      SPLIT=$((HALF-1))
      EMBEDNM=${ROI}_movie_split${SPLIT}_${ZDIM}dimension_embedding_${EMBED}.npy
      LABELP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/forrest_movie_labels_coded_expanded.csv
      # DATANM=${ROI}_movie_all_runs.npy
      DATANM=${ROI}_forrest_movie.npy

      python3 ind_mr_AE.py --ROI=$ROI --volsurf=$VOLSURF --pt=$pt \
      --n_TR=$TR --TR_range=$HALF --n_pt=1 --hidden_dim=$HIDDIM --zdim=$ZDIM \
      --datanaming=$DATANM --embednaming=$EMBEDNM --labelpath=$LABELP \
      --lam_mani=$LAMMANI
    done
  done
done