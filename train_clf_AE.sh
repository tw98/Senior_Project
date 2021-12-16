#!/bin/bash

#SBATCH --partition=psych_day
#SBATCH --job-name=forrest_manifold_mlp
#SBATCH --output=results/classification_fMRI_AE.txt
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

BS=64
LAMBDA=0.01 #the common hidden lambda
MANILAM=0.01 # manifold embedding lambda
CLFLAM=0.1

NEPOCHS=4000
SAVEFREQ=1000

DATAP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/sherlock/MNI152_3mm_data/denoised_filtered_smoothed/ROI_data/${ROI}/data
DATANM=${ROI}_sherlock_movie.npy

EMBEDP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/sherlock/MNI152_3mm_data/denoised_filtered_smoothed/ROI_data/${ROI}/embeddings
EMBEDNM=${ROI}_split0_${ZDIM}dimension_embedding_PHATE.npy

LABELNAME=IN_OUT
LABELP=/home/tw496/data_sherlock/sherlock_label_inout.npy

HALF=1
REG=mse
AE=mlp_md
#LR=1E-2
PTNUM=16 # only 16 patients now.
N_TR=1976

RUN_LABEL=sherlock_clf_${ROI}_${HIDDIM}_${ZDIM}_${LAMBDA}_${MANILAM}_${CLFLAM}_HALF${HALF}

SUMMARY_FILE=results/fMRI_clf_AE_sherlock.csv

python3 fMRI_classification_AE.py  --datapath=$DATAP --datanaming=$DATANM \
--embedpath=$EMBEDP --embednaming=$EMBEDNM \
--n_TR=$N_TR --TR_range=$HALF --hidden_dim=$HIDDIM --zdim=$ZDIM \
--batch_size=$BS --lam=$LAMBDA --lam_mani=$MANILAM --lam_clf=$CLFLAM \
--n_epochs=$NEPOCHS --reg=$REG --reg_ref --mmd_specsig \
--ae_type=$AE --shuffle_reg --n_pt=$PTNUM \
--save_model --save_freq=$SAVEFREQ \
--downstream --labeltype=$LABELNAME --labelpath=$LABELP --experiment_label=$RUN_LABEL \
--summary_file=$SUMMARY_FILE

HALF=2
EMBEDNM=${ROI}_split1_${ZDIM}dimension_embedding_PHATE.npy
RUN_LABEL=sherlock_clf_${ROI}_${HIDDIM}_${ZDIM}_${LAMBDA}_${MANILAM}_${CLFLAM}_HALF${HALF}

python3 fMRI_classification_AE.py  --datapath=$DATAP --datanaming=$DATANM \
--embedpath=$EMBEDP --embednaming=$EMBEDNM \
--n_TR=$N_TR --TR_range=$HALF --hidden_dim=$HIDDIM --zdim=$ZDIM \
--batch_size=$BS --lam=$LAMBDA --lam_mani=$MANILAM --lam_clf=$CLFLAM \
--n_epochs=$NEPOCHS --reg=$REG --reg_ref --mmd_specsig \
--ae_type=$AE --shuffle_reg --n_pt=$PTNUM \
--save_model --save_freq=$SAVEFREQ \
--downstream --labeltype=$LABELNAME --labelpath=$LABELP --experiment_label=$RUN_LABEL \
--summary_file=$SUMMARY_FILE