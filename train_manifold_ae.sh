#!/bin/bash

#SBATCH --partition=psych_day
#SBATCH --job-name=forrest_manifold_mlp
#SBATCH --output=results/forrest_viz_mlp_mani_lam0.01_1half.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tom.wallenstein@yale.edu
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --mem=50G

module load miniconda
conda activate tjl

# ROI=late_visual_neurosynth
# HIDDIM=64
# ZDIM=20

# LAMBDA=100 #the common hidden lambda
# MANILAM=0.1 # manifold embedding lambda

# BS=64
# LAMBDA=0.1 #the common hidden lambda
# MANILAM=0.1 # manifold embedding lambda
# NEPOCHS=4000
# SAVEFREQ=1000
# DATAP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/MNI152_3mm_data/denoised_filtered_smoothed/ROI_data/${ROI}/data
# # DATANM=${ROI}_movie_all_runs.npy
# DATANM=${ROI}_forrest_movie.npy
# EMBEDP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/MNI152_3mm_data/denoised_filtered_smoothed/ROI_data/${ROI}/embeddings
# EMBEDNM=${ROI}_forrest_movie_${ZDIM}dimension_embedding_PHATE.npy
# TESTP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/MNI152_3mm_data/denoised_filtered_smoothed/ROI_data/${ROI}/data
# # TESTNM=${ROI}_localizer_all_runs.npy
# TESTNM=${ROI}_forrest_localizer.npy
# N_TEST=624
# LABELP=/home/tw496/data_StudyForrest/localizer_labels_sub-01_all_runs.csv
# #LOADP=/gpfs/milgram/scratch60/turk-browne/jh2752/results/sherlock_viz_17pt_hiddendim64_bs64_300epochs_reg_lam0.01_amlp_1half
# #LOADE=150
# # HALF=2
# REG=mse
# AE=mlp_md
# #LR=1E-2
# PTNUM=10 # only 16 patients now.
# N_TR=3599
# RUN_LABEL=preprocess1_forrest_paper_${ROI}_${HIDDIM}_${ZDIM}_${LAMBDA}_${MANILAM}

# SUMMARY_FILE=results/fMRI_manifoldAE_localizer_result_summary2.csv

# python3 fMRI_Manifold_AE.py  --datapath=$DATAP --datanaming=$DATANM \
# --embedpath=$EMBEDP --embednaming=$EMBEDNM \
# --test_path=$TESTP --testnaming=$TESTNM --test_range=$N_TEST \
# --n_TR=$N_TR --hidden_dim=$HIDDIM --zdim=$ZDIM \
# --batch_size=$BS --lam=$LAMBDA --lam_mani=$MANILAM --n_epochs=$NEPOCHS \
# --reg=$REG --reg_ref --mmd_specsig \
# --ae_type=$AE --shuffle_reg --n_pt=$PTNUM \
# --save_model --save_freq=$SAVEFREQ \
# --downstream --labelpath=$LABELP --experiment_label=$RUN_LABEL  --summary_file=$SUMMARY_FILE

####################################################################################################################################

ROI=late_visual_neurosynth
HIDDIM=64
ZDIM=20

BS=64
LAMBDA=0.01 #the common hidden lambda
MANILAM=2 # manifold embedding lambda
NEPOCHS=4000
SAVEFREQ=1000
DATAP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/MNI152_3mm_data/denoised_filtered_smoothed/ROI_data/${ROI}/data
# DATANM=${ROI}_movie_all_runs.npy
DATANM=${ROI}_forrest_movie.npy
EMBEDP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/MNI152_3mm_data/denoised_filtered_smoothed/ROI_data/${ROI}/embeddings
EMBEDNM=${ROI}_movie_split0_${ZDIM}dimension_embedding_PHATE.npy
LABELP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/forrest_movie_labels_coded_expanded.csv
#LOADP=/gpfs/milgram/scratch60/turk-browne/jh2752/results/sherlock_viz_17pt_hiddendim64_bs64_300epochs_reg_lam0.01_amlp_1half
#LOADE=150
HALF=1
REG=mse
AE=mlp_md
#LR=1E-2
PTNUM=10 # only 16 patients now.
N_TR=3599
RUN_LABEL=preprocess1_forrest_movie_${ROI}_${HIDDIM}_${ZDIM}_${LAMBDA}_${MANILAM}_${HALF}

SUMMARY_FILE=results/fMRI_manifoldAE_movie_result_summary2.csv


# ROI=early_visual
# BRAIN_HALF=rh
# HIDDIM=20
# ZDIM=10

# BS=64
# LAMBDA=0.01 #the common hidden lambda
# MANILAM=1E-2 # manifold embedding lambda
# NEPOCHS=4000
# SAVEFREQ=1000
# DATAP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/fsaverage_data/denoised_filtered_smoothed/ROI_data/${ROI}/data
# DATANM=movie_all_runs_${ROI}_${BRAIN_HALF}.npy
# EMBEDP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/fsaverage_data/denoised_filtered_smoothed/ROI_data/${ROI}/embeddings
# EMBEDNM=${ROI}_movie_split1_${ZDIM}dimension_embedding_PHATE_${BRAIN_HALF}.npy
# # TESTP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/fsaverage_data/denoised_filtered_smoothed/ROI_data/early_visual/data
# # TESTNM=localizer_all_runs_early_visual_rh.npy
# N_TEST=624
# LABELP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/forrest_movie_labels_coded_expanded.csv
# #LOADP=/gpfs/milgram/scratch60/turk-browne/jh2752/results/sherlock_viz_17pt_hiddendim64_bs64_300epochs_reg_lam0.01_amlp_1half
# #LOADE=150
# HALF=2
# REG=mse
# AE=mlp_md
# #LR=1E-2
# PTNUM=10 # only 16 patients now.
# N_TR=3599
# RUN_LABEL=good_forrest_${BRAIN_HALF}_${ROI}_TRAIN_HALF${HALF}

python3 fMRI_Manifold_AE.py  --datapath=$DATAP --datanaming=$DATANM \
--embedpath=$EMBEDP --embednaming=$EMBEDNM \
--n_TR=$N_TR --TR_range=$HALF --hidden_dim=$HIDDIM --zdim=$ZDIM \
--batch_size=$BS --lam=$LAMBDA --lam_mani=$MANILAM --n_epochs=$NEPOCHS \
--reg=$REG --reg_ref --mmd_specsig \
--ae_type=$AE --shuffle_reg --n_pt=$PTNUM \
--save_model --save_freq=$SAVEFREQ \
--downstream --labelpath=$LABELP --experiment_label=$RUN_LABEL \
--summary_file=$SUMMARY_FILE

HALF=2
EMBEDNM=${ROI}_movie_split1_${ZDIM}dimension_embedding_PHATE.npy
RUN_LABEL=preprocess1_forrest_movie_${ROI}_${HIDDIM}_${ZDIM}_${LAMBDA}_${MANILAM}_${HALF}

python3 fMRI_Manifold_AE.py  --datapath=$DATAP --datanaming=$DATANM \
--embedpath=$EMBEDP --embednaming=$EMBEDNM \
--n_TR=$N_TR --TR_range=$HALF --hidden_dim=$HIDDIM --zdim=$ZDIM \
--batch_size=$BS --lam=$LAMBDA --lam_mani=$MANILAM --n_epochs=$NEPOCHS \
--reg=$REG --reg_ref --mmd_specsig \
--ae_type=$AE --shuffle_reg --n_pt=$PTNUM \
--save_model --save_freq=$SAVEFREQ \
--downstream --labelpath=$LABELP --experiment_label=$RUN_LABEL \
--summary_file=$SUMMARY_FILE