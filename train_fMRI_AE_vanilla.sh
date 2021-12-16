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

######################################################################################################################
#########      LOCALIZER       ######################################################################################

ROI=late_visual_neurosynth
HIDDIM=64
ZDIM=20

BS=64
LAMBDA=0 #the common hidden lambda
MANILAM=1E-2 # manifold embedding lambda
NEPOCHS=4000
SAVEFREQ=1000
DATAP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/MNI152_3mm_data/denoised_filtered_smoothed/ROI_data/${ROI}/data
# DATANM=${ROI}_movie_all_runs.npy
DATANM=${ROI}_forrest_movie.npy
TESTP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/MNI152_3mm_data/denoised_filtered_smoothed/ROI_data/${ROI}/data
# TESTNM=${ROI}_localizer_all_runs.npy
TESTNM=${ROI}_forrest_localizer.npy
N_TEST=624
LABELP=/home/tw496/data_StudyForrest/localizer_labels_sub-01_all_runs.csv
#LOADP=/gpfs/milgram/scratch60/turk-browne/jh2752/results/sherlock_viz_17pt_hiddendim64_bs64_300epochs_reg_lam0.01_amlp_1half
#LOADE=150
REG=mse
AE=mlp
#LR=1E-2
PTNUM=10 # only 16 patients now.
N_TR=3599
RUN_LABEL=preprocess1_forrest_paper_${ROI}_${ZDIM}vanilla

SUMMARY_FILE=results/fMRI_AE_vanilla_localizer_result_summary.csv


python3 fMRI_AE_vanilla.py  --datapath=$DATAP --datanaming=$DATANM \
--test_path=$TESTP --testnaming=$TESTNM --test_range=$N_TEST \
--n_TR=$N_TR --hidden_dim=$HIDDIM --zdim=$ZDIM \
--batch_size=$BS --lam=$LAMBDA --n_epochs=$NEPOCHS \
--reg=$REG --reg_ref --mmd_specsig \
--ae_type=$AE --shuffle_reg --n_pt=$PTNUM \
--save_model --save_freq=$SAVEFREQ \
--downstream --labelpath=$LABELP --experiment_label=$RUN_LABEL  --summary_file=$SUMMARY_FILE
--lr=$LR


#########################################################################################################################
##########      WITHIN MOVIE       ######################################################################################

# ROI=late_visual_neurosynth
# HIDDIM=64
# ZDIM=20

BS=64
LAMBDA=0 #the common hidden lambda
MANILAM=1E-2 # manifold embedding lambda
NEPOCHS=4000
SAVEFREQ=1000
DATAP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/MNI152_3mm_data/denoised_filtered_smoothed/ROI_data/${ROI}/data
# DATANM=${ROI}_movie_all_runs.npy
DATANM=${ROI}_forrest_movie.npy
HALF=1
LABELP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/forrest_movie_labels_coded_expanded.csv
#LOADP=/gpfs/milgram/scratch60/turk-browne/jh2752/results/sherlock_viz_17pt_hiddendim64_bs64_300epochs_reg_lam0.01_amlp_1half
#LOADE=150
REG=mse
AE=mlp
#LR=1E-2
PTNUM=10 # only 16 patients now.
N_TR=3599
RUN_LABEL=preprocess1_forrest_movie_${ROI}_vanilla_${HALF}

SUMMARY_FILE=results/fMRI_AE_vanilla_movie_result_summary.csv

python3 fMRI_AE_vanilla.py  --datapath=$DATAP --datanaming=$DATANM \
--n_TR=$N_TR --TR_range=$HALF --hidden_dim=$HIDDIM --zdim=$ZDIM \
--batch_size=$BS --lam=$LAMBDA --n_epochs=$NEPOCHS \
--reg=$REG --reg_ref --mmd_specsig \
--ae_type=$AE --shuffle_reg --n_pt=$PTNUM \
--save_model --save_freq=$SAVEFREQ \
--downstream --labelpath=$LABELP --experiment_label=$RUN_LABEL  --summary_file=$SUMMARY_FILE


HALF=2
RUN_LABEL=preprocess1_forrest_movie_${ROI}_vanilla_${HALF}

python3 fMRI_AE_vanilla.py  --datapath=$DATAP --datanaming=$DATANM \
--n_TR=$N_TR --TR_range=$HALF --hidden_dim=$HIDDIM --zdim=$ZDIM \
--batch_size=$BS --lam=$LAMBDA --n_epochs=$NEPOCHS \
--reg=$REG --reg_ref --mmd_specsig \
--ae_type=$AE --shuffle_reg --n_pt=$PTNUM \
--save_model --save_freq=$SAVEFREQ \
--downstream --labelpath=$LABELP --experiment_label=$RUN_LABEL  --summary_file=$SUMMARY_FILE