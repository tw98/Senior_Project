###########################################################
##  reconstruct one subject's test half using another subjects test half
##  get MSE
##  input: X_i
##  encoder + decoder_j
##  reconstruct output X^i_j
##  MSE(X^i_j, X_j)
##  MSE(X^_j, X_j) compare
###########################################################

import warnings  # Ignore sklearn future warning
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import pandas as pd
import numpy as np
from lib.helper import drive_decoding
import os
import torch
from lib.autoencoder import Encoder_conv,Decoder_conv, Encoder_basic, Decoder_basic, Decoder_Manifold
from lib.helper import extract_hidden_reps, drive_decoding, extract_hidden_n_rconst
from lib.fMRI import fMRI_Time_Subjs_Dataset, fMRI_Time_Subjs_Embed_Dataset, fMRIAutoencoderDataset
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--n_pt', type=int, default =16)
parser.add_argument('--n_TR', type = int, default = 1976)
parser.add_argument('--ROI', type = str, default = 'early_visual')
parser.add_argument('--hiddim', type = int, default = 64)
parser.add_argument('--zdim', type = int, default = 20)
parser.add_argument('--input_subj', type = int, default=1) # the subject used as input data X_i
parser.add_argument('--load_epoch', type = int, default=4000)
parser.add_argument('--volsurf', type = str, default='MNI152_3mm_data') # default is volumetric data , alternatively fsaverage_data for surface data
parser.add_argument('--embedpath',type=str,default=None)
parser.add_argument('--TR_range', type = int, default = None) #TR_range=1 is first half, 2 second
parser.add_argument('--ae_type', type =str, default='mlp_md') # choices: 'conv', 'mlp', 'mlp_md'(mlp with manifold bottleneck decoder)
parser.add_argument('--not_regref', action ='store_true')

def checkexist(targetdf, df, rowk, cols, additional ):
    for col in cols :
        targetdf = targetdf.loc[targetdf[col]==df.loc[rowk,col]]
        if targetdf.shape[0]<1:
            return False
    for key in additional.keys():
        targetdf = targetdf.loc[targetdf[key] == additional[key]]
        if targetdf.shape[0]<1:
            return False
    return True




def main():

    folder = '/home/tw496/fMRI_AE'
    SUMMARYFILE = 'results/cross_subject_translation.csv'  # the volumetric input data

    args = parser.parse_args()
    outfile = 'results/fMRI_manifoldAE_result_xsubj_translation.csv' #record the embeddings and MSE

    if args.not_regref:
        SUMMARYFILE='results/cross_subject_translation.csv'
        outfile = 'results/fMRI_manifoldAE_result_xsubj_translation_pairsRegcommon.csv'
    datapath = f"/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/{args.volsurf}/denoised_filtered_smoothed/ROI_data/{args.ROI}/data"
    datanaming = f"{args.ROI}_movie_all_runs.npy"
    embedpath = f'/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/{args.volsurf}/denoised_filtered_smoothed/ROI_data/{args.ROI}/embeddings'
    if args.embedpath is None:
        args.embedpath = embedpath
    else:
        embedpath = args.embedpath

    df = pd.read_csv(os.path.join(folder, SUMMARYFILE))
    # print(df)
    # don't do the not training large lambda ones
    # df = df.loc[df.lambda_common<10]

    df = df.loc[(df.ROI==args.ROI) &
                (df.common_hiddendim==args.hiddim) &
                (df.manifold_embed_dim==args.zdim) &
                (df.train_half == args.TR_range)][['ROI','common_hiddendim','manifold_embed_dim',
                                                   'lambda_common','lambda_mani','load_path','train_half']].reset_index(drop=True)


    # randomly select a different subject j and reconstruct X^i_j using X_i
    subj_i = args.input_subj
    # subj_j = np.random.choice(np.setxor1d(np.arange(1, args.n_pt + 1), [args.input_subj]))


    outdf = None
    if os.path.exists(outfile):
        outdf_old = pd.read_csv(outfile)
    else:
        outdf_old = None

    print(f'rows in dataframe {df.shape[0]}')
    for k in range(df.shape[0]):
        if outdf_old is not None:
            exist = checkexist(outdf_old, df, k, df.columns.tolist(), {'input_subj':args.input_subj})
            if exist:
                print(df.loc[k,:], 'already run')
                continue
        path = df.loc[k, 'load_path']
        print(df.loc[k,['lambda_common','lambda_mani']])
        checkpoint = torch.load(os.path.join(path, f'ae_e{args.load_epoch}.pt'))

        data_3d = False
        train_half = df.loc[k, 'train_half']
        if train_half is not None:
            args.TR_range = train_half
            args.n_TR = args.n_TR // 2
            if train_half == 1:
                TR_range = np.arange(args.n_TR, 2 * args.n_TR)  # train on 1st half, test on 2nd half
            else:
                TR_range = np.arange(args.n_TR)  # train on 2nd half, test on 1st half
        else: TR_range = np.arange(args.n_TR)
        dataset = fMRIAutoencoderDataset([args.input_subj],
                                         datapath,
                                         TR_range,
                                         data_3d=data_3d,
                                         data_name_suffix=datanaming)

        input_size=dataset.get_TR_dims()[0]
        embed_size = args.zdim

        encoder = Encoder_basic(input_size, args.hiddim * 4, args.hiddim * 2, args.hiddim)
        decoders = []
        for i in range(args.n_pt):
            decoder = Decoder_Manifold(embed_size, args.hiddim, args.hiddim * 2, args.hiddim * 4,
                                       input_size)
            decoders.append(decoder)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        for i in range(1, args.n_pt + 1):
            decoders[i - 1].load_state_dict(checkpoint[f'decoder_{i}_state_dict'])
        print('loaded models')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder.to(device)
        encoder.eval()

        patient_ids = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15]
        for subj_j in np.setxor1d(patient_ids, [args.input_subj]):

            print(f"translate subj{subj_i} to subj{subj_j}")

            decoder = decoders[patient_ids.index(subj_j)-1]
            decoder.to(device)
            decoder.eval()

            _,_,Xhat_ij = extract_hidden_n_rconst(encoder, [decoder], dataset, device, None, args)

            # Also do the Xhat_j
            dataset_j = fMRIAutoencoderDataset([subj_j],
                                         datapath,
                                         TR_range,
                                         data_3d=data_3d,
                                         data_name_suffix=datanaming)
            _,_, Xhat_j = extract_hidden_n_rconst(encoder, [decoder], dataset_j, device, None, args)


            X_j = np.load(os.path.join(datapath, f"sub-{subj_j:02}_{datanaming}"))[TR_range]

            # X_j = torch.from_numpy(X_j) # make the target subject data as torch tensor
            # Xhat_ij = torch.from_numpy(Xhat_ij)
            # Xhat_j = torch.from_numpy(Xhat_j)
            # # print(Xhat_ij.shape, Xhat_j.shape, X_j.shape)
            #
            # loss_criterion = torch.nn.MSELoss()
            #
            # loss_translate = loss_criterion(Xhat_ij.view(X_j.shape), X_j)
            # loss_rconst = loss_criterion(Xhat_j.view(X_j.shape), X_j)

            loss_translate = mean_squared_error(Xhat_ij.reshape(X_j.shape), X_j)
            loss_rconst = mean_squared_error(Xhat_j.reshape(X_j.shape), X_j)

            #permute the ordering of Xhat_ij and check MSE of this random construction
            # idx = torch.randperm(X_j.shape[0])
            idx = np.random.permutation(X_j.shape[0])
            loss_permtranslate = mean_squared_error(Xhat_ij[idx], X_j)

            corr_translate = np.corrcoef(np.ravel(Xhat_ij), np.ravel(X_j))[0,1]
            corr_rconst = np.corrcoef(np.ravel(Xhat_j), np.ravel(X_j) )[0,1]
            corr_permtranslate = np.corrcoef(np.ravel(Xhat_ij[idx]), np.ravel(X_j) )[0,1]

            cols =['ROI','common_hiddendim','manifold_embed_dim',
                   'lambda_common','lambda_mani','load_path','train_half',
                   'input_subj', 'decode_subj','translation_MSE', 'reconstruction_MSE', 'permute_translation_MSE',
                   'translation_corr', 'reconstruction_corr', 'permute_translation_corr']
            entry = [df.loc[k,col] for col in cols if col in df.columns]
            entry.extend([subj_i, subj_j, loss_translate, loss_rconst, loss_permtranslate,
                          corr_translate, corr_rconst, corr_permtranslate])
            print(entry)

            if outdf is None:
                outdf = pd.DataFrame(columns = cols)
            outdf.loc[len(outdf)]=entry

    if os.path.exists(outfile):
        outdf_old = pd.read_csv(outfile)
        outdf = pd.concat([outdf_old,outdf])

    outdf.to_csv(outfile, index=False)



if __name__=='__main__':
    main()