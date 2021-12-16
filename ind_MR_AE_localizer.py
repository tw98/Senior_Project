# Individual manifold regularized autoencoder
# 1. classification on the second half (train on the first half)
# 2. interpolate manifold embedding on slices of data and calculate the mse with real embedding
#######################
import warnings  # Ignore sklearn future warning
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import argparse
import torch
import random
from lib.fMRI import fMRIAutoencoderDataset, fMRI_Time_Subjs_Embed_Dataset
from lib.helper import extract_hidden_reps, drive_decoding, get_models, checkexist
from torch.utils.data import DataLoader
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--ROI', type = str, default = 'early_visual')
parser.add_argument('--volsurf', type = str, default='MNI152_3mm_data') # default is volumetric data , alternatively fsaverage_data for surface data
parser.add_argument('--pt', type = int, default =1, help='which patient we work with')
parser.add_argument('--TR_range', type = int, default = None) #TR_range=1 is first half, 2 second
parser.add_argument('--n_TR', type = int, default = 3599)
parser.add_argument('--n_pt', type = int, default =1, help='numbe of subjects the model deal with')
parser.add_argument('--hidden_dim', type = int, default = 64)
parser.add_argument('--zdim', type = int, default =2)

parser.add_argument('--input_size', type = int, default =None, help='voxel dimensions or dimensions of each timepoint')
parser.add_argument('--embednaming', type=str, default='early_visual_SRM_10dimensions_PHATE.npy')
parser.add_argument('--lr', type = float, default=0.001)
parser.add_argument('--n_epochs', type = int, default = 4000)
parser.add_argument('--labelpath', type = str, default='/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/forrest_movie_labels_coded_expanded.csv')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--lam_mani', type = float, default = 1)
parser.add_argument('--batch_size', type = int, default =64)
parser.add_argument('--symm', action ='store_true') # use the symmetric config for encoder as decoder, so the latent encoder dim is the same as manifold dim
parser.add_argument('--ae_type', type =str, default='mlp_md')
parser.add_argument('--datanaming', type=str, default=None)
parser.add_argument('--test_path', type = str, default=None)
parser.add_argument('--testnaming', type = str, default=None)
parser.add_argument('--test_range', type = int, default=None)

SUMMARYFILE = 'results/ind_mrAE_classification_localizer.csv'
def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    datapath = f"/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/{args.volsurf}/denoised_filtered_smoothed/ROI_data/{args.ROI}/data"
    embedpath = f'/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/{args.volsurf}/denoised_filtered_smoothed/ROI_data/{args.ROI}/embeddings'

    savepath = f"/gpfs/milgram/scratch60/turk-browne/tw496/results/forrest_{args.volsurf}" \
               f"_{args.ROI}_ind_mrAE"

    # datanaming = f"{args.ROI}_movie_all_runs.npy"
    datanaming = args.datanaming

    # if args.TR_range is not None:
    #     args.n_TR = args.n_TR // 2
    #     if args.TR_range == 1:
    #         savepath += f'_{1}half'
    #         TRs = np.arange(args.n_TR)
    #     else:
    #         savepath += f'_{2}half'
    #         TRs = np.arange(args.n_TR, 2 * args.n_TR)
    # else:
    TRs = np.arange(args.n_TR)

    savepath = os.path.join(savepath,f"hiddim{args.hidden_dim}_zdim{args.zdim}_lam_mani{args.lam_mani}_bs{args.batch_size}")

    outdf = None
    cols = ['ROI', 'subj', 'hidden_dim', 'zdim',
            'lam_mani', 'bs',
            'train_half', 'savepath', 'symm_design']
    entry = [args.ROI, args.pt, args.hidden_dim, args.zdim,
             args.lam_mani, args.batch_size, args.TR_range, savepath, args.symm]
    if os.path.exists(SUMMARYFILE):
        outdf_old = pd.read_csv(SUMMARYFILE)
    else:
        outdf_old = None
    if outdf_old is not None:
        exist = checkexist(outdf_old, dict(zip(cols, entry)))
        if exist:
            print(f"{entry} exists")
            return
    patient_ids = [args.pt]
    dataset = fMRI_Time_Subjs_Embed_Dataset(patient_ids,
                                            datapath,
                                            embedpath,
                                            TRs,
                                            emb_name_suffix=args.embednaming,
                                            data_3d=False,
                                            data_name_suffix=datanaming)
    train_means, train_stds = dataset.preprocess_center_normalize1()

    if args.input_size is None:
        args.input_size = dataset.get_TR_dims()[0]
    if args.zdim is None:
        args.zdim = dataset.get_embed_dims()
    if args.zdim!=dataset.get_embed_dims():
        print('ERROR: manifold layer dim and embedding reg dim not match')
        return
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder = get_models(args)
    encoder.to(device)
    decoder = decoder[0]
    decoder.to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    criterion = torch.nn.MSELoss()
    mr_criterion = torch.nn.MSELoss()

    losses = np.array([])
    rconst_losses = np.array([])
    manifold_reg_losses = np.array([])

    for epoch in range(1, args.n_epochs +1):
        epoch_losses = 0.0
        epoch_rconst_losses = 0.0
        epoch_manifold_reg_losses = 0.0
        for data_batch, embed_batch in dataloader:
            optimizer.zero_grad()
            data_batch = data_batch.reshape((data_batch.shape[0] * data_batch.shape[1], -1)).float()
            data_batch = data_batch.to(device)
            embed_batch = embed_batch.reshape((embed_batch.shape[0] * embed_batch.shape[1], -1)).float()
            embed_batch = embed_batch.to(device)

            hidden = encoder(data_batch)
            embed, output = decoder(hidden)

            # print(f"output shape:{output.shape}, data_batch shape:{data_batch.shape}, embed shape:{embed.shape}")

            loss_reconstruct = criterion(output, data_batch)
            loss_manifold_reg = mr_criterion(embed, embed_batch)
            loss = loss_reconstruct+loss_manifold_reg
            loss.backward()
            optimizer.step()

            epoch_losses += loss.item() * data_batch.size(0)
            epoch_rconst_losses += loss_reconstruct.item() * data_batch.size(0)
            epoch_manifold_reg_losses += loss_manifold_reg.item() * data_batch.size(0)

        epoch_losses = epoch_losses / (args.n_TR * len(patient_ids))  # change to report epoch loss
        epoch_rconst_losses = epoch_rconst_losses / (args.n_TR * len(patient_ids))
        epoch_manifold_reg_losses = epoch_manifold_reg_losses / (args.n_TR * len(patient_ids))

        losses = np.append(losses, epoch_losses)
        rconst_losses = np.append(rconst_losses, epoch_rconst_losses)
        manifold_reg_losses = np.append(manifold_reg_losses, epoch_manifold_reg_losses)

    all_losses = np.stack((losses, rconst_losses, manifold_reg_losses), axis=1)

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    np.save(os.path.join(savepath, f'pt{args.pt}_symm{args.symm}_all_train_losses.npy'), all_losses)

    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'epoch': epoch
    }, os.path.join(savepath, f"model_e{args.n_epochs}.pt"))

    # load the second half data and get the embeddings
    # if args.TR_range==1:
    #     TRs = np.arange(args.n_TR, 2* args.n_TR)
    # elif args.TR_range==2:
    #     TRs = np.arange(args.n_TR)

    TRs = np.arange(args.test_range)

    dataset = fMRIAutoencoderDataset([args.pt],
                                     args.test_path,
                                     TRs,
                                     data_3d=False,
                                     data_name_suffix=args.testnaming)
    dataset.preprocess_center_normalize1(train_means, train_stds)

    encoder.eval()
    decoder.eval()
    hidden, alhidden = extract_hidden_reps(encoder, [decoder], dataset, device, None, args)

    hidden = hidden.reshape(args.n_pt, args.test_range, -1)
    alhidden = alhidden.reshape(args.n_pt, args.test_range, -1)
    saveto = os.path.join(savepath, f"pt{args.pt}e{args.n_epochs}_symm{args.symm}_amlp_localizer.npy")
    np.save(saveto, alhidden)
    saveto = os.path.join(savepath, f"pt{args.pt}e{args.n_epochs}_symm{args.symm}_manihidden_localizer.npy")
    np.save(saveto, hidden)

    labeldf = pd.read_csv(args.labelpath)

    for labeltype in labeldf.columns.tolist()[1:]:
        labels = labeldf[labeltype].values[TRs]
        results = drive_decoding(hidden, labels, balance_min=False)
        cols.append(labeltype+'_mani_acc')
        entry.append([results.accuracy.values])

    if outdf is None:
        outdf = pd.DataFrame(columns=cols)
    outdf.loc[len(outdf)]=entry

    outdf.loc[len(outdf)-1, 'time' ]=datetime.datetime.now()

    if os.path.exists(SUMMARYFILE):
        outdf_old = pd.read_csv(SUMMARYFILE)
        outdf = pd.concat([outdf_old, outdf])
    outdf.to_csv(SUMMARYFILE, index=False)

if __name__=='__main__':
    main()



