###########################################################################
## latent manifold embedding regularized autoencoder for fMRI data
## one encoder, multiple decoders
## 2021-06-25
## Jessie Huang (create)
## (1) manifold embedding bottleneck in per subject decoder: one common embedding (hidden dim) and individual phate operator idea
##
##
###########################################################################

import warnings  # Ignore sklearn future warning
warnings.simplefilter(action='ignore', category=FutureWarning)
from lib.utils import setup_save_n_log, get_per_pt, set_grad_req, setup_save_dir
import numpy as np
import torch
import logging
from lib.fMRI import fMRIAutoencoderDataset, fMRI_Time_Subjs_Embed_Label_Dataset
from torch.utils.data import DataLoader
from lib.autoencoder import Encoder_conv,Decoder_conv, Encoder_basic, Decoder_basic, Decoder_Manifold, Classifier
import random
from lib.utils import save_checkpoint_allin1
import os
import matplotlib.pyplot as plt
import argparse
from lib.helper import balance_data_indices, pearsonr_loss, MMD_loss, extract_hidden_reps, drive_decoding
from lib.utils_Erica import time_segment_matching, ISC
import pandas as pd
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--n_pt', type=int, default =17)
parser.add_argument('--n_TR', type = int, default = 1976)
parser.add_argument('--hidden_dim', type = int, default = 64)
parser.add_argument('--zdim', type = int, default =2) # the bottleneck manifold layer
parser.add_argument('--batch_size', type = int, default =16 )
parser.add_argument('--n_epochs', type = int, default = 10)
parser.add_argument('--lr', type = float, default=0.001)
parser.add_argument('--datapath', type = str, default = None)
parser.add_argument('--datanaming', type=str, default='') #file name base, s.t. filename = sub-??_datanaming
parser.add_argument('--embedpath',type=str,default=None)
parser.add_argument('--embednaming', type=str, default='early_visual_SRM_10dimensions_PHATE.npy')
parser.add_argument('--lam', type = float, default = 10)
parser.add_argument('--lam_mani', type = float, default = 1)
parser.add_argument('--lam_clf', type = float, default=0.0)
parser.add_argument('--lam_decay', type = float, default = 1)
parser.add_argument('--save_freq', type = int, default = 1)
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--TR_range', type = int, default = None) #TR_range=1 is first half, 2 second
parser.add_argument('--shuffle_reg', action='store_true')
parser.add_argument('--loadpath', type =str, default=None) #directory that saves checkpoints e.g. results/sherlock_viz_17pt_hiddendim64_bs64_20epochs_reg_lam0.001_1half
parser.add_argument('--load_epoch', type = int, default=0)
parser.add_argument('--ae_type', type =str, default='conv') # choices: 'conv', 'mlp', 'mlp_md'(mlp with manifold bottleneck decoder)
parser.add_argument('--reg', type = str, default = 'mse') #choices: 'mse', 'pear'(for peason corr), 'mmd'
parser.add_argument('--reg_ref', action = 'store_true') # for regularization, use pt[0] (shuffle or not) as reference; if not activated, random nonoverlapping pairs cycle
parser.add_argument('--mmd_kmul', type = float, default = 2.0)
parser.add_argument('--mmd_nmul', type= int, default = 5)
parser.add_argument('--mmd_specsig', action = 'store_true' )
parser.add_argument('--downstream', action = 'store_true') #follow with downstream analyses
parser.add_argument('--labeltype', type = str, default=None)
parser.add_argument('--labelpath', type = str, default=None)
parser.add_argument('--test_path', type = str, default=None)
parser.add_argument('--test_range', type = int, default=None)
parser.add_argument('--testnaming', type = str, default='"ses-localizer_task-objectcategories_all_runs_early_visual.npy"')
parser.add_argument('--experiment_label', type=str, default=None)
parser.add_argument('--summary_file', type=str, default='results/fMRI_AE_classification_result_summary.csv')

SUMMARYFILE = 'results/fMRI_AE_classification_result_summary.csv'
# SUMMARYFILE = 'results/fMRI_manifoldAE_paper_movie_result_summary.csv'

def main():
    # params
    args = parser.parse_args()

    global SUMMARYFILE
    SUMMARYFILE = args.summary_file

    savepath = f'results/forrest_paper_{args.n_pt}pt_hiddendim{args.hidden_dim}_bs{args.batch_size}_' \
               f'{args.ae_type}_{args.reg}_reg_lam{args.lam}_manilam{args.lam_mani}'

    scratchpath = '/gpfs/milgram/scratch60/turk-browne/tw496'
    if '_PHATE' in args.embednaming:
        savepath+='_phate'
    if '_TPHATE' in args.embednaming:
        savepath += '_tphate'
    if args.shuffle_reg:
        savepath+='_shf'

    print(f"save to {savepath}")
    # set up TR_range for dataset
    if args.TR_range is not None:
        args.n_TR = args.n_TR // 2
        if args.TR_range ==1:
            savepath+=f'_{1}half'

            TR_range = np.arange(args.n_TR)
            # print('data file contains full time length.')

        else:
            savepath += f'_{2}half'
            TR_range = np.arange(args.n_TR, 2*args.n_TR)
    else:
        TR_range = np.arange(args.n_TR)

    # make training reproducible
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    savepath = setup_save_n_log(savepath)
    print('set up log save to %s' % savepath)
    if args.TR_range ==1:
        if 'split1' in args.embednaming:
            logging.info('ERROR: train on first half need split0 embedding file!')
    elif args.TR_range==2:
        if 'split0' in args.embednaming:
            logging.info('ERROR: train on second half need split1 embedding file!')
    chkpt_savepath = os.path.join(scratchpath, savepath)
    chkpt_savepath = setup_save_dir(chkpt_savepath)
    logging.info(f'checkpoints save to: {chkpt_savepath}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    logging.info('Device:%s' % device)

    logging.info(f"AE model, manifold regularized")
    logging.info(f"Network params: hidden dim={args.hidden_dim}, manifold embedding dim={args.zdim}")
    logging.info(f"batch size={args.batch_size}, lr={args.lr}, seed={args.seed}")
    logging.info(f'save checkpoint: {args.save_model}')
    logging.info(f'common embedding regularization: {args.reg}')
    if args.lam==0:
        logging.info(f'common embedding regularization not used.')
    if args.TR_range is not None:
        logging.info(f'TR: {TR_range[0]}-{TR_range[-1]}')

    if 'StudyForrest' in args.datapath:
        # patient_ids = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 17, 18, 19, 20]
        patient_ids = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15]
    else:
        patient_ids = np.arange(1,args.n_pt+1)

    if args.ae_type=='conv':
        data_3d = True
    else: data_3d = False

    start_train = datetime.datetime.now()

    # embed_name_suffix = '_early_visual_SRM_10dimensions_PHATE.npy'
    embed_name_suffix = args.embednaming
    dataset = fMRI_Time_Subjs_Embed_Label_Dataset(patient_ids,
                                            args.datapath,
                                            args.embedpath,
                                            TR_range,
                                            emb_name_suffix=embed_name_suffix,
                                            data_3d=data_3d,
                                            data_name_suffix = args.datanaming,
                                            label_path = args.labelpath)

    # train_means, train_stds = dataset.preprocess_center_normalize1()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    input_size = dataset.get_TR_dims()[0]
    embed_size = dataset.get_embed_dims()
    args.zdim = embed_size # this is to set the manifold embedding shape
    logging.info(f"input size={input_size}")
    logging.info(f"manifold embedding size={args.zdim}")

    encoder = Encoder_basic(input_size, args.hidden_dim * 4, args.hidden_dim * 2, args.hidden_dim)
    if args.ae_type == 'mlp_md':
        encoder = Encoder_basic(input_size, args.hidden_dim * 4, args.hidden_dim * 2, args.hidden_dim)
        decoders = []
        for i in range(args.n_pt):
            decoder = Decoder_Manifold(embed_size, args.hidden_dim, args.hidden_dim*2, args.hidden_dim*4, input_size)
            decoder.to(device)
            decoders.append(decoder)

        if args.lam_clf:
            classifier = Classifier(args.hidden_dim)
            classifier.to(device)  

    encoder.to(device)

    # initialize optimizer
    params = list(encoder.parameters())
    for decoder in decoders:
        params = params + list(decoder.parameters())
    params = params + list(classifier.parameters())
    
    optimizer = torch.optim.Adam(params, lr=args.lr)  # either set initial lr or default 0.001

    # load pretrained and continue
    if args.loadpath is not None:
        print(f'loading model from {args.loadpath} at checkpoint epoch {args.load_epoch}')
        checkpoint = torch.load(os.path.join(args.loadpath, f'ae_e{args.load_epoch}.pt'))
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        for i in range(1, args.n_pt + 1):
            decoders[i - 1].load_state_dict(checkpoint[f'decoder_{i}_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info('loaded pretrained model for continue training')

    criterion = torch.nn.MSELoss()  # reconstruction loss criterion
    mr_criterion = torch.nn.MSELoss() # manifold embedding regularization criterion

    # common latent regularization
    # Todo: not used here but possible for later
    reg_criterion = torch.nn.MSELoss()

    clf_criterion = torch.nn.NLLLoss()

    losses = np.array([])
    rconst_losses = np.array([])
    reg_losses = np.array([])
    manifold_reg_losses = np.array([])
    
    clf_reg_losses = np.array([])

    pt_list = np.arange(args.n_pt)

    for epoch in range(args.load_epoch + 1, args.n_epochs + 1):
        epoch_losses = 0.0
        epoch_rconst_losses = 0.0
        epoch_manifold_reg_losses = 0.0
        epoch_reg_losses=0.0
        epoch_clf_reg_losses = 0.0

        for data_batch, embed_batch, label_batch in dataloader:
            optimizer.zero_grad()
            current_bs = data_batch.size()[0]
            data_batch = data_batch.reshape((data_batch.shape[0]*data_batch.shape[1], -1)).float()
            data_batch = data_batch.to(device)
            embed_batch = embed_batch.reshape((embed_batch.shape[0]*embed_batch.shape[1],-1)).float()
            embed_batch = embed_batch.to(device)

            label_batch = label_batch.reshape((label_batch.shape[0]*label_batch.shape[1],-1)).float()
            label_batch = label_batch.to(device)

            clf_indices = balance_data_indices(torch.flatten(label_batch).numpy())
            label_batch = label_batch[clf_indices]

            hidden = encoder(data_batch)
            hiddens = [hidden[i * current_bs:(i + 1) * current_bs] for i in range(args.n_pt)]
            outputs = []
            embeds = []
            for i in range(args.n_pt):
                set_grad_req(decoders, i)
                embed, output = decoders[i](hiddens[i])
                outputs.append(output)
                embeds.append(embed)

            if args.lam_clf:
                # print(hidden.shape)
                preds = classifier(hidden[clf_indices])

            if args.shuffle_reg:
                random.shuffle(pt_list)

            loss_reg = reg_criterion(hiddens[pt_list[0]], hiddens[pt_list[1]])
            if args.reg_ref:
                for z1 in range(1, args.n_pt):
                    loss_reg += reg_criterion(hiddens[pt_list[0]], hiddens[pt_list[z1]])

            else:
                for z1 in range(1, args.n_pt - 1):  # consecutive pairs (cycle)
                    z2 = z1 + 1
                    loss_reg += reg_criterion(hiddens[pt_list[z1]], hiddens[pt_list[z2]])

            loss_reconstruct = criterion(torch.stack(outputs).view(data_batch.shape), data_batch)
            loss_manifold_reg = mr_criterion(torch.stack(embeds).view(embed_batch.shape), embed_batch)

            # print(preds.shape)
            # print(label_batch.shape)
            loss_clf_reg = clf_criterion(preds, torch.flatten(label_batch).long())

            loss = loss_reconstruct + args.lam_mani*loss_manifold_reg
            
            print(f'reconstruction loss: {loss_reconstruct}')

            if args.lam > 0:
                loss += args.lam * loss_reg

            if args.lam_clf > 0:
                loss += args.lam_clf * loss_clf_reg
                
                print(f'classifier loss: {loss_clf_reg}')

            print(f'total loss: {loss}')

            args.lam_mani = args.lam_mani * args.lam_decay
            args.lam = args.lam * args.lam_decay
            args.lam_clf = args.lam_clf * args.lam_decay

            loss.backward()
            optimizer.step()

            epoch_losses += loss.item() * data_batch.size(0)
            epoch_rconst_losses += loss_reconstruct.item()*data_batch.size(0)
            epoch_reg_losses += loss_reg.item() * data_batch.size(0)
            epoch_manifold_reg_losses += loss_manifold_reg.item() * data_batch.size(0)

            epoch_clf_reg_losses += loss_clf_reg.item() * data_batch.size(0)

        epoch_losses = epoch_losses/ len(dataloader)  # change to report epoch loss
        epoch_rconst_losses = epoch_rconst_losses / len(dataloader)
        epoch_reg_losses = epoch_reg_losses / len(dataloader)
        epoch_manifold_reg_losses = epoch_manifold_reg_losses/len(dataloader)


        epoch_clf_reg_losses = epoch_clf_reg_losses / len(dataloader)

        logging.info(
            f"Epoch {epoch}\tLoss={epoch_losses:.4f}\tloss_rconst={epoch_rconst_losses:.4f}\tloss_manfold_reg={epoch_manifold_reg_losses:.4f}\tloss_reg={epoch_reg_losses:.4f}")

        losses = np.append(losses, epoch_losses)
        rconst_losses = np.append(rconst_losses, epoch_rconst_losses)
        reg_losses = np.append(reg_losses, epoch_reg_losses)
        manifold_reg_losses = np.append(manifold_reg_losses, epoch_manifold_reg_losses)

        clf_reg_losses = np.append(clf_reg_losses, epoch_clf_reg_losses)

        if (epoch % args.save_freq == 0 or epoch == args.n_epochs) and args.save_model:
            model_namelist = ['encoder'] + ['decoder_%d' % i for i in range(1, len(decoders) + 1)]
            save_checkpoint_allin1([encoder] + decoders, model_namelist,
                                   optimizer,
                                   os.path.join(chkpt_savepath, 'ae_e%d.pt' % epoch),
                                   epoch)
            logging.info(f'saved checkpoint at epoch{epoch}')

    all_losses = np.stack((losses, rconst_losses, manifold_reg_losses, reg_losses, clf_reg_losses), axis=1)
    np.save(os.path.join(savepath, 'all_train_losses.npy'), all_losses)

    fig, ax = plt.subplots()
    for j, l in enumerate(['total loss', 'reconstruction loss', 'manifold_reg_loss',
                                                                'regularization loss', 'classifier loss']):
        ax.scatter(range(args.load_epoch, args.n_epochs),
                   all_losses[:, j] * (len(dataloader)) / len(dataset.TRs) / args.n_pt, label=l)

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    # plt.yscale('log')
    # plt.xscale('log')

    plt.show()
    plt.savefig(os.path.join(savepath, 'all_losses'))
    logging.info(f'Finished AE training {args.n_epochs} epochs')

    end_train = datetime.datetime.now()

    if args.downstream:

        # record all results

        start_downstream = datetime.datetime.now()

        cols = ['data', 'n_pt', 'train_half', 'common_hiddendim', 'manifold_embed_dim',
                'common_latent_regularization', 'lambda_common', 'manifold_regularization', 'lambda_mani', 'lambda_clf',
                'encoder', 'decoder', 'reg_shuffle', 'load_path', 'load_epoch']
        if args.experiment_label:
            datasource = args.experiment_label
        elif ('sherlock' in args.datapath):
            datasource = 'sherlock'
        else:
            datasource = 'forrest'

        entry =[datasource, args.n_pt, args.TR_range, args.hidden_dim, args.zdim,
                args.reg, args.lam, 'mse', args.lam_mani, args.lam_clf,
                args.ae_type, args.ae_type, args.shuffle_reg, chkpt_savepath, args.n_epochs]

        if os.path.exists(SUMMARYFILE):
            resultsdf = pd.read_csv(SUMMARYFILE)
            resultsdf.loc[len(resultsdf)] = dict(zip(cols, entry))
        else:
            resultsdf = pd.DataFrame([entry], columns=cols)

        args.loadpath = chkpt_savepath
        args.load_epoch = args.n_epochs
        print('extract hidden rep and calculate metrics and produce phate plots')
        del dataloader
        del dataset

        if args.test_path is None:
            if args.TR_range is not None:
                # cross-validate the other half. e.g. trained model on the first half data, then apply the trained model on the second half
                if args.TR_range == 2:
                    TR_range = np.arange(args.n_TR)
                else:
                    TR_range = np.arange(args.n_TR, 2 * args.n_TR)
            else:
                TR_range = np.arange(args.n_TR)

            logging.info(f'apply trained model on {args.TR_range} half to data[{TR_range[0]}-{TR_range[-1]}]')

            # checkpoint = torch.load(os.path.join(args.loadpath, f'ae_e{args.load_epoch}.pt'))

            # load dataset
            if 'StudyForrest' in args.datapath:
                # patient_ids = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 17, 18, 19, 20]
                patient_ids = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15]
            else:
                patient_ids = np.arange(1, args.n_pt + 1)

            dataset = fMRIAutoencoderDataset(patient_ids,
                                            args.datapath,
                                            TR_range,
                                            data_3d=data_3d,
                                            data_name_suffix=args.datanaming)
            # dataset.preprocess_center_normalize1(train_means, train_stds)

            if len(dataset) != args.n_TR * args.n_pt:
                print('Error: dataset timepoints not equal to desired timepoints.')
        else:
            TR_range = np.arange(args.test_range)

            logging.info(f'apply trained model on {args.test_path} -- timesteps: {TR_range[0]}-{TR_range[-1]}]')

            # checkpoint = torch.load(os.path.join(args.loadpath, f'ae_e{args.load_epoch}.pt'))

            # load dataset
            if 'StudyForrest' in args.datapath:
                # patient_ids = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 17, 18, 19, 20]
                patient_ids = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15]
            else:
                patient_ids = np.arange(1, args.n_pt + 1)

            dataset = fMRIAutoencoderDataset(patient_ids,
                                            args.test_path,
                                            TR_range,
                                            data_3d=data_3d,
                                            data_name_suffix=args.testnaming)
            # dataset.preprocess_center_normalize1(train_means, train_stds)

        # continue to use the last state of encoder and decoder from training so that no need to load the saved checkpoints.
        latent_mlps = None

        encoder.eval()
        hidden, al_hidden = extract_hidden_reps(encoder, decoders, dataset, device, latent_mlps, args)
        if args.test_range:
            hidden = hidden.reshape(args.n_pt, args.test_range, -1) # hidden is all patients, and interested timepoints
        else:
            hidden = hidden.reshape(args.n_pt, args.n_TR, -1)

        # if args.amlp or args.ae_type == 'mlp_md':
        if args.ae_type == 'mlp_md':
            if args.test_range:
                al_hidden = al_hidden.reshape(args.n_pt, args.test_range, -1)
            else:
                al_hidden = al_hidden.reshape(args.n_pt, args.n_TR, -1)

        print('hidden representation shape: ',hidden.shape)
        datatype = os.path.split(args.datapath)[-1]
        modeldata = os.path.split(args.loadpath)[-1]
        if args.test_path:
            saveto = os.path.join(args.loadpath,
                                f'{datatype}_{modeldata}_model_on_test_data_e{args.load_epoch}.npy')
        else:
            saveto = os.path.join(args.loadpath,
                                f'{datatype}_{modeldata}_model_on_otherhalf_data_e{args.load_epoch}.npy')

        np.save(saveto, hidden)
        resultsdf.loc[len(resultsdf)-1, 'hiddenfile']=saveto
        print('saved extracted latent representation to : ', saveto)
        logging.info (f'saved extracted latent representation to : {saveto}')
        tsm = np.nan
        isc = np.nan
        tsm_std = np.nan
        isc_std = np.nan
        # if args.amlp or args.ae_type == 'mlp_md': # the amlp in mlp_md case is the common embedding
        if args.ae_type == 'mlp_md':
            if args.test_path:
                saveto = os.path.join(args.loadpath,
                                    f'{datatype}_{modeldata}_model_on_test_data_e{args.load_epoch}_amlp.npy')
            else:
                saveto = os.path.join(args.loadpath,
                                    f'{datatype}_{modeldata}_model_on_{args.TR_range}half_data_e{args.load_epoch}_amlp.npy')
            np.save(saveto, al_hidden)
            resultsdf.loc[len(resultsdf) - 1, 'aligned_hiddenfile'] = saveto
            print('amlp hidden saved to: ', saveto)
            logging.info(f"aligned encoder latent representation saved to: {saveto}")

            tsm = time_segment_matching(al_hidden)
            tsm_std = tsm.std()
            tsm = tsm.mean()

            isc = ISC(al_hidden)
            isc_std = isc.std()
            isc = isc.mean()

        resultsdf.loc[len(resultsdf)-1, 'tsm_mean' ]=tsm
        resultsdf.loc[len(resultsdf) - 1, 'tsm_std']=tsm_std
        resultsdf.loc[len(resultsdf) - 1, 'isc_mean']=isc
        resultsdf.loc[len(resultsdf) - 1, 'isc_std']=isc_std

        # SVC prediction
        if args.labelpath is None:
            logging.info('ERROR: need label to perform classification. set labelpath=')
        else:
            
            labels = np.load(args.labelpath)[TR_range]
            results = drive_decoding(hidden, labels,balance_min = False) #augment train data
            
            resultsdf.loc[len(resultsdf)-1, 'label'] = args.labeltype
            resultsdf.loc[len(resultsdf)-1, 'acc']=results['accuracy'].mean()
            resultsdf.loc[len(resultsdf)-1, 'std']=results['accuracy'].std()

            # resultsdf.loc[len(resultsdf)-1, 'time' ]=datetime.datetime.now()
            # resultsdf.loc[len(resultsdf)-1, 'train_time' ] = end_train - start_train
            # resultsdf.loc[len(resultsdf)-1, 'downstream_time' ] = datetime.datetime.now() - start_downstream

            results.to_csv(os.path.join(args.loadpath,
                f"{(SUMMARYFILE.split('.')[0]).split('/')[-1]}_row_{len(resultsdf)-1}_{args.labeltype}_accuracies.csv"),
                            index=False)

        resultsdf.loc[len(resultsdf)-1, 'time' ]=datetime.datetime.now()
        resultsdf.loc[len(resultsdf)-1, 'train_time' ] = end_train - start_train
        resultsdf.loc[len(resultsdf)-1, 'downstream_time' ] = datetime.datetime.now() - start_downstream
        resultsdf.to_csv(SUMMARYFILE, index=False)

if __name__ == '__main__':
    main()