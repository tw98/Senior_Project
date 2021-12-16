import torch
from lib.fMRI import fMRIAutoencoderDataset, fMRI_Time_Subjs_Dataset
from lib.autoencoder import Encoder_conv, Decoder_conv, Encoder_basic, Decoder_basic, Decoder_Manifold, Decoder_Manifold_btlnk, Encoder_basic_btlnk
import numpy as np
from scipy.stats import zscore
from brainiak.fcma.util import compute_correlation
import audtorch
import random
import pandas as pd
from sklearn.svm import SVC
import time
import multiprocessing as mp

def gaussian_kernel(x, y, kernel_mul, kernel_num, sigma_list):
    n_samples = int(x.size()[0]) + int(y.size()[0])
    total = torch.cat([x, y])
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if sigma_list is None:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        sigma_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        print(sigma_list)

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in sigma_list]

    return sum(kernel_val)/len(sigma_list)

class MMD_loss(torch.nn.Module):
    # adapted from ZongxianLee/MMD_Loss.Pytorch
    def __init__(self, kernel_mul = 2.0, kernel_num = 5, sigma_list= None):
        """
        :param kernel_mul: geometric seq ratio
        :param kernel_num: number of sigmas to use in multiscale RBF
        :param sigma_list: list of custom specify torch.Tensor s
        """
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.sigma_list = sigma_list



    def forward(self, source, target):
        """
        :param source: batch_size x hidden_dim
        :param target: batch_size x hidden_dim
        :return: MMD loss
        """
        batch_size = int(source.size()[0])
        kernels = gaussian_kernel(source, target, self.kernel_mul, self.kernel_num, self.sigma_list)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss





class pearsonr_loss(torch.nn.Module):
    """
    x, y are (bs x hidden_dim)
    sum_j^hidden_dim pearson(x[:,j], y[:,j])
    """

    def __init__(self):
        super(pearsonr_loss, self).__init__()

    def forward(self, x, y):
        loss = audtorch.metrics.functional.pearsonr(torch.transpose(x, 0, 1), torch.transpose(y, 0, 1),
                                                    batch_first=True)
        loss = loss.sum()/x.size(1)
        return loss

def get_models_n_dataloaders(n_pt, device, batch_size=16,  hidden_dim=64):
    dataloaders = []
    decoders = []
    for pt in range(1, n_pt+1):
        dataset = fMRIAutoencoderDataset([pt],
                                         "/gpfs/milgram/pi/turk-browne/jh2752/fMRI_manifold/data/auditory_3d",
                                         range(1976))
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=1)
        input_size = dataset.get_TR_dims()
        decoder = Decoder_conv(hidden_dim, input_size, dataset.mask)
        decoder.to(device)

        dataloaders.append(dataloader)
        decoders.append(decoder)

    encoder = Encoder_conv(input_size, hidden_dim, dataset.mask)  # mask not really useful for encoder
    encoder.to(device)
    return encoder, decoders, dataloaders


def get_Subjs_Dataset_DataLoader(n_pt, timelength, datapath, batch_size=16, ):
    dataset = fMRI_Time_Subjs_Dataset(np.arange(1,n_pt+1),
                                      datapath,
                                      range(timelength))
    return dataset, torch.utils.data.DataLoader(dataset, batch_size =batch_size, shuffle = True, num_workers=1)



def adjust_learning_rate(init_lr, optimizer, totalepoch):
    # Sets the learning rate to decay when each call by 0.95
    lr = init_lr * (0.95**totalepoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def extract_hidden_reps(encoder, decoders, dataset, device, amlps, args):
    # idx is pt-1 to index the mlp
    encoder.eval()
    for decoder in decoders:
        decoder.eval()
    if amlps is not None:
        for mlp in amlps:
            mlp.eval()

    hidden_reps = []
    aligned_hidden_reps = []
    for i in range(len(dataset)):
        input_TR = dataset[i]
        input_TR = torch.from_numpy(input_TR)
        input_TR = input_TR.unsqueeze(0).unsqueeze(0)
        input_TR = input_TR.float().to(device)
        if args.ae_type == 'conv':
            hidden, _,_ = encoder(input_TR)
        elif args.ae_type =='mlp':
            hidden = encoder(input_TR)
        elif args.ae_type =='mlp_md':
            common_hidden = encoder(input_TR)
            pt = i//args.n_TR
            decoders[pt].eval()
            hidden, _=decoders[pt](common_hidden)
            aligned_hidden_reps.append(common_hidden.detach().cpu().numpy().flatten())

        if amlps is not None:
            idx = i//args.n_TR
            aligned_hidden = amlps[idx](hidden)
            aligned_hidden = aligned_hidden.detach().cpu().numpy().flatten()
            aligned_hidden_reps.append(aligned_hidden)


        hidden = hidden.detach().cpu().numpy().flatten()
        hidden_reps.append(hidden)
    hidden_reps = np.vstack(hidden_reps)
    if amlps or args.ae_type=='mlp_md':
        aligned_hidden_reps = np.vstack(aligned_hidden_reps)
    return hidden_reps, aligned_hidden_reps

def extract_hidden_n_rconst(encoder, decoders, dataset, device, amlps, args):
    # idx is pt-1 to index the mlp
    encoder.eval()
    for decoder in decoders:
        decoder.eval()
    if amlps is not None:
        for mlp in amlps:
            mlp.eval()

    hidden_reps = []
    aligned_hidden_reps = []
    outputs = []
    for i in range(len(dataset)):
        input_TR = dataset[i]
        input_TR = torch.from_numpy(input_TR)
        input_TR = input_TR.unsqueeze(0).unsqueeze(0)
        input_TR = input_TR.float().to(device)
        if args.ae_type == 'conv':
            hidden, _, _ = encoder(input_TR)
        elif args.ae_type == 'mlp':
            hidden = encoder(input_TR)
        elif args.ae_type == 'mlp_md':
            common_hidden = encoder(input_TR)
            if len(decoders) == 1:
                decoder = decoders[0]
                decoder.eval()
                hidden, output = decoder(common_hidden)
                aligned_hidden_reps.append(common_hidden.detach().cpu().numpy().flatten())
            else:
                pt = i // args.n_TR
                decoders[pt].eval()
                hidden, _ = decoders[pt](common_hidden)
                aligned_hidden_reps.append(common_hidden.detach().cpu().numpy().flatten())

        if amlps is not None:
            idx = i // args.n_TR
            aligned_hidden = amlps[idx](hidden)
            aligned_hidden = aligned_hidden.detach().cpu().numpy().flatten()
            aligned_hidden_reps.append(aligned_hidden)

        hidden = hidden.detach().cpu().numpy().flatten()
        hidden_reps.append(hidden)
        output = output.detach().cpu().numpy().flatten()
        outputs.append(output)
    hidden_reps = np.vstack(hidden_reps)
    outputs = np.vstack(outputs)
    if amlps or args.ae_type == 'mlp_md':
        aligned_hidden_reps = np.vstack(aligned_hidden_reps)
    return hidden_reps, aligned_hidden_reps, outputs

def pair_ISC(data):
    """

    :param data: array of shape (n_subjects, n_samples, n_features)
    :return: array of shape (n_subjects, n_features) where
            each entry is the correlation of timeseries at feature_j
            between test subject i and a random i' subject, all 17 pairs of (i,i') are non-repeating
    """
    n_subjects, n_timepoints, n_features = np.shape(data)
    iscs = np.empty((n_subjects, n_features))

    subj_list = np.arange(n_subjects)
    random.shuffle(subj_list)

    for i in range(n_subjects):
        train_subject = subj_list[i]
        test_subject = subj_list[(i+1)%n_subjects] #train_subject last  test_subject first in subject list

        train_data = data[train_subject]
        test_data = data[test_subject]

        for feature in range(n_features):
            corr = np.corrcoef(train_data[:,feature], test_data[:,feature])[0,1]
            iscs[test_subject, feature]=corr
    return iscs

def pair_tsm(data, window_size =10):
    data = np.swapaxes(np.array(data),2,1)
    nsubjs, ndim, nsamples = data.shape
    accuracy=np.zeros(shape=nsubjs)
    nseg=nsamples-window_size

    subj_list = np.arange(nsubjs)
    random.shuffle(subj_list)
    for i in range(nsubjs):
        trn_subj = subj_list[i]
        tst_subj = subj_list[(i+1)%nsubjs]
        trn_data = np.zeros((ndim*window_size, nseg), order='f')
        tst_data = np.zeros((ndim*window_size, nseg),order='f')
        for w in range(window_size):
            tst_data[w * ndim:(w + 1) * ndim, :] = data[tst_subj][:, w:(w + nseg)]
            trn_data[w * ndim:(w + 1) * ndim, :] = data[trn_subj][:, w:(w + nseg)]

        #zscore or not
        A = np.nan_to_num(zscore(trn_data, axis=0, ddof=1))
        B = np.nan_to_num(zscore(tst_data, axis=0, ddof=1))

        corr_mtx = compute_correlation(B.T, A.T)
        for i in range(nseg):
            for j in range(nseg):
                # exclude segments overlapping with the testing segment
                if abs(i - j) < window_size and i != j:
                    corr_mtx[i, j] = -np.inf
        max_idx = np.argmax(corr_mtx, axis=1)
        accuracy[tst_subj]=sum(max_idx==range(nseg))//nseg
    return accuracy
# def pair_MSE(data):


def balance_data_indices(labels, minimize=False):
    # assuming binary labels here
    (unique, counts) = np.unique(labels, return_counts=True)
    ind_max = np.where(counts == max(counts))
    ind_min = np.where(counts == min(counts))
    
 
    if len(counts[ind_max]) == 1 and counts[ind_max] / np.sum(counts) >= 0.52:
        where_max = np.where(labels == unique[ind_max])
        where_min = np.where(labels == unique[ind_min])
        min_indices = list(where_min[0])
        max_indices = list(where_max[0])

        if minimize:
            print(
                f"reducing training for label = {unique[ind_max]} from count = {counts[ind_max]} to count = {counts[ind_min]}")
            max_indices_subsamp = list(random.sample(sorted(where_max[0]), k=len(min_indices)))

            indices = min_indices + max_indices_subsamp
            print(labels.shape)
            print(len(indices))

        else:
            print(
                f"augmenting training for label = {unique[ind_min]} from count = {counts[ind_min]} to count = {counts[ind_max]}")

            num2add = len(max_indices) - len(min_indices)
            min_indices_upsamp = list(random.choices(where_min[0], k=num2add)) + list(where_min[0])
            indices = min_indices_upsamp + max_indices

    else:
        indices = np.arange(len(labels))

    return indices 

def balance_train_data(X_train, Y_train, minimize=True):
    # assuming binary labels here
    (unique, counts) = np.unique(Y_train, return_counts=True)
    ind_max = np.where(counts == max(counts))
    ind_min = np.where(counts == min(counts))

    if len(counts[ind_max]) == 1 and counts[ind_max] / np.sum(counts) >= 0.52:
        where_max = np.where(Y_train == unique[ind_max])
        where_min = np.where(Y_train == unique[ind_min])
        min_indices = list(where_min[0])
        max_indices = list(where_max[0])
        if minimize:
            print(
                f"reducing training for label = {unique[ind_max]} from count = {counts[ind_max]} to count = {counts[ind_min]}")
            max_indices_subsamp = list(random.choices(where_max[0], k=len(min_indices)))
            train_indices = min_indices + max_indices_subsamp
        else:
            print(
                f"augmenting training for label = {unique[ind_min]} from count = {counts[ind_min]} to count = {counts[ind_max]}")

            num2add = len(max_indices) - len(min_indices)
            min_indices_upsamp = list(random.choices(where_min[0], k=num2add)) + list(where_min[0])
            train_indices = min_indices_upsamp + max_indices
        train_indices.sort()

        X_train = X_train[train_indices]
        Y_train = Y_train[train_indices]
    return X_train, Y_train

from sklearn.model_selection import KFold
def decode_timeseries(PARAMETERS):
    data, labels, balance_min, subject_id, n_folds = PARAMETERS
    """
    data: n_TR x hidden_dim
    labels: n_TR
    balance_min: None, True, False
    subject_id: 1--n_pt
    n_folds: kfold 
    """
    kf = KFold(n_splits = n_folds, shuffle=True)
    results = pd.DataFrame(columns=['subject_ID', 'fold', 'accuracy'])
    for split, (train, test) in enumerate(kf.split(np.arange(data.shape[0]))):
        model = SVC(kernel='rbf', C=10)
        X_train=data[train,:]
        y_train = labels[train]
        if balance_min is not None:
            X_train, y_train = balance_train_data(X_train, y_train, minimize=balance_min)
        model.fit(X_train, y_train)
        score = model.score(data[test,:], labels[test])
        results.loc[len(results)] = {'subject_ID':subject_id, 'fold':split, 'accuracy':score}

    return results

def drive_decoding(data_allpt, labels, kfold=5, balance_min = None, datasource='Sherlock', ROI='early_vis' ):
    t0=time.time()
    iterable_data = [(data_allpt[i], labels, balance_min, i+1, kfold) for i in range(len(data_allpt))]
    pool = mp.Pool(len(data_allpt))
    results = pool.map(decode_timeseries, iterable_data)
    results = pd.concat(results)
    print('time consumption= ', time.time()-t0)
    return results

def get_models(args):
    if args.symm:
        encoder = Encoder_basic_btlnk(args.input_size, args.hidden_dim * 4, args.hidden_dim * 2, args.hidden_dim, args.zdim)
    else:
        encoder = Encoder_basic(args.input_size, args.hidden_dim * 4, args.hidden_dim * 2, args.hidden_dim)

    decoders = []
    for i in range(args.n_pt):
        if args.symm:
            decoder = Decoder_Manifold_btlnk(args.zdim, args.hidden_dim, args.hidden_dim * 2, args.hidden_dim * 4,
                                             args.input_size)
        else:
            decoder = Decoder_Manifold(args.zdim, args.hidden_dim, args.hidden_dim * 2, args.hidden_dim * 4,
                                       args.input_size)
        decoders.append(decoder)

    return encoder, decoders

def checkexist(targetdf, additional ):
    if len(targetdf)<1:
        return False
    for key in additional.keys():
        targetdf = targetdf.loc[targetdf[key] == additional[key]]
        if targetdf.shape[0]<1:
            return False
    return True