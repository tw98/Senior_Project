import numpy as np
import os, sys, glob
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.image import math_img, resample_img
from scipy.stats import zscore

ROI_dir = '/gpfs/milgram/scratch60/turk-browne/neuromanifold/ROIs/'
sherlock_dir = '/gpfs/milgram/scratch60/turk-browne/neuromanifold/sherlock/'
forrest_dir = '/gpfs/milgram/scratch60/turk-browne/neuromanifold/StudyForrest/'
simulated_vol = '/gpfs/milgram/project/turk-browne/users/elb77/sherlock/simulated_data/simulated_sherlock_timeseries.nii.gz'
results_dir = '/gpfs/milgram/scratch60/turk-browne/elb77/manifold_analyses/'
sherlock_embedding_dir = '/gpfs/milgram/scratch60/turk-browne/elb77/manifold_analyses/sherlock_results/embeddings/'
forrest_embedding_dir = '/gpfs/milgram/scratch60/turk-browne/elb77/manifold_analyses/forrest_results/embeddings/'

forrest_subjects = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 17, 18, 19, 20]


def mask_timeseries(ROI_file, dts, z=True, zaxis=0, cluster=None):
    if cluster is not None:
        ROI_file = math_img(f'img == {cluster}', img=ROI_file)

    masker = NiftiMasker(mask_img=ROI_file, target_affine=dts.affine)
    ds_masked = masker.fit_transform(dts)
    if z:
        ds_masked = zscore(ds_masked, axis=zaxis)
    return ds_masked


def resample_mask_timeseries(ROI_nii, ds, z=True, zaxis=0):
    resampled_ROI = resample_img(ROI_nii, target_affine=ds.affine, target_shape=ds.shape[:-1])
    resampled_ROI = resampled_ROI.get_fdata()
    resampled_ROI_mask = np.where(resampled_ROI > 0.95, 1, 0)
    ds = ds.get_fdata()
    masked_data = ds[resampled_ROI_mask == 1, :]
    if z:
        masked_data = zscore(masked_data, axis=zaxis)
    return masked_data


def load_original_sherlock_data(ROI_nii, start_subj=1, n_subj=17):
    dss = []
    if n_subj == 1:
        data = mask_timeseries(ROI_nii, nib.load(sherlock_dir + f'sherlock_movie_s{start_subj}.nii'))
        return np.nan_to_num(data)
    for subj in np.arange(start_subj, n_subj + 1):
        data = mask_timeseries(ROI_nii, nib.load(sherlock_dir + f'sherlock_movie_s{subj}.nii'))
        dss.append(np.nan_to_num(data))
    dss = np.array(dss)
    return dss


def load_original_forrest_localizer_data(ROI_nii, start_subj=1, n_subj=15, run='all'):
    dss = []
    base_fn = 'ses-localizer_task-objectcategories_'
    if run == 'all':
        base_fn = '-' + base_fn + 'full_timeseries.nii.gz'
    else:
        base_fn = f'_{base_fn}run-{run}_func2standard.nii.gz'

    if n_subj == 1:
        sub_id = forrest_subjects[start_subj - 1]
        fn = os.path.join(forrest_dir, f'sub-{sub_id:02d}', 'func', 'localizer', f'sub-{sub_id:02d}{base_fn}')
        data = resample_mask_timeseries(ROI_nii, nib.load(fn))
        return np.nan_to_num(data)

    for sub_id in forrest_subjects:
        # sub_id = forrest_subjects[s]
        fn = os.path.join(forrest_dir, f'sub-{sub_id:02d}', 'func', 'localizer', f'sub-{sub_id:02d}{base_fn}')
        data = resample_mask_timeseries(ROI_nii, nib.load(fn))
        print(sub_id, data.shape)
        dss.append(np.nan_to_num(data))
    return np.array(dss)


def load_forrest_localizer_labels(start_subj=1, n_subj=15, run='all'):
    labels = []

    if run == 'all':
        to_load = [1, 2, 3, 4]
    else:
        to_load = [run]

    if n_subj == 1:
        sub_id = forrest_subjects[start_subj - 1]
        fns = [os.path.join(forrest_dir, f'sub-{sub_id:02d}', 'func', 'localizer', 'labels',
                            f'sub-{sub_id:02d}_run_{r}_tr_labels.npy') for r in to_load]
        labels = np.concatenate([np.load(f) for f in fns])
        return labels
    for sub_id in forrest_subjects:
        # sub_id = forrest_subjects[s]
        fns = [os.path.join(forrest_dir, f'sub-{sub_id:02d}', 'func', 'localizer', 'labels',
                            f'sub-{sub_id:02d}_run_{r}_tr_labels.npy') for r in to_load]
        these_labels = np.concatenate([np.load(f) for f in fns])
        labels.append(these_labels)
    return labels


def load_original_forrest_movie_data(ROI_nii, start_subj=1, n_subj=15, run='all'):
    dss = []
    base_fn = 'ses-movie_task-movie_'
    if run == 'all':
        base_fn = '-' + base_fn + 'full_timeseries.nii.gz'
    else:
        base_fn = f'_{base_fn}run-{run}_func2standard.nii.gz'

    if n_subj == 1:
        sub_id = forrest_subjects[start_subj]
        fn = os.path.join(forrest_dir, f'sub-{sub_id:02d}', 'func', 'movie', f'sub-{sub_id:02d}{base_fn}')
        data = resample_mask_timeseries(ROI_nii, nib.load(fn))
        return np.nan_to_num(data)

    for sub_id in forrest_subjects:
        # sub_id = forrest_subjects[s]
        fn = os.path.join(forrest_dir, f'sub-{sub_id:02d}', 'func', 'movie', f'sub-{sub_id:02d}{base_fn}')
        data = resample_mask_timeseries(ROI_nii, nib.load(fn))
        print(sub_id, data.shape)
        dss.append(np.nan_to_num(data))
    return np.array(dss)


def load_embedding_sherlock_data(ROI_name, manifold_type, dimensions, n_subj=17, other=None):
    dss = []
    for subj in np.arange(1, n_subj + 1):
        fn = f'{embedding_dir}/whole_embeddings/{ROI_name}_s{subj}_dim{dimensions}_{manifold_type}_embedding.npy'
        dss.append(np.nan_to_num(np.load(fn)))
    dss = np.array(dss)
    return dss


def load_split_half_sherlock_embeddings(ROI_name, manifold_type, dimensions, n_subj=17):
    dss = []
    for subj in np.arange(1, n_subj + 1):
        fn = f'{embedding_dir}/split_half_embeddings/{ROI_name}_s{subj}_{dimensions}_{manifold_type}_split1.npy'
        fn2 = f'{embedding_dir}/split_half_embeddings/{ROI_name}_s{subj}_{dimensions}_{manifold_type}_split2.npy'
        dss.append(np.nan_to_num(np.load(fn)))
        dss.append(np.nan_to_num(np.load(fn2)))
    dss = np.array(dss)
    return dss


def load_sherlock_regressors():
    import pandas as pd
    regressors = pd.read_csv('/gpfs/milgram/scratch60/turk-browne/neuromanifold/Sherlock_Segments_master.csv')
    return regressors


def load_binarized_sherlock_regressors(feature=None):
    import pandas as pd
    regressors = pd.read_csv('/gpfs/milgram/scratch60/turk-browne/neuromanifold/binarized_sherlock_regressors.csv')
    if feature != None:
        regressors = (regressors[feature]).values
    return regressors


import numpy as np
import os, sys, glob
from sklearn.model_selection import KFold
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform, cdist
from brainiak.fcma.util import compute_correlation


def ISC(data):
    """
    Intersubject correlation of timeseries responses across features.

    Parameters:
    ----------
    data: array or list of shape (n_subjects, n_samples, n_features)


    Returns:
    --------
    data: array of shape (n_subjects, n_features) where each entry is the
    correlation of the timeseries at subject_i feature_j compared with the
    group average timeseries at feature_j
    """

    n_subjects, n_timepoints, n_features = np.shape(data)
    iscs = np.empty((n_subjects, n_features))
    for test_subject in range(n_subjects):
        train_subjects = np.setdiff1d(np.arange(n_subjects), test_subject)
        test_data = data[test_subject]
        train_data = np.mean(data[train_subjects], axis=0)
        for feature in range(n_features):
            corr = np.corrcoef(train_data[:, feature], test_data[:, feature])[0, 1]
            iscs[test_subject, feature] = corr
    return iscs


# Take in array or list of shape (n_subjects, n_samples, n_features)
# Also specify how big the time segment is to be matched.

def time_segment_matching(data, window_size=10):
    data = np.swapaxes(np.array(data), 2, 1)  # swap to be (n_subjects, n_features, n_samples)
    nsubjs, ndim, nsamples = data.shape
    accuracy = np.zeros(shape=nsubjs)
    nseg = nsamples - window_size
    trn_data = np.zeros((ndim * window_size, nseg), order='f')
    # the training data also include the test data, but will be subtracted when calculating A
    for m in range(nsubjs):
        for w in range(window_size):
            trn_data[w * ndim:(w + 1) * ndim, :] += data[m][:, w:(w + nseg)]
    for tst_subj in range(nsubjs):
        tst_data = np.zeros((ndim * window_size, nseg), order='f')
        for w in range(window_size):
            tst_data[w * ndim:(w + 1) * ndim, :] = data[tst_subj][:, w:(w + nseg)]
        A = np.nan_to_num(zscore((trn_data - tst_data), axis=0, ddof=1))
        B = np.nan_to_num(zscore(tst_data, axis=0, ddof=1))
        # compute correlation matrix
        corr_mtx = compute_correlation(B.T, A.T)
        # The correlation classifier.
        for i in range(nseg):
            for j in range(nseg):
                # exclude segments overlapping with the testing segment
                if abs(i - j) < window_size and i != j:
                    corr_mtx[i, j] = -np.inf
        max_idx = np.argmax(corr_mtx, axis=1)
        accuracy[tst_subj] = sum(max_idx == range(nseg)) / nseg
    return accuracy
