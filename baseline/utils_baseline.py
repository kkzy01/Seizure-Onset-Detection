import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.svm import OneClassSVM
import scipy.signal as sig
import pandas as pd
from numbers import Number
from ieeg.auth import Session
#### feature extraction
def teager_operator(x):
    """
    Computes the Teager Energy Operator for a 1D signal.
    The output will be 2 samples shorter than the input.
    """
    return x[1:-1]**2 - x[:-2] * x[2:]
def mean_curve_length(x):
    return np.log1p(np.mean(np.abs(np.diff(x))))

def mean_energy(x):
    return np.log1p(np.mean(x**2))

def mean_teager_energy(x):
    te = teager_operator(x)
    return np.log1p(np.mean(te))

def extract_features(eeg, fs=200, win_len=1.0, step_size=0.5):
    """
    epoch: 1D np.array EEG signal
    fs: sampling frequency (e.g. 200 Hz)
    win_len: window length in seconds
    step_size: window overlap step in seconds
    """
    N = int(win_len * fs)
    step = int(step_size * fs)
    features = []

    for start in range(0, len(eeg) - N + 1, step):
        x = eeg[start:start + N]
        cl = mean_curve_length(x)
        e = mean_energy(x)
        te = mean_teager_energy(x)
        features.append([cl, e, te])
    return np.array(features)
#### classifier 
def train_one_class_svm(X_train, nu=0.1, gamma=1.0):
    model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
    model.fit(X_train)
    return model
def compute_novelty_scores(model, X_test):
    preds = model.predict(X_test)  # +1 or -1
    return preds

def estimate_outlier_fraction(y_pred, n=10):
    # running window estimate of outlier fraction
    y = (y_pred == -1).astype(int)  # 1 if outlier
    return uniform_filter1d(y.astype(float), size=n, mode='constant')

# === Seizure Detection Based on ν̂ Hypothesis Test ===
def detect_seizure(nu_hat, threshold=0.8):
    return (nu_hat > threshold).astype(int)

# === Apply Persistence Filter ===
def apply_persistence(z, refractory_sec=100, step_sec=0.5):
    refractory_steps = int(refractory_sec / step_sec)
    detection = np.zeros_like(z)
    i = 0
    while i < len(z):
        if z[i]:
            detection[i] = 1
            i += refractory_steps
        else:
            i += 1
    return detection


# seizure onset detection
def get_onset_and_spread(sz_prob,threshold=None,
                        ret_smooth_mat = True, #True
                        filter_w = 5, # seconds 
                        rwin_size = 5, # seconds #10
                        rwin_req = 4, # seconds #9
                        w_size = 1,
                        w_stride = 0.5
                        ): 

        sz_clf = (sz_prob>threshold).reset_index(drop=True)
        filter_w_idx = np.floor((filter_w - w_size)/w_stride).astype(int) + 1
        sz_clf = pd.DataFrame(sig.ndimage.median_filter(sz_clf,size=filter_w_idx,mode='nearest',origin=0),columns=sz_prob.columns)
        seized_idxs = np.any(sz_clf,axis=1)
        rwin_size_idx = np.floor((rwin_size - w_size)/w_stride).astype(int) + 1
        rwin_req_idx = np.floor((rwin_req - w_size)/w_stride).astype(int) + 1
        sz_spread_idxs_all = sz_clf.rolling(window=rwin_size_idx,center=False).apply(lambda x: (x == 1).sum()>rwin_req_idx).dropna().reset_index(drop=True)
        sz_spread_idxs = sz_spread_idxs_all.loc[seized_idxs]
        extended_seized_idxs = np.any(sz_spread_idxs,axis=1)
        if sum(extended_seized_idxs) > 0:
            # Get indices into the sz_prob matrix and times since start of matrix that the seizure started
            first_sz_idxs = sz_spread_idxs.loc[extended_seized_idxs].idxmax(axis=0)
            sz_idxs_arr = np.array(first_sz_idxs)
            sz_order = np.argsort(first_sz_idxs)
            sz_idxs_arr = first_sz_idxs.iloc[sz_order].to_numpy()
            sz_ch_arr = first_sz_idxs.index[sz_order].to_numpy()
            # sz_times_arr = self.get_win_times(len(sz_clf))[sz_idxs_arr]
            # sz_times_arr -= np.min(sz_times_arr)
            # sz_ch_arr = np.array([s.split("-")[0] for s in sz_ch_arr]).flatten()
        else:
            sz_ch_arr = []
            sz_idxs_arr = np.array([])
        sz_idxs_df = pd.DataFrame(sz_idxs_arr.reshape(1,-1),columns=sz_ch_arr)
        sz_idxs_df.drop(columns=sz_idxs_df.columns[(sz_idxs_df == 0).all()]) # zhiyu: drop those start at 0
        if ret_smooth_mat:
            return sz_idxs_df,sz_spread_idxs_all
        else:
            '''sz_idx_df is the onset time of each channel'''
            return sz_idxs_df

# data processing
def get_iEEG_data(
    username,
    password_bin_file,
    iEEG_filename,
    start_time_usec,
    stop_time_usec,
    select_electrodes=None,
):

    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec
    with open(password_bin_file, "r") as f:
        s = Session(username, f.read())
    ds = s.open_dataset(iEEG_filename)
    all_channel_labels = ds.get_channel_labels()

    if select_electrodes is not None:
        if isinstance(select_electrodes[0], Number):
            channel_ids = select_electrodes
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(select_electrodes[0], str):
            channel_ids = [
                i for i, e in enumerate(all_channel_labels) if e in select_electrodes
            ]
            channel_names = select_electrodes
        else:
            print("Electrodes not given as a list of ints or strings")

    try:
        data = ds.get_data(start_time_usec, duration, channel_ids)
    except:
        # clip is probably too big, pull chunks and concatenate
        clip_size = 60 * 1e6
        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            if data is None:
                data = ds.get_data(clip_start, clip_size, channel_ids)
            else:
                data = np.concatenate(
                    ([data, ds.get_data(clip_start, clip_size, channel_ids)]), axis=0
                )
            clip_start = clip_start + clip_size
        data = np.concatenate(
            ([data, ds.get_data(clip_start, stop_time_usec - clip_start, channel_ids)]),
            axis=0,
        )

    df = pd.DataFrame(data, columns=channel_names)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate  # get sample rate
    return df, fs
def notch_filter(data, hz, fs):
    b,a = sig.iirnotch(hz, 30, fs)
    return sig.filtfilt(b,a,data)

def band_pass_filter(data,high,low,fs):
    b, a = sig.butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return sig.filtfilt(b, a, data)

def preprocess(df,fs, current_channel_order, new_channel_order,bipolar_channels):
    segment_data = df.values.T

    fs = int(fs)
    notch_data = notch_filter(segment_data, hz=60, fs=fs)
    filtered_data = band_pass_filter(notch_data,high=15,low=3, fs=fs).T

    signal_len = int(filtered_data.shape[0]/fs*200)
    downsampled_data = sig.resample(filtered_data,signal_len,axis=0)
    
    # Process and reshape data
    indices = [2, 12, 13, 10, 11]
    pz_mean = np.mean(downsampled_data[:, indices], axis=1)

    downsampled_data_with_pz = np.column_stack((downsampled_data, pz_mean))
    reorder_index = [current_channel_order.index(ch) for ch in new_channel_order]
    reordered_data = downsampled_data_with_pz[:, reorder_index]
    #car_data = reordered_data - np.mean(reordered_data, axis=1, keepdims=True)
    bipolar_ids = np.array([[new_channel_order.index(bc.split('-')[0]), new_channel_order.index(bc.split('-')[1])] for bc in bipolar_channels])
    bipolar_data = reordered_data[:, bipolar_ids[:, 0]] - reordered_data[:, bipolar_ids[:, 1]]
    #combined_eeg = np.hstack((car_data, bipolar_data))
    return bipolar_data

