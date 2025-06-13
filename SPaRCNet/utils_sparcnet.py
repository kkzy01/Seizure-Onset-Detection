import numpy as np
import pandas as pd
from ieeg.auth import Session
import os
from numbers import Number
import pyedflib
import warnings
import pickle
warnings.filterwarnings("ignore")
from scipy import signal as sig
import scipy as sc
import time
import re
def _pull_iEEG(ds, start_usec, duration_usec, channel_ids):
    """
    Pull data while handling iEEGConnectionError
    """
    i = 0
    while True:
        if i == 50:
            logger = logging.getLogger()
            logger.error(
                f"failed to pull data for {ds.name}, {start_usec / 1e6}, {duration_usec / 1e6}, {len(channel_ids)} channels"
            )
            return None
        try:
            data = ds.get_data(start_usec, duration_usec, channel_ids)
            return data
        except Exception as _:
            time.sleep(1)
            i += 1
def clean_labels(channel_li: list, pt: str) -> list:
    """This function cleans a list of channels and returns the new channels

    Args:
        channel_li (list): _description_

    Returns:
        list: _description_
    """

    new_channels = []
    for i in channel_li:
        i = i.replace("-", "")
        i = i.replace("GRID", "G")  # mne has limits on channel name size
        # standardizes channel names
        pattern = re.compile(r"([A-Za-z0-9]+?)(\d+)$")
        regex_match = pattern.match(i)

        if regex_match is None:
            new_channels.append(i)
            continue

        # if re.search('Cz|Fz|C3|C4|EKG',i):
        #     continue
        lead = regex_match.group(1).replace("EEG", "").strip()
        contact = int(regex_match.group(2))
        if pt in ("HUP75_phaseII", "HUP075", "sub-RID0065"):
            if lead == "Grid":
                lead = "G"

        if pt in ("HUP78_phaseII", "HUP078", "sub-RID0068"):
            if lead == "Grid":
                lead = "LG"

        if pt in ("HUP86_phaseII", "HUP086", "sub-RID0018"):
            conv_dict = {
                "AST": "LAST",
                "DA": "LA",
                "DH": "LH",
                "Grid": "LG",
                "IPI": "LIPI",
                "MPI": "LMPI",
                "MST": "LMST",
                "OI": "LOI",
                "PF": "LPF",
                "PST": "LPST",
                "SPI": "RSPI",
            }
            if lead in conv_dict:
                lead = conv_dict[lead]
        
        if pt in ("HUP93_phaseII", "HUP093", "sub-RID0050"):
            if lead.startswith("G"):
                lead = "G"
    
        if pt in ("HUP89_phaseII", "HUP089", "sub-RID0024"):
            if lead in ("GRID", "G"):
                lead = "RG"
            if lead == "AST":
                lead = "AS"
            if lead == "MST":
                lead = "MS"

        if pt in ("HUP99_phaseII", "HUP099", "sub-RID0032"):
            if lead == "G":
                lead = "RG"

        if pt in ("HUP112_phaseII", "HUP112", "sub-RID0042"):
            if "-" in i:
                new_channels.append(f"{lead}{contact:02d}-{i.strip().split('-')[-1]}")
                continue
        if pt in ("HUP116_phaseII", "HUP116", "sub-RID0175"):
            new_channels.append(f"{lead}{contact:02d}".replace("-", ""))
            continue

        if pt in ("HUP123_phaseII_D02", "HUP123", "sub-RID0193"):
            if lead == "RS": 
                lead = "RSO"
            if lead == "GTP":
                lead = "RG"
        
        new_channels.append(f"{lead}{contact:02d}")

        if pt in ("HUP189", "HUP189_phaseII", "sub-RID0520"):
            conv_dict = {"LG": "LGr"}
            if lead in conv_dict:
                lead = conv_dict[lead]
                
    return new_channels
def get_iEEG_data(
    username: str,
    password_bin_file: str,
    iEEG_filename: str, 
    start_time_usec: float,
    stop_time_usec: float,
    select_electrodes=None,
    ignore_electrodes=None,
    outputfile=None,
    force_pull = False
):
    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec

    with open(password_bin_file, "r") as f:
        pwd = f.read()

    iter = 0
    while True:
        try:
            if iter == 50:
                raise ValueError("Failed to open dataset")
            s = Session(username, pwd)
            ds = s.open_dataset(iEEG_filename)
            all_channel_labels = ds.get_channel_labels()
            break
            
        except Exception as e:
            time.sleep(1)
            iter += 1
    all_channel_labels = clean_labels(all_channel_labels, iEEG_filename)
    
    if select_electrodes is not None:
        if isinstance(select_electrodes[0], Number):
            channel_ids = select_electrodes
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(select_electrodes[0], str):
            select_electrodes = clean_labels(select_electrodes, iEEG_filename)
            if any([i not in all_channel_labels for i in select_electrodes]):
                if force_pull:
                    select_electrodes = [e for e in select_electrodes
                                          if e in all_channel_labels]
                else:
                    raise ValueError("Channel not in iEEG")

            channel_ids = [
                i for i, e in enumerate(all_channel_labels) if e in select_electrodes
            ]
            channel_names = select_electrodes
        else:
            print("Electrodes not given as a list of ints or strings")

    elif ignore_electrodes is not None:
        if isinstance(ignore_electrodes[0], int):
            channel_ids = [
                i
                for i in np.arange(len(all_channel_labels))
                if i not in ignore_electrodes
            ]
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(ignore_electrodes[0], str):
            ignore_electrodes = clean_labels(ignore_electrodes, iEEG_filename)
            channel_ids = [
                i
                for i, e in enumerate(all_channel_labels)
                if e not in ignore_electrodes
            ]
            channel_names = [
                e for e in all_channel_labels if e not in ignore_electrodes
            ]
        else:
            print("Electrodes not given as a list of ints or strings")

    else:
        channel_ids = np.arange(len(all_channel_labels))
        channel_names = all_channel_labels

    # if clip is small enough, pull all at once, otherwise pull in chunks
    if (duration < 120 * 1e6) and (len(channel_ids) < 100):
        data = _pull_iEEG(ds, start_time_usec, duration, channel_ids)
    elif (duration > 120 * 1e6) and (len(channel_ids) < 100):
        # clip is probably too big, pull chunks and concatenate
        clip_size = 60 * 1e6

        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            if data is None:
                data = _pull_iEEG(ds, clip_start, clip_size, channel_ids)
            else:
                new_data = _pull_iEEG(ds, clip_start, clip_size, channel_ids)
                data = np.concatenate((data, new_data), axis=0)
            clip_start = clip_start + clip_size

        last_clip_size = stop_time_usec - clip_start
        new_data = _pull_iEEG(ds, clip_start, last_clip_size, channel_ids)
        data = np.concatenate((data, new_data), axis=0)
    else:
        # there are too many channels, pull chunks and concatenate
        channel_size = 20
        channel_start = 0
        data = None
        while channel_start + channel_size < len(channel_ids):
            if data is None:
                data = _pull_iEEG(
                    ds,
                    start_time_usec,
                    duration,
                    channel_ids[channel_start : channel_start + channel_size],
                )
            else:
                new_data = _pull_iEEG(
                    ds,
                    start_time_usec,
                    duration,
                    channel_ids[channel_start : channel_start + channel_size],
                )
                data = np.concatenate((data, new_data), axis=1)
            channel_start = channel_start + channel_size

        last_channel_size = len(channel_ids) - channel_start
        new_data = _pull_iEEG(
            ds,
            start_time_usec,
            duration,
            channel_ids[channel_start : channel_start + last_channel_size],
        )
        data = np.concatenate((data, new_data), axis=1)

    df = pd.DataFrame(data, columns=channel_names)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate  # get sample rate

    if outputfile:
        with open(outputfile, "wb") as f:
            pickle.dump([df, fs], f)
    else:
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
    #downsample to 200 Hz
    signal_len = int(filtered_data.shape[0]/fs*200)
    data_bpd = sig.resample(filtered_data,signal_len,axis=0)
    # Process and reshape data
    reorder_index = [current_channel_order.index(ch) for ch in new_channel_order]
    reordered_data = data_bpd[:, reorder_index]
    bipolar_ids = np.array([[new_channel_order.index(bc.split('-')[0]), new_channel_order.index(bc.split('-')[1])] for bc in bipolar_channels])
    bipolar_data = reordered_data[:, bipolar_ids[:, 0]] - reordered_data[:, bipolar_ids[:, 1]]
    return bipolar_data



def get_onset_and_spread(sz_prob,threshold=None,
                        ret_smooth_mat = True, #True
                        filter_w = 5, # seconds 
                        rwin_size = 10, # seconds #10
                        rwin_req = 8, # seconds #9
                        w_size = 1,
                        w_stride = 0.5
                        ): 

        sz_clf = (sz_prob>threshold).reset_index(drop=True)
        filter_w_idx = np.floor((filter_w - w_size)/w_stride).astype(int) + 1
        sz_clf = pd.DataFrame(sc.ndimage.median_filter(sz_clf,size=filter_w_idx,mode='nearest',origin=0),columns=sz_prob.columns)
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
