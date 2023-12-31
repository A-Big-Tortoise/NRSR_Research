#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 23:05:13 2022

@author: Song
"""

import numpy as np
# from utils import read_influx, epoch_time_local, int_to_mac
from scipy import signal
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.signal import hilbert, periodogram
from BCG_preproccessing import get_envelope, annotate_R_peaks, low_pass_filter, band_pass_filter, wavelet_denoise
# import heartpy as hp
from statsmodels.tsa.stattools import acf
from BCG_preproccessing import signal_quality_assessment, bed_status_detection, J_peak_detection, peak_correction

# load data
target_datasets = ['train', 'test', 'good', 'bad', 'overall']
target_dataset = target_datasets[4]
print('===============================')
print('load data......')
if target_dataset == 'train':
    bcg_ppg_dataset = np.load('../data/good_BCG_PPG_train.npy')
    dataset = np.load('../data/bcg_mask_pair_train.npy')
elif target_dataset == 'test':
    bcg_ppg_dataset = np.load('../data/good_BCG_PPG_test.npy')
    dataset = np.load('../data/bcg_mask_pair_test.npy')
elif target_dataset == 'overall':
    bcg_ppg_train_dataset = np.load('../data/good_BCG_PPG_train.npy')
    dataset_train = np.load('../data/bcg_mask_pair_train.npy')
    bcg_ppg_test_dataset = np.load('../data/good_BCG_PPG_test.npy')
    dataset_test = np.load('../data/bcg_mask_pair_test.npy')
    bcg_ppg_dataset = np.vstack((bcg_ppg_train_dataset, bcg_ppg_test_dataset))
    dataset = np.vstack((dataset_train, dataset_test))
    
elif target_dataset == 'good':
    dataset = np.load('../data/good_bcg_mask_dataset.npy')
elif target_dataset == 'bad':
    dataset = np.load('../data/bad_bcg_mask_dataset.npy')

if target_dataset == 'train' or target_dataset == 'test' or target_dataset == 'overall':
    bcg_dataset = bcg_ppg_dataset[:, :1400]
else:
    bcg_dataset = dataset[:, :1400]

mask_labels = dataset[:, -1000:]
ppg_fs = 100
overall_error = 0
good_cnt = 0
bad_pred_id = []
preds = []
true_label = []

for i in range(0, bcg_dataset.shape[0]): #bcg_ppg.shape[0]
    print('%d======================================' % (i+1))
    BCG = bcg_dataset[i, :]
    mask = mask_labels[i, :]
    
    BCG = (BCG - np.min(BCG))/(np.max(BCG) - np.min(BCG))
    envelope = get_envelope(x=BCG, n_decomposition=6, Fs=100)
    envelope = envelope[:len(BCG)]
    envelope = (envelope - np.min(envelope))/(np.max(envelope) - np.min(envelope))
    acf_env = acf(envelope, nlags=len(envelope))
    acf_env = band_pass_filter(data=acf_env, Fs=ppg_fs, low=0.6, high=3, order=3)
    acf_env = (acf_env - np.min(acf_env))/(np.max(acf_env) - np.min(acf_env))
    acf_env = acf_env[round(100 * 2):-round(100 * 2)]
    
    res = signal_quality_assessment(x=BCG, n_decomposition=6, Fs=100, n_lag=len(BCG)//2,
                                    target='HR', show=False)
    if res[0] != 'good data':
        print(res[0])
        continue

    hr = round(res[2] * 60)
    peaks_id = []
    for j in range(len(mask)):
        if mask[j] == 1:
            peaks_id.append(j)
    peaks_id = np.array(peaks_id)
    
    J_peaks_id, segment_lines = J_peak_detection(acf_env, hr)
    
    if target_dataset != 'bad':
        if abs(len(J_peaks_id) - len(peaks_id)) > 1:
            bad_pred_id.append(i)
            J_peaks_id = np.array(J_peaks_id)
            J_J_inv = J_peaks_id[1:] - J_peaks_id[:-1]
            J_J_inv = np.mean(J_J_inv * 10)
            
            IBI = peaks_id[1:] - peaks_id[:-1]
            IBI = np.mean(IBI * 10)
            overall_error += abs(J_J_inv - IBI)
        else:
            if len(peaks_id) - len(J_peaks_id) == 1:
                peaks_id = peaks_id[1:]
            elif len(J_peaks_id) - len(peaks_id) == 1:
                J_peaks_id = J_peaks_id[1:]
            good_cnt += 1
            J_peaks_id = np.array(J_peaks_id)
            J_J_inv = J_peaks_id[1:] - J_peaks_id[:-1]
            J_J_inv = J_J_inv * 10
            for j in range(len(J_J_inv)):
                preds.append(J_J_inv[j])
            
            IBI = peaks_id[1:] - peaks_id[:-1]
            IBI = IBI * 10
            for j in range(len(IBI)):
                true_label.append(IBI[j])
            
            overall_error += np.mean(abs(J_J_inv - IBI))
    
    
    # plt.figure(num = i+1, figsize = (16,4))
    # plt.plot(acf_env)
    # plt.scatter(J_peaks_id, acf_env[J_peaks_id], c = 'g')
    # for j in range(len(segment_lines)):
    #     plt.vlines(segment_lines[j][0], np.min(acf_env), np.max(acf_env),
    #                 linestyles = '--', colors = 'g')
    #     plt.vlines(segment_lines[j][1], np.min(acf_env), np.max(acf_env), 
    #                 linestyles = '--', colors = 'r')
    # plt.plot(mask)
    
if good_cnt:
    print('MAE:%.2f' % (overall_error/good_cnt), 'ms')#27ms for training #27 for testing
    sorted_id = sorted(range(len(true_label)), key=lambda k: true_label[k])
    plt.figure()
    plt.title(label='HRV testing result')
    plt.scatter(list(range(len(preds))), np.array(preds)[sorted_id], 
                s=1, alpha=0.5, label='prediction')
    plt.scatter(list(range(len(true_label))), np.array(true_label)[sorted_id], 
                s=1, alpha=0.5, label='true_label')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title(label='HRV testing result')
    plt.scatter(list(range(len(preds))), np.array(preds)[sorted_id], 
                s=1, alpha=0.5, label='prediction')
    plt.legend()
    plt.show()
