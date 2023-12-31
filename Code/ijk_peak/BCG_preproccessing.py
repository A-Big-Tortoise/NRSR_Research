#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:25:07 2022

@author: Song
"""

import pytz
from datetime import datetime
# from influxdb import InfluxDBClient
import operator
from scipy import signal
from statsmodels.tsa.stattools import acf
import pywt
import numpy as np
from scipy.signal import hilbert, savgol_filter, periodogram
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
# from scipy.fft import fft, fftfreq, ifft

def get_covariance_matrix(x, step, window_size):
    N = (len(x) - window_size)//step + 1

    segments = []
    covariance = np.zeros((N, N))
    start = 0
    for i in range(N):
        start = int(i * step)
        end = start + window_size
        segment = x[start:end]
        if len(segment) > 1:
            segment = (segment - np.mean(segment))/np.std(segment)
        segments.append(segment)
    segments = np.array(segments)
    # print(segments.shape)
    covariance1 = np.cov(segments)
    covariance2 = np.cov(segments.T)
        # for j in range(len(segments)):
        #     b = segments[j]
        #     if len(segment) > 1:
        #         auto_cov = sum(segment * b)/(len(segment) -1)
        #         auto_cov /= (np.std(segment) * np.std(b))
        #     else:
        #         auto_cov = segment * b
        #     covariance[i,j] = auto_cov
        #     covariance[j, i] = auto_cov
    return covariance1, covariance2
        
def peak_correction(x, peak_ids):
    corrected_peak_ids = []
    continue_flag = 0
    diff = peak_ids[1:] - peak_ids[:-1]
    for j in range(len(diff)):
        if diff[j] < np.median(diff) - 12 * (100/31.25): 
            if continue_flag:
                continue_flag = 0
                continue
            
            if np.array(x)[peak_ids[j]] > np.array(x)[peak_ids[j + 1]]:
                max_id = peak_ids[j]
                continue_flag = 1
            elif np.array(x)[peak_ids[j]] < np.array(x)[peak_ids[j + 1]]:
                max_id = peak_ids[j+1]
            elif abs(diff[j-1] - np.median(diff)) < abs(diff[j+1] - np.median(diff)):
                max_id = peak_ids[j]
                continue_flag = 1
            elif abs(diff[j-1] - np.median(diff)) > abs(diff[j+1] - np.median(diff)):
                max_id = peak_ids[j+1]
            else:
                max_id = round((peak_ids[j+1] + peak_ids[j])/2)
                continue_flag = 1
            
            if len(corrected_peak_ids) == 0 or max_id != corrected_peak_ids[-1]:
                corrected_peak_ids.append(max_id)
                
        elif len(corrected_peak_ids) == 0 or peak_ids[j] != corrected_peak_ids[-1]:
            if not continue_flag:
                corrected_peak_ids.append(peak_ids[j])
            else:
                continue_flag = 0
            if j == len(diff) - 1:
                corrected_peak_ids.append(peak_ids[j+1])
    corrected_peak_ids = np.array(corrected_peak_ids)
    return corrected_peak_ids

def get_J_peak(segment, start_index):
    peaks_id = signal.find_peaks(segment)[0]
    segment = np.array(segment)
    # if len(peaks_id) == 0:
    if len(peaks_id) > 2:
        max_id = np.argmax(segment[peaks_id])
        # if max_id > 0:
        #     pre_max_id = max_id - 1
        #     if max_id == len(peaks_id) - 1:
        #         peaks_id_2 = peaks_id[:max_id]
        #     else:
        #         peaks_id_2 = np.hstack((peaks_id[:max_id], peaks_id[max_id + 1 :]))
        #     second_max_id = np.argmax(segment[peaks_id_2])
        #     if pre_max_id == second_max_id:
        #         max_id = pre_max_id
        R_peak = peaks_id[max_id] + start_index
    elif len(peaks_id) == 2:
        if abs(segment[peaks_id[0]] - segment[peaks_id[1]]) >= 0.2:
            max_id = np.argmax(segment[peaks_id])
        else:
            max_id = 0
        R_peak = peaks_id[max_id] + start_index
    elif len(peaks_id) == 1:
        R_peak = peaks_id[0] + start_index
    else:
        return None
    return R_peak
    
def J_peak_detection(x, HR):
    segment_lines = []
    last_seg = 0
    # 10s / segment
    n_heartbeat = round(HR//6)
    n_win_len = round(len(x)/n_heartbeat)
    most_win_len = round(len(x)/(n_heartbeat - 1))
    R_peaks_id = []

    for i in range(n_heartbeat):
        start_index = i * n_win_len
        if i == 0:
            start_index = 0
            R_peak = start_index
        else:
            start_index = R_peak + int(3/4 * n_win_len)
        
        if i == 0:
            end_index = R_peak + n_win_len #- 12
        else:
            end_index = R_peak + most_win_len + 12

        if i < n_heartbeat - 1:
            segment = x[start_index:end_index]
        elif end_index < len(x) - 1:
            segment = x[start_index:end_index]
            last_seg = 1
        else:
            segment = x[start_index:]

        R_peak = get_J_peak(segment, start_index)
        # print('=============================')
        # print(start_index, end_index, R_peak)
        if i == 0:
            if R_peak and segment[R_peak] < np.mean(x):
                R_peak = np.argmax(segment)
                continue
            while(not R_peak):
                end_index += 12
                segment = x[start_index:end_index]
                R_peak = get_J_peak(segment, start_index)
                
        elif not R_peak:
            middle_index = (start_index + end_index)//2
            start_index = middle_index - most_win_len//2
            end_index = middle_index + most_win_len//2
            R_peak = get_J_peak(segment, start_index)
            if not R_peak:
                R_peak = middle_index
        # print(start_index, end_index, R_peak)
        # elif not R_peak:
        #     continue
        segment_lines.append([start_index, end_index])
        R_peaks_id.append(R_peak)

    if last_seg:
        start_index = R_peak + int(3/4 * n_win_len)
        end_index = R_peak + most_win_len + 12
        if start_index < len(x):
            segment = x[start_index:]
            R_peak = get_J_peak(segment, start_index)
            if R_peak:
                R_peaks_id.append(R_peak)
    return R_peaks_id, segment_lines

def ACF(x, lag):
    acf = []
    mean_x = np.mean(x)
    var = sum((x - mean_x) ** 2)
    for i in range(lag):
        if i == 0:
            lag_x = x
            original_x = x
        else:
            lag_x = x[i:]
            original_x = x[:-i]
        new_x = sum((lag_x - mean_x) * (original_x - mean_x))/var
        new_x = new_x/len(lag_x)
        acf.append(new_x)
    return np.array(acf)

def annotate_R_peaks(x, HR):
    if HR/6 - (HR//6) > 0.5:
        n_heartbeat = int(HR//6) + 1
    else:
        n_heartbeat = int(HR//6)
    n_win_len = round(len(x)/n_heartbeat)
    a_win_len = round(len(x)/(n_heartbeat-1))
    side_win_len = (a_win_len - n_win_len)//2
    R_peaks_id = []
    for i in range(n_heartbeat):
        start_index = i * n_win_len
        if i > 0:
            start_index -= side_win_len
        
        if i < n_heartbeat - 1:
            end_index = (i + 1) * n_win_len
            end_index += side_win_len
            segment = x[start_index:end_index]
            # print(start_index, end_index)
        else:
            segment = x[start_index:]
        peaks_id = signal.find_peaks(segment)[0]
        segment = np.array(segment)
        if len(peaks_id) == 0:
            # R_peak = np.argwhere(segment == np.max(segment))[0][0] + start_index
            continue
        elif len(peaks_id) > 1:
            max_id = np.argmax(segment[peaks_id])
            R_peak = peaks_id[max_id] + start_index
        else:
            R_peak = peaks_id[0] + start_index
        if x[R_peak] < (np.mean(x) + (0.5 * np.std(x))):
            continue
        R_peaks_id.append(R_peak)
    return R_peaks_id

def kurtosis(x):
    x = x - np.mean(x)
    a = 0
    b = 0
    for i in range(len(x)):
        a += x[i] ** 4
        b += x[i] ** 2
    a = a/len(x)
    b = b/len(x)
    k = a/(b**2)
    return k

#define high-order butterworth low-pass filter
def low_pass_filter(data, Fs, low, order):
    b, a = signal.butter(order, low/(Fs * 0.5), 'low')
    filtered_data = signal.filtfilt(b, a, data)
    # filtered_data = signal.filtfilt(b, a, data, method = 'gust')
    return filtered_data

def high_pass_filter(data, Fs, high, order):
    b, a = signal.butter(order, high/(Fs * 0.5), 'high')
    filtered_data = signal.filtfilt(b, a, data)
    # filtered_data = signal.filtfilt(b, a, data, method = 'gust')
    return filtered_data

def band_pass_filter(data, Fs, low, high, order):
    b, a = signal.butter(order, [low/(Fs * 0.5), high/(Fs * 0.5)], 'bandpass')
    # perform band pass filter
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def wavelet_denoise(data, wave, Fs = None, n_decomposition = None):
    a = data
    w = wave
    ca = []
    cd = []
    rec_a = []
    rec_d = []
    freq_range = []
    for i in range(n_decomposition):
        if i == 0:
            freq_range.append(Fs/2)
        freq_range.append(Fs/2/(2** (i+1)))
        (a, d) = pywt.dwt(a, w)
        ca.append(a)
        cd.append(d)

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec = pywt.waverec(coeff_list, w)
        rec_a.append(rec)
        # ax3[i].plot(Fre, FFT_y1)
        # print(max_freq)

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))
         
    return rec_a, rec_d

def zero_cross_rate(x):
    cnt  = 0
    for i in range(1,len(x)):
        if x[i] > 0 and x[i-1] < 0:
            cnt += 1
        elif x[i] < 0 and x[i-1] > 0:
            cnt += 1
    return cnt

def outlier_detection(x):
    cnt = 0
    x = (x - np.mean(x))/np.std(x)
    for i in range(len(x)):
        if x[i] > 2.96 or x[i] < -2.96:
            cnt += 1
    return cnt

def get_rr_evnvelope(x):
    max_ids = argrelextrema(x, np.greater)[0]
    peaks_val = x[max_ids]
    
    min_ids = argrelextrema(x, np.less)[0]
    valley_val = x[min_ids]
    
    if len(max_ids) > 3 and len(min_ids) > 3:
        if max_ids[0] < min_ids[0]:
            valley_val = np.hstack((x[0], valley_val))
            peaks_val = np.hstack((x[max_ids[0]], peaks_val))
    
        elif max_ids[0] > min_ids[0]:
            valley_val = np.hstack((x[max_ids[0]], valley_val))
            peaks_val = np.hstack((x[0], peaks_val))
        
        if max_ids[-1] > min_ids[-1]:
            valley_val = np.hstack((valley_val, x[-1]))
            peaks_val = np.hstack((peaks_val, x[max_ids[-1]]))
        elif max_ids[-1] < min_ids[-1]:
            valley_val = np.hstack((valley_val, x[max_ids[-1]]))
            peaks_val = np.hstack((peaks_val, x[-1]))
        
        max_ids = np.hstack((0, max_ids))
        max_ids = np.hstack((max_ids, len(x) - 1))
        
        min_ids = np.hstack((0, min_ids))
        min_ids = np.hstack((min_ids, len(x) - 1))
    
        upper_envelope = interp1d(max_ids, peaks_val, kind = 'cubic', bounds_error = False)(list(range(len(x)))) # (x[max_ids[0]], x[max_ids[-1]])
        lower_envelope = interp1d(min_ids, valley_val, kind = 'cubic', bounds_error = False)(list(range(len(x)))) #(x[min_ids[0]], x[min_ids[-1]])
        rr_envelope = (upper_envelope + lower_envelope)/2
        return upper_envelope, lower_envelope, rr_envelope
    else:
        return [], [], []

def rr_estimation(x, n_lag):
    rr_acf = acf(x, nlags = n_lag)
    # rr_acf = rr_acf - np.mean(rr_acf)
    rr_acf = (rr_acf - np.min(rr_acf))/(np.max(rr_acf) - np.min(rr_acf))
    rr_peak_ids, _ = signal.find_peaks(rr_acf, height = 0.3)
    time_diff = rr_peak_ids[1:] - rr_peak_ids[:-1]
    if len(rr_peak_ids) < 2:
        return -1
    
    if len(rr_peak_ids) > 3 and np.std(time_diff) > 32:
        # for i in range(1, len(time_diff)):
        #     if abs(time_diff[i] - time_diff[i-1]) > 32:
                return -1
    
    # check if the peaks is periodic
    median_hr = np.median(time_diff)
    frequency = 1/(median_hr/100)
    
    if frequency > 0.5 or frequency < 0.1:
        return -1
    # plt.figure(figsize = (16,4))
    # plt.plot(rr_acf)
    # # plt.scatter(rr_peak_ids, rr_acf[rr_peak_ids], c = 'r')
    # rr_f, mag = periodogram(rr_acf, fs = 100, nfft = 4096)
    # # print(rr_f[:100])
    # frequency = rr_f[np.argmax(mag)]
    return frequency

def false_J_peak_detection(x, J_peaks_index, median_diff):
    time_diff = np.array(J_peaks_index[1:]) - np.array(J_peaks_index[:-1])
    index_to_delete = []
    for i in range(len(time_diff)):
        if time_diff[i] < median_diff * 0.7:
            diff1 = abs(J_peaks_index[i] - J_peaks_index[i-1] - median_diff)
            diff2 = abs(J_peaks_index[i + 1] - J_peaks_index[i-1] - median_diff)
            
            if diff1 > diff2:
                index_to_delete.append(i)
            else:
                index_to_delete.append(i + 1)
            
            # if x[J_peaks_index[i]] > x[J_peaks_index[i + 1]]:
            #     index_to_delete.append(i+1)
            # elif x[J_peaks_index[i]] < x[J_peaks_index[i + 1]]:
            #     index_to_delete.append(i)
            # else:
            #     if diff1 > diff2:
            #         index_to_delete.append(i)
            #     else:
            #         index_to_delete.append(i + 1)
    
    if len(index_to_delete) > 0:
        for i in range(len(index_to_delete)):
            if i == 0:
                new_J_peaks_index = J_peaks_index[:index_to_delete[i]]
            else:
                new_J_peaks_index += J_peaks_index[index_to_delete[i-1]+1 : index_to_delete[i]]
        new_J_peaks_index += J_peaks_index[index_to_delete[-1] + 1:]
        J_peaks_index = new_J_peaks_index
    
    return J_peaks_index

def missing_J_peak_detection(x, J_peaks_index, median_diff):
    time_diff = np.array(J_peaks_index[1:]) - np.array(J_peaks_index[:-1])
    new_J_peaks = []
    for i in range(len(time_diff)):
        if time_diff[i] - median_diff > median_diff//3:
            segmentation = x[J_peaks_index[i] + median_diff//3 : J_peaks_index[i + 1] - median_diff//3]
            new_index = peak_selection(segmentation, J_peaks_index[i] + median_diff//3, median_diff)
            if new_index:
                new_J_peaks.append(new_index)
    if len(new_J_peaks):
        J_peaks_index = J_peaks_index + new_J_peaks
        J_peaks_index = sorted(J_peaks_index)
    return J_peaks_index

def peak_selection(segmentation, start, median_diff):
    max_ids = argrelextrema(segmentation, np.greater)[0]
    if len(max_ids) > 0:
        for index in max_ids:
            if index == max_ids[0]:
                diff = abs(median_diff - index)
                J_peak_index = index + start
            elif diff > abs(median_diff - index):
                diff = abs(median_diff - index)
                J_peak_index = index + start
    else:
        J_peak_index = None
    return J_peak_index

def J_peaks_detection(x, f):
    window_len = round(1/f * 100)
    start = 0
    J_peaks_index = []
    
    #get J peaks
    while start < len(x):
        end = start + window_len
        if start > 0:
            segmentation = x[start -1 : end]
        else:
            segmentation = x[start : end]
        # J_peak_index = np.argmax(segmentation) + start
        max_ids = argrelextrema(segmentation, np.greater)[0]
        for index in max_ids:
            if index == max_ids[0]:
                max_val = segmentation[index]
                J_peak_index = index + start
            elif max_val < segmentation[index]:
                max_val = segmentation[index]
                J_peak_index = index + start
        
        if len(max_ids) > 0 and x[J_peak_index] > 0:
            if len(J_peaks_index) == 0 or J_peak_index != J_peaks_index[-1]:
                J_peaks_index.append(J_peak_index)
        start = start + window_len//2
    
    #correct J peak false detection
    median_diff = window_len
    # median_diff = np.median(time_diff)
    ##false J peak detection
    # J_peaks_index = false_J_peak_detection(x, J_peaks_index, median_diff)
    # # ## missing J peak detection
    # J_peaks_index = missing_J_peak_detection(x, J_peaks_index, median_diff)
    # ##false J peak detection
    # J_peaks_index = false_J_peak_detection(x, J_peaks_index, median_diff)
    # print(J_peaks_index)
    
    return J_peaks_index

def bed_status_detection(x):
    x = (x - np.mean(x))/np.std(x)
    zcr = zero_cross_rate(x)
    
    rec_a, rec_d = wavelet_denoise(data = x, wave = 'db4', Fs = 100, n_decomposition = 6)
    min_len = min(len(rec_d[-4]), len(rec_d[-3])) #len(rec_d[-5])
    denoised_sig = rec_d[-4][:min_len] + rec_d[-3][:min_len]
    
    denoise_zcr = zero_cross_rate(denoised_sig)

    if denoise_zcr > 165:
        if zcr < 180:
            bed_status = 1
        else:
            bed_status = -1
    elif denoise_zcr < 150:
        bed_status = 1
    elif zcr > 340:
        bed_status = -1
    else:
        bed_status = 1
    return bed_status, denoise_zcr, zcr

def get_envelope(x, n_decomposition, Fs):
    x = (x - np.mean(x))/np.std(x)
    rec_a, rec_d = wavelet_denoise(data=x, wave='db4', Fs=Fs, n_decomposition=n_decomposition)
    
    min_len = min(len(rec_d[-1]), len(rec_d[-2]), len(rec_d[-3]), len(rec_d[-4]), len(rec_a[-1])) # len(rec_d[-5])
    denoised_sig = rec_d[-1][:min_len] + rec_d[-2][:min_len] + rec_d[-4][:min_len] + rec_d[-3][:min_len] + rec_a[-1][:min_len]
    z = hilbert(denoised_sig) #form the analytical signal
    envelope = np.abs(z)
    
    # smoothed_envelope = band_pass_filter(data = np.copy(envelope), Fs = 100, 
    #                                 high = 3, low = 0.1, order = 3)
    smoothed_envelope = savgol_filter(envelope, 41, 2, mode='nearest')
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
    
    return smoothed_envelope

def energy_envelope(x, N):
    energy = x ** 2
    envelope = np.convolve(energy, np.ones(N), mode='valid')
    return envelope

def signal_quality_assessment(
        x, n_decomposition, Fs, n_lag, target='HR', denoised_method='DWT', show=False):
    x = (x - np.mean(x))/np.std(x)

    # if round(kurtosis(x)) > 5:
    #     return ['movement']
    #     # return ['bad data']
    if denoised_method == 'DWT':
        rec_a, rec_d = wavelet_denoise(data=x, wave='db7', Fs=Fs, n_decomposition=n_decomposition)
        
        min_len = min(len(rec_d[-1]), len(rec_d[-2]), len(rec_d[-3]), len(rec_d[-4])) #len(rec_a[-1]) len(rec_d[-5])
        denoised_sig = rec_d[-1][:min_len] + rec_d[-2][:min_len] + rec_d[-4][:min_len] + rec_d[-3][:min_len] #+ rec_a[-1][:min_len]
    elif denoised_method == 'bandpass':
        denoised_sig = band_pass_filter(data=x, Fs=100, low=0.6, high=10, order=5)
    # denoised_sig = acf(x, nlags = len(x))/np.arange(len(x), 0, -1)
    # fig, ax = plt.subplots(2, 1, figsize=(16,8))
    # ax[0].plot(x)
    # ax[1].plot(denoised_sig)
    
    # min_len = min(len(rec_d[-1]), len(rec_d[-2]), len(rec_d[-3]), len(rec_d[-4]), len(rec_a[-1])) # len(rec_d[-5])
    # denoised_sig = rec_d[-1][:min_len] + rec_d[-2][:min_len] + rec_d[-4][:min_len] + rec_d[-3][:min_len] + rec_a[-1][:min_len]

    index = 0
    window_size = 100
    z = hilbert(denoised_sig) #form the analytical signal
    envelope = np.abs(z)
    
    smoothed_envelope = savgol_filter(envelope, 41, 2, mode='nearest')
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
    
    trend = savgol_filter(smoothed_envelope, 201, 2, mode='nearest')
    smoothed_envelope = smoothed_envelope - trend
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)


    acf_x = acf(smoothed_envelope, nlags=n_lag)
    # acf_x = acf_x[200:]
    acf_x = acf_x - np.mean(acf_x)
    f, Pxx_den = periodogram(acf_x, fs=100, nfft=1024)
    # print(f)
    
    sig_means = []
    index = 0
    frequency = f[np.argmax(Pxx_den)]
    power = max(Pxx_den)
    # print('ACF frequency:', frequency)
    
    if show:
        fig, ax = plt.subplots(6, 1, figsize=(16, 18))
        ax[0].plot(x, label='raw data')
        ax[1].plot(denoised_sig, label='wavelet denoised data')
        ax[2].plot(envelope, label='envelope extraction by Hilbert transform')
        ax[3].plot(smoothed_envelope, label='smoothed envelope')
        # ax[3].plot(rr_envelope)
        if target == 'HR':
            ax[4].plot(acf_x, label='ACF of smoothed envelope')
            ax[5].plot(f, Pxx_den, label='spectrum of ACF')
            for i in range(len(ax)):
                ax[i].legend()

    window_size = 100
    while index + window_size < len(acf_x):
        sig_means.append(np.mean(acf_x[index:index + window_size]))
        index = index + window_size

    # 方差、频率、能量都是smoothed_envelope的acf结果
    if np.std(sig_means) < 0.1 and 0.6< frequency < 2.5 and power > 0.1:
        # peak_ids, _ = signal.find_peaks(acf_x, height = 0.1)
        peak_ids, _ = signal.find_peaks(acf_x, height=np.mean(acf_x))
        time_diff = peak_ids[1:] - peak_ids[:-1]
        
        # check if the peaks is periodic
        cadidates = []
        for peak_id in peak_ids:
            if (peak_id > 51 and peak_id < 167):
                cadidates.append(peak_id)
        
        if len(cadidates) > 1:
            sorted_ids = sorted(range(len(acf_x[cadidates])), key=lambda k: acf_x[cadidates][k])
            height = acf_x[cadidates][sorted_ids[-1]]
            # print(acf_x[cadidates])
            # print(cadidates)
            if (height - acf_x[cadidates][sorted_ids[-2]] > 0.2) or \
                (height - acf_x[cadidates][sorted_ids[-2]] > 0.5 * height):
                median_hr = cadidates[sorted_ids[-1]]
            elif sorted_ids[-2] > sorted_ids[-1]:
                median_hr = cadidates[sorted_ids[-1]]
            elif len(cadidates) > 2:
                # if height - acf_x[cadidates][sorted_ids[-3]] > 0.2 or \
                #     (height - acf_x[cadidates][sorted_ids[-3]] > 0.5 * height):
                #     median_hr = cadidates[sorted_ids[-2]]
                # elif sorted_ids[-2] > sorted_ids[-3]:
                #     median_hr = cadidates[sorted_ids[-3]]
                # else:
                #     median_hr = cadidates[sorted_ids[-2]]
                median_hr = cadidates[np.argmax(acf_x[cadidates])]
            else:
                median_hr = cadidates[sorted_ids[-2]]
                # print(median_hr)
            # median_hr = cadidates[np.argmax(acf_x[cadidates])]
        elif len(cadidates) == 1:
            median_hr = cadidates[0]
        else:
            median_hr = np.median(time_diff)
        
        frequency = 1/(median_hr/100)
        # frequency = peak_ids[1] - peak_ids[0]
        if len(peak_ids) > 3:
            res = ['good data', np.std(sig_means), frequency, power]
            if show:
                if target == 'HR':
                    # ax[4].plot(acf_x)
                    ax[4].scatter(peak_ids, acf_x[peak_ids])
                fig.suptitle('good data')
            # else:
            #     if show:
            #         ax[4].scatter(peak_ids, acf_x[peak_ids], c = 'r')
            #     res = ['bad data', np.std(sig_means), frequency, power]
        else:
            res = ['bad data', np.std(sig_means), frequency, power]
        # res = [f[np.argmax(Pxx_den)], max(Pxx_den), 'rx']
    else:
        res = ['bad data', np.std(sig_means), frequency, power]
        if show:
            fig.suptitle('bad data')
        # res = [f[np.argmax(Pxx_den)], max(Pxx_den), 'go']
    return res
