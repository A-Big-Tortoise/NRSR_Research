# -*- coding: utf-8 -*-
import math
import random

import numpy as np
import scipy

from signal_distort import signal_distort
from signal_resample import signal_resample
import matplotlib.pyplot as plt
import pywt


def scg_simulate(
        duration=10, length=None, sampling_rate=100, noise=0.01, heart_rate=60, heart_rate_std=1, respiratory_rate=15,
        systolic=120, diastolic=80, method="simple", random_state=None, template=False
):
    """Simulate an scg/EKG signal.

    Generate an artificial (synthetic) scg signal of a given duration and sampling rate using either
    the scgSYN dynamical model (McSharry et al., 2003) or a simpler model based on Daubechies wavelets
    to roughly approximate cardiac cycles.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    length : int
        The desired length of the signal (in samples).
    noise : float
        Noise level (amplitude of the laplace noise).
    heart_rate : int
        Desired simulated heart rate (in beats per minute). The default is 70. Note that for the
        scgSYN method, random fluctuations are to be expected to mimick a real heart rate. These
        fluctuations can cause some slight discrepancies between the requested heart rate and the
        empirical heart rate, especially for shorter signals.
    heart_rate_std : int
        Desired heart rate standard deviation (beats per minute).
    method : str
        The model used to generate the signal. Can be 'simple' for a simulation based on Daubechies
        wavelets that roughly approximates a single cardiac cycle.
        Jasper Chen: Now method can be 'simple'/'daubechies', 'biorthogonal', 'symlet', 'meyer' and 'coiflets'
    random_state : int
        Seed for the random number generator.

    Returns
    -------
    array
        Vector containing the scg signal.

    """
    # Seed the random generator for reproducible results
    np.random.seed(random_state)

    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = duration * sampling_rate
    if duration is None:
        duration = length / sampling_rate

    # Run appropriate method

    if method.lower() in ["simple", "daubechies"]:
        # print("method is:", method)
        scg = _scg_simulate_daubechies(
            duration=duration, length=length, sampling_rate=sampling_rate, heart_rate=heart_rate,
            respiratory_rate=respiratory_rate, systolic=systolic, diastolic=diastolic, template=template
        )
    # Coiflets, Biorthogonal
    if method.lower() in ["symlets", "symlet", "sym"]:
        scg = _scg_simulate_symlets(
            duration=duration, length=length, sampling_rate=sampling_rate, heart_rate=heart_rate,
            respiratory_rate=respiratory_rate, systolic=systolic, diastolic=diastolic, template=template
        )

    # Coiflets, Biorthogonal
    if method.lower() in ["coiflets", "coiflet", "coif"]:
        scg = _scg_simulate_coiflets(
            duration=duration, length=length, sampling_rate=sampling_rate, heart_rate=heart_rate,
            respiratory_rate=respiratory_rate, systolic=systolic, diastolic=diastolic, template=template
        )

    # Add random noise
    if noise > 0:
        scg = signal_distort(
            scg,
            sampling_rate=sampling_rate,
            noise_amplitude=noise,
            noise_frequency=[5, 10, 100],
            noise_shape="laplace",
            random_state=random_state,
            silent=True,
        )

    # Reset random seed (so it doesn't affect global)
    np.random.seed(None)
    return scg


# =============================================================================
# Daubechies
# =============================================================================
def _scg_simulate_daubechies(
        duration=10,
        length=None,
        sampling_rate=100,
        heart_rate=70,
        respiratory_rate=15,
        systolic=120,
        diastolic=80,
        template=False):
    """Generate an artificial (synthetic) scg signal of a given duration and sampling rate.

    It uses a 'Daubechies' wavelet that roughly approximates a single cardiac cycle.
    This function is based on `this script <https://github.com/diarmaidocualain/scg_simulation>`_.

    """
    # The "Daubechies" wavelet is a rough approximation to a real, single, cardiac cycle
    # distance = np.exp((heart_rate - (systolic + diastolic)/2)/10)
    # p = int(round(25/(np.exp(8) - np.exp(-9)) * distance + 9))

    # distance = np.exp((heart_rate - (systolic + diastolic)/2)/10)
    # p = int(round(25/(np.exp(8) - np.exp(-9)) * distance + 9))

    # # print(p)
    # # min_p = 9 max_p = 34
    # cardiac_s = scipy.signal.wavelets.daub(int(p))
    # cardiac_d = scipy.signal.wavelets.daub(int(p)) * (diastolic/systolic)
    # print(f"cardiac_s: {len(cardiac_s)}, cardiac_d: {len(cardiac_d)}")

    # cardiac_s = scipy.signal.wavelets.daub(int(systolic/10)) * int(math.sqrt(pow(systolic,2)+pow(heart_rate,2)))
    # # print("cardiac_s:", len(cardiac_s))

    # cardiac_d = scipy.signal.wavelets.daub(int(diastolic/10)) * int(math.sqrt(pow(diastolic,2)+pow(heart_rate,2))*0.3)
    # print("cardiac_d:", len(cardiac_d))

    # Add the gap after the pqrst when the heart is resting.
    # cardiac = np.concatenate([cardiac, np.zeros(10)])
    # cardiac = np.concatenate([cardiac_s, cardiac_d])

    cardiac_length = int(100 * sampling_rate / heart_rate)  # sampling_rate #

    ind = random.randint(17, 34)
    cardiac_s = scipy.signal.wavelets.daub(ind)
    cardiac_d = scipy.signal.wavelets.daub(ind) * 0.3 * diastolic / 80  # change height to 0.3

    cardiac_s = scipy.signal.resample(cardiac_s, 100)
    cardiac_d = scipy.signal.resample(cardiac_d, 100)

    cardiac_s = cardiac_s[0:40]

    distance = 180 - systolic  # systolic 81-180
    # distance = cardiac_length - len(cardiac_s) - len(cardiac_d) - systolic # here 140 = 40 (cardiac_s) + 100 (cardiac_d) as below
    zero_signal = np.zeros(distance)
    cardiac = np.concatenate([cardiac_s, zero_signal, cardiac_d])
    # cardiac = scipy.signal.resample(cardiac, 100) # fix every cardiac length to 100
    cardiac = scipy.signal.resample(cardiac, cardiac_length)  # fix every cardiac length to 1000/heart_rate
    template = False


    if template == False:
        # Caculate the number of beats in capture time period
        num_heart_beats = int(duration * heart_rate / 60)

        # Concatenate together the number of heart beats needed
        scg = np.tile(cardiac, num_heart_beats)

        # Resample
        scg = signal_resample(
            scg, sampling_rate=int(len(scg) / 10), desired_length=length, desired_sampling_rate=sampling_rate
        )
        # print(scg.shape)

    elif template == True:
        scg = cardiac
        scg = scipy.signal.resample(scg, int(60 * sampling_rate / heart_rate))  # fix every cardiac length to 1000/heart_rate

    ### add rr
    num_points = duration * sampling_rate
    x_space = np.linspace(0, 1, num_points)
    seg_fre = respiratory_rate / (60 / duration)
    seg_amp = max(scg) * 0.00001
    rr_component = seg_amp * np.sin(2 * np.pi * seg_fre * x_space)
    # scg *= rr_component
    # plt.figure(figsize = (16,2))
    # plt.plot(scg)
    # plt.plot(rr_component * 1000)
    # plt.scatter(peaks, scg[peaks], c = 'r')

    # #modeified rr component
    # for i in range(len(scg)):
    #     if scg[i] > 0:
    #         scg[i] *= (rr_component[i] + 2 * seg_amp)
    #     elif scg[i] < 0:
    #         scg[i] *= (rr_component[i] + 2 * seg_amp)
    scg *= (rr_component + 2 * seg_amp)
    # scg *= seg_amp
    # plt.figure(figsize = (16,2))
    # plt.plot(scg)

    # import matplotlib.pyplot as plt
    # # plt.plot(rr_component,'r')
    # plt.plot(scg)
    # plt.show()

    # import pdb; pdb.set_trace()
    return scg

# =============================================================================
# Symlets
# =============================================================================
def _scg_simulate_symlets(
        duration=10,
        length=None,
        sampling_rate=100,
        heart_rate=70,
        respiratory_rate=15,
        systolic=120,
        diastolic=80,
        template=False):

    cardiac_length = int(100 * sampling_rate / heart_rate)  # sampling_rate #

    # ind = random.randint(17, 34)
    # cardiac_s = scipy.signal.wavelets.daub(ind)
    # cardiac_d = scipy.signal.wavelets.daub(ind) * 0.3 * diastolic / 80  # change height to 0.3

    ind = random.randint(11, 21)
    wavelet = np.array(pywt.DiscreteContinuousWavelet('sym'+str(ind)).rec_hi)
    cardiac_s = wavelet
    cardiac_d = wavelet * 0.3 * diastolic / 80

    cardiac_s = scipy.signal.resample(cardiac_s, 100)
    cardiac_d = scipy.signal.resample(cardiac_d, 100)

    cardiac_s = cardiac_s[0:40]

    distance = 180 - systolic  # systolic 81-180
    # distance = cardiac_length - len(cardiac_s) - len(cardiac_d) - systolic # here 140 = 40 (cardiac_s) + 100 (cardiac_d) as below
    zero_signal = np.zeros(distance)
    cardiac = np.concatenate([cardiac_s, zero_signal, cardiac_d])
    # cardiac = scipy.signal.resample(cardiac, 100) # fix every cardiac length to 100
    cardiac = scipy.signal.resample(cardiac, cardiac_length)  # fix every cardiac length to 1000/heart_rate


    if template == False:
        # Caculate the number of beats in capture time period
        num_heart_beats = int(duration * heart_rate / 60)

        # Concatenate together the number of heart beats needed
        scg = np.tile(cardiac, num_heart_beats)

        # Resample
        scg = signal_resample(
            scg, sampling_rate=int(len(scg) / 10), desired_length=length, desired_sampling_rate=sampling_rate
        )
        # print(scg.shape)

    elif template == True:
        scg = cardiac
        scg = scipy.signal.resample(scg, int(60 * sampling_rate / heart_rate))  # fix every cardiac length to 1000/heart_rate

    ### add rr
    num_points = duration * sampling_rate
    x_space = np.linspace(0, 1, num_points)
    seg_fre = respiratory_rate / (60 / duration)
    seg_amp = max(scg) * 0.00001
    rr_component = seg_amp * np.sin(2 * np.pi * seg_fre * x_space)

    scg *= (rr_component + 2 * seg_amp)
    # scg *= seg_amp

    return scg


# =============================================================================
# Coflits
# =============================================================================
def _scg_simulate_coiflets(
        duration=10,
        length=None,
        sampling_rate=100,
        heart_rate=70,
        respiratory_rate=15,
        systolic=120,
        diastolic=80,
        template=False):

    cardiac_length = int(100 * sampling_rate / heart_rate)  # sampling_rate #

    ind = random.randint(3, 18)
    wavelet = np.array(pywt.DiscreteContinuousWavelet('coif'+str(ind)).rec_hi)
    cardiac_s = wavelet
    cardiac_d = wavelet * 0.3 * diastolic / 80

    cardiac_s = scipy.signal.resample(cardiac_s, 100)
    cardiac_d = scipy.signal.resample(cardiac_d, 100)

    cardiac_s = cardiac_s[0:40]

    distance = 180 - systolic  # systolic 81-180
    # distance = cardiac_length - len(cardiac_s) - len(cardiac_d) - systolic # here 140 = 40 (cardiac_s) + 100 (cardiac_d) as below
    zero_signal = np.zeros(distance)
    cardiac = np.concatenate([cardiac_s, zero_signal, cardiac_d])
    # cardiac = scipy.signal.resample(cardiac, 100) # fix every cardiac length to 100
    cardiac = scipy.signal.resample(cardiac, cardiac_length)  # fix every cardiac length to 1000/heart_rate


    if template == False:
        # Caculate the number of beats in capture time period
        num_heart_beats = int(duration * heart_rate / 60)

        # Concatenate together the number of heart beats needed
        scg = np.tile(cardiac, num_heart_beats)

        # Resample
        scg = signal_resample(
            scg, sampling_rate=int(len(scg) / 10), desired_length=length, desired_sampling_rate=sampling_rate
        )
        # print(scg.shape)

    elif template == True:
        scg = cardiac
        scg = scipy.signal.resample(scg, int(60 * sampling_rate / heart_rate))  # fix every cardiac length to 1000/heart_rate

    ### add rr
    num_points = duration * sampling_rate
    x_space = np.linspace(0, 1, num_points)
    seg_fre = respiratory_rate / (60 / duration)
    seg_amp = max(scg) * 0.00001
    rr_component = seg_amp * np.sin(2 * np.pi * seg_fre * x_space)

    scg *= (rr_component + 2 * seg_amp)
    # scg *= seg_amp

    return scg
