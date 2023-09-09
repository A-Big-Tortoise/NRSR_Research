import numpy as np

def wiener_filter(signal, noise, show=False):

    clean_signal = signal - noise

    signal_power = np.abs(np.fft.fft(clean_signal))**2
    noise_power = np.abs(np.fft.fft(noise))**2
    noise_power = np.mean(noise_power)

    snr = signal_power / noise_power
    wiener_ = 1 / (1 + 1 / snr)

    filtered_signal = np.fft.fft(signal) * wiener_
    filtered_signal = np.fft.ifft(filtered_signal)

    if show:
        pass

    return filtered_signal