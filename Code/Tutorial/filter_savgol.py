
import scipy.signal
def savgol_filter(signal, window_length=64, polyorder=1, show=False):
    filtered_signal = scipy.signal.savgol_filter(signal, window_length, polyorder)
    if show:
        pass
    return filtered_signal