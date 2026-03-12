import numpy as np
import scipy as sp
from copy import deepcopy


####get noise
def get_noise_from_time_domain(tdi_td, dt, nperseg=15000):
    f, psdA = sp.signal.welch(tdi_td["A"], fs=1.0/dt, nperseg=nperseg)
    f, psdE = sp.signal.welch(tdi_td["E"], fs=1.0/dt, nperseg=nperseg)
    f, psdT = sp.signal.welch(tdi_td["T"], fs=1.0/dt, nperseg=nperseg)
    return f, psdA, psdE, psdT

def median_windows(y, window_size):
    medians = deepcopy(y)
    for i in range(int(len(y)/window_size*2)-1):
        start_index = int(i/2*window_size)
        end_index = int((i/2+1)*window_size)
        median = np.median(y[start_index:end_index])
        outliers = np.abs(medians[start_index:end_index]) > median*2
        medians[start_index:end_index][outliers] = median
    return medians

def smooth_psd(psd, f):
    smoothed = median_windows(psd, 30)
    smoothed[:40] = psd[:40]
    index_cut = np.searchsorted(f, 0.0008)  # 0.0008 for 1,2 years
    index_cut_lower = np.searchsorted(f, 3*10**-4)
    psd_fit = np.ones_like(smoothed)
    psd_fit_low = sp.signal.savgol_filter(smoothed, 10, 1)
    psd_fit_high = sp.signal.savgol_filter(smoothed, 70, 1) # 70 for 1,2 years
    psd_fit[:index_cut] = psd_fit_low[:index_cut] 
    psd_fit[index_cut:] = psd_fit_high[index_cut:] 
    psd_fit[:index_cut_lower] = smoothed[:index_cut_lower]
    psd_fit_savgol = sp.signal.savgol_filter(psd_fit, 5, 1)
    return psd_fit_savgol, psd_fit, smoothed


def get_psd_estimate(tdi_td, freq_new, dt, time_per_segment=75000):
    nperseg = time_per_segment/dt
    psd = {}

    frequencies, psdA_welch, psdE_welch, psdT_welch = get_noise_from_time_domain(tdi_td, dt=dt, nperseg=nperseg)

    psdA, psdA2, smoothedA = smooth_psd(psdA_welch, frequencies)
    psdE, psdE2, smoothedE = smooth_psd(psdE_welch, frequencies)
    psdT, psdT2, smoothedT = smooth_psd(psdT_welch, frequencies)
    psdA = sp.interpolate.interp1d(frequencies, psdA)(freq_new)
    psdE = sp.interpolate.interp1d(frequencies, psdE)(freq_new)
    psdT = sp.interpolate.interp1d(frequencies, psdT)(freq_new)
    if freq_new[0] == 0:
        psdA[0] = 1
        psdE[0] = 1
    psd['A'] = psdA
    psd['E'] = psdE
    psd['T'] = psdT
    psd['f'] = freq_new
    return psd

