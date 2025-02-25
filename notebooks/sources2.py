import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy
from scipy.optimize import differential_evolution
import numpy as np
import xarray as xr
import time
from copy import deepcopy
import multiprocessing as mp
import pandas as pd
import os
import h5py
import sys
import pickle

from ldc.lisa.noise import get_noise_model
# from ldc.common.series import TimeSeries, window
import ldc.waveform.fastGB as fastGB
from ldc.common.tools import compute_tdi_snr

# customized settings
plot_parameter = {  # 'backend': 'ps',
    "font.family": "serif",
    "font.serif": "times",
    "font.size": 16,
    "axes.labelsize": "medium",
    "axes.titlesize": "medium",
    "legend.fontsize": "medium",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "grid.color": "k",
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "savefig.dpi": 150,
}

# tell matplotlib about your param_plots
rcParams.update(plot_parameter)
# set nice figure sizes
fig_width_pt = 1.5*464.0  # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1.0 / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * ratio  # height in inches
fig_size = [fig_width, fig_height]
fig_size_squared = [fig_width, fig_width]
rcParams.update({"figure.figsize": fig_size})
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def frequency_derivative(f: float, Mc: float) -> float:
    """Computes the frequency derivative based on chirp mass Mc."""
    G, c = 6.674e-11, 3e8
    Mc_s = Mc * 2e30 * G / c**3
    return 96 / (5 * np.pi * Mc_s**2) * (np.pi * Mc_s * f) ** (11 / 3)

def frequency_derivative_tyson_upper(f):
    return 8*10**-7*f**(11/3)

def frequency_derivative_tyson_lower(f):
    return -5*10**-6*f**(13/3)

def get_noise_from_frequency_domain(tdi_fs, num_windows=100):
    """Computes noise from the frequency domain representation.
    tdi_fs: frequency domain representation of the TDI data
    num_windows: number of windows to use for the PSD estimation
    """
    tdi_ts = xr.Dataset({k: tdi_fs[k].ts.ifft() for k in ["X", "Y", "Z"]})
    tdi_ts["A"] = (tdi_ts["Z"] - tdi_ts["X"]) / np.sqrt(2.0)
    tdi_ts["E"] = (tdi_ts["Z"] - 2.0 * tdi_ts["Y"] + tdi_ts["X"]) / np.sqrt(6.0)
    tdi_ts["T"] = (tdi_ts["Z"] + tdi_ts["Y"] + tdi_ts["X"]) / np.sqrt(3.0)
    dt = float(tdi_ts["X"].t[1] - tdi_ts["X"].t[0])
    
    f, psdA = scipy.signal.welch(tdi_ts["A"], fs=1.0/dt, nperseg=len(tdi_ts["X"])/num_windows)
    f, psdE = scipy.signal.welch(tdi_ts["E"], fs=1.0/dt, nperseg=len(tdi_ts["X"])/num_windows)
    f, psdT = scipy.signal.welch(tdi_ts["T"], fs=1.0/dt, nperseg=len(tdi_ts["X"])/num_windows)
    
    return f, psdA, psdE, psdT

def median_windows(y, window_size):
    """Applies a median filter to the data.
    y: data to filter
    window_size: size of the window
    """
    medians = deepcopy(y)
    for i in range(int(len(y)/window_size*2)-1):
        start_index = int(i/2*window_size)
        end_index = int((i/2+1)*window_size)
        median = np.median(y[start_index:end_index])
        outliers = np.abs(medians[start_index:end_index]) > median*2
        medians[start_index:end_index][outliers] = median
    return medians

def smooth_psd(psd, f):
    """Smooths the PSD.
    psd: power spectral density
    f: frequency
    """
    smoothed = median_windows(psd, 20)
    smoothed[:40] = psd[:40]
    index_cut = np.searchsorted(f, 0.0008)
    index_cut_lower = np.searchsorted(f, 3*10**-4)
    psd_fit = np.ones_like(smoothed)
    psd_fit_low = scipy.signal.savgol_filter(smoothed, 10, 1)
    psd_fit_high = scipy.signal.savgol_filter(smoothed, 70, 1)
    psd_fit[:index_cut] = psd_fit_low[:index_cut] 
    psd_fit[index_cut:] = psd_fit_high[index_cut:] 
    psd_fit[:index_cut_lower] = smoothed[:index_cut_lower]
    return psd_fit

def scaletooriginal(previous_max, boundaries, parameters, parameters_log_uniform):
    """Scales the parameters back to their original values.
    previous_max: maximum values of the parameters
    boundaries: boundaries of the parameters
    parameters: list of parameters
    parameters_log_uniform: list of parameters that are log-uniform
    """
    maxpGB = {}
    for parameter in parameters:
        if parameter in ["EclipticLatitude"]:
            maxpGB[parameter] = np.arcsin((previous_max[parameters.index(parameter)] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        elif parameter in ["Inclination"]:
            maxpGB[parameter] = np.arccos((previous_max[parameters.index(parameter)] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        elif parameter in parameters_log_uniform:
            maxpGB[parameter] = 10**((previous_max[parameters.index(parameter)] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        else:
            maxpGB[parameter] = (previous_max[parameters.index(parameter)] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0]
    return maxpGB

def scaletooriginalparameter(previous_max, boundaries, parameters, parameters_log_uniform):
    """Scales the parameters back to their original values.
    previous_max: maximum values of the parameters
    boundaries: boundaries of the parameters
    parameters: list of parameters
    parameters_log_uniform: list of parameters that are log-uniform
    """
    maxpGB = {}
    for parameter in parameters:
        if parameter in ["EclipticLatitude"]:
            maxpGB[parameter] = np.arcsin((previous_max[parameter] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        elif parameter in ["Inclination"]:
            maxpGB[parameter] = np.arccos((previous_max[parameter] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        elif parameter in parameters_log_uniform:
            maxpGB[parameter] = 10**((previous_max[parameter] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        else:
            maxpGB[parameter] = (previous_max[parameter] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0]
    return maxpGB

def scaleto01(previous_max, boundaries, parameters, parameters_log_uniform):
    """Scales the parameters to the range [0, 1].
    previous_max: maximum values of the parameters
    boundaries: boundaries of the parameters
    parameters: list of parameters
    parameters_log_uniform: list of parameters that are log-uniform
    """
    maxpGB = {}
    for parameter in parameters:
        if parameter in ["EclipticLatitude"]:
            maxpGB[parameter] = (np.sin(previous_max[parameter]) - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
        elif parameter in ["Inclination"]:
            maxpGB[parameter] = (np.cos(previous_max[parameter]) - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
        elif parameter in parameters_log_uniform:
            maxpGB[parameter] = (np.log10(previous_max[parameter]) - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
        else:
            maxpGB[parameter] = (previous_max[parameter] - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
    return maxpGB

def reduce_boundaries(maxpGB, boundaries, parameters, ratio=0.1):
    """Reduces the boundaries of the parameters.
    maxpGB: maximum values of the parameters
    boundaries: boundaries of the parameters
    parameters: list of parameters
    ratio: ratio of the boundaries
    """
    boundaries_reduced = deepcopy(boundaries)
    for parameter in parameters:
        length = boundaries[parameter][1] - boundaries[parameter][0]
        if parameter == "EclipticLatitude":
            boundaries_reduced[parameter] = [
                np.sin(maxpGB[parameter]) - length * ratio / 2,
                np.sin(maxpGB[parameter]) + length * ratio / 2,
            ]
        elif parameter == "Inclination":
            boundaries_reduced[parameter] = [
                np.cos(maxpGB[parameter]) - length * ratio / 2,
                np.cos(maxpGB[parameter]) + length * ratio / 2,
            ]
        elif parameter == "Frequency":
            boundaries_reduced[parameter] = [
                maxpGB[parameter] - length * ratio / 2,
                maxpGB[parameter] + length * ratio / 2,
            ]
        elif parameter == "FrequencyDerivative":
            boundaries_reduced[parameter] = [
                maxpGB[parameter] - length * ratio / 2,
                maxpGB[parameter] + length * ratio / 2,
            ]
        elif parameter == "Amplitude":
            boundaries_reduced[parameter] = [
                np.log10(maxpGB[parameter]) - length * ratio / 2,
                np.log10(maxpGB[parameter]) + length * ratio / 2,
            ]
        elif parameter in ["InitialPhase",'Polarization']:
            boundaries_reduced[parameter] = [
                maxpGB[parameter] - length * ratio / 2,
                maxpGB[parameter] + length * ratio / 2,
            ]
        elif parameter == "EclipticLongitude":
            boundaries_reduced[parameter] = [
                maxpGB[parameter] - length * ratio / 2,
                maxpGB[parameter] + length * ratio / 2,
            ]
        if boundaries_reduced[parameter][0] < boundaries[parameter][0]:
            boundaries_reduced[parameter][0] = boundaries[parameter][0]
        if boundaries_reduced[parameter][1] > boundaries[parameter][1]:
            boundaries_reduced[parameter][1] = boundaries[parameter][1]
    return boundaries_reduced
    
def max_signal_bandwidth(frequency, Tobs, chandrasekhar_limit=1.4):
    """Computes the maximum signal bandwidth.
    frequency: frequency of the signal
    Tobs: observation time
    chandrasekhar_limit: Chandrasekhar limit
    """
    M_chirp_upper_boundary = (chandrasekhar_limit**2)**(3/5)/(2*chandrasekhar_limit)**(1/5)
    f_smear = frequency *2* 10**-4
    f_deviation = frequency_derivative(frequency, M_chirp_upper_boundary)*Tobs
    # window_length = np.max([f_smear, f_deviation])
    window_length = f_smear + f_deviation
    window_length += 4/31536000*2
    return window_length

def create_frequency_windows(search_range, Tobs, chandrasekhar_limit=1.4):
    """Creates frequency windows.
    search_range: search range
    Tobs: observation time
    chandrasekhar_limit: Chandrasekhar limit
    """
    frequencies = []
    current_frequency = search_range[0]
    while current_frequency < search_range[1]:
        window_length = max_signal_bandwidth(current_frequency, Tobs, chandrasekhar_limit)
        upper_limit = current_frequency+window_length*2
        frequencies.append([current_frequency, upper_limit])
        current_frequency = deepcopy(upper_limit)
    return frequencies

class Search():
    def __init__(self,tdi_fs,Tobs, lower_frequency, upper_frequency, noise_model =  "SciRDv1", recombination=0.75, dt=None, update_noise=True,
    parameters = [
    "Amplitude",
    "EclipticLatitude",
    "EclipticLongitude",
    "Frequency",
    "FrequencyDerivative",
    "Inclination",
    "InitialPhase",
    "Polarization",
], parameters_log_uniform = ['Amplitude']):
        '''
        Search classs for the search of a single source in the LISA data
        tdi_f t with the TDI data
        Tobs: observation time
        lower_frequency: lower frequency of the search
        upper_frequency: upper frequency of the search
        noise_model: noise model for the data

        '''
        self.parameters = parameters
        self.parameters_no_amplitude = deepcopy(parameters)
        self.parameters_no_amplitude.remove('Amplitude')
        self.intrinsic_parameters = ['EclipticLatitude','EclipticLongitude','Frequency', 'FrequencyDerivative']
        self.parameters_log_uniform = parameters_log_uniform
        self.N_samples = (len(tdi_fs['X'].f)-1)*2
        self.tdi_fs = tdi_fs
        if dt is None:
            dt =   Tobs/self.N_samples # 0 and f_Nyquist are both included
        self.dt = dt
        self.fs = 1/dt
        self.GB = fastGB.FastGB(delta_t=dt, T=Tobs)
        self.reduced_frequency_boundaries = None
        self.recombination = recombination
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        chandrasekhar_limit = 1.4
        M_chirp_upper_boundary = (chandrasekhar_limit**2)**(3/5)/(2*chandrasekhar_limit)**(1/5)
        self.padding = max_signal_bandwidth(lower_frequency, Tobs, chandrasekhar_limit)/2
        self.frequency_T_threshold = 19.1*10**-3/2
        
        self.use_T_component = False
        if self.upper_frequency + self.padding > self.frequency_T_threshold:
            self.use_T_component = True

        frequencyrange =  [lower_frequency-self.padding, upper_frequency+self.padding]
        self.indexes = np.logical_and(tdi_fs['X'].f > frequencyrange[0], tdi_fs['X'].f < frequencyrange[1]) 
        self.dataX = tdi_fs["X"][self.indexes]
        self.dataY = tdi_fs["Y"][self.indexes]
        self.dataZ = tdi_fs["Z"][self.indexes]

        self.DAf = (self.dataZ - self.dataX)/np.sqrt(2.0)
        self.DEf = (self.dataZ - 2.0*self.dataY + self.dataX)/np.sqrt(6.0)
        self.DTf = (self.dataZ + self.dataY + self.dataX)/np.sqrt(3.0)

        fmin, fmax = float(self.dataX.f[0]), float(self.dataX.f[-1] + self.dataX.attrs["df"])
        freq = np.array(self.dataX.sel(f=slice(fmin, fmax)).f)
        Nmodel = get_noise_model(noise_model, freq)
        self.Sn = Nmodel.psd(freq=freq, option="X")
        self.SA = Nmodel.psd(freq=freq, option="A")
        self.SE = Nmodel.psd(freq=freq, option="E")
        self.ST = Nmodel.psd(freq=freq, option="T")
        self.SE_theory = Nmodel.psd(freq=freq, option="E")
        if update_noise:
            self.update_noise()

        f_0 = fmin
        f_transfer = 19.1*10**-3
        snr = 7
        amplitude_lower = 2*snr/(Tobs * np.sin(f_0/ f_transfer)**2/self.SA[0])**0.5
        snr = 1000
        amplitude_upper = 2*snr/(Tobs * np.sin(f_0/ f_transfer)**2/self.SA[0])**0.5
        amplitude = [amplitude_lower, amplitude_upper]
        fd_range = [frequency_derivative_tyson_lower(lower_frequency),frequency_derivative(upper_frequency,M_chirp_upper_boundary)]
        self.boundaries = {
            "Amplitude": [np.log10(amplitude[0]),np.log10(amplitude[1])],
            "EclipticLatitude": [-1.0, 1.0],
            "EclipticLongitude": [-np.pi, np.pi],
            "Frequency": frequencyrange,
            "FrequencyDerivative": fd_range,
            "Inclination": [-1.0, 1.0],
            "InitialPhase": [0.0, 2.0 * np.pi],
            "Polarization": [0.0, 1.0 * np.pi],
        }

        if self.boundaries['FrequencyDerivative'][0] > self.boundaries['FrequencyDerivative'][1]:
            c = self.boundaries['FrequencyDerivative'][0]
            self.boundaries['FrequencyDerivative'][0] = self.boundaries['FrequencyDerivative'][1]
            self.boundaries['FrequencyDerivative'][1] = c
        
        previous_max = np.random.rand(8)
        i = 0
        self.pGBs = {}
        for parameter in parameters:
            # if parameter in ["FrequencyDerivative"]:
            #     i -= 1
            if parameter in ["EclipticLatitude"]:
                self.pGBs[parameter] = np.arcsin((previous_max[i] * (self.boundaries[parameter][1] - self.boundaries[parameter][0])) + self.boundaries[parameter][0])
            elif parameter in ["Inclination"]:
                self.pGBs[parameter] = np.arccos((previous_max[i] * (self.boundaries[parameter][1] - self.boundaries[parameter][0])) + self.boundaries[parameter][0])
            elif parameter in parameters_log_uniform:
                self.pGBs[parameter] = 10**((previous_max[i] * (self.boundaries[parameter][1] - self.boundaries[parameter][0])) + self.boundaries[parameter][0])
            else:
                self.pGBs[parameter] = (previous_max[i] * (self.boundaries[parameter][1] - self.boundaries[parameter][0])) + self.boundaries[parameter][0]
            i += 1

    def update_noise(self, pGB=None):
        """Updates the noise model.
        pGB: gravitational wave parameters
        """ 
        if pGB != None:
            freqs = None
            if self.GB.T < 365.26*24*3600/4:
                freqs = np.linspace(self.lower_frequency, self.upper_frequency, 128)
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGB, oversample=4, freqs=freqs)
            Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
            Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
            Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]

            Af = (Zs_total - Xs_total)/np.sqrt(2.0)
            Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
            Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
        else:
            Af = 0
            Ef = 0
            Tf = 0
        
        self.SA = np.ones_like(self.SA)*np.median((np.abs(self.DAf-Af).data/self.dt)**2/(self.fs*self.N_samples)*2)
        self.SE = np.ones_like(self.SE)*np.median((np.abs(self.DEf-Ef).data/self.dt)**2/(self.fs*self.N_samples)*2)
        self.ST = np.ones_like(self.ST)*np.median((np.abs(self.DTf-Tf).data/self.dt)**2/(self.fs*self.N_samples)*2)

    def scalarproduct(self, a, b):
        """Computes the scalar product of two signals.
        a: signal 1
        b: signal 2
        """
        diff = np.real(a[0].values * np.conjugate(b[0].values)) ** 2 + np.real(a[1].values * np.conjugate(b[1].values)) ** 2 + np.real(a[2].values * np.conjugate(b[2].values)) ** 2
        res = 4*float(np.sum(diff / self.Sn) * self.dataX.df)
        return res

    def SNR(self, pGBs):
        """Computes the signal-to-noise ratio.
        pGBs: list of gravitational wave parameters
        """
        for i in range(len(pGBs)):
            freqs = None
            if self.GB.T < 365.26*24*3600/4:
                freqs = np.linspace(self.lower_frequency, self.upper_frequency, 128)
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4, freqs=freqs)#, tdi2=self.tdi2)
            # Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4)
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        if self.use_T_component:
            Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/self.SA + np.absolute(Tf.data)**2 /self.ST)
            SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA + np.real(self.DTf * np.conjugate(Tf.data))/self.ST )
        else:
            SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA )
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
        SNR = 4.0*Xs.df* hh
        SNR2 = 4.0*Xs.df* SNR2
        SNR3 = SNR2 / np.sqrt(SNR)
        return SNR3.values


    def loglikelihood(self, pGBs):
        """Computes the log-likelihood.
        pGBs: list of gravitational wave parameters
        """
        for i in range(len(pGBs)):
            freqs = None
            if self.GB.T < 365.26*24*3600/4:
                freqs = np.linspace(self.lower_frequency, self.upper_frequency, 128)
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4, freqs=freqs)
            # Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4)
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
        if self.use_T_component:
            Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/self.SA + np.absolute(Tf.data)**2 /self.ST)
            SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA + np.real(self.DTf * np.conjugate(Tf.data))/self.ST )
        else:
            SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA )
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
        # dd = np.sum((np.absolute(self.DAf.data)**2 + np.absolute(self.DEf.data)**2) /self.SA)
        plotIt = False
        if plotIt:
            fig, ax = plt.subplots(nrows=2, sharex=True) 
            ax[0].plot(Af.f, np.abs(self.DAf))
            ax[0].plot(Af.f, np.abs(Af.data))
            
            ax[1].plot(Af.f, np.abs(self.DEf))
            ax[1].plot(Af.f, np.abs(Ef.data))
            plt.show()
        logliks = 4.0*Xs.df*( SNR2 - 0.5 * hh )
        return logliks.values

    def intrinsic_SNR(self, pGBs):
        """Computes the intrinsic signal-to-noise ratio.
        pGBs: list of gravitational wave parameters
        """
        for i in range(len(pGBs)):
            freqs = None
            if self.GB.T < 365.26*24*3600/4:
                freqs = np.linspace(self.lower_frequency, self.upper_frequency, 128)
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4, freqs=freqs)
            # Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4)
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        if self.use_T_component:
            Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/self.SA + np.absolute(Tf.data)**2 /self.ST)
        else:
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
        SNR = 4.0*Xs.df* hh
        return np.sqrt(SNR)



    def differential_evolution_search(self, frequency_boundaries, initial_guess = None, number_of_signals = 1):
        """Performs a differential evolution search.
        frequency_boundaries: boundaries of the frequency
        initial_guess: initial guess of the parameters
        number_of_signals: number of signals
        """
        bounds = []
        for signal in range(number_of_signals):
            for i in range(7):
                bounds.append((0,1))

        maxpGB = []
        self.boundaries_reduced = deepcopy(self.boundaries)
        self.boundaries_reduced['Frequency'] = frequency_boundaries
        if initial_guess != None:
            initial_guess01 = np.zeros((len(self.parameters)-1)*number_of_signals)
            for signal in range(number_of_signals):
                pGBstart01 = scaleto01(initial_guess[signal], self.boundaries_reduced, self.parameters, self.parameters_log_uniform)

                for count, parameter in enumerate(self.parameters_no_amplitude):
                    if pGBstart01[parameter] < 0:
                        pGBstart01[parameter] = 0
                    if pGBstart01[parameter] > 1:
                        pGBstart01[parameter] = 1
                    initial_guess01[count+(len(self.parameters_no_amplitude))*signal] = pGBstart01[parameter]
            start = time.time()
            res = differential_evolution(self.from01tologlikelihood_without_amplitude, bounds=bounds, disp=False, strategy='best1exp', popsize=8,tol= 1e-8 , maxiter=1000, recombination= self.recombination, mutation=(0.5,1), x0=initial_guess01)
            print('time',time.time()-start)
        else:
            start = time.time()
            res = differential_evolution(self.from01tologlikelihood_without_amplitude, bounds=bounds, disp=False, strategy='best1exp', popsize=8, tol= 1e-8 , maxiter=1000, recombination= self.recombination, mutation=(0.5,1))
            print('time',time.time()-start)
        for signal in range(number_of_signals):
            pGB01 = [0.5] + res.x[signal*7:signal*7+7].tolist()
            maxpGB.append(scaletooriginal(pGB01,self.boundaries_reduced, self.parameters, self.parameters_log_uniform))
        print(res)
        print(maxpGB)
        # print('log-likelihood',self.loglikelihood(maxpGB))
        # print(pGB)
        return [maxpGB], res.nfev

    def optimize(self, pGBmodes, boundaries = None):
        """
        This function optimizes the parameters of the pGBs. It uses the scipy.optimize.minimize function with the SLSQP method.
        pgBmodes: list of dictionaries with the parameters of the pGBs
        boundaries: dictionary with the boundaries of the parameters
        return: list of dictionaries with the optimized parameters of the pGBs
        """
        if boundaries == None:
            boundaries = self.boundaries
        bounds = ()
        number_of_signals_optimize = len(pGBmodes[0])
        for signal in range(number_of_signals_optimize):
            bounds += ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
        for i in range(len(pGBmodes)):
            maxpGB = []
            boundaries_reduced = []
            pGBs01 = []

            # print('initial loglikelihood noise matrix', self.loglikelihood_noise_matrix(pGBmodes[0]),pGBmodes[0][0]['Frequency'])
            # print('initial loglikelihood', self.loglikelihood(pGBmodes[0]),pGBmodes[0][0]['Frequency'])
            for j in range(2):
                x = []
                for signal in range(number_of_signals_optimize):
                    if j == 0:
                        maxpGB.append({})
                        boundaries_reduced.append({})
                        for parameter in self.parameters:
                            maxpGB[signal][parameter] = pGBmodes[i][signal][parameter]
                    # print(maxpGB)
                    # boundaries_reduced[signal] = reduce_boundaries(maxpGB[signal], boundaries,ratio=0.1)
                    if j > 0:
                        boundaries_reduced[signal] = reduce_boundaries(maxpGB[signal], boundaries,ratio=0.4, parameters=self.parameters)
                    if j in [0]:
                        boundaries_reduced[signal] = deepcopy(boundaries)
                    pGBs01.append({})
                    for parameter in self.parameters:
                        if parameter in ["EclipticLatitude"]:
                            pGBs01[signal][parameter] = (np.sin(maxpGB[signal][parameter]) - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                        elif parameter in ["Inclination"]:
                            pGBs01[signal][parameter] = (np.cos(maxpGB[signal][parameter]) - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                        elif parameter in self.parameters_log_uniform:
                            pGBs01[signal][parameter] = (np.log10(maxpGB[signal][parameter]) - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                        else:
                            pGBs01[signal][parameter] = (maxpGB[signal][parameter] - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                    for parameter in self.parameters:
                        x.append(pGBs01[signal][parameter])
                # print(loglikelihood(maxpGB))
                res = scipy.optimize.minimize(self.from01tologlikelihood, x, args=boundaries_reduced, method='SLSQP', bounds=bounds, tol=1e-5)#, options= {'maxiter':100})
                # res = scipy.optimize.minimize(self.from01tologlikelihood, x, args=boundaries_reduced, method='Nelder-Mead', tol=1e-6)
                # res = scipy.optimize.least_squares(self.from01tologlikelihood, x, args=boundaries_reduced, bounds=bounds)
                for signal in range(number_of_signals_optimize):
                    maxpGB[signal] = scaletooriginal(res.x[signal*8:signal*8+8],boundaries_reduced[signal], self.parameters, self.parameters_log_uniform)
                # print('optimized loglikelihood', loglikelihood(maxpGB),maxpGB)
                # print('boundaries reduced', boundaries_reduced)

            best_value = self.loglikelihood(maxpGB) 
            if i == 0:
                current_best_value = best_value
                current_maxpGB = maxpGB
            try:
                if current_best_value < best_value:
                    current_best_value = best_value
                    current_maxpGB = maxpGB
            except:
                pass
            # print('optimized loglikelihood', self.loglikelihood(maxpGB),self.loglikelihood([self.pGB]))
        maxpGB = current_maxpGB
        # print('final optimized loglikelihood noise matrix', self.loglikelihood_noise_matrix(maxpGB),maxpGB[0]['Frequency'])
        # print('final optimized loglikelihood', self.loglikelihood(maxpGB),maxpGB[0]['Frequency'])
        return maxpGB

    def optimize_without_amplitude(self, pGBmodes, boundaries = None):
        """
        This function optimizes the parameters of the pGBs without the amplitude. It uses the scipy.optimize.minimize function with the SLSQP method.
        pgBmodes: list of dictionaries with the parameters of the pGBs
        boundaries: dictionary with the boundaries of the parameters
        return: list of dictionaries with the optimized parameters of the pGBs
        """
        if boundaries == None:
            boundaries = self.boundaries
        bounds = ()
        number_of_signals_optimize = len(pGBmodes[0])
        for signal in range(number_of_signals_optimize):
            bounds += ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
        for i in range(len(pGBmodes)):
            maxpGB = []
            pGBs01 = []

            for j in range(2):
                x = []
                for signal in range(number_of_signals_optimize):
                    if j == 0:
                        maxpGB.append({})
                        self.boundaries_reduced = {}
                        for parameter in self.parameters_no_amplitude:
                            maxpGB[signal][parameter] = pGBmodes[i][signal][parameter]
                    # print(maxpGB)
                    self.boundaries_reduced = reduce_boundaries(maxpGB[signal], boundaries,ratio=0.1, parameters=self.parameters)
                    if j == 2:
                        self.boundaries_reduced = reduce_boundaries(maxpGB[signal], boundaries,ratio=0.1, parameters=self.parameters)
                    if j in [0,1]:
                        self.boundaries_reduced = deepcopy(boundaries)
                    pGBs01.append({})
                    for parameter in self.parameters_no_amplitude:
                        if parameter in ["EclipticLatitude"]:
                            pGBs01[signal][parameter] = (np.sin(maxpGB[signal][parameter]) - self.boundaries_reduced[parameter][0]) / (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])
                        elif parameter in ["Inclination"]:
                            pGBs01[signal][parameter] = (np.cos(maxpGB[signal][parameter]) - self.boundaries_reduced[parameter][0]) / (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])
                        elif parameter in self.parameters_log_uniform:
                            pGBs01[signal][parameter] = (np.log10(maxpGB[signal][parameter]) - self.boundaries_reduced[parameter][0]) / (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])
                        else:
                            pGBs01[signal][parameter] = (maxpGB[signal][parameter] - self.boundaries_reduced[parameter][0]) / (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])
                    for parameter in self.parameters_no_amplitude:
                        x.append(pGBs01[signal][parameter])
                # print(loglikelihood(maxpGB))
                res = scipy.optimize.minimize(self.from01tologlikelihood_without_amplitude, x, method='SLSQP', bounds=bounds, tol=1e-10)
                # res = scipy.optimize.minimize(self.from01tologlikelihood, x, args=self.boundaries_reduced, method='Nelder-Mead', tol=1e-10)
                # res = scipy.optimize.least_squares(self.from01tologlikelihood, x, args=self.boundaries_reduced, bounds=bounds)
                for signal in range(number_of_signals_optimize):
                    pGB01 = [0.5] + res.x[signal*7:signal*7+7].tolist()
                    maxpGB[signal] = scaletooriginal(pGB01,self.boundaries_reduced, self.parameters, self.parameters_log_uniform)
                    # maxpGB[signal] = scaletooriginal(res.x[signal*7:signal*7+7],self.boundaries_reduced)
                # print('optimized loglikelihood', loglikelihood(maxpGB),maxpGB)
                # print('boundaries reduced', self.boundaries_reduced)

            best_value = self.loglikelihood(maxpGB)
            if i == 0:
                current_best_value = best_value
                current_maxpGB = maxpGB
            try:
                if current_best_value < best_value:
                    current_best_value = best_value
                    current_maxpGB = maxpGB
            except:
                pass
            # print('optimized loglikelihood', self.loglikelihood(maxpGB),self.loglikelihood([self.pGB]))
        maxpGB = current_maxpGB
        print('final optimized loglikelihood', self.loglikelihood(maxpGB),maxpGB[0]['Frequency'])
        return maxpGB

    def calculate_Amplitude(self, pGBs):
        """Calculates the amplitude of the signal analyticaly based on the parameters pGBs.
        pGBs: list of dictionaries with the parameters of the pGBs
        """
        for i in range(len(pGBs)):
            freqs = None
            if self.GB.T < 365.26*24*3600/4:
                freqs = np.linspace(self.lower_frequency, self.upper_frequency, 128)
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4, freqs=freqs)
            # Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4)
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA )
        hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
        logliks = 4.0*Xs.df*( SNR2 - 0.5 * hh )
        scalar_product_hh = 4.0*Xs.df* hh
        scalar_product_dh = 4.0*Xs.df* SNR2
        A = scalar_product_dh / scalar_product_hh
        return A


    def optimizeA(self, pGBmodes, boundaries = None):
        """
        This function optimizes the amplitude of the pGBs. It uses the scipy.optimize.minimize function with the trust-constr method.
        pgBmodes: list of dictionaries with the parameters of the pGBs
        boundaries: dictionary with the boundaries of the parameters
        return: list of dictionaries with the optimized parameters of the pGBs
        """
        if boundaries == None:
            boundaries = self.boundaries
        bounds = ()
        number_of_signals_optimize = len(pGBmodes[0])
        for signal in range(number_of_signals_optimize):
            bounds += ((0,1))
        for i in range(len(pGBmodes)):
            maxpGB = []
            boundaries_reduced = []
            pGBs01 = []

            for j in range(2):
                x = []
                pGBx = []
                for signal in range(number_of_signals_optimize):
                    if j == 0:
                        maxpGB.append({})
                        boundaries_reduced.append({})
                        for parameter in self.parameters:
                            maxpGB[signal][parameter] = pGBmodes[i][signal][parameter]
                    # print(maxpGB)
                    boundaries_reduced[signal] = reduce_boundaries(maxpGB[signal], boundaries,ratio=0.1, parameters=self.parameters)
                    if j == 2:
                        boundaries_reduced[signal] = reduce_boundaries(maxpGB[signal], boundaries,ratio=0.1, parameters=self.parameters)
                    if j in [0,1]:
                        boundaries_reduced[signal] = deepcopy(boundaries)
                    pGBs01.append({})
                    for parameter in self.parameters:
                        if parameter in ["EclipticLatitude"]:
                            pGBs01[signal][parameter] = (np.sin(maxpGB[signal][parameter]) - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                        elif parameter in ["Inclination"]:
                            pGBs01[signal][parameter] = (np.cos(maxpGB[signal][parameter]) - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                        elif parameter in self.parameters_log_uniform:
                            pGBs01[signal][parameter] = (np.log10(maxpGB[signal][parameter]) - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                        else:
                            pGBs01[signal][parameter] = (maxpGB[signal][parameter] - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                    for parameter in ['Amplitude']:
                        x.append(pGBs01[signal][parameter])
                    for parameter in self.parameters:
                        pGBx.append(pGBs01[signal][parameter])
                self.pGBx = pGBx
                res = scipy.optimize.minimize(self.from01tologlikelihooda, x, args=(pGBx, boundaries_reduced), method='trust-constr', bounds=[bounds], tol=1e-1)
                # res = scipy.optimize.minimize(self.from01tologlikelihood, x, args=boundaries_reduced, method='Nelder-Mead', tol=1e-10)
                # res = scipy.optimize.least_squares(self.from01tologlikelihood, x, args=boundaries_reduced, bounds=bounds)
                for signal in range(number_of_signals_optimize):
                    pGB01 = deepcopy(pGBx)
                    pGB01[signal*8] = res.x[signal]
                    maxpGB[signal] = scaletooriginal(pGB01,boundaries_reduced[signal], self.parameters, self.parameters_log_uniform)
                # print('optimized loglikelihood', loglikelihood(maxpGB),maxpGB)
                # print('boundaries reduced', boundaries_reduced)

            best_value = self.loglikelihood(maxpGB)
            if i == 0:
                current_best_value = best_value
                current_maxpGB = maxpGB
            try:
                if current_best_value < best_value:
                    current_best_value = best_value
                    current_maxpGB = maxpGB
            except:
                pass
            # print('optimized loglikelihood', self.loglikelihood(maxpGB),self.loglikelihood([self.pGB]))
        maxpGB = current_maxpGB
        print('final optimized loglikelihood', self.loglikelihood(maxpGB),maxpGB[0]['Frequency'])
        return maxpGB

    def from01tologlikelihood(self, pGBs01, boundaries_reduced):
        """Computes the log-likelihood from the parameters in range [0,1]
        pGBs01: list of gravitational wave parameters in range [0,1]
        boundaries_reduced: boundaries of the parameters
        """
        pGBs = []
        for signal in range(int(len(pGBs01)/8)):
            pGBs.append({})
            i = 0
            for parameter in self.parameters:
                if parameter in ["EclipticLatitude"]:
                    pGBs[signal][parameter] = np.arcsin((pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                elif parameter in ["Inclination"]:
                    shifted_inclination = pGBs01[signal*8:signal*8+8][i]
                    if pGBs01[signal*8:signal*8+8][i] < 0:
                        shifted_inclination = pGBs01[signal*8:signal*8+8][i] + 1
                    if pGBs01[signal*8:signal*8+8][i] > 1:
                        shifted_inclination = pGBs01[signal*8:signal*8+8][i] - 1
                    pGBs[signal][parameter] = np.arccos((shifted_inclination * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                elif parameter in self.parameters_log_uniform:
                    pGBs[signal][parameter] = 10**((pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                else:
                    pGBs[signal][parameter] = (pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0]
                i += 1
        p = -self.loglikelihood(pGBs)
        return p#/10**4

    def from01tologlikelihood_without_amplitude(self, pGBs01, number_of_signals = 1):
        """Computes the log-likelihood from the parameters in range [0,1] without the amplitude
        pGBs01: list of gravitational wave parameters in range [0,1] without the amplitude
        """
        pGBs = []
        # pGBs01 = np.nan_to_num(pGBs01, nan=0.5)
        for signal in range(number_of_signals):
            pGBs.append({})
            i = 0
            for parameter in self.parameters:
                if parameter in ["EclipticLatitude"]:
                    pGBs[signal][parameter] = np.arcsin((pGBs01[signal*7:signal*7+7][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                elif parameter in ["Inclination"]:
                    pGBs[signal][parameter] = np.arccos((pGBs01[signal*7:signal*7+7][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                elif parameter in ['Amplitude']:
                    i -= 1
                    pGBs[signal][parameter] = 10**((0.1 * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                # elif parameter in ["FrequencyDerivative"]:
                #     pGBs[signal][parameter] = 10**((pGBs01[signal*7:signal*7+7][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                else:
                    pGBs[signal][parameter] = (pGBs01[signal*7:signal*7+7][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0]
                i += 1
        self.pGBs01 = pGBs01
        p = -self.SNR(pGBs)
        # if np.isnan(p):
        #     p = -np.inf
        return p
