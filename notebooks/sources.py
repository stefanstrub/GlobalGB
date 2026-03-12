import matplotlib.pyplot as plt
from matplotlib import markers
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
from ldc.common.series import TimeSeries
import ldc.waveform.fastGB as fastGB

try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

# customized settings
plot_parameter = {  # 'backend': 'ps',
    "font.family": "DeJavu Serif",
    "font.serif": "Times",
    "mathtext.fontset": "cm",
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
fig_size = [fig_width*2, fig_height]
fig_size_squared = [fig_width, fig_width]
rcParams.update({"figure.figsize": fig_size})
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def frequency_derivative(f, Mc):
    G = 6.674*10**(-11)
    c = 3*10**8
    Mc = Mc * 2*10**30
    Mc_s = Mc*G/c**3
    return 96/(5*np.pi*Mc_s**2)*(np.pi*Mc_s*f)**(11/3)
def frequency_derivative_tyson(f):
    return 8*10**-7*f**(11/3)
def frequency_derivative_tyson_lower(f):
    return -5*10**-6*f**(13/3)

def get_noise_from_frequency_domain(tdi_fs, number_of_windows=100):
    tdi_ts = xr.Dataset(dict([(k, tdi_fs[k].ts.ifft()) for k in ["X", "Y", "Z"]]))
    tdi_ts["E"] = deepcopy(tdi_ts["X"])
    tdi_ts["A"] = (tdi_ts["Z"] - tdi_ts["X"])/np.sqrt(2.0)
    tdi_ts["E"].values = (tdi_ts["Z"] - 2.0*tdi_ts["Y"] + tdi_ts["X"])/np.sqrt(6.0)
    tdi_ts["T"] = (tdi_ts["Z"] + tdi_ts["Y"] + tdi_ts["X"])/np.sqrt(3.0)
    dt = float(tdi_ts["X"].t[1]-tdi_ts["X"].t[0])
    f, psdA =  scipy.signal.welch(tdi_ts["A"], fs=1.0/dt, nperseg=len(tdi_ts["X"])/number_of_windows)#, average='mean', window= 'boxcar')
    f, psdE =  scipy.signal.welch(tdi_ts["E"], fs=1.0/dt, nperseg=len(tdi_ts["X"])/number_of_windows)#, average='mean', window= 'boxcar')
    f, psdT =  scipy.signal.welch(tdi_ts["T"], fs=1.0/dt, nperseg=len(tdi_ts["X"])/number_of_windows)#, average='mean', window= 'boxcar')
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


def scaletooriginal(previous_max, boundaries, parameters, parameters_log_uniform):
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

def scaletooriginal_array(previous_pGB, boundaries, parameters, parameters_log_uniform):
    start = time.time()
    no_array = False
    if len(np.shape(previous_pGB)) == 1:
        no_array = True
        previous_pGB = np.array([previous_pGB])
    original_pGB = np.zeros(np.shape(previous_pGB))
    i = 0
    for parameter in parameters:
        if parameter in ["EclipticLatitude"]:
            original_pGB[:,i] = np.arcsin((previous_pGB[:,parameters.index(parameter)] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        elif parameter in ["Inclination"]:
            original_pGB[:,i] = np.arccos((previous_pGB[:,parameters.index(parameter)] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        elif parameter in parameters_log_uniform:
            original_pGB[:,i] = 10**((previous_pGB[:,parameters.index(parameter)] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        else:
            original_pGB[:,i] = (previous_pGB[:,parameters.index(parameter)] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0]
        i += 1
    if no_array:
        original_pGB = original_pGB[0]
    # print('time rescale', time.time()-start)
    return original_pGB

def scaletooriginalparameter(previous_max, boundaries, parameters, parameters_log_uniform):
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

def CoordinateMC(n, pGBs, boundaries, parameters_recorded, loglikelihood, n_trials=50, number_of_signals =1):
    maxpGB = []
    parameters_recorded1 = []
    no_improvement_counter = 0
    for i in range(number_of_signals):
        maxpGB.append(deepcopy(pGBs))
        parameters_recorded1.append({})
        for parameter in parameters_no_amplitude:
            parametervalue = np.random.uniform(boundaries[parameter][0], boundaries[parameter][1])
            maxpGB[i][parameter] = parametervalue
            if parameter in ["EclipticLatitude"]:
                maxpGB[i][parameter] = np.arcsin(parametervalue)
            elif parameter in ["Inclination"]:
                maxpGB[i][parameter] = np.arccos(parametervalue)
            elif parameter in ['Amplitude',"FrequencyDerivative"]:
                maxpGB[i][parameter] = 10**(parametervalue)
            parameters_recorded1[i][parameter] = []
            parameters_recorded1[i][parameter].append(maxpGB[i][parameter])
        parameters_recorded1[i]["Loglikelihood"] = []
    # if n == 0:
    #     maxpGB = deepcopy(pGBs)
    try:
        best_value
    except:
        best_value = loglikelihood(maxpGB)
    previous_best = deepcopy(best_value)
    maxpGB2 = deepcopy(maxpGB)
    for i in range(100):
        # if i > 48:
        #     n_trials = 50
        parameter1 = parameters_no_amplitude[i % 7]
        parameter2 = parameters_no_amplitude[np.random.randint(0, 6)]
        parameter3 = parameters_no_amplitude[np.random.randint(0, 6)]
        # parameter2 = 'InitialPhase'
        # parameter1 = 'Inclination'
        while parameter2 == parameter1:
            parameter2 = parameters_no_amplitude[np.random.randint(0, 6)]
        while parameter3 == parameter1 or parameter3 == parameter2:
            parameter3 = parameters_no_amplitude[np.random.randint(0, 6)]
        # if parameter1 == 'Frequency':
        #     parameter2 = 'Polarization'
        changeableparameters = [parameter1, parameter2]
        signal = i % number_of_signals
        for j in range(n_trials):
            k = 0
            for parameter in changeableparameters:
                parametervalue = np.random.uniform(boundaries[parameter][0], boundaries[parameter][1],1)[0]
                # if k == 0:
                #     parametervalue = np.random.uniform(j%number_per_row * 1/number_per_row, (j%number_per_row+1)*1/number_per_row)
                # else:
                #     parametervalue = np.random.uniform(int(j/number_per_row) * 1/number_per_row, (int(j/number_per_row)+1)*1/number_per_row)
                # parametervalue = parametervalue * (boundaries[parameter][1] - boundaries[parameter][0]) + boundaries[parameter][0]
                maxpGB2[signal][parameter] = parametervalue
                if parameter in ["EclipticLatitude"]:
                    maxpGB2[signal][parameter] = np.arcsin(parametervalue)
                elif parameter in ["Inclination"]:
                    maxpGB2[signal][parameter] = np.arccos(parametervalue)
                elif parameter in ['Amplitude',"FrequencyDerivative"]:
                    maxpGB2[signal][parameter] = 10**(parametervalue)
                k += 1
            suggestion = loglikelihood(maxpGB2)
            if suggestion > previous_best:
                no_improvement_counter = 0
                previous_best = suggestion
                for parameter in changeableparameters:
                    maxpGB[signal][parameter] = maxpGB2[signal][parameter]
        if no_improvement_counter != 0:
            no_improvement_counter += 1
        if no_improvement_counter > 16:
            past_mean = 0
            sum_count = 0
            for l in range(n):
                try:
                    past_mean += parameters_recorded[l]["Loglikelihood"][i]
                    sum_count += 1
                except:
                    pass
            try:
                past_mean = past_mean / sum_count
                if previous_best > past_mean:
                    no_improvement_counter = 0
                else:
                    # print("no improvement")
                    break
            except:
                # no_improvement_counter = 0
                break

        if i in [30,40,50,60,70]:
            past_mean = 0
            sum_count = 0
            for l in range(n):
                try:
                    past_mean += parameters_recorded[l][0]["Loglikelihood"][i]
                    sum_count += 1
                except:
                    pass
            try:
                past_mean = past_mean / sum_count
                if previous_best > past_mean:
                    pass
                else:
                    break
            except:
                pass

        # start = time.time()
        # print(n, i, previous_best, loglikelihood(maxpGB), maxpGB)
        parameters_recorded1[0]["Loglikelihood"].append(loglikelihood(maxpGB))
        for i in range(number_of_signals):
            for parameter in parameters_no_amplitude:
                parameters_recorded1[i][parameter].append(maxpGB[i][parameter])

        maxpGB2 = deepcopy(maxpGB)
    if previous_best > best_value:
        best_value = previous_best
    parameters_recorded[n] = parameters_recorded1
    return parameters_recorded1

class Distribution(object):
    """
    draws samples from a one dimensional probability distribution,
    by means of inversion of a discrete inverstion of a cumulative density function

    the pdf can be sorted first to prevent numerical error in the cumulative sum
    this is set as default; for big density functions with high contrast,
    it is absolutely necessary, and for small density functions,
    the overhead is minimal

    a call to this distibution object returns indices into density array
    """
    def __init__(self, pdf, sort = True, interpolation = True, transform = lambda x: x):
        self.shape          = pdf.shape
        self.pdf            = pdf.ravel()
        self.sort           = sort
        self.interpolation  = interpolation
        self.transform      = transform

        #a pdf can not be negative
        assert(np.all(pdf>=0))

        #sort the pdf by magnitude
        if self.sort:
            self.sortindex = np.argsort(self.pdf, axis=None)
            self.pdf = self.pdf[self.sortindex]
        #construct the cumulative distribution function
        self.cdf = np.cumsum(self.pdf)
    @property
    def ndim(self):
        return len(self.shape)
    @property
    def sum(self):
        """cached sum of all pdf values; the pdf need not sum to one, and is imlpicitly normalized"""
        return self.cdf[-1]
    def __call__(self, N):
        """draw """
        #pick numbers which are uniformly random over the cumulative distribution function
        choice = np.random.uniform(high = self.sum, size = N)
        #find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        pdfs = self.pdf[index]
        #if necessary, map the indices back to their original ordering
        if self.sort:
            index = self.sortindex[index]
        #map back to multi-dimensional indexing
        index = np.unravel_index(index, self.shape)
        index = np.vstack(index)
        #is this a discrete or piecewise continuous distribution?
        if self.interpolation:
            index = index + np.random.uniform(size=index.shape)
        return self.transform(index), pdfs
    
def max_signal_bandwidth(frequency, Tobs, chandrasekhar_limit=1.4):
    M_chirp_upper_boundary = (chandrasekhar_limit**2)**(3/5)/(2*chandrasekhar_limit)**(1/5)
    f_smear = frequency *2* 10**-4
    f_deviation = frequency_derivative(frequency, M_chirp_upper_boundary)*Tobs
    # window_length = np.max([f_smear, f_deviation])
    window_length = f_smear + f_deviation
    window_length += 4/31536000*2
    return window_length

def create_frequency_windows(search_range, Tobs, chandrasekhar_limit=1.4):
    frequencies = []
    current_frequency = search_range[0]
    while current_frequency < search_range[1]:
        window_length = max_signal_bandwidth(current_frequency, Tobs, chandrasekhar_limit)
        upper_limit = current_frequency+window_length*2
        frequencies.append([current_frequency, upper_limit])
        current_frequency = deepcopy(upper_limit)
    return frequencies
    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Search():
    def __init__(self,tdi_fs,Tobs, lower_frequency, upper_frequency, noise_model =  "SciRDv1", recombination=0.75, dt=None, update_noise=True, noise=None, gb_gpu=None, use_gpu=False, tdi2=False, t_start=0,
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
        self.parameters = parameters
        self.parameters_no_amplitude = deepcopy(parameters)
        self.parameters_no_amplitude.remove('Amplitude')
        self.intrinsic_parameters = ['EclipticLatitude','EclipticLongitude','Frequency', 'FrequencyDerivative']
        self.parameters_log_uniform = parameters_log_uniform
        self.N_samples = (len(tdi_fs['X'].f)-1)*2
        self.tdi_fs = tdi_fs
        self.tdi2 = tdi2
        self.t_start = t_start
        self.Tobs = Tobs
        self.gb_gpu = gb_gpu
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np
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
        # self.tdi_ts = xr.Dataset(dict([(k, tdi_fs[k].ts.ifft()) for k in ["X", "Y", "Z"]]))
        # self.tdi_ts['A'] = (self.tdi_ts['Z'] - self.tdi_ts['X'])/np.sqrt(2.0)
        # self.tdi_ts['E'] = (self.tdi_ts['Z'] - 2.0* self.tdi_ts['Y'] + self.tdi_ts['X'])/np.sqrt(6.0)
        # f, psdX =  scipy.signal.welch(self.tdi_ts["X"], fs=1.0/dt, nperseg=len(self.tdi_ts["X"])/1)
        # f, psdY =  scipy.signal.welch(self.tdi_ts["Y"], fs=1.0/dt, nperseg=len(self.tdi_ts["X"])/1)
        # f, psdZ =  scipy.signal.welch(self.tdi_ts["Z"], fs=1.0/dt, nperseg=len(self.tdi_ts["X"])/1)
        # self.psdf, self.psdA =  scipy.signal.welch(self.tdi_ts["A"], fs=1.0/dt, nperseg=len(self.tdi_ts["X"])/10)
        # self.psdf, self.psdE =  scipy.signal.welch(self.tdi_ts["E"], fs=1.0/dt, nperseg=len(self.tdi_ts["X"])/10)
        # psd = psdX + psdY + psdZ
        # indexes = np.logical_and(f>lower_frequency-self.padding, f<upper_frequency+self.padding)
        # psd = psd[indexes]
        self.frequency_T_threshold = 19.1*10**-3/2
        
        self.use_T_component = False
        if self.upper_frequency + self.padding > self.frequency_T_threshold:
            self.use_T_component = True
        # plt.figure()
        # plt.plot(f[indexes], psdX[indexes])
        # plt.plot(f[indexes], psd)
        # plt.show()

        self.frequencyrange =  [lower_frequency-self.padding, upper_frequency+self.padding]
        self.indexes = np.logical_and(tdi_fs['X'].f > self.frequencyrange[0], tdi_fs['X'].f < self.frequencyrange[1]) 
        self.dataX = tdi_fs["X"][self.indexes]
        self.dataY = tdi_fs["Y"][self.indexes]
        self.dataZ = tdi_fs["Z"][self.indexes]
        try:
            self.DAf = tdi_fs["A"][self.indexes]
            self.DEf = tdi_fs["E"][self.indexes]
            self.DTf = tdi_fs["T"][self.indexes]
        except:
            self.DAf = (self.dataZ - self.dataX)/np.sqrt(2.0)
            self.DEf = (self.dataZ - 2.0*self.dataY + self.dataX)/np.sqrt(6.0)
            self.DTf = (self.dataZ + self.dataY + self.dataX)/np.sqrt(3.0)

        if gb_gpu:
            self.dataX_full_f = tdi_fs["X"]
            self.dataY_full_f = tdi_fs["Y"]
            self.dataZ_full_f = tdi_fs["Z"]
            try:
                self.DAf_full_f = tdi_fs["A"]
                self.DEf_full_f = tdi_fs["E"]
                self.DTf_full_f = tdi_fs["T"]
            except:
                self.DAf_full_f = (self.dataZ_full_f - self.dataX_full_f)/np.sqrt(2.0)
                self.DEf_full_f = (self.dataZ_full_f - 2.0*self.dataY_full_f + self.dataX_full_f)/np.sqrt(6.0)
                self.DTf_full_f = (self.dataZ_full_f + self.dataY_full_f + self.dataX_full_f)/np.sqrt(3.0)

        # plt.figure()
        # plt.plot(f[self.indexes], np.abs(self.DAf))
        # plt.plot(f[indexes], np.abs(self.DEf))
        # plt.plot(f[indexes], np.abs(self.DAf)+np.abs(self.DEf))spos
        # plt.show()
        # print('frequencyrange',frequencyrange)

        if noise is None:
            fmin, fmax = float(self.dataX.f[0]), float(self.dataX.f[-1] + self.dataX.attrs["df"])
            freq = np.array(self.dataX.sel(f=slice(fmin, fmax)).f)
            Nmodel = get_noise_model(noise_model, freq)
            self.Sn = Nmodel.psd(freq=freq, option="X")
            self.SA = Nmodel.psd(freq=freq, option="A")
            self.SE = Nmodel.psd(freq=freq, option="E")
            self.ST = Nmodel.psd(freq=freq, option="T")
            if gb_gpu:
                self.SA_full_f = Nmodel.psd(freq=self.DAf_full_f.f, option="A")
                self.SE_full_f = Nmodel.psd(freq=self.DEf_full_f.f, option="E")
                self.ST_full_f = Nmodel.psd(freq=self.DAf_full_f.f, option="T")
        else:
            self.SA_full_f = noise['A']
            self.SE_full_f = noise['E']
            self.ST_full_f = noise['T']
            fmin, fmax = float(self.dataX.f[0]), float(self.dataX.f[-1] + self.dataX.attrs["df"])
            freq = np.array(self.dataX.sel(f=slice(fmin, fmax)).f)
            self.SA = self.SA_full_f[self.indexes]
            self.SE = self.SE_full_f[self.indexes]
            self.ST = self.ST_full_f[self.indexes]

        if gb_gpu:
            self.data_GPU = [self.xp.array(self.DAf_full_f),
                    self.xp.array(self.DEf_full_f),
            ]
            
            self.PSD_GPU =  [self.xp.array(self.SA_full_f),
                    self.xp.array(self.SA_full_f),
            ]

        # if update_noise:
        self.update_noise()

        f_0 = fmin
        f_transfer = 19.1*10**-3
        snr = 7
        amplitude_lower = 2*snr/(Tobs * np.sin(f_0/ f_transfer)**2/self.SA[0])**0.5
        snr = 1000
        amplitude_upper = 2*snr/(Tobs * np.sin(f_0/ f_transfer)**2/self.SA[0])**0.5
        amplitude = [amplitude_lower, amplitude_upper]
        # print('lower frequency', lower_frequency)
        # print('amplitude boundaries', amplitude)
        # print('amplitude boundaries previous', np.sqrt(np.max(psd))/1000,np.sqrt(np.max(psd))/10)
        # print('amplitude boundaries previous', np.max(np.abs(self.DAf.data))/10**7,np.max(np.abs(self.DAf.data))/10**5)
        # amplitude = np.sqrt(np.max(psd))

        # indexes = np.argsort(p.get('Frequency'))
        # index_low = np.searchsorted(p.get('Frequency')[indexes], frequencyrange[0])
        # index_high = np.searchsorted(p.get('Frequency')[indexes], frequencyrange[1])

        # pGB = deepcopy(pGBadded)
        # self.pGB = {'Amplitude': 3.676495e-22, 'EclipticLatitude': 0.018181, 'EclipticLongitude': 1.268061, 'Frequency': 0.01158392, 'FrequencyDerivative': 8.009579e-15, 'Inclination': 0.686485, 'InitialPhase': 4.201455, 'Polarization': 2.288223}

        # fd_range = [np.log10(frequency_derivative(lower_frequency,0.1)),np.log10(frequency_derivative(lower_frequency,M_chirp_upper_boundary))]
        # fd_range = [frequency_derivative(lower_frequency,0.1),frequency_derivative(lower_frequency,M_chirp_upper_boundary)]
        # fd_range = [frequency_derivative_tyson_lower(lower_frequency),frequency_derivative_tyson(lower_frequency)]
        fd_range = [frequency_derivative_tyson_lower(lower_frequency),frequency_derivative(upper_frequency,M_chirp_upper_boundary)]
        self.boundaries = {
            "Amplitude": [np.log10(amplitude[0]),np.log10(amplitude[1])],
            # "Amplitude": [-23.5,-21],
            # "Amplitude": [np.log10(self.pGB['Amplitude'])-2,np.log10(self.pGB['Amplitude'])+1],
            "EclipticLatitude": [-1.0, 1.0],
            "EclipticLongitude": [-np.pi, np.pi],
            # "Frequency": [self.pGB["Frequency"] * 0.99995, self.pGB["Frequency"] * 1.00015],
            # "Frequency": [self.pGB["Frequency"] - 3e-7, self.pGB["Frequency"] + 3e-7],
            "Frequency": self.frequencyrange,
            "FrequencyDerivative": fd_range,
            # "FrequencyDerivative": [np.log10(5e-6*self.pGB['Frequency']**(13/3)),np.log10(8e-8*self.pGB['Frequency']**(11/3))],
            "Inclination": [-1.0, 1.0],
            "InitialPhase": [0.0, 2.0 * np.pi],
            "Polarization": [0.0, 1.0 * np.pi],
        }
        # print('fd_rage',self.boundaries['FrequencyDerivative'], 'at f=', lower_frequency)

        # print('fd * Tobs', 10**self.boundaries['FrequencyDerivative'][1] * Tobs)
        # print('smear f', 300*lower_frequency * 10**3 / 10**9)

        if self.boundaries['FrequencyDerivative'][0] > self.boundaries['FrequencyDerivative'][1]:
            c = self.boundaries['FrequencyDerivative'][0]
            self.boundaries['FrequencyDerivative'][0] = self.boundaries['FrequencyDerivative'][1]
            self.boundaries['FrequencyDerivative'][1] = c
        
        previous_max = np.random.rand(8)
        # previous_max[0] = np.random.rand(1)*0.1 +0.5
        # previous_max[3] = np.random.rand(1)*0.1 +0.5
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
        # self.pGBs = {'Amplitude': 4.0900673126042746e-22, 'EclipticLatitude': 0.8718477251317046, 'EclipticLongitude': 0.48599945403230693, 'Frequency': 0.003995220986111426, 'FrequencyDerivative': 1.0985841703423861e-16, 'Inclination': 1.0262955111380103, 'InitialPhase': 5.453865686076588, 'Polarization': 1.089057196561609}

        # cutoff_ratio = 1000
        # Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=self.pGBs, oversample=4, tdi2=self.tdi2)
        # psd_signal = np.abs(Xs.values) ** 2 + np.abs(Ys.values) ** 2 + np.abs(Zs.values) ** 2
        # highSNR = psd_signal > np.max(psd_signal) / cutoff_ratio
        # lowerindex = np.where(highSNR)[0][0] - 30
        # higherindex = np.where(highSNR)[0][-1] + 30
        # self.dataX = tdi_fs["X"].isel(f=slice(Xs.kmin, Xs.kmin + len(Xs)))[lowerindex:higherindex]
        # self.dataY = tdi_fs["Y"].isel(f=slice(Ys.kmin, Ys.kmin + len(Ys)))[lowerindex:higherindex]
        # self.dataZ = tdi_fs["Z"].isel(f=slice(Zs.kmin, Zs.kmin + len(Zs)))[lowerindex:higherindex]

        # self.DAf = (2/3*self.dataX - self.dataY - self.dataZ)/3.0
        # self.DEf = (self.dataZ - self.dataY)/np.sqrt(3.0)

        # Xs, Ys, Zs = (
        #     Xs[lowerindex:higherindex],
        #     Ys[lowerindex:higherindex],
        #     Zs[lowerindex:higherindex],
        # )

        # diff = np.abs(self.dataX - Xs.values) ** 2 + np.abs(self.dataY - Ys.values) ** 2 + np.abs(self.dataZ - Zs.values) ** 2
        # p1 = float(np.sum(diff / (self.Sn + noise)) * Xs.df) / 2.0
        # p1 = -p1
        # diff = np.abs(self.dataX) ** 2 + np.abs(self.dataY) ** 2 + np.abs(self.dataZ) ** 2
        # null_pGBs = deepcopy(self.pGBs)
        # null_pGBs['Amplitude'] = 4*10**-25
        # print('pGB', self.pGB, self.loglikelihood([self.pGB]))

    def update_noise(self, pGB=None):
        if pGB != None:
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGB, oversample=4, tdi2=self.tdi2)
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
        # self.SA = np.abs(self.DAf-Af).data**2/(dt*len(self.tdi_fs['X']))*2
        # self.SE = np.abs(self.DEf-Ef).data**2/(dt*len(self.tdi_fs['X']))*2
        # self.ST = np.abs(self.DTf-Tf).data**2/(dt*len(self.tdi_fs['X']))*2

        # start = time.time()
        # f, psdA, psdE, psdT = get_noise_from_frequency_domain(self.tdi_fs, number_of_windows=1)
        # psdA_fit = smooth_psd(psdA, f)
        # psdE_fit = smooth_psd(psdE, f)
        # psdT_fit = smooth_psd(psdT, f)
        # psdA_fit = scipy.interpolate.InterpolatedUnivariateSpline(f, psdA)(self.tdi_fs.f)
        # psdE_fit = scipy.interpolate.InterpolatedUnivariateSpline(f, psdE)(self.tdi_fs.f)
        # psdT_fit = scipy.interpolate.InterpolatedUnivariateSpline(f, psdT)(self.tdi_fs.f)
        # self.SA = psdA[self.indexes]
        # self.SE = psdE[self.indexes]
        # self.ST = psdT[self.indexes]
        # self.SA = np.ones_like(self.SA)*np.median(psdA[self.indexes])
        # self.SE = np.ones_like(self.SE)*np.median(psdE[self.indexes])
        # self.ST = np.ones_like(self.ST)*np.median(psdT[self.indexes])
        # print(time.time()-start)

        # start = time.time()
        # self.psdf, self.psdE =  scipy.signal.welch(self.tdi_ts["E"], fs=1.0/dt, nperseg=len(self.tdi_ts["X"])/10)
        # print(time.time()-start)
        

        self.SA = np.ones_like(self.SA)*np.median((np.abs(self.DAf-Af).data/self.dt)**2/(self.fs*self.N_samples)*2)
        self.SE = np.ones_like(self.SE)*np.median((np.abs(self.DEf-Ef).data/self.dt)**2/(self.fs*self.N_samples)*2)
        self.ST = np.ones_like(self.ST)*np.median((np.abs(self.DTf-Tf).data/self.dt)**2/(self.fs*self.N_samples)*2)

        # self.SA = np.ones_like(self.SA)*(np.median(np.abs(self.DAf-Af).data))**2/(self.dt*len(self.tdi_fs['X']))*2
        # self.SE = np.ones_like(self.SE)*(np.median(np.abs(self.DEf-Ef).data))**2/(self.dt*len(self.tdi_fs['X']))*2
        # self.ST = np.ones_like(self.ST)*(np.median(np.abs(self.DTf-Tf).data))**2/(self.dt*len(self.tdi_fs['X']))*2

        # self.SA_psd = (np.abs(self.DAf-Af).data)**2/(dt*len(self.tdi_fs['X']))*2
        # self.SE = (np.abs(self.DEf-Ef).data)**2/(dt*len(self.tdi_fs['X']))*2
        # self.ST = (np.abs(self.DTf-Tf).data)**2/(dt*len(self.tdi_fs['X']))*2

        # self.SA_psd = median_windows(self.SA_psd, len(self.SA_psd))
        # self.SA = np.ones_like(self.SA_psd)*np.mean(self.SA_psd)

        # plt.figure()
        # plt.loglog(self.dataX.f, self.SA)
        # plt.loglog(self.dataX.f, self.SA_psd)
        # plt.loglog(self.dataX.f, (np.abs(self.DAf-Af).data)**2/(dt*len(self.tdi_fs['X']))*2)
        # plt.loglog(self.dataX.f, np.ones_like(self.SA)*(np.median(np.abs(self.DAf-Af).data))**2/(dt*len(self.tdi_fs['X']))*2)
        # plt.loglog(self.dataX.f, np.ones_like(self.SA)*np.median((np.abs(self.DAf-Af).data)**2)/(dt*len(self.tdi_fs['X']))*2)
        # plt.show()

        # plt.figure()
        # plt.loglog(self.dataX.f, self.SA)
        # plt.loglog(self.dataX.f, self.SAs)
        # plt.show()
        # plt.figure()
        # plt.loglog(self.tdi_fs.f, psdA_fit)
        # plt.show()

        # self.SE = Nmodel.psd(freq=freq, option="E")

    def plot(self, maxpGBs=None, pGBadded=None, second_data = None , found_sources_in= [], found_sources_not_matched= [], pGB_injected = [], pGB_injected_matched = [], added_label='Injection2', saving_label =None, vertical_lines = []):
        # plt.figure(figsize=fig_size)
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=fig_size)
        # plt.plot(dataX_training.f*1000,dataX_training.values, label='data')
        # ax1.plot(self.dataX.f * 1000, self.dataX.values.real, label="data", marker="o", zorder=5)

        # Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=self.pGB, oversample=4, tdi2=self.tdi2)
        # index_low = np.searchsorted(Xs.f, self.dataX.f[0])
        # Xs = Xs[index_low : index_low + len(self.dataX)]
        # Ys = Ys[index_low : index_low + len(self.dataY)]
        # Zs = Zs[index_low : index_low + len(self.dataZ)]

        # Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=self.pGB, oversample=8, tdi2=self.tdi2)
        # index_low = np.searchsorted(Xs.f, self.dataX.f[0])
        # Xs = Xs[index_low : index_low + len(self.dataX)]
        # Ys = Ys[index_low : index_low + len(self.dataY)]
        # Zs = Zs[index_low : index_low + len(self.dataZ)]
        # ax1.plot(Xs.f * 1000, Xs.values.real, label="VGB2", marker=".", zorder=5)
                    
        # Af = (Zs - Xs)/np.sqrt(2.0)
        ax1.plot(self.DAf.f*10**3,self.DAf,'k',zorder= 1, linewidth = 2, label = 'Data')
        # ax1.plot(self.DAf.f*10**3,np.sqrt(self.SA),'r',zorder= 1, linewidth = 2, label = 'Noise')
        # ax1.plot(self.psdf*10**3,np.sqrt(self.psdA),'b',zorder= 1, linewidth = 2, label = 'Noise welch')
        # ax1.plot(self.DAf.f*10**3,np.sqrt(self.SA_median),'g',zorder= 1, linewidth = 2, label = 'Noise median')
        ax2.plot(self.DEf.f*10**3,np.abs(self.DEf),'k',zorder= 1, linewidth = 2, label = 'Data')
        # ax2.plot(self.DEf.f*10**3,np.abs(np.sqrt(self.SE)),'r',zorder= 1, linewidth = 2, label = 'Noise')
        # ax2.plot(self.DEf.f*10**3,np.abs(np.sqrt(self.SE_theory)),'g',zorder= 1, linewidth = 2, label = 'Noise theory')
        # ax2.plot(self.psdf*10**3,np.abs(np.sqrt(self.psdE)),'b',zorder= 1, linewidth = 2, label = 'Noise welch')
        # ax2.plot(self.DEf.f*10**3,np.abs(np.sqrt(self.SE_median)),'g',zorder= 1, linewidth = 2, label = 'Noise median')
        # ax1.plot(tdi_fs_long_subtracted.f[range_index],np.abs(tdi_fs_long_subtracted['X'][range_index])**2,'b',zorder= 5)

        if second_data != None:
            a,Xs = xr.align(self.dataX, second_data['X'], join='left',fill_value=0)
            a,Ys = xr.align(self.dataY, second_data['Y'], join='left',fill_value=0)
            a,Zs = xr.align(self.dataZ, second_data['Z'], join='left',fill_value=0)
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f*10**3,Af,'k--',zorder= 1, linewidth = 2, label = 'Data subtracted')
            ax2.plot(Ef.f*10**3,np.abs(Ef),'k--',zorder= 1, linewidth = 2, label = 'Data subtracted')

        for j in range(len( pGB_injected)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template= pGB_injected[j], oversample=4, tdi2=self.tdi2)
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            try:
                if np.abs(self.dataX.f[0] - Xs.f[index_low-1]) < np.abs(self.dataX.f[0] - Xs.f[index_low]):
                    index_low = index_low-1
            except:
                pass
            Xs = Xs[index_low : index_low + len(self.dataX)]
            Ys = Ys[index_low : index_low + len(self.dataY)]
            Zs = Zs[index_low : index_low + len(self.dataZ)]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f*10**3,Af.data, color='grey', linewidth = 5, alpha = 0.5)
            ax2.plot(Ef.f*10**3,np.abs(Ef.data), color='grey', linewidth = 5, alpha = 0.5)

        for j in range(len(pGB_injected_matched)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template= pGB_injected_matched[j], oversample=4, tdi2=self.tdi2)
            a,Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)
            a,Ys = xr.align(self.dataY, Ys, join='left',fill_value=0)
            a,Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f*10**3,Af.data, color=colors[j%10], linewidth = 5, alpha = 0.5, label = 'Injection')
            ax2.plot(Ef.f*10**3,np.abs(Ef.data), color=colors[j%10], linewidth = 5, alpha = 0.5)


        if pGBadded != None:
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBadded, oversample=4, tdi2=self.tdi2)
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            Xs = Xs[index_low : index_low + len(self.dataX)]
            Ys = Ys[index_low : index_low + len(self.dataY)]
            Zs = Zs[index_low : index_low + len(self.dataZ)]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f* 1000, Af.data, marker='.', label=added_label)
            ax2.plot(Ef.f* 1000, np.abs(Ef.data), marker='.', label=added_label)

        for j in range(len(found_sources_in)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=found_sources_in[j], oversample=4, tdi2=self.tdi2)
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            try:
                if np.abs(self.dataX.f[0] - Xs.f[index_low-1]) < np.abs(self.dataX.f[0] - Xs.f[index_low]):
                    index_low = index_low-1
            except:
                pass
            Xs = Xs[index_low : index_low + len(self.dataX)]
            Ys = Ys[index_low : index_low + len(self.dataY)]
            Zs = Zs[index_low : index_low + len(self.dataZ)]
            # Xs = xr.align(self.dataX, Xs, join='inner',fill_value=0)[1]
            # Ys = xr.align(self.dataX, Ys, join='left',fill_value=0)[1]
            # Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f* 1000, Af.data,'--', color= colors[j%10], linewidth = 1.6)
            ax2.plot(Ef.f* 1000, np.abs(Ef.data),'--', color= colors[j%10], linewidth = 1.6)


        for j in range(len(found_sources_not_matched)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=found_sources_not_matched[j], oversample=4, tdi2=self.tdi2)
            # index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
            Ys = xr.align(self.dataX, Ys, join='left',fill_value=0)[1]
            Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f* 1000,Af.data,'.', color= colors[j%10], linewidth = 1.6)
            ax2.plot(Ef.f* 1000, np.abs(Ef.data),'.', color= colors[j%10], linewidth = 1.6)

        # ax1.plot(Xs_added2.f * 1000, Xs_added2.values.real, label="VGB2", marker=".", zorder=5)
        ax1.axvline(self.lower_frequency* 1000, color= 'red', label='Boundaries')
        ax1.axvline(self.upper_frequency* 1000, color= 'red')
        ax2.axvline(self.lower_frequency* 1000, color= 'red')
        ax2.axvline(self.upper_frequency* 1000, color= 'red')
        for j in range(len(vertical_lines)):
            ax1.axvline(vertical_lines[j]* 1000, color= 'red')
            ax2.axvline(vertical_lines[j]* 1000, color= 'red')
        # ax2.axvline(self.lower_frequency* 1000- 4*32*10**-6, color= 'green')
        # ax2.axvline(self.upper_frequency* 1000+ 4*32*10**-6, color= 'green')
        # if self.reduced_frequency_boundaries != None:
        #     ax1.axvline(self.reduced_frequency_boundaries[0]* 1000, color= 'green', label='Reduced Boundaries')
        #     ax1.axvline(self.reduced_frequency_boundaries[1]* 1000, color= 'green')

        # ax1.plot(Xs.f * 1000, dataX.values.real - Xs.values.real, label="residual", alpha=0.8, color="red", marker=".")
        plt.xlabel('f (mHz)')
        ax1.set_ylabel('A')    
        ax2.set_ylabel('|E|') 
        # ax1.set_yscale('log')  
        ax2.set_yscale('log')   
        ax1.set_xlim((self.lower_frequency-self.padding)*10**3, (self.upper_frequency+self.padding)*10**3)
        ax2.set_xlim((self.lower_frequency-self.padding)*10**3, (self.upper_frequency+self.padding)*10**3)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax1.legend()
        ax2.legend()
        # plt.legend()
        if saving_label != None:
            plt.savefig(saving_label,dpi=300,bbox_inches='tight')
        plt.show()
        # print("p true", self.loglikelihood([pGB]), "null hypothesis", self.loglikelihood([null_pGBs]))

    def A_E_total(self, pGBs):
        Xs_total = deepcopy(self.dataX)
        Ys_total = deepcopy(self.dataY)
        Zs_total = deepcopy(self.dataZ)
        Xs_total.data = np.zeros_like(self.dataX.data)
        Ys_total.data = np.zeros_like(self.dataY.data)
        Zs_total.data = np.zeros_like(self.dataZ.data)
        for i in range(len(pGBs)):
            # freqs = np.array(self.dataX.f)
            # if len(freqs) < 16:
            #     freqs = np.linspace(self.lower_frequency, self.upper_frequency, 16)
            # Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4, freqs=freqs)#, tdi2=self.tdi2)
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4)#, tdi2=self.tdi2)

            index_low = np.searchsorted(Xs.f, Xs_total.f[0])
            index_high = np.searchsorted(Xs.f, Xs_total.f[-1])
            if index_high > len(Xs.f)-1:
                index_high = len(Xs.f)-1
            try:
                if np.abs(Xs_total.f[0] - Xs.f[index_low-1]) < np.abs(Xs_total.f[0] - Xs.f[index_low]):
                # if Xs.f[index_low] > Xs_total.f[0]:
                    index_low = index_low-1
            except:
                pass
            try:
                if np.abs(Xs_total.f[-1] - Xs.f[index_high-1]) < np.abs(Xs_total.f[-1] - Xs.f[index_high]):
                    index_high = index_high-1
            except:
                pass
            index_low_total = np.searchsorted(Xs_total.f, Xs.f[index_low])
            index_high_total = np.searchsorted(Xs_total.f, Xs.f[index_high])
            if index_high_total > len(Xs_total.f)-1:
                index_high_total = len(Xs_total.f)-1
            try:
                if np.abs(Xs_total.f[index_low_total-1] - Xs.f[index_low]) < np.abs(Xs_total.f[index_low_total] - Xs.f[index_low]):
                    index_low_total = index_low_total-1
            except:
                pass
            try:
                if np.abs(Xs_total.f[index_high_total-1] - Xs.f[index_high]) < np.abs(Xs_total.f[index_high_total] - Xs.f[index_high]):
                    index_high_total = index_high_total-1
            except:
                pass
            Xs_total[index_low_total:index_low_total+(index_high-index_low)] += Xs[index_low:index_high]
            Ys_total[index_low_total:index_low_total+(index_high-index_low)] += Ys[index_low:index_high]
            Zs_total[index_low_total:index_low_total+(index_high-index_low)] += Zs[index_low:index_high]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        if self.use_T_component:
            Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
            return Af, Ef, Tf
        return Af, Ef
    
    def get_dh_hh(self, pGBs):
        if self.use_T_component:
            Af, Ef, Tf = self.A_E_total(pGBs)
        else:
            Af, Ef = self.A_E_total(pGBs)

            
        index_data = np.searchsorted(self.dataX.f, Af.f[0])
        try:
            if np.abs(self.dataX.f[index_data-1] - Af.f[0]) < np.abs(self.dataX.f[index_data] - Af.f[0]):
            # if self.dataX.f[index_data] > Xs.f[0]:
                index_data = index_data-1
        except:
            pass
        if len(self.DAf) > len(Af.f) or index_data != 0:
            Af_full = xr.DataArray(np.zeros(len(self.DAf), dtype=complex), dims='f', coords={'f':self.DAf.f})
            Ef_full = xr.DataArray(np.zeros(len(self.DEf), dtype=complex), dims='f', coords={'f':self.DEf.f})
            if self.use_T_component:
                Tf_full = xr.DataArray(np.zeros(len(self.DTf), dtype=complex), dims='f', coords={'f':self.DTf.f})
            try:
                Af_full.data[index_data:index_data+len(Af)] = Af.data
                Ef_full.data[index_data:index_data+len(Ef)] = Ef.data
                if self.use_T_component:
                    Tf_full.data[index_data:index_data+len(Tf)] = Tf.data
            except:
                Af_full.data[index_data:] = Af.data[:len(Af_full.data[index_data:])]
                Ef_full.data[index_data:] = Ef.data[:len(Ef_full.data[index_data:])]
                if self.use_T_component:
                    Tf_full.data[index_data:] = Tf.data[:len(Tf_full.data[index_data:])]
            Af = Af_full
            Ef = Ef_full
            if self.use_T_component:
                Tf = Tf_full
        if self.use_T_component:
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/self.SA + np.absolute(Tf.data)**2 /self.ST)
            dh = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA + np.real(self.DTf * np.conjugate(Tf.data))/self.ST )
        else:
            dh = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA )
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
        hh = 4.0*self.dataX.df* hh
        dh = 4.0*self.dataX.df* dh
        return dh, hh
    
    def SNR(self, pGBs):
        dh, hh = self.get_dh_hh(pGBs)
        SNR = dh / np.sqrt(hh)
        return SNR.values

    def SNR_scaled(self, pGBs):
        for i in range(len(pGBs)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4, tdi2=self.tdi2)
            if i == 0:
                index_low = np.searchsorted(Xs.f, self.dataX.f[0])
                try:
                    if np.abs(self.dataX.f[0] - Xs.f[index_low-1]) < np.abs(self.dataX.f[0] - Xs.f[index_low]):
                        index_low = index_low-1
                except:
                    pass
                Xs_total = Xs[index_low : index_low + len(self.dataX)]
                Ys_total = Ys[index_low : index_low + len(self.dataY)]
                Zs_total = Zs[index_low : index_low + len(self.dataZ)]
            else:
                index_low = np.searchsorted(Xs.f, self.dataX.f[0])
                try:
                    if np.abs(self.dataX.f[0] - Xs.f[index_low-1]) < np.abs(self.dataX.f[0] - Xs.f[index_low]):
                        index_low = index_low-1
                except:
                    pass
                Xs_total += Xs[index_low : index_low + len(self.dataX)]
                Ys_total += Ys[index_low : index_low + len(self.dataY)]
                Zs_total += Zs[index_low : index_low + len(self.dataZ)]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        if self.use_T_component:
            Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/self.SA + np.absolute(Tf.data)**2 /self.ST)
            dd = np.sum((np.absolute(self.DAf)**2 + np.absolute(self.DEf)**2)/self.SA + np.absolute(self.DTf)**2 /self.ST)
            SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA + np.real(self.DTf * np.conjugate(Tf.data))/self.ST )
        else:
            SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA )
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
            dd = np.sum((np.absolute(self.DAf)**2 + np.absolute(self.DEf)**2) /self.SA)
        SNR = 4.0*Xs.df* hh
        SNR2 = 4.0*Xs.df* SNR2
        SNR3 = SNR2 / np.sqrt(SNR) / np.sqrt(4.0*Xs.df* dd)
        return SNR3.values

    def plotA(self, maxpGBs=None, pGBadded=None, found_sources_in= [], found_sources_not_matched= [], pGB_injected = [], pGB_injected_matched = [], chains=[], added_label='Injection2', saving_label =None, vertical_lines = [], second_data= None):
        # plt.figure(figsize=fig_size)
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=False, figsize=np.array(fig_size)*[1,1.5])

        parameter_x = 'Frequency'
        parameter_y = 'Amplitude'
        parameter_x2 = 'EclipticLongitude'
        parameter_y2 = 'EclipticLatitude'
                    
        # Af = (Zs - Xs)/np.sqrt(2.0)
        ax1.plot(self.DAf.f*10**3,np.abs(self.DAf),'k',zorder= 1, linewidth = 3, label = 'Data')
        # ax1.plot(tdi_fs_long_subtracted.f[range_index],np.abs(tdi_fs_long_subtracted['X'][range_index])**2,'b',zorder= 5)
        if second_data != None:
            a,Xs = xr.align(self.dataX, second_data['X'], join='left',fill_value=0)
            a,Ys = xr.align(self.dataY, second_data['Y'], join='left',fill_value=0)
            a,Zs = xr.align(self.dataZ, second_data['Z'], join='left',fill_value=0)
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f*10**3,Af,'c--',zorder= 1, linewidth = 2, label = 'Data subtracted')
            ax2.plot(Ef.f*10**3,np.abs(Ef),'c--',zorder= 1, linewidth = 2, label = 'Data subtracted')
        for j in range(len( pGB_injected)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template= pGB_injected[j], oversample=4, tdi2=self.tdi2)
            a,Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)
            a,Ys = xr.align(self.dataY, Ys, join='left',fill_value=0)
            a,Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f*10**3,np.abs(Af.data), color=colors[j%10], linewidth = 5, alpha = 0.5)
            ax2.plot(pGB_injected[j][parameter_x]*10**3,pGB_injected[j][parameter_y],'o', color=colors[j%10], markersize=7, alpha = 1, markerfacecolor='None')
            for k in range(len(pGB_injected[j][parameter_x2].shape)):
                if pGB_injected[j][parameter_x2][k] < 0:
                    pGB_injected[j][parameter_x2][k] += 2*np.pi
            # pGB_injected[j][parameter_x2][pGB_injected[j][parameter_x2] < 0] += 2*np.pi
            # ax3.plot(pGB_injected[j][parameter_x2],pGB_injected[j][parameter_y2],'o', color=colors[j%10], markersize=7, alpha = 1, markerfacecolor='None')
            # ax3.plot(pGB_injected[j]['EclipticLongitude']*10**3,pGB_injected[j]['EclipticLatitude'],'o', color=colors[j%10], markersize=7, alpha = 0.5)
            # ax4.plot(pGB_injected[j]['Inclination']*10**3,pGB_injected[j]['FrequencyDerivative'],'o', color=colors[j%10], markersize=7, alpha = 0.5)
            # ax5.plot(pGB_injected[j]['InitialPhase']*10**3,pGB_injected[j]['Polarization'],'o', color=colors[j%10], markersize=7, alpha = 0.5)
        for j in range(len(pGB_injected_matched)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template= pGB_injected_matched[j], oversample=4, tdi2=self.tdi2)
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            try:
                if np.abs(self.dataX.f[0] - Xs.f[index_low-1]) < np.abs(self.dataX.f[0] - Xs.f[index_low]):
                    index_low = index_low-1
            except:
                pass
            Xs = Xs[index_low : index_low + len(self.dataX)]
            Ys = Ys[index_low : index_low + len(self.dataY)]
            Zs = Zs[index_low : index_low + len(self.dataZ)]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f*10**3,np.abs(Af.data), color=colors[j%10], linewidth = 5, alpha = 0.5)
            ax2.plot(pGB_injected_matched[j][parameter_x]*10**3,pGB_injected_matched[j][parameter_y],'o', color=colors[j%10], markersize=7, alpha = 1, markerfacecolor='None', label='true')
            for k in range(len(pGB_injected_matched[j][parameter_x2].shape)):
                if pGB_injected_matched[j][parameter_x2][k] < 0:
                    pGB_injected_matched[j][parameter_x2][k] += 2*np.pi
            # ax3.plot(pGB_injected_matched[j][parameter_x2],pGB_injected_matched[j][parameter_y2],'o', color=colors[j%10], markersize=7, alpha = 1, markerfacecolor='None')
            # ax3.plot(pGB_injected_matched[j]['EclipticLongitude']*10**3,pGB_injected_matched[j]['EclipticLatitude'],'o', color=colors[j%10], markersize=7, alpha = 0.5)
        if pGBadded != None:
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBadded, oversample=4, tdi2=self.tdi2)
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            Xs = Xs[index_low : index_low + len(self.dataX)]
            Ys = Ys[index_low : index_low + len(self.dataY)]
            Zs = Zs[index_low : index_low + len(self.dataZ)]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f* 1000, np.abs(Af.data), marker='.', label=added_label)
            # ax2.plot(Ef.f* 1000, np.abs(Ef.data), marker='.', label=added_label)
        # for j in range(len(found_sources_in)):
        #     Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=found_sources_in[j], oversample=4, tdi2=self.tdi2)
        #     index_low = np.searchsorted(Xs.f, self.dataX.f[0])
        #     Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
        #     Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
        #     Af = (Zs - Xs)/np.sqrt(2.0)
        #     Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
        #     ax1.plot(Af.f* 1000, np.abs(Af.data),'--', color= colors[j%10], linewidth = 1.6)
        #     ax2.plot(found_sources_in[j][parameter_x]*10**3,found_sources_in[j][parameter_y],'.', color=colors[j%10], markersize=7)
        #     # ax2.plot(Ef.f* 1000, np.abs(Ef.data),'--', color= colors[j%10], linewidth = 1.6)
        # for j in range(len(found_sources_in)):
        #     Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=found_sources_in[j], oversample=4, tdi2=self.tdi2)
        #     index_low = np.searchsorted(Xs.f, self.dataX.f[0])
        #     if j == 0:
        #         Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
        #         Ys = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
        #         Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
        #     # ax2.plot(found_sources_in[j][parameter_x]*10**3,found_sources_in[j][parameter_y],'o', color='grey', markersize=7)
        #     Af = (Zs - Xs)/np.sqrt(2.0)
        #     Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
        #     ax1.plot(Af.f* 1000, np.abs(Af.data),linewidth = 5, alpha = 0.5, color= 'grey')
        #     # ax2.plot(Ef.f* 1000, np.abs(Ef.data),'--', color= colors[j%10], linewidth = 1.6)
        for j in range(len(found_sources_in)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=found_sources_in[j], oversample=4, tdi2=self.tdi2)
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            try:
                if np.abs(self.dataX.f[0] - Xs.f[index_low-1]) < np.abs(self.dataX.f[0] - Xs.f[index_low]):
                    index_low = index_low-1
            except:
                pass
            Xs = Xs[index_low : index_low + len(self.dataX)]
            Ys = Ys[index_low : index_low + len(self.dataY)]
            Zs = Zs[index_low : index_low + len(self.dataZ)]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f* 1000, np.abs(Af.data),'--', color= colors[j%10], linewidth = 1.6)
            ax2.plot(found_sources_in[j][parameter_x]*10**3,found_sources_in[j][parameter_y],'.', color=colors[j%10], markersize=12, linewidth=6, label='recovered')
        for j in range(len(found_sources_not_matched)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=found_sources_not_matched[j], oversample=4, tdi2=self.tdi2)
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
            Ys = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
            Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f* 1000, np.abs(Af.data),'--', color= colors[j%10], linewidth = 1.6)
            ax2.plot(found_sources_not_matched[j][parameter_x]*10**3,found_sources_not_matched[j][parameter_y],'.', color=colors[j%10], markersize=12, linewidth=6)

        for j in range(len(chains)):
            number_of_samples = 1000
            ax2.plot(chains[j][parameter_x][:number_of_samples], chains[j][parameter_y][:number_of_samples], '.', alpha = 0.01, markersize=2, color=colors[j%10], zorder = -1)
            chains[j][parameter_x2][chains[j][parameter_x2] < 0] += 2*np.pi
            # ax3.plot(chains[j][parameter_x2], chains[j][parameter_y2], '.', alpha = 0.01, markersize=2, color=colors[j%10], zorder = -1)
            mean_x = np.mean(chains[j][parameter_x2])
            std_x = np.std(chains[j][parameter_x2])
            mean_y = np.mean(chains[j][parameter_y2])
            std_y = np.std(chains[j][parameter_y2])
            # ax3.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y, capsize=6, color=colors[j%10], markersize=10, zorder = 2)

        ax1.axvline(self.lower_frequency* 1000, color= 'red', label='Boundaries')
        ax1.axvline(self.upper_frequency* 1000, color= 'red')
        ax2.axvline(self.lower_frequency* 1000, color= 'red')
        ax2.axvline(self.upper_frequency* 1000, color= 'red')
        for j in range(len(vertical_lines)):
            ax1.axvline(vertical_lines[j]* 1000, color= 'red')
            ax2.axvline(vertical_lines[j]* 1000, color= 'red')
            # ax1.axvline((vertical_lines[j]-self.padding)* 1000, color= 'green')
            # ax2.axvline((vertical_lines[j]-self.padding)* 1000, color= 'green')
            # ax1.axvline((vertical_lines[j]+self.padding)* 1000, color= 'green')
            # ax2.axvline((vertical_lines[j]+self.padding)* 1000, color= 'green')

        ax2.set_xlabel(parameter_x)
        if parameter_x == 'Frequency':
            ax2.set_xlabel(r'$f$ (mHz)')
        # ax3.set_xlabel(r'$\lambda$'+' (rad)')
        # ax3.set_ylabel(r'$\beta$'+' (rad)')
        ax1.set_ylabel(r'$|A|$')
        ax2.set_ylabel(r'$\mathcal{A}$')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        # ax1.set_xlim((self.lower_frequency-self.padding)*10**3, (self.upper_frequency+self.padding)*10**3)
        # ax2.set_xlim((self.lower_frequency-self.padding)*10**3, (self.upper_frequency+self.padding)*10**3)
        ax1.set_xlim((vertical_lines[1]-self.padding)*10**3, (vertical_lines[-2]+self.padding)*10**3)
        ax2.set_xlim((vertical_lines[1]-self.padding)*10**3, (vertical_lines[-2]+self.padding)*10**3)
        ax1.set_ylim(10**-19,10**-15)
        # ax2.set_ylim(10**-23,4*10**-23)
        # ax1.set_xlim((self.lower_frequency)*10**3, (self.upper_frequency)*10**3)
        # ax2.set_xlim((self.lower_frequency)*10**3, (self.upper_frequency)*10**3)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
        # ax2.xaxis.set_major_locator(plt.MaxNLocator(4))

        labels_plot = ['data','true', 'recovered']
        custom_lines1 = [plt.Line2D([0], [0], color='k', lw=2, linestyle='solid'),
                         plt.Line2D([0], [0], color=colors[0], lw=5, linestyle='solid', alpha=0.5),
                         plt.Line2D([0], [0], color=colors[0], lw=2, linestyle='dashed')]
        # custom_lines2 = [markers([0], [0], color=colors[0], lw=2, linestyle='solid'),
        #                 plt.Line2D([0], [0], color=colors[0], lw=2, linestyle='dashed')]
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend([handles[0], handles[len(pGB_injected_matched)]], [labels[0],labels[len(pGB_injected_matched)]], loc='upper left')
        ax1.legend(custom_lines1, labels_plot, loc='upper left')
        # plt.legend()
        plt.tight_layout()
        if saving_label != None:
            plt.savefig(saving_label,dpi=300,bbox_inches='tight')
        plt.show()
        # print("p true", self.loglikelihood([pGB]), "null hypothesis", self.loglikelihood([null_pGBs]))


    def plotAE(self, maxpGBs=None, pGBadded=None, found_sources_in= [], found_sources_not_matched= [], pGB_injected = [], pGB_injected_matched = [], chains=[], added_label='Injection2', saving_label =None, vertical_lines = [], second_data= None):
        # plt.figure(figsize=fig_size)
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=False, figsize=np.array(fig_size)*[1,1.5])

        parameter_x = 'Frequency'
        parameter_y = 'Amplitude'
        parameter_x2 = 'EclipticLongitude'
        parameter_y2 = 'EclipticLatitude'
                    
        # Af = (Zs - Xs)/np.sqrt(2.0)
        ax1.plot(self.DAf.f*10**3,self.DAf,'k',zorder= 1, linewidth = 3, label = 'Data')
        ax2.plot(self.DEf.f*10**3,self.DEf,'k',zorder= 1, linewidth = 3, label = 'Data')
        # ax1.plot(tdi_fs_long_subtracted.f[range_index],np.abs(tdi_fs_long_subtracted['X'][range_index])**2,'b',zorder= 5)
        if second_data != None:
            a,Xs = xr.align(self.dataX, second_data['X'], join='left',fill_value=0)
            a,Ys = xr.align(self.dataY, second_data['Y'], join='left',fill_value=0)
            a,Zs = xr.align(self.dataZ, second_data['Z'], join='left',fill_value=0)
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f*10**3,Af,'c--',zorder= 1, linewidth = 2, label = 'Data subtracted')
            ax2.plot(Ef.f*10**3,Ef,'c--',zorder= 1, linewidth = 2, label = 'Data subtracted')
        for j in range(len( pGB_injected)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template= pGB_injected[j], oversample=4, tdi2=self.tdi2)
            a,Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)
            a,Ys = xr.align(self.dataY, Ys, join='left',fill_value=0)
            a,Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f*10**3,Af.data, color=colors[j%10], linewidth = 5, alpha = 0.5)
            ax2.plot(Ef.f*10**3,Ef.data, color=colors[j%10], linewidth = 5, alpha = 0.5)
            # ax2.plot(pGB_injected[j][parameter_x]*10**3,pGB_injected[j][parameter_y],'o', color=colors[j%10], markersize=7, alpha = 1, markerfacecolor='None')
            for k in range(len(pGB_injected[j][parameter_x2].shape)):
                if pGB_injected[j][parameter_x2][k] < 0:
                    pGB_injected[j][parameter_x2][k] += 2*np.pi
            # pGB_injected[j][parameter_x2][pGB_injected[j][parameter_x2] < 0] += 2*np.pi
            # ax3.plot(pGB_injected[j][parameter_x2],pGB_injected[j][parameter_y2],'o', color=colors[j%10], markersize=7, alpha = 1, markerfacecolor='None')
            # ax3.plot(pGB_injected[j]['EclipticLongitude']*10**3,pGB_injected[j]['EclipticLatitude'],'o', color=colors[j%10], markersize=7, alpha = 0.5)
            # ax4.plot(pGB_injected[j]['Inclination']*10**3,pGB_injected[j]['FrequencyDerivative'],'o', color=colors[j%10], markersize=7, alpha = 0.5)
            # ax5.plot(pGB_injected[j]['InitialPhase']*10**3,pGB_injected[j]['Polarization'],'o', color=colors[j%10], markersize=7, alpha = 0.5)
        for j in range(len(pGB_injected_matched)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template= pGB_injected_matched[j], oversample=4, tdi2=self.tdi2)
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            try:
                if np.abs(self.dataX.f[0] - Xs.f[index_low-1]) < np.abs(self.dataX.f[0] - Xs.f[index_low]):
                    index_low = index_low-1
            except:
                pass
            Xs = Xs[index_low : index_low + len(self.dataX)]
            Ys = Ys[index_low : index_low + len(self.dataY)]
            Zs = Zs[index_low : index_low + len(self.dataZ)]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f*10**3,Af.data, color=colors[j%10], linewidth = 5, alpha = 0.5)
            ax2.plot(Ef.f*10**3,Ef.data, color=colors[j%10], linewidth = 5, alpha = 0.5)
            # ax2.plot(pGB_injected_matched[j][parameter_x]*10**3,pGB_injected_matched[j][parameter_y],'o', color=colors[j%10], markersize=7, alpha = 1, markerfacecolor='None', label='true')
            for k in range(len(pGB_injected_matched[j][parameter_x2].shape)):
                if pGB_injected_matched[j][parameter_x2][k] < 0:
                    pGB_injected_matched[j][parameter_x2][k] += 2*np.pi
            # ax3.plot(pGB_injected_matched[j][parameter_x2],pGB_injected_matched[j][parameter_y2],'o', color=colors[j%10], markersize=7, alpha = 1, markerfacecolor='None')
            # ax3.plot(pGB_injected_matched[j]['EclipticLongitude']*10**3,pGB_injected_matched[j]['EclipticLatitude'],'o', color=colors[j%10], markersize=7, alpha = 0.5)
        if pGBadded != None:
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBadded, oversample=4, tdi2=self.tdi2)
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            Xs = Xs[index_low : index_low + len(self.dataX)]
            Ys = Ys[index_low : index_low + len(self.dataY)]
            Zs = Zs[index_low : index_low + len(self.dataZ)]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f* 1000, np.abs(Af.data), marker='.', label=added_label)
            # ax2.plot(Ef.f* 1000, np.abs(Ef.data), marker='.', label=added_label)
        # for j in range(len(found_sources_in)):
        #     Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=found_sources_in[j], oversample=4, tdi2=self.tdi2)
        #     index_low = np.searchsorted(Xs.f, self.dataX.f[0])
        #     Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
        #     Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
        #     Af = (Zs - Xs)/np.sqrt(2.0)
        #     Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
        #     ax1.plot(Af.f* 1000, np.abs(Af.data),'--', color= colors[j%10], linewidth = 1.6)
        #     ax2.plot(found_sources_in[j][parameter_x]*10**3,found_sources_in[j][parameter_y],'.', color=colors[j%10], markersize=7)
        #     # ax2.plot(Ef.f* 1000, np.abs(Ef.data),'--', color= colors[j%10], linewidth = 1.6)
        # for j in range(len(found_sources_in)):
        #     Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=found_sources_in[j], oversample=4, tdi2=self.tdi2)
        #     index_low = np.searchsorted(Xs.f, self.dataX.f[0])
        #     if j == 0:
        #         Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
        #         Ys = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
        #         Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
        #     # ax2.plot(found_sources_in[j][parameter_x]*10**3,found_sources_in[j][parameter_y],'o', color='grey', markersize=7)
        #     Af = (Zs - Xs)/np.sqrt(2.0)
        #     Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
        #     ax1.plot(Af.f* 1000, np.abs(Af.data),linewidth = 5, alpha = 0.5, color= 'grey')
        #     # ax2.plot(Ef.f* 1000, np.abs(Ef.data),'--', color= colors[j%10], linewidth = 1.6)
        for j in range(len(found_sources_in)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=found_sources_in[j], oversample=4, tdi2=self.tdi2)
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            try:
                if np.abs(self.dataX.f[0] - Xs.f[index_low-1]) < np.abs(self.dataX.f[0] - Xs.f[index_low]):
                    index_low = index_low-1
            except:
                pass
            Xs = Xs[index_low : index_low + len(self.dataX)]
            Ys = Ys[index_low : index_low + len(self.dataY)]
            Zs = Zs[index_low : index_low + len(self.dataZ)]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f* 1000, Af.data,'--', color= colors[j%10], linewidth = 1.6)
            ax2.plot(Ef.f* 1000, Ef.data,'--', color= colors[j%10], linewidth = 1.6)
            # ax2.plot(found_sources_in[j][parameter_x]*10**3,found_sources_in[j][parameter_y],'.', color=colors[j%10], markersize=12, linewidth=6, label='recovered')
        for j in range(len(found_sources_not_matched)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=found_sources_not_matched[j], oversample=4, tdi2=self.tdi2)
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
            Ys = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
            Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f* 1000, Af.data,'--', color= colors[j%10], linewidth = 1.6)
            ax2.plot(Ef.f* 1000, Ef.data,'--', color= colors[j%10], linewidth = 1.6)
            # ax2.plot(found_sources_not_matched[j][parameter_x]*10**3,found_sources_not_matched[j][parameter_y],'.', color=colors[j%10], markersize=12, linewidth=6)

        for j in range(len(chains)):
            number_of_samples = 1000
            # ax2.plot(chains[j][parameter_x][:number_of_samples], chains[j][parameter_y][:number_of_samples], '.', alpha = 0.01, markersize=2, color=colors[j%10], zorder = -1)
            chains[j][parameter_x2][chains[j][parameter_x2] < 0] += 2*np.pi
            # ax3.plot(chains[j][parameter_x2], chains[j][parameter_y2], '.', alpha = 0.01, markersize=2, color=colors[j%10], zorder = -1)
            mean_x = np.mean(chains[j][parameter_x2])
            std_x = np.std(chains[j][parameter_x2])
            mean_y = np.mean(chains[j][parameter_y2])
            std_y = np.std(chains[j][parameter_y2])
            # ax3.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y, capsize=6, color=colors[j%10], markersize=10, zorder = 2)

        # ax1.axvline(self.lower_frequency* 1000, color= 'red', label='Boundaries')
        # ax1.axvline(self.upper_frequency* 1000, color= 'red')
        # ax2.axvline(self.lower_frequency* 1000, color= 'red')
        # ax2.axvline(self.upper_frequency* 1000, color= 'red')
        # for j in range(len(vertical_lines)):
        #     ax1.axvline(vertical_lines[j]* 1000, color= 'red')
        #     ax2.axvline(vertical_lines[j]* 1000, color= 'red')
            # ax1.axvline((vertical_lines[j]-self.padding)* 1000, color= 'green')
            # ax2.axvline((vertical_lines[j]-self.padding)* 1000, color= 'green')
            # ax1.axvline((vertical_lines[j]+self.padding)* 1000, color= 'green')
            # ax2.axvline((vertical_lines[j]+self.padding)* 1000, color= 'green')

        ax2.set_xlabel(parameter_x)
        if parameter_x == 'Frequency':
            ax2.set_xlabel(r'$f$ (mHz)')
        # ax3.set_xlabel(r'$\lambda$'+' (rad)')
        # ax3.set_ylabel(r'$\beta$'+' (rad)')
        ax1.set_ylabel(r'real($A$)')
        ax2.set_ylabel(r'real($E$)')
        # ax1.set_yscale('log')
        # ax2.set_yscale('log')
        # ax1.set_xlim((self.lower_frequency-self.padding)*10**3, (self.upper_frequency+self.padding)*10**3)
        # ax2.set_xlim((self.lower_frequency-self.padding)*10**3, (self.upper_frequency+self.padding)*10**3)
        ax1.set_xlim((vertical_lines[1]-self.padding)*10**3, (vertical_lines[-2]+self.padding)*10**3)
        ax2.set_xlim((vertical_lines[1]-self.padding)*10**3, (vertical_lines[-2]+self.padding)*10**3)
        # ax1.set_ylim(10**-19,10**-15)
        # ax2.set_ylim(10**-19,10**-15)
        # ax2.set_ylim(10**-23,4*10**-23)
        # ax1.set_xlim((self.lower_frequency)*10**3, (self.upper_frequency)*10**3)
        # ax2.set_xlim((self.lower_frequency)*10**3, (self.upper_frequency)*10**3)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(4))

        labels_plot = ['data','injected', 'recovered']
        custom_lines1 = [plt.Line2D([0], [0], color='k', lw=2, linestyle='solid'),
                         plt.Line2D([0], [0], color=colors[0], lw=5, linestyle='solid', alpha=0.5),
                         plt.Line2D([0], [0], color=colors[0], lw=2, linestyle='dashed')]
        # custom_lines2 = [markers([0], [0], color=colors[0], lw=2, linestyle='solid'),
        #                 plt.Line2D([0], [0], color=colors[0], lw=2, linestyle='dashed')]
        handles, labels = ax2.get_legend_handles_labels()
        # ax2.legend([handles[0], handles[len(pGB_injected_matched)]], [labels[0],labels[len(pGB_injected_matched)]], loc='upper left')
        ax1.legend(custom_lines1, labels_plot, loc='upper left')
        # plt.legend()
        plt.tight_layout()
        if saving_label != None:
            plt.savefig(saving_label,dpi=300,bbox_inches='tight')
        plt.show()
        # print("p true", self.loglikelihood([pGB]), "null hypothesis", self.loglikelihood([null_pGBs]))

    def loglikelihood_gpu(self, parameters, start_freq_ind=0, **kwargs):
        # N_index = np.searchsorted(self.N_values,int(len(self.dataX)))
        N = 256
        self.gb_gpu.d_d = 0
        # parameters[4] *= -1
        partial_length = 1*10**4
        full_length = parameters.shape[-1]
        like = np.zeros(parameters.shape[-1])
        if len(parameters.shape) == 2:
            parameters = np.array([parameters])
        for n in range(int(full_length/partial_length)):
            like[n*partial_length:(n+1)*partial_length] = self.gb_gpu.get_ll(parameters[:,:,(n)*partial_length:(n+1)*partial_length], self.data_GPU, self.PSD_GPU, N=N, oversample=4, dt=self.dt, T=self.Tobs, start_freq_ind=start_freq_ind, tdi2=self.tdi2, t_start=self.t_start, **kwargs)
        try:
            like[int(full_length/partial_length)*partial_length:] = self.gb_gpu.get_ll(parameters[:,:,int(full_length/partial_length)*partial_length:], self.data_GPU, self.PSD_GPU, N=N, oversample=4, dt=self.dt, T=self.Tobs, start_freq_ind=start_freq_ind, tdi2=self.tdi2, t_start=self.t_start, **kwargs)
        except:
            pass
        return like
    
    def initialize_for_gaps(self, end_points, start_points, gap_less_duration, data_gaps_gpu, PSD_gaps_gpu, get_SNR):
        self.end_points = end_points
        self.start_points = start_points
        self.gap_less_duration = gap_less_duration
        self.data_gaps_gpu = data_gaps_gpu
        self.PSD_gaps_gpu = PSD_gaps_gpu
        self.get_SNR = get_SNR

    def loglikelihood_gaps_gpu(self, params, get_SNR=None, data_gaps_gpu=None):
        N = 256
        if get_SNR == None:
            get_SNR = self.get_SNR
        self.loglikelihood_list = []
        if len(np.shape(params)) == 2:
            params = np.array([params])
        else:
            params = np.array(params)
        # initial_phase = np.copy(pGB_gpu[:,4])
        for i in range(len(self.data_gaps_gpu)):
            # pGB_gpu[:,4] = initial_phase - pGB_gpu[:,1]*start_points[i] * np.pi*2 - pGB_gpu[:,2]*t_start**2 * np.pi - pGB_gpu[:,3]*t_start**3 * np.pi/3
            # pGB_gpu = np.array([np.full(num_bin, pGB_gpu[parameter]) for parameter in pGB_gpu_gpgpu])
            self.loglikelihood_list.append(self.gb_gpu.get_ll(params, self.data_gaps_gpu[i], self.PSD_gaps_gpu[i], N=N, oversample=4, dt=self.dt, T=self.end_points[i]-self.start_points[i], start_freq_ind=0, tdi2=True, t_start=self.start_points[i], get_SNR=get_SNR))

        self.total_gapless_time = np.sum(self.gap_less_duration)
        loglikelihood_average = 0
        for i in range(len(self.loglikelihood_list)):
            loglikelihood_average += self.loglikelihood_list[i] * self.gap_less_duration[i]
        loglikelihood_average /= self.total_gapless_time
        return loglikelihood_average
    
    def calculate_Amplitude_gaps(self, params):
        N = 256
        self.amplitude_list = []
        if len(np.shape(params)) == 2:
            pGB_gpu = np.array([params])
        else:
            pGB_gpu = np.array(params)
        # initial_phase = np.copy(pGB_gpu[:,4])
        for i in range(len(self.data_gaps_gpu)):
            # pGB_gpu[:,4] = initial_phase - pGB_gpu[:,1]*start_points[i] * np.pi*2 - pGB_gpu[:,2]*t_start**2 * np.pi - pGB_gpu[:,3]*t_start**3 * np.pi/3
            # pGB_gpu = np.array([np.full(num_bin, pGB_gpu[parameter]) for parameter in pGB_gpu_gpgpu])
            self.amplitude_list.append(self.gb_gpu.get_ll(pGB_gpu, self.data_gaps_gpu[i], self.PSD_gaps_gpu[i], N=N, oversample=4, dt=self.dt, T=self.end_points[i]-self.start_points[i], start_freq_ind=0, tdi2=True, t_start=self.start_points[i], get_SNR=False, get_dh_hh_ratio=True))

        self.total_gapless_time = np.sum(self.gap_less_duration)
        Amplitude_average = 0
        for i in range(len(self.amplitude_list)):
            Amplitude_average += self.amplitude_list[i] * self.gap_less_duration[i]
        Amplitude_average /= self.total_gapless_time
        return Amplitude_average
    
    def loglikelihood(self, pGBs):
        dh, hh = self.get_dh_hh(pGBs)
        logliks = dh - 0.5 * hh
        return logliks.values

    def intrinsic_SNR(self, pGBs):
        # for i in range(len(pGBs)):
        #     Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4, tdi2=self.tdi2)
        #     if i == 0:
        #         index_low = np.searchsorted(Xs.f, self.dataX.f[0])
        #         try:
        #             if np.abs(self.dataX.f[0] - Xs.f[index_low-1]) < np.abs(self.dataX.f[0] - Xs.f[index_low]):
        #                 index_low = index_low-1
        #         except:
        #             pass
        #         Xs_total = Xs[index_low : index_low + len(self.dataX)]
        #         Ys_total = Ys[index_low : index_low + len(self.dataY)]
        #         Zs_total = Zs[index_low : index_low + len(self.dataZ)]
        #     else:
        #         index_low = np.searchsorted(Xs.f, self.dataX.f[0])
        #         try:
        #             if np.abs(self.dataX.f[0] - Xs.f[index_low-1]) < np.abs(self.dataX.f[0] - Xs.f[index_low]):
        #                 index_low = index_low-1
        #         except:
        #             pass
        #         Xs_total += Xs[index_low : index_low + len(self.dataX)]
        #         Ys_total += Ys[index_low : index_low + len(self.dataY)]
        #         Zs_total += Zs[index_low : index_low + len(self.dataZ)]
            
        # Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        # Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        # if self.use_T_component:
        #     Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
        #     hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/self.SA + np.absolute(Tf.data)**2 /self.ST)
        # else:
        #     hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
        # SNR = 4.0*Xs.df* hh
        dh, hh = self.get_dh_hh(pGBs)
        return np.sqrt(hh)

    def differential_evolution_search(self, frequency_boundaries=None, initial_guess = None, number_of_signals = 1):
        bounds = []
        for signal in range(number_of_signals):
            for i in range(7):
                bounds.append((0,1))

        maxpGB = []
        self.boundaries_reduced = deepcopy(self.boundaries)
        if frequency_boundaries != None:
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
            res = differential_evolution(self.function_evolution, bounds=bounds, disp=False, strategy='best1exp', popsize=8,tol= 1e-8 , maxiter=1000, recombination= self.recombination, mutation=(0.5,1), x0=initial_guess01)
            print('time',time.time()-start)
        else:
            start = time.time()
            res = differential_evolution(self.function_evolution, bounds=bounds, disp=False, strategy='best1exp', popsize=8, tol= 1e-8 , maxiter=1000, recombination= self.recombination, mutation=(0.5,1))
            print('time',time.time()-start)
        for signal in range(number_of_signals):
            pGB01 = [0.5] + res.x[signal*7:signal*7+7].tolist()
            maxpGB.append(scaletooriginal(pGB01,self.boundaries_reduced, self.parameters, self.parameters_log_uniform))
        print(res)
        print(maxpGB)
        print('log-likelihood',self.loglikelihood(maxpGB))
        print('SNR', self.SNR(maxpGB))
        print('evolution SNR', self.function_evolution(res.x))
        # print('log-likelihood',self.loglikelihood(maxpGB))
        # print(pGB)
        return [maxpGB], res.nfev

    def searchCD(self, number_of_signals = 1):
        # np.random.seed(42)

        parameters_recorded = [None] * 10

        n_trials = 50
        number_of_evaluations = 0
        for n in range(len(parameters_recorded)):
            start = time.time()
            parameters_recorded[n] = CoordinateMC(n, self.pGBs, self.boundaries, parameters_recorded, self.SNR, n_trials=n_trials)

            print('n',n+1,'time', int(time.time()-start), np.round(parameters_recorded[n][0]['Loglikelihood'][-1],2),len(parameters_recorded[n][0]['Loglikelihood']))
            number_of_evaluations += len(parameters_recorded[n][0]['Loglikelihood']) * n_trials
        # pbar = tqdm(total=len(parameters_recorded))
        # pool = mp.Pool(mp.cpu_count())
        # parameters_recorded = pool.map(CoordinateMC, [n for n in range(len(parameters_recorded))])
        # pool.close()
        # pool.join()
        # pbar.close()

        best_run = 0
        loglikelihoodofruns = np.zeros(len(parameters_recorded))
        for i in range(len(parameters_recorded)):
            loglikelihoodofruns[i] = parameters_recorded[i][0]['Loglikelihood'][-1]
        best_value = np.max(loglikelihoodofruns)
        best_run = np.argmax(loglikelihoodofruns)
        good_runs = loglikelihoodofruns > 0
        indices = (-loglikelihoodofruns).argsort()[:len(good_runs)]
        pGBmodes = []
        for i in range(len(good_runs)):
            if good_runs[i]:
                pGBmodes.append({})
        indices = (-loglikelihoodofruns).argsort()[:len(pGBmodes)]
        pGBmodes = []
        for n in indices:
            pGBmodes.append([])
            for i in range(number_of_signals):
                pGBmodes[-1].append({})
                for parameter in self.parameters_no_amplitude + ['Loglikelihood']:
                    if parameter == 'Loglikelihood':
                        pGBmodes[-1][i][parameter] = parameters_recorded[n][0][parameter][-1]
                    else:
                        pGBmodes[-1][i][parameter] = parameters_recorded[n][i][parameter][-1]
        if len(pGBmodes) > 5:
            for signal in range(number_of_signals):
                pGBmodes = pGBmodes[:5]
        pGBmodes_opt = self.optimize_without_amplitude(pGBmodes)
        return [pGBmodes_opt], number_of_evaluations

    def optimize(self, pGBmodes, boundaries = None):
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
                # for signal in range(number_of_signals_optimize):
                #     pGBs01 = scaleto01(maxpGB[signal], self.boundaries_reduced[signal], self.parameters, self.parameters_log_uniform)
                #     for parameter in self.parameters:
                #         x.append(pGBs01[signal][parameter])

                # print(loglikelihood(maxpGB))
                res = scipy.optimize.minimize(self.function, x, args=boundaries_reduced, method='SLSQP', bounds=bounds, tol=1e-5)#, options= {'maxiter':100})
                # res = scipy.optimize.minimize(self.function, x, args=boundaries_reduced, method='Nelder-Mead', tol=1e-6)
                # res = scipy.optimize.least_squares(self.function, x, args=boundaries_reduced, bounds=bounds)
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
                res = scipy.optimize.minimize(self.function_evolution, x, method='SLSQP', bounds=bounds, tol=1e-10)
                # res = scipy.optimize.minimize(self.function, x, args=self.boundaries_reduced, method='Nelder-Mead', tol=1e-10)
                # res = scipy.optimize.least_squares(self.function, x, args=self.boundaries_reduced, bounds=bounds)
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
        dh, hh = self.get_dh_hh(pGBs)
        A = dh / hh
        return A

    def optimizeA(self, pGBmodes, boundaries = None):
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
                res = scipy.optimize.minimize(self.functiona, x, args=(pGBx, boundaries_reduced), method='trust-constr', bounds=[bounds], tol=1e-1)
                # res = scipy.optimize.minimize(self.function, x, args=boundaries_reduced, method='Nelder-Mead', tol=1e-10)
                # res = scipy.optimize.least_squares(self.function, x, args=boundaries_reduced, bounds=bounds)
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

    def function(self, pGBs01, boundaries_reduced):
        pGBs = []

        for signal in range(int(len(pGBs01)/8)):
            pGBs.append(scaletooriginal(pGBs01[signal*8:signal*8+8],boundaries_reduced[signal], self.parameters, self.parameters_log_uniform))
        # for signal in range(int(len(pGBs01)/8)):
        #     pGBs.append({})
        #     i = 0
        #     for parameter in self.parameters:
        #         if parameter in ["EclipticLatitude"]:
        #             pGBs[signal][parameter] = np.arcsin((pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
        #         elif parameter in ["Inclination"]:
        #             shifted_inclination = pGBs01[signal*8:signal*8+8][i]
        #             if pGBs01[signal*8:signal*8+8][i] < 0:
        #                 shifted_inclination = pGBs01[signal*8:signal*8+8][i] + 1
        #             if pGBs01[signal*8:signal*8+8][i] > 1:
        #                 shifted_inclination = pGBs01[signal*8:signal*8+8][i] - 1
        #             pGBs[signal][parameter] = np.arccos((shifted_inclination * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
        #         elif parameter in self.parameters_log_uniform:
        #             pGBs[signal][parameter] = 10**((pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
        #         else:
        #             pGBs[signal][parameter] = (pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0]
        #         i += 1
        p = -self.loglikelihood(pGBs)
        return p#/10**4

    def function_evolution(self, pGBs01, number_of_signals = 1):
        pGBs = []

        for signal in range(number_of_signals):
            pGB01 = [0.5] + pGBs01[signal*7:signal*7+7].tolist()
            pGBs.append(scaletooriginal(pGB01,self.boundaries_reduced, self.parameters, self.parameters_log_uniform))

        # for signal in range(number_of_signals):
        #     pGBs.append({})
        #     i = 0
        #     for parameter in self.parameters:
        #         if parameter in ["EclipticLatitude"]:
        #             pGBs[signal][parameter] = np.arcsin((pGBs01[signal*7:signal*7+7][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
        #         elif parameter in ["Inclination"]:
        #             pGBs[signal][parameter] = np.arccos((pGBs01[signal*7:signal*7+7][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
        #         elif parameter in ['Amplitude']:
        #             i -= 1
        #             pGBs[signal][parameter] = 10**((0.1 * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
        #         # elif parameter in ["FrequencyDerivative"]:
        #         #     pGBs[signal][parameter] = 10**((pGBs01[signal*7:signal*7+7][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
        #         else:
        #             pGBs[signal][parameter] = (pGBs01[signal*7:signal*7+7][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0]
        #         i += 1
        p = -self.SNR(pGBs)
        return p


    # def fisher_information(self, maxpGB):
    #     maxpGB_changed = deepcopy(maxpGB)
    #     maxpGB01 = scaleto01(maxpGB, self.boundaries, self.parameters, self.parameters_log_uniform)
    #     maxpGB01_changed = deepcopy(maxpGB01)
    #     step_size = {}
    #     pGB_low = {}
    #     pGB_high = {}
    #     derivativeAf = {}
    #     derivativeEf = {}
    #     inner_product = {}
    #     for i in range(1):
    #         for parameter in self.parameters:
    #             if i == 0:
    #                 step_size[parameter] = 1e-9
    #                 # if parameter == 'Frequency':
    #                 #     step_size[parameter] = 0.00001
    #             else:
    #                 step_size[parameter] = 0.001/np.sqrt(inner_product[parameter][parameter])
    #             # if step_size[parameter] > 1e-9:
    #             #     step_size[parameter] = 1e-9
    #             pGB_low = maxpGB01[parameter] - step_size[parameter]/2
    #             pGB_high = maxpGB01[parameter] + step_size[parameter]/2
    #             # print(parameter, step_size[parameter],i)
    #             # print(parameter, pGB_low, pGB_high)
    #             if pGB_low < 0:
    #                 pGB_low = 0
    #             if pGB_high > 1:
    #                 pGB_high = 1
    #             maxpGB01_changed[parameter] = pGB_low
    #             maxpGB_changed = scaletooriginalparameter(maxpGB01_changed,self.boundaries, self.parameters, self.parameters_log_uniform)
    #             # print(maxpGB_changed)

    #             Af_low, Ef_low = self.A_E_total([maxpGB_changed])

    #             maxpGB01_changed[parameter] = pGB_high
    #             maxpGB_changed = scaletooriginalparameter(maxpGB01_changed,self.boundaries, self.parameters, self.parameters_log_uniform)
    #             Af_high, Ef_high = self.A_E_total([maxpGB_changed])

    #             derivativeAf[parameter] = (Af_high - Af_low)/step_size[parameter]
    #             derivativeEf[parameter] = (Ef_high - Ef_low)/step_size[parameter]

    #             maxpGB01_changed[parameter] = maxpGB01[parameter]

    #         for parameter1 in self.parameters:
    #             inner_product[parameter1] = {}
    #             for parameter2 in self.parameters:
    #                 AE = derivativeAf[parameter1]*np.conjugate(derivativeAf[parameter2]) + derivativeEf[parameter1]*np.conjugate(derivativeEf[parameter2])
    #                 inner_product[parameter1][parameter2] = 4*float(np.real(np.sum(AE / self.SA) * self.dataX.df))
    #         print(step_size['Amplitude'],inner_product['Amplitude']['Amplitude'],step_size['Frequency'],inner_product['Frequency']['Frequency'])
    #     return inner_product
    
    def fisher_information(self, maxpGB):
        maxpGB_changed = deepcopy(maxpGB)
        maxpGB01 = scaleto01(maxpGB, self.boundaries, self.parameters, self.parameters_log_uniform)
        maxpGB01_changed = deepcopy(maxpGB01)
        step_size = {}
        pGB_low = {}
        pGB_high = {}
        derivativeAf = {}
        derivativeEf = {}
        inner_product = {}
        for i in range(1):
            for parameter in self.parameters:
                if i == 0:
                    step_size[parameter] = 1e-9
                    # if parameter == 'Frequency':
                    #     step_size[parameter] = 0.00001
                else:
                    step_size[parameter] = 0.001/np.sqrt(inner_product[parameter][parameter])
                # if step_size[parameter] > 1e-9:
                #     step_size[parameter] = 1e-9
                pGB_low = maxpGB01[parameter] - step_size[parameter]/2
                pGB_high = maxpGB01[parameter] + step_size[parameter]/2
                # print(parameter, step_size[parameter],i)
                # print(parameter, pGB_low, pGB_high)
                if pGB_low < 0:
                    pGB_low = 0
                if pGB_high > 1:
                    pGB_high = 1
                maxpGB01_changed[parameter] = pGB_low
                maxpGB_changed = scaletooriginalparameter(maxpGB01_changed,self.boundaries, self.parameters, self.parameters_log_uniform)
                # print(maxpGB_changed)
                Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=maxpGB_changed, oversample=4, tdi2=self.tdi2)
                index_low = np.searchsorted(Xs.f, self.dataX.f[0])
                try:
                    if np.abs(self.dataX.f[0] - Xs.f[index_low-1]) < np.abs(self.dataX.f[0] - Xs.f[index_low]):
                        index_low = index_low-1
                except:
                    pass
                Xs_total = Xs[index_low : index_low + len(self.dataX)]
                Ys_total = Ys[index_low : index_low + len(self.dataY)]
                Zs_total = Zs[index_low : index_low + len(self.dataZ)]
                Af_low = (Zs_total - Xs_total)/np.sqrt(2.0)
                Ef_low = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)

                maxpGB01_changed[parameter] = pGB_high
                maxpGB_changed = scaletooriginalparameter(maxpGB01_changed,self.boundaries, self.parameters, self.parameters_log_uniform)
                Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=maxpGB_changed, oversample=4, tdi2=self.tdi2)
                index_low = np.searchsorted(Xs.f, self.dataX.f[0])
                try:
                    if np.abs(self.dataX.f[0] - Xs.f[index_low-1]) < np.abs(self.dataX.f[0] - Xs.f[index_low]):
                        index_low = index_low-1
                except:
                    pass
                Xs_total = Xs[index_low : index_low + len(self.dataX)]
                Ys_total = Ys[index_low : index_low + len(self.dataY)]
                Zs_total = Zs[index_low : index_low + len(self.dataZ)]
                Af_high = (Zs_total - Xs_total)/np.sqrt(2.0)
                Ef_high = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)

                derivativeAf[parameter] = (Af_high - Af_low)/step_size[parameter]
                derivativeEf[parameter] = (Ef_high - Ef_low)/step_size[parameter]

                maxpGB01_changed[parameter] = maxpGB01[parameter]

            for parameter1 in self.parameters:
                inner_product[parameter1] = {}
                for parameter2 in self.parameters:
                    AE = derivativeAf[parameter1]*np.conjugate(derivativeAf[parameter2]) + derivativeEf[parameter1]*np.conjugate(derivativeEf[parameter2])
                    inner_product[parameter1][parameter2] = 4*float(np.real(np.sum(AE / self.SA[:len(AE)]) * self.dataX.df))
            print(step_size['Amplitude'],inner_product['Amplitude']['Amplitude'],step_size['Frequency'],inner_product['Frequency']['Frequency'])
        return inner_product
    


