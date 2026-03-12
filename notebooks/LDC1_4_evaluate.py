#%%
from re import A
# from matplotlib.lines import _LineStyle
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
import pickle
import sys
sys.path.append('/cluster/home/sstrub/Repositories/LDC/lib/lib64/python3.8/site-packages/ldc-0.1-py3.8-linux-x86_64.egg')

from astropy import units as u
import astropy.coordinates as coord

from ldc.lisa.noise import get_noise_model
from ldc.common.series import TimeSeries
import ldc.waveform.fastGB as fastGB
from ldc.common.tools import compute_tdi_snr, window

from sources import *

# customized settings
plot_parameter = {  # 'backend': 'ps',
    "font.family": "DeJavu Serif",
    "font.serif": "Times",
    "font.size": 16,
    "mathtext.fontset": "cm",
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

parameters = [
    "Amplitude",
    "EclipticLatitude",
    "EclipticLongitude",
    "Frequency",
    "FrequencyDerivative",
    "Inclination",
    "InitialPhase",
    "Polarization",
]
# parameters_log_uniform = ['Amplitude','FrequencyDerivative']
parameters_log_uniform = ['Amplitude']
parameters_no_amplitude = parameters[1:]
intrinsic_parameters = ['EclipticLatitude','EclipticLongitude','Frequency', 'FrequencyDerivative']

# get current directory
path = os.getcwd()

# parent directory
parent = os.path.dirname(path)
# grandparent directory
grandparent = os.path.dirname(parent)

Radler = False
version = '2'
reduction = 1
weeks = int(sys.argv[5])
print('week',weeks)
Tobs = float(weeks*7*24*3600)

if Radler:
    DATAPATH = grandparent+"/LDC/Radler/data/"
    SAVEPATH = grandparent+"/LDC/pictures/LDC1-4/"
    SAVEPATH = grandparent+"/LDC/Radler/LDC1-4_evaluation/"
else:
    DATAPATH = grandparent+"/LDC/Sangria/data/"
    SAVEPATH = grandparent+"/LDC/pictures/Sangria/"

if Radler:
    # sangria_fn = DATAPATH + "/LDC1-4_GB_v2.hdf5"
    sangria_fn = DATAPATH + "/LDC1-4_GB_v"+version+".hdf5"
    # sangria_fn = DATAPATH + "/LDC1-3_VGB_v2_FD_noiseless.hdf5"
    # sangria_fn = DATAPATH + "/LDC1-3_VGB_v2.hdf5"
else:
    sangria_fn = DATAPATH + "/LDC2_sangria_training_v2.h5"
fid = h5py.File(sangria_fn)


# get TDI 
if Radler:
    td = np.array(fid["H5LISA/PreProcess/TDIdata"])
    td = np.rec.fromarrays(list(td.T), names=["t", "X", "Y", "Z"])
    dt = float(np.array(fid['H5LISA/GWSources/GalBinaries']['Cadence']))
    Tobs = float(int(np.array(fid['H5LISA/GWSources/GalBinaries']['ObservationDuration']))/reduction)
else:
    td = fid["obs/tdi"][()]
    td = np.rec.fromarrays(list(td.T), names=["t", "X", "Y", "Z"])
    td = td['t']
    dt = int(td["t"][1]-td["t"][0])
    
    td_mbhb = fid["sky/mbhb/tdi"][()]
    # cat_mbhb = fid["sky/mbhb/cat"]
    td_mbhb  = np.rec.fromarrays(list(td_mbhb .T), names=["t", "X", "Y", "Z"])
    td_mbhb  = td_mbhb ['t']
    # tdi_ts_mbhb = dict([(k, TimeSeries(td_mbhb[k][:int(len(td_mbhb[k][:])/reduction)], dt=dt)) for k in ["X", "Y", "Z"]])
    # tdi_fs_mbhb = xr.Dataset(dict([(k, tdi_ts_mbhb[k].ts.fft(win=window)) for k in ["X", "Y", "Z"]]))

    td_dgb = fid["sky/dgb/tdi"][()]
    cat_dgb = fid["sky/dgb/cat"]
    td_dgb  = np.rec.fromarrays(list(td_dgb .T), names=["t", "X", "Y", "Z"])
    td_dgb  = td_dgb ['t']
    # tdi_ts_dgb = dict([(k, TimeSeries(td_dgb[k][:int(len(td_dgb[k][:])/reduction)], dt=dt)) for k in ["X", "Y", "Z"]])
    # tdi_fs_dgb = xr.Dataset(dict([(k, tdi_ts_dgb[k].ts.fft(win=window)) for k in ["X", "Y", "Z"]]))

    td_igb = fid["sky/igb/tdi"][()]
    cat_igb = fid["sky/igb/cat"]
    td_igb  = np.rec.fromarrays(list(td_igb .T), names=["t", "X", "Y", "Z"])
    td_igb  = td_igb ['t']
    # tdi_ts_igb = dict([(k, TimeSeries(td_igb[k][:int(len(td_igb[k][:])/reduction)], dt=dt)) for k in ["X", "Y", "Z"]])
    # tdi_fs_igb = xr.Dataset(dict([(k, tdi_ts_igb[k].ts.fft(win=window)) for k in ["X", "Y", "Z"]]))

    # td_vgb = fid["sky/vgb/tdi"][()]
    # cat_vgb = fid["sky/vgb/cat"]
    # td_vgb  = np.rec.fromarrays(list(td_vgb .T), names=["t", "X", "Y", "Z"])
    # td_vgb  = td_vgb ['t']
    # tdi_ts_vgb = dict([(k, TimeSeries(td_vgb[k][:int(len(td_vgb[k][:])/reduction)], dt=dt)) for k in ["X", "Y", "Z"]])
    # tdi_fs_vgb = xr.Dataset(dict([(k, tdi_ts_vgb[k].ts.fft(win=window)) for k in ["X", "Y", "Z"]]))

    # Tobs = float(int(np.array(fid['obs/config/t_max']))/reduction)
    for k in ["X", "Y", "Z"]:
        td[k] = td[k] - td_mbhb[k]


# Build timeseries and frequencyseries object for X,Y,Z
t_max_index = np.searchsorted(td['t'], Tobs)
tdi_ts = dict([(k, TimeSeries(td[k][:t_max_index], dt=dt, t0=td.t[0])) for k in ["X", "Y", "Z"]])
# tdi_ts = dict([(k, TimeSeries(td[k][:int(len(td[k][:])/reduction)], dt=dt, t0=td.t[0])) for k in ["X", "Y", "Z"]])
tdi_fs = xr.Dataset(dict([(k, tdi_ts[k].ts.fft(win=window)) for k in ["X", "Y", "Z"]]))
GB = fastGB.FastGB(delta_t=dt, T=Tobs)  # in seconds

if Radler:
    noise_model = "SciRDv1"
else:
    noise_model = "sangria"
Nmodel = get_noise_model(noise_model, np.logspace(-5, -1, 100))

pGB = {}
ind = 0
found_sources = []
target_sources = []
first_start = time.time()
np.random.seed(42) #40
number_of_signals = 1
signals_per_subtraction = 1

f = 0.0115248
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
def frequency_derivative2(f, Mc_s):
    return 96/5*np.pi**(8/3)*Mc_s**(5/3)*(f)**(11/3)
print('frequency derivative', frequency_derivative(f,0.1),frequency_derivative(f,2),' at f=', f)
chandrasekhar_limit = 1.4
M_chirp_upper_boundary = (chandrasekhar_limit**2)**(3/5)/(2*chandrasekhar_limit)**(1/5)


def tdi_subtraction(tdi_fs,found_sources_mp_subtract, frequencies_search):
    
    #subtract the found sources from original
    tdi_fs_subtracted2 = deepcopy(tdi_fs)
    for i in range(len(found_sources_mp_subtract)):
        # for j in range(len(found_sources_to_subtract[i])):
            if GB.T < 365.26*24*3600/4:
                freqs = np.linspace(found_sources_mp_subtract['Frequency'], found_sources_mp_subtract['Frequency']+0.00001, 128)
            Xs_subtracted, Ys_subtracted, Zs_subtracted = GB.get_fd_tdixyz(template=found_sources_mp_subtract[i], oversample=4, freqs=freqs)
            source_subtracted = dict({"X": Xs_subtracted, "Y": Ys_subtracted, "Z": Zs_subtracted})
            index_low = np.searchsorted(tdi_fs_subtracted2["X"].f, Xs_subtracted.f[0])
            index_high = index_low+len(Xs_subtracted)
            for k in ["X", "Y", "Z"]:
                tdi_fs_subtracted2[k].data[index_low:index_high] -= source_subtracted[k].data
    return tdi_fs_subtracted2

try:
    cat = np.load(SAVEPATH+'cat_sorted.npy', allow_pickle = True)
    # cat = np.load(SAVEPATH+'/cat_sorted_v'+version+'.npy', allow_pickle = True)
    print('cat sorted loaded')
except:
    # get the source parameters
    # Radler
    if Radler:
        names = np.array(fid['H5LISA/GWSources/GalBinaries']) # radler
        params = [fid['H5LISA/GWSources/GalBinaries'][k] for k in names]
        reduced_names = []
        i = 0
        for p in params:
            i += 1
            if p.shape:
                reduced_names.append(names[i-1])
        params = [np.array(p) for p in params if p.shape]
        names = reduced_names
    # Sangria
    else:
        names_dgb = fid["sky/dgb/cat"].dtype.names # Sangria
        params_dgb = [np.array(fid["sky/dgb/cat"][k]).squeeze() for k in names_dgb]
        names_igb = fid["sky/igb/cat"].dtype.names # Sangria
        params_igb = [np.array(fid["sky/igb/cat"][k]).squeeze() for k in names_igb]
        names_vgb = fid["sky/vgb/cat"].dtype.names # Sangria
        params_vgb = [np.array(fid["sky/vgb/cat"][k]).squeeze() for k in names_vgb]

    parameter_list = ['Name'] + parameters
    cat_array = []
    for parameter in parameter_list:
        cat_array.append(np.append(np.append(params_vgb[names_vgb.index(parameter)],params_igb[names_igb.index(parameter)]),params_dgb[names_dgb.index(parameter)]))
    cat = np.rec.fromarrays(cat_array, names=parameter_list)

    indexes = np.argsort(cat['Frequency'])
    cat = cat[indexes]
    np.save(SAVEPATH+'cat_sorted_all.npy',cat)

cat_sorted = cat

# cat_window = cat_sorted[cat_sorted['Frequency']>0.005208333333333333]
# cat_window = cat_window[cat_window['Frequency']<0.005208333333333333+512/62914560]


DAt = (tdi_ts['Z'] - tdi_ts['X'])/np.sqrt(2.0)
DEt = (tdi_ts['Z'] - 2.0*tdi_ts['Y'] + tdi_ts['X'])/np.sqrt(6.0)
DTt = (tdi_ts['Z'] + tdi_ts['Y'] + tdi_ts['X'])/np.sqrt(3.0)

DAf = (tdi_fs.Z - tdi_fs.X)/np.sqrt(2.0)
DEf = (tdi_fs.Z - 2.0*tdi_fs.Y + tdi_fs.X)/np.sqrt(6.0)
DTf = (tdi_fs.Z + tdi_fs.Y + tdi_fs.X)/np.sqrt(3.0)


# fig = plt.figure(figsize=fig_size)
# plt.semilogy(tdi_fs.f[tdi_fs.f<0.03][1000:]*10**3,np.abs(DAf[tdi_fs.f<0.03][1000:]), 'k')
# plt.xlabel('Frequency (mHz)')
# plt.ylabel('|A|')
# plt.show()

# fig = plt.figure(figsize=fig_size)
# plt.plot(tdi_ts['X'].t,DAt, 'k')
# plt.xlabel('Time (s)')
# plt.ylabel('TDI A')
# plt.show()


# LDC1-4 #####################################
frequencies = []
frequencies_even = []
frequencies_odd = []
# search_range = [0.00398, 0.0041]
# search_range = [0.0039885, 0.0040205]
# search_range = [0.0039935, 0.0039965]
f_Nyquist = 1/dt/2
search_range = [0.0003, f_Nyquist]
# search_range = [0.0002, 0.0003]
# search_range = [0.0019935, 0.0020135]
# search_range = [0.0029935, 0.0030135]
# window_length = 1*10**-7 # Hz

frequencies = create_frequency_windows(search_range, Tobs)


# frequencies = frequencies[:32]
frequencies_even = frequencies[::2]
frequencies_odd = frequencies[1::2]

start_index = np.searchsorted(np.asarray(frequencies_odd)[:,0], 0.0040489)-1

# save_name = 'Sangria_1_full_cut'
# save_name = 'Sangria_12m_filled_anticorrelated'

save_name_injected = 'Sangria_'+str(weeks)+'w'
seed2 = str(sys.argv[1])
save_name = 'original_'+save_name_injected+'_mbhb_SNR9_seed1'
# save_name2 = 'original_'+save_name_injected+'_mbhb_SNR9_seed'+seed2
# save_name = 'Radler_24m'
# save_name = 'Radler_24m_filled_anticorrelated'
# save_name = 'Radler_24m_redone'
# save_name = 'Radler_half_even_dynamic_noise'
# save_name = 'LDC1-4_2_optimized_second' ### ETH submission
# save_name = 'Montana'
# save_name = 'APC'
# save_name = 'LDC1-4_half_year'

# duration = '3932160'
# duration = '7864320'
# duration = '15728640'
duration = '31457280'
injected_SNR = 5
# save_name = 'Montana2022_'+duration
frequencies_search = frequencies
# batch_index = int(sys.argv[1])
batch_index = 28
start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.003977)-1
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.019)
start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.004654)-1
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.007977)
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], cat_sorted[-2]['Frequency'])-1
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.0004)-1
batch_size = 2
# start_index = batch_size*batch_index
print('batch',batch_index, start_index)
# frequencies_search = frequencies_search[start_index:start_index+batch_size]
# print(i, frequencies_search[0])
### highest + padding has to be less than f Nyqist
while frequencies_search[-1][1] + (frequencies_search[-1][1] - frequencies_search[-1][0])/2 > f_Nyquist:
    frequencies_search = frequencies_search[:-1]

frequencies_min_montana = 0.00398148268199693
frequencies_max_montana = 0.004003419101309928
frequencies_min_APC = 0.003999281
frequencies_max_APC = 0.00410031292
frequencies_min = 0.0095
frequencies_max = 0.00955
start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], frequencies_min_APC)-1
end_index = np.searchsorted(np.asarray(frequencies_search)[:,0], frequencies_max_APC)
# frequencies_search = frequencies_search[start_index:end_index]

search_range = [frequencies_search[0][0],frequencies_search[-1][1]]
# search_range = [1619472*10**-8,2689639*10**-8]
print('search range '+ str(int(np.round(search_range[0]*10**8)))+'to'+ str(int(np.round(search_range[1]*10**8))))

# i = 44
# search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i][1])
# search1.plot()

def l2_norm_match(pGB_injected, pGB_found):
    if GB.T < 365.26*24*3600/4:
        freqs = np.linspace(pGB_found['Frequency'], pGB_found['Frequency']+0.00001, 128)
    Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGB_found, oversample=4, freqs=freqs)
    Xs_injected, Ys_injected, Zs_injected = GB.get_fd_tdixyz(template=pGB_injected, oversample=4)
    Xs_aligned = xr.align(Xs_injected, Xs, join='left',fill_value=0)[1]
    Ys_aligned = xr.align(Ys_injected, Ys, join='left',fill_value=0)[1]
    Zs_aligned = xr.align(Zs_injected, Zs, join='left',fill_value=0)[1]
        
    fmin, fmax = float(Xs_injected.f[0]), float(Xs_injected.f[-1] + Xs_injected.attrs["df"])
    freq = np.array(Xs_injected.sel(f=slice(fmin, fmax)).f)
    Nmodel = get_noise_model(noise_model, freq)

    Af = (Zs_aligned - Xs_aligned)/np.sqrt(2.0)
    Ef = (Zs_aligned - 2.0*Ys_aligned + Xs_aligned)/np.sqrt(6.0)
    Af_injected = (Zs_injected - Xs_injected)/np.sqrt(2.0)
    Ef_injected = (Zs_injected - 2.0*Ys_injected + Xs_injected)/np.sqrt(6.0)
    frequency_T_threshold = 19.1*10**-3/2
    if Af.f[-1] > frequency_T_threshold:
        Tf = (Zs_aligned + Ys_aligned + Xs_aligned)/np.sqrt(3.0)
        Tf_injected = (Zs_injected + Ys_injected + Xs_injected)/np.sqrt(3.0)
        norm = np.sqrt(np.linalg.norm(Af_injected - Af.data)**2 + np.linalg.norm(Ef_injected - Ef.data)**2 + np.linalg.norm(Tf_injected - Tf.data)**2)
        norm_s = np.sqrt(np.linalg.norm(Af.data)**2 + np.linalg.norm(Ef.data)**2 + np.linalg.norm(Tf.data)**2)
        norm_relative = norm/norm_s
    else:
        norm = np.sqrt(np.linalg.norm(Af_injected - Af.data)**2 + np.linalg.norm(Ef_injected - Ef.data)**2)
        norm_s = np.sqrt(np.linalg.norm(Af.data)**2 + np.linalg.norm(Ef.data)**2)
        norm_relative = norm/norm_s
    return norm_relative

def correlation_match(pGB_injected, pGB_found):
    freqs = None
    if GB.T < 365.26*24*3600/4:
        freqs = np.linspace(pGB_found['Frequency'], pGB_found['Frequency']+0.00001, 128)
    Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGB_found, oversample=4, freqs=freqs)
    Xs_injected, Ys_injected, Zs_injected = GB.get_fd_tdixyz(template=pGB_injected, oversample=4, freqs=freqs)
    Xs_aligned = xr.align(Xs_injected, Xs, join='left',fill_value=0)[1]
    Ys_aligned = xr.align(Ys_injected, Ys, join='left',fill_value=0)[1]
    Zs_aligned = xr.align(Zs_injected, Zs, join='left',fill_value=0)[1]
        
    fmin, fmax = float(Xs_injected.f[0]), float(Xs_injected.f[-1] + Xs_injected.attrs["df"])
    freq = np.array(Xs_injected.sel(f=slice(fmin, fmax)).f)
    Nmodel = get_noise_model(noise_model, freq)
    SA = Nmodel.psd(freq=freq, option="A")
    ST = Nmodel.psd(freq=freq, option="T")

    Af = (Zs_aligned - Xs_aligned)/np.sqrt(2.0)
    Ef = (Zs_aligned - 2.0*Ys_aligned + Xs_aligned)/np.sqrt(6.0)
    Af_injected = (Zs_injected - Xs_injected)/np.sqrt(2.0)
    Ef_injected = (Zs_injected - 2.0*Ys_injected + Xs_injected)/np.sqrt(6.0)
    frequency_T_threshold = 19.1*10**-3/2
    if Af.f[-1] > frequency_T_threshold:
        Tf = (Zs_aligned + Ys_aligned + Xs_aligned)/np.sqrt(3.0)
        Tf_injected = (Zs_injected + Ys_injected + Xs_injected)/np.sqrt(3.0)
        SNR2 = np.sum( np.real(Af_injected * np.conjugate(Af.data) + Ef_injected * np.conjugate(Ef.data))/SA + np.real(Tf_injected * np.conjugate(Tf.data))/ST)
        hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/SA + np.absolute(Tf.data)**2 /ST)
        ss = np.sum((np.absolute(Af_injected.data)**2 + np.absolute(Ef_injected.data)**2)/SA + np.absolute(Tf_injected.data)**2 /ST)
    else:
        SNR2 = np.sum( np.real(Af_injected * np.conjugate(Af.data) + Ef_injected * np.conjugate(Ef.data))/SA)
        hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /SA)
        ss = np.sum((np.absolute(Af_injected.data)**2 + np.absolute(Ef_injected.data)**2) /SA)
    SNR = 4.0*Xs.df* hh
    SNR2 = 4.0*Xs.df* SNR2
    SNR3 = SNR2 / (np.sqrt(SNR)*np.sqrt(4.0*Xs.df* ss))
    return SNR3.values

def SNR_match_scaled(pGB_injected, pGB_found):
    freqs = None
    if GB.T < 365.26*24*3600/4:
        freqs = np.linspace(pGB_found['Frequency'], pGB_found['Frequency']+0.00001, 128)
    Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGB_found, oversample=4, freqs=freqs)
    Xs_injected, Ys_injected, Zs_injected = GB.get_fd_tdixyz(template=pGB_injected, oversample=4, freqs=freqs)
    Xs_aligned = xr.align(Xs_injected, Xs, join='left',fill_value=0)[1]
    Ys_aligned = xr.align(Ys_injected, Ys, join='left',fill_value=0)[1]
    Zs_aligned = xr.align(Zs_injected, Zs, join='left',fill_value=0)[1]
        
    fmin, fmax = float(Xs_injected.f[0]), float(Xs_injected.f[-1] + Xs_injected.attrs["df"])
    freq = np.array(Xs_injected.sel(f=slice(fmin, fmax)).f)
    Nmodel = get_noise_model(noise_model, freq)
    SA = Nmodel.psd(freq=freq, option="A")
    ST = Nmodel.psd(freq=freq, option="T")

    Af = (Zs_aligned - Xs_aligned)/np.sqrt(2.0)
    Ef = (Zs_aligned - 2.0*Ys_aligned + Xs_aligned)/np.sqrt(6.0)
    Af_injected = (Zs_injected - Xs_injected)/np.sqrt(2.0)
    Ef_injected = (Zs_injected - 2.0*Ys_injected + Xs_injected)/np.sqrt(6.0)
    frequency_T_threshold = 19.1*10**-3/2
    if Af.f[-1] > frequency_T_threshold:
        Tf = (Zs_aligned + Ys_aligned + Xs_aligned)/np.sqrt(3.0)
        Tf_injected = (Zs_injected + Ys_injected + Xs_injected)/np.sqrt(3.0)
        SNR2 = np.sum((np.absolute(Af_injected-Af.data)**2 + np.absolute(Ef_injected-Ef.data)**2)/SA + np.absolute(Tf_injected-Tf.data)**2 /ST)
        # SNR2_complex_mult = np.sum( (np.real((Af_injected-Af.data) * np.conjugate((Af_injected-Af.data))) + np.real((Ef_injected-Ef.data) * np.conjugate((Ef_injected-Ef.data))))/SA + np.real((Tf_injected-Tf.data) * np.conjugate(Tf_injected-Tf.data))/ST)
        hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/SA + np.absolute(Tf.data)**2 /ST)
        ss = np.sum((np.absolute(Af_injected.data)**2 + np.absolute(Ef_injected.data)**2)/SA + np.absolute(Tf_injected.data)**2 /ST)
    else:
        # SNR2 = np.sum( np.real((Af_injected-Af.data) * np.conjugate((Af_injected-Af.data)) + (Ef_injected-Ef.data) * np.conjugate((Ef_injected-Ef.data)))/SA)
        SNR2 = np.sum((np.absolute(Af_injected-Af.data)**2 + np.absolute(Ef_injected-Ef.data)**2)/SA)
        hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /SA)
        ss = np.sum((np.absolute(Af_injected.data)**2 + np.absolute(Ef_injected.data)**2) /SA)
    # SNR = 4.0*Xs.df* hh
    SNR3 = SNR2/(np.sqrt(ss)*np.sqrt(hh))
    return SNR3.values

def SNR_match(pGB_injected, pGB_found):
    freqs = None
    if GB.T < 365.26*24*3600/4:
        freqs = np.linspace(pGB_found['Frequency'], pGB_found['Frequency']+0.00001, 128)
    Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGB_found, oversample=4, freqs=freqs)
    Xs_injected, Ys_injected, Zs_injected = GB.get_fd_tdixyz(template=pGB_injected, oversample=4, freqs=freqs)
    Xs_aligned = xr.align(Xs_injected, Xs, join='left',fill_value=0)[1]
    Ys_aligned = xr.align(Ys_injected, Ys, join='left',fill_value=0)[1]
    Zs_aligned = xr.align(Zs_injected, Zs, join='left',fill_value=0)[1]
        
    fmin, fmax = float(Xs_injected.f[0]), float(Xs_injected.f[-1] + Xs_injected.attrs["df"])
    freq = np.array(Xs_injected.sel(f=slice(fmin, fmax)).f)
    Nmodel = get_noise_model(noise_model, freq)
    SA = Nmodel.psd(freq=freq, option="A")
    ST = Nmodel.psd(freq=freq, option="T")

    Af = (Zs_aligned - Xs_aligned)/np.sqrt(2.0)
    Ef = (Zs_aligned - 2.0*Ys_aligned + Xs_aligned)/np.sqrt(6.0)
    Af_injected = (Zs_injected - Xs_injected)/np.sqrt(2.0)
    Ef_injected = (Zs_injected - 2.0*Ys_injected + Xs_injected)/np.sqrt(6.0)
    frequency_T_threshold = 19.1*10**-3/2
    if Af.f[-1] > frequency_T_threshold:
        Tf = (Zs_aligned + Ys_aligned + Xs_aligned)/np.sqrt(3.0)
        Tf_injected = (Zs_injected + Ys_injected + Xs_injected)/np.sqrt(3.0)
        SNR2 = np.sum((np.absolute(Af_injected-Af.data)**2 + np.absolute(Ef_injected-Ef.data)**2)/SA + np.absolute(Tf_injected-Tf.data)**2 /ST)
        # SNR2_complex_mult = np.sum( (np.real((Af_injected-Af.data) * np.conjugate((Af_injected-Af.data))) + np.real((Ef_injected-Ef.data) * np.conjugate((Ef_injected-Ef.data))))/SA + np.real((Tf_injected-Tf.data) * np.conjugate(Tf_injected-Tf.data))/ST)
        # hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/SA + np.absolute(Tf.data)**2 /ST)
        # ss = np.sum((np.absolute(Af_injected.data)**2 + np.absolute(Ef_injected.data)**2)/SA + np.absolute(Tf_injected.data)**2 /ST)
    else:
        # SNR2 = np.sum( np.real((Af_injected-Af.data) * np.conjugate((Af_injected-Af.data)) + (Ef_injected-Ef.data) * np.conjugate((Ef_injected-Ef.data)))/SA)
        SNR2 = np.sum((np.absolute(Af_injected-Af.data)**2 + np.absolute(Ef_injected-Ef.data)**2)/SA)
        # hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /SA)
        # ss = np.sum((np.absolute(Af_injected.data)**2 + np.absolute(Ef_injected.data)**2) /SA)
    # SNR = 4.0*Xs.df* hh
    SNR3 = SNR2
    return SNR3.values

def SNR_match_amplitude_condsiered(pGB_injected, pGB_found):
    if GB.T < 365.26*24*3600/4:
        freqs = np.linspace(pGB_found['Frequency'], pGB_found['Frequency']+0.00001, 128)
    Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGB_found, oversample=4, freqs=freqs)
    Xs_injected, Ys_injected, Zs_injected = GB.get_fd_tdixyz(template=pGB_injected, oversample=4, freqs=freqs)
    Xs_aligned = xr.align(Xs_injected, Xs, join='left',fill_value=0)[1]
    Ys_aligned = xr.align(Ys_injected, Ys, join='left',fill_value=0)[1]
    Zs_aligned = xr.align(Zs_injected, Zs, join='left',fill_value=0)[1]
        
    fmin, fmax = float(Xs_injected.f[0]), float(Xs_injected.f[-1] + Xs_injected.attrs["df"])
    freq = np.array(Xs_injected.sel(f=slice(fmin, fmax)).f)
    Nmodel = get_noise_model(noise_model, freq)
    SA = Nmodel.psd(freq=freq, option="A")
    ST = Nmodel.psd(freq=freq, option="T")

    Af = (Zs_aligned - Xs_aligned)/np.sqrt(2.0)
    Ef = (Zs_aligned - 2.0*Ys_aligned + Xs_aligned)/np.sqrt(6.0)
    Af_injected = (Zs_injected - Xs_injected)/np.sqrt(2.0)
    Ef_injected = (Zs_injected - 2.0*Ys_injected + Xs_injected)/np.sqrt(6.0)
    frequency_T_threshold = 19.1*10**-3/2
    if Af.f[-1] > frequency_T_threshold:
        Tf = (Zs_aligned + Ys_aligned + Xs_aligned)/np.sqrt(3.0)
        Tf_injected = (Zs_injected + Ys_injected + Xs_injected)/np.sqrt(3.0)
        SNR2 = np.sum( np.real(Af_injected * np.conjugate(Af.data) + Ef_injected * np.conjugate(Ef.data))/SA + np.real(Tf_injected * np.conjugate(Tf.data))/ST)
        hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/SA + np.absolute(Tf.data)**2 /ST)
        ss = np.sum((np.absolute(Af_injected.data)**2 + np.absolute(Ef_injected.data)**2)/SA + np.absolute(Tf_injected.data)**2 /ST)
    else:
        SNR2 = np.sum( np.real(Af_injected * np.conjugate(Af.data) + Ef_injected * np.conjugate(Ef.data))/SA)
        hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /SA)
        ss = np.sum((np.absolute(Af_injected.data)**2 + np.absolute(Ef_injected.data)**2) /SA)
    SNR = 4.0*Xs.df* hh
    scalar_ss = 4.0*Xs.df* ss
    SNR2 = 4.0*Xs.df* SNR2
    amplitude_factor = np.abs(np.log10(scalar_ss/SNR))/2+1
    SNR3 = SNR2 / (np.sqrt(SNR)*np.sqrt(scalar_ss))/amplitude_factor
    return SNR3.values, amplitude_factor, SNR2 / (np.sqrt(SNR)*np.sqrt(scalar_ss))

try:
    found_sources_in_flat = np.load(SAVEPATH+'found_sources_' +save_name+'_flat.pkl', allow_pickle = True)
    # found_sources_in_flat = np.asarray(found_sources_in_flat[:-1])
except:
    found_sources_mp = np.load(SAVEPATH+'found_sources_' +save_name+'.pkl', allow_pickle = True)
    # found_sources_mp = np.load(SAVEPATH+'found_sources_' +save_name+'.npy', allow_pickle = True)

    # found_sources_mp_best = []
    # found_sources_mp_all = []
    # for i in range(len(found_sources_mp)):
    #     found_sources_mp_best.append(found_sources_mp[i][0])
    #     found_sources_in_window = []
    #     for j in range(len(found_sources_mp[i][1])):
    #         found_sources_in_window.append(found_sources_mp[i][1][j][0][0])
    #     found_sources_mp_all.append(found_sources_in_window)

    found_sources_in_flat = []
    for i in range(len(found_sources_mp)):
        for j in range(len(found_sources_mp[i][3])):
            found_sources_in_flat.append(found_sources_mp[i][3][j])
    found_sources_in_flat = np.asarray(found_sources_in_flat)

    np.save(SAVEPATH+'found_sources_' +save_name+'_flat.pkl', np.asarray(found_sources_in_flat))

sort_found = False
if sort_found:
    found_sources_in_flat_frequency = []
    for i in range(len(found_sources_in_flat)):
        found_sources_in_flat_frequency.append(found_sources_in_flat[i]['Frequency'])
    found_sources_in_flat_frequency = np.asarray(found_sources_in_flat_frequency)
    found_sources_in_flat = np.asarray(found_sources_in_flat)
    indexes_in = np.argsort(found_sources_in_flat_frequency)
    found_sources_in_flat_frequency = found_sources_in_flat_frequency[indexes_in]
    found_sources_in_flat = found_sources_in_flat[indexes_in]

    found_sources_in_flat_array = {attribute: np.asarray([x[attribute] for x in found_sources_in_flat]) for attribute in parameters}
    found_sources_in_flat_df = pd.DataFrame(found_sources_in_flat_array)
    found_sources_in_flat_df = found_sources_in_flat_df.sort_values('Frequency')
    found_sources_in = []
    for i in range(len(frequencies_search)):
        found_sources_in.append(found_sources_in_flat_df[(found_sources_in_flat_df['Frequency'] > frequencies_search[i][0]) & (found_sources_in_flat_df['Frequency'] < frequencies_search[i][1])].to_dict(orient='records'))

    new_dt = np.dtype(cat_sorted.dtype.descr + [('IntrinsicSNR','<f8')])
    cat_sorted_SNR = np.zeros(cat_sorted.shape, dtype=cat_sorted.dtype.descr + [('IntrinsicSNR','<f8')])
    for parameter in parameters:
        cat_sorted_SNR[parameter] = cat_sorted[parameter]
    cat_sorted = cat_sorted_SNR

# pGB_injected_SNR = np.load(SAVEPATH+'/found_sources_pGB_injected_in_out_intrinsic_SNR_sorted'+ str(int(np.round(search_range[0]*10**8)))+'to'+ str(int(np.round(search_range[1]*10**8))) +save_name+'.npy', allow_pickle = True)
# pGB_injected_SNR = np.load(SAVEPATH+'/found_sources_pGB_injected_in_out_intrinsic_SNR_sortedsecond'+ str(int(np.round(search_range[0]*10**8)))+'to'+ str(int(np.round(search_range[1]*10**8))) +save_name+'.npy', allow_pickle = True)

# pGB_injected_no_overlap = []
# pGB_injected_flat = []
# for i in range(len(pGB_injected_SNR)):
#     pGB_injected_no_overlap.append([])
#     list_count = 0
#     for j in range(len(pGB_injected_SNR[i])):
#         if pGB_injected_SNR[i][j]['Frequency'] > frequencies_search[i][0] and pGB_injected_SNR[i][j]['Frequency'] < frequencies_search[i][1]:
#             pGB_injected_flat.append(pGB_injected_SNR[i][j])
#             pGB_injected_no_overlap[-1].append(pGB_injected_SNR[i][j])
#             list_count += 1
#         if list_count > 35:
#             break
# pGB_injected_flat_array = np.asarray(pGB_injected_flat)
# pGB_injected_SNR = np.load(SAVEPATH+'/found_sources_pGB_injected_in_out_intrinsic_SNR_sorted30000to3316929LDC1-4_half_year.npy', allow_pickle = True)

# for i in range(len(pGB_injected)):
#     pGB_injected[i] = np.asarray(pGB_injected[i])
# pGB_injected_flat = np.concatenate(np.asarray(pGB_injected))

def get_SNR(pGB_injected, lower_frequency, upper_frequency):
    search1 = Search(tdi_fs,Tobs, lower_frequency, upper_frequency, update_noise=False)
    intrinsic_SNR_injected = []
    # print(lower_frequency)
    for i in range(len(pGB_injected)):
        pGB_injected_dict = {}
        for parameter in parameters:
            try:
                pGB_injected_dict[parameter] = pGB_injected[i][parameter]
            except:
                pGB_injected_dict[parameter] = pGB_injected.iloc[i][parameter]
        intrinsic_SNR_injected.append(search1.intrinsic_SNR([pGB_injected_dict]))
        if i > 30:
            break
    return intrinsic_SNR_injected

try:
    # found_sources_in = np.load(SAVEPATH+'/found_sources_in_SNR_'+save_name+'.npy', allow_pickle= True)
    fn = SAVEPATH+'/found_sources_in_SNR_'+save_name+'.pkl'
    found_sources_in = pickle.load(open(fn, 'rb'))

# delete 7400 element from montana solution
# found_sources_in[5218] = np.delete(found_sources_in[5218],2)
# np.save(SAVEPATH+'/found_sources_in_SNR'+ str(int(np.round(search_range[0]*10**8)))+'to'+ str(int(np.round(search_range[1]*10**8))) +save_name+'.npy', np.asarray(found_sources_in))

except:
    #### parallel found sources
    input = []
    index = []
    for i in range(len(found_sources_in)):
    # for i in range(16):
        if len(found_sources_in[i]) > 0:
            index.append(i)
            input.append((found_sources_in[i],frequencies_search[i][0], frequencies_search[i][1]))
    start = time.time()
    pool = mp.Pool(16)
    SNR_intrinsic = pool.starmap(get_SNR, input)
    pool.close()
    pool.join()
    print('time to calculate SNR for', len(frequencies_search), 'windows: ', time.time()-start)
    for i in range(len(SNR_intrinsic)):
        for j in range(len(SNR_intrinsic[i])):
            if len(found_sources_in[index[i]]) > 0:
                found_sources_in[index[i]][j]['IntrinsicSNR'] = SNR_intrinsic[i][j]

    fn = SAVEPATH+'/found_sources_in_SNR_'+save_name+'.pkl'
    pickle.dump(found_sources_in, open(fn, "wb"))
    # np.save(SAVEPATH+'/found_sources_in_SNR'+ str(int(np.round(search_range[0]*10**8)))+'to'+ str(int(np.round(search_range[1]*10**8))) +save_name+'.npy',found_sources_in)

# found_sources_in = np.load(SAVEPATH+'/found_sources_in_SNR30000to3312283Montana2022_31457280.npy', allow_pickle= True)

# try:
#     fn = SAVEPATH+'/pGB_injected_no_SNR' +save_name_injected+'.pkl'
#     pGB_injected = pickle.load(open(fn, 'rb'))
#     # pGB_injected = np.load(SAVEPATH+'/found_sources_pGB_injected_no_SNR'+ str(int(np.round(search_range[0]*10**8)))+'to'+ str(int(np.round(search_range[1]*10**8))) +save_name_injected+'.npy', allow_pickle = True)
# except:
#     get_injected_sources = pd.DataFrame(cat_sorted)
#     pGB_injected = []
#     for j in range(len(frequencies_search)):
#         if j%10 == 0:
#             print(j)
#         # padding = (frequencies_search[j][1] - frequencies_search[j][0])/2*0
#         # index_low = np.searchsorted(get_injected_sources['Frequency'], frequencies_search[j][0])
#         # index_high = np.searchsorted(get_injected_sources['Frequency'], frequencies_search[j][1])
#         # try:
#         #     if get_injected_sources['Frequency'][index_high] < frequencies_search[j][1]:
#         #         index_high -= 1
#         # except:
#         #     pass
#         # pGB_injected.append(get_injected_sources[index_low:index_high].sort_values(by='Amplitude', ascending=False))

#         pGB_injected.append(get_injected_sources[(get_injected_sources['Frequency']> frequencies_search[j][0]) & (get_injected_sources['Frequency']< frequencies_search[j][1])].sort_values(by='Amplitude', ascending=False))


#     fn = SAVEPATH+'/pGB_injected_no_SNR' +save_name_injected+'.pkl'
#     pickle.dump(pGB_injected, open(fn, "wb"))
#     # np.save(SAVEPATH+'/found_sources_pGB_injected_no_SNR'+ str(int(np.round(search_range[0]*10**8)))+'to'+ str(int(np.round(search_range[1]*10**8))) +save_name+'.npy', np.asarray(pGB_injected))

# number_of_injected_signals= 0
# for i in range(len(pGB_injected)):
#     number_of_injected_signals += len(pGB_injected[i])

# pGB_injected_flat = np.concatenate(pGB_injected)



#### parallel
# for i in range(len(pGB_injected)):
#     if i in index:
#         continue
#     else:
#         pGB_injected[i]['IntrinsicSNR'] = np.zeros(len(pGB_injected[i]))
# input = []
# index = []
# for i in range(len(pGB_injected)):
# # for i in range(16):
#     if i == 6021:
#         continue
#     if len(pGB_injected[i]) > 0:
#         index.append(i)
#         input.append((pGB_injected[i],frequencies_search[i][0], frequencies_search[i][1]))
# start = time.time()
# pool = mp.Pool(16)
# SNR_intrinsic = pool.starmap(get_SNR, input)
# pool.close()
# pool.join()
# print('time to calculate SNR for', len(frequencies_search), 'windows: ', time.time()-start)
# # np.save(SAVEPATH+'/SNR_intrinsic' +save_name+'.npy', np.asarray(SNR_intrinsic))
# # np.save(SAVEPATH+'/index_intrinsic' +save_name+'.npy', np.asarray(index))
# for i in range(len(SNR_intrinsic)):
#     for j in range(len(SNR_intrinsic[i])):
#         if len(pGB_injected[index[i]]) > 0:
#             pGB_injected[index[i]]['IntrinsicSNR'].iloc[j] = SNR_intrinsic[i][j]
# # outside of normal range
# for i in range(len(pGB_injected)):
#     pGB_injected[i] = pGB_injected[i][:30]

# for i in range(len(pGB_injected)):
#     pGB_injected[i] = pGB_injected[i].to_records()

# pGB_injected = list(pGB_injected)

# pd.options.mode.chained_assignment = None 
# # #### sequential
# for i in range(len(pGB_injected)):
#     print(i)
#     pGB_injected[i]['IntrinsicSNR'] = np.zeros(len(pGB_injected[i]))
#     # if i != 5:
#     #     continue
#     if len(pGB_injected[i]) > 0:
#         search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i][1])
#     for j in range(len(pGB_injected[i])):
#         pGB_injected_dict = {}
#         for parameter in parameters:
#             pGB_injected_dict[parameter] = pGB_injected[i].iloc[j][parameter]
#         pGB_injected[i]['IntrinsicSNR'].iloc[j] = search1.intrinsic_SNR([pGB_injected_dict])
#         if j > 30:
#             break
        # print('SNR for noise model', noise_model, intrinsic_SNR_injected[-1], 'loglikelihood ratio',search1.loglikelihood([pGB_injected[i][j]]), 'SNR data',search1.loglikelihood_SNR([pGB_injected[i][j]]))

# fn = SAVEPATH+'/pGB_injected_intrinsic_SNR' +save_name_injected+'.pkl'
# pickle.dump(pGB_injected, open(fn, "wb"))

# pGB_injected_SNR_sorted = []
# for i in range(len(pGB_injected)):
#     indexesSNR = np.argsort(-pGB_injected[i]['IntrinsicSNR'])
#     pGB_injected_SNR_sorted.append(pGB_injected[i].iloc[indexesSNR])
# pGB_injected = pGB_injected_SNR_sorted

# fn = SAVEPATH+'/pGB_injected_intrinsic_SNR_sorted' +save_name+'.pkl'
# pickle.dump(pGB_injected, open(fn, "wb"))

# np.save(SAVEPATH+'/pGB_injected_intrinsic_SNR_sorted'+ str(int(np.round(search_range[0]*10**8)))+'to'+ str(int(np.round(search_range[1]*10**8))) +save_name+'.npy', np.asarray(pGB_injected_SNR_sorted))
# pGB_injected = np.load(SAVEPATH+'/found_sources_pGB_injected'+ str(int(np.round(search_range[0]*10**8)))+'to'+ str(int(np.round(search_range[1]*10**8))) +save_name+'.npy', allow_pickle = True)
# pGB_injected_even = np.load(SAVEPATH+'/found_sources_pGB_injected_array2_half_year30000to3305084LDC1-4_half_even_T'+'.npy', allow_pickle = True)
# pGB_injected_odd = np.load(SAVEPATH+'/found_sources_pGB_injected_in_out_intrinsic_SNR_sorted30035to3316929LDC1-4_half_odd_T'+'.npy', allow_pickle = True)

if Radler:
    if reduction == 1:
        # pGB_injected = np.load(DATAPATH+'/pGB_injectedthird30000to3293090LDC1-4_2_years_full.npy', allow_pickle = True)
        pGB_injected = np.load(SAVEPATH+'/pGB_injected_intrinsic_SNR_sorted30000to3293090LDC1-4_2_optimized_second.npy', allow_pickle = True)
    elif reduction == 2:
        pGB_injected = np.load(SAVEPATH+'/pGB_injected_intrinsic_SNR_sorted30000to3312283Montana2022_31457280.npy', allow_pickle = True)
    elif reduction == 4:
        pGB_injected = np.load(SAVEPATH+'/pGB_injected_intrinsic_SNR_sorted30000to3316929LDC1-4_half_year.npy', allow_pickle = True)
    for i in range(len(pGB_injected)):
        pGB_injected[i] = pGB_injected[i].to_records()
else:
    # fn = SAVEPATH+'/pGB_injected_intrinsic_SNR' +save_name_injected+'.pkl'
    fn = SAVEPATH+'/pGB_injected_intrinsic_SNRSangria_12m.pkl'
    pGB_injected = pickle.load(open(fn, 'rb'))
    # for i in range(len(pGB_injected)):
    #     pGB_injected[i] = pGB_injected[i].to_records()


get_pGB_injected = True
if get_pGB_injected:
    ### get injected on desired windows
    pGB_injected_flat = np.concatenate(pGB_injected)
    indexes_in = np.argsort(pGB_injected_flat['Frequency'])
    pGB_injected_flat = pGB_injected_flat[indexes_in]

    pGB_injected = []
    for j in range(len(frequencies_search)):
        # padding = (frequencies_search[j][1] - frequencies_search[j][0])/2*0
        index_low = np.searchsorted(pGB_injected_flat['Frequency'], frequencies_search[j][0])
        index_high = np.searchsorted(pGB_injected_flat['Frequency'], frequencies_search[j][1])
        try:
            if pGB_injected_flat['Frequency'][index_high] < frequencies_search[j][1]:
                index_high -= 1
        except:
            pass
        pGB_injected.append(pGB_injected_flat[index_low:index_high])
    # pGB_injected = np.array(pGB_injected)
    for i in range(len(pGB_injected)):
        # pGB_injected_amplitude = pGB_injected[i]['IntrinsicSNR']
        pGB_injected[i] = pGB_injected[i][np.argsort(-pGB_injected[i]['IntrinsicSNR'])]

    pGB_injected_no_overlap = []
    for i in range(len(frequencies_search)):
        pGB_injected_no_overlap.append(pGB_injected[i][(pGB_injected[i]['Frequency']> frequencies_search[i][0]) & (pGB_injected[i]['Frequency']< frequencies_search[i][1])])

    ### get injected windows with overlaps
    # pGB_injected = np.concatenate((pGB_injected_even, pGB_injected_odd))

    pGB_injected_flat = np.concatenate(pGB_injected)
    indexesfrequency = np.argsort(pGB_injected_flat['Frequency'])
    pGB_injected_flat = pGB_injected_flat[indexesfrequency]
    get_injected_sources = pGB_injected_flat

    pGB_injected_overlap = []
    for j in range(len(frequencies_search)):
        padding = (frequencies_search[j][1] - frequencies_search[j][0])/2
        index_low = np.searchsorted(get_injected_sources['Frequency'], frequencies_search[j][0]-padding)
        index_high = np.searchsorted(get_injected_sources['Frequency'], frequencies_search[j][1]+padding)
        try:
            if get_injected_sources['Frequency'][index_high] < frequencies_search[j][1]:
                index_high -= 1
        except:
            pass
        # indexesA = np.argsort(-get_injected_sources[index_low:index_high]['Amplitude'])
        pGB_injected_overlap.append(get_injected_sources[index_low:index_high])

    pGB_injected_flat_overlap = np.concatenate(pGB_injected_overlap)
    pGB_injected_SNR_sorted_overlap = []
    for i in range(len(pGB_injected_overlap)):
        indexesSNR = np.argsort(-pGB_injected_overlap[i]['IntrinsicSNR'])
        # indexesSNR = np.argsort(-pGB_injected_overlap[i]['Amplitude'])
        pGB_injected_SNR_sorted_overlap.append(pGB_injected_overlap[i][indexesSNR])

    ### Reduce the number of signals per window
    for i in range(len(pGB_injected_overlap)):
        pGB_injected_SNR_sorted_overlap[i] = pGB_injected_SNR_sorted_overlap[i][:50]

    ### just take the SNR > threshold signals
    for i in range(len(pGB_injected_overlap)):
        pGB_injected_SNR_sorted_overlap[i] = pGB_injected_SNR_sorted_overlap[i][pGB_injected_SNR_sorted_overlap[i]['IntrinsicSNR'] > injected_SNR]

# i = 6837
# pGB_injected_overlap = pGB_injected_SNR_sorted_overlap
# pGB_injected_overlap_flat = np.concatenate(pGB_injected_overlap)

try:
    fn = SAVEPATH+'/found_sources_not_anticorrelated_'+save_name+'.pkl'
    found_sources_in = pickle.load(open(fn, 'rb'))
    # fn = SAVEPATH+'/found_sources_not_anticorrelated_'+save_name2+'.pkl'
    # found_sources_in2 = pickle.load(open(fn, 'rb'))
except:
    threshold_overlap = -0.7
    found_sources_anitcorrelated = []
    found_sources_not_anitcorrelated = deepcopy(found_sources_in)
    number_of_matched_signals = 0
    correlation_list = []
    start = time.time()
    percentage = 0
    for i in range(len(found_sources_in)):
        found_sources_anitcorrelated.append([])
        # if i > 100:
        #     continue
        try:
            if i%int(len(found_sources_in)/100) == 0:
                percentage += 1
                print(percentage,'%')
        except:
            pass
        for j in range(len(found_sources_in[i])):
            found_match = False
            found_match_max = False
            correlation_list_of_one_signal = []
            for k in range(len(found_sources_in[i])):
                if k == j:
                    continue
                found_second_dict = {}
                found_dict = {}
                for parameter in parameters:
                    found_second_dict[parameter] = found_sources_in[i][k][parameter]
                    found_dict[parameter] = found_sources_in[i][j][parameter]
                correlation = correlation_match(found_second_dict,found_dict)
                correlation_list_of_one_signal.append(correlation)
                if k > 19:
                    print('k')
                    break
            if 0 == len(correlation_list_of_one_signal):
                break
            min_index = np.argmin(correlation_list_of_one_signal)
            if correlation_list_of_one_signal[min_index] < -0.7:
                found_match = True
            if found_match:
                found_sources_anitcorrelated[-1].append(found_sources_in[i][j])
                correlation_list.append(correlation_list_of_one_signal[min_index])
                found_sources_not_anitcorrelated[i][j] = None
                found_sources_not_anitcorrelated[i][min_index] = None
                number_of_matched_signals += 1

            max_index = np.argmax(correlation_list_of_one_signal)
            if correlation_list_of_one_signal[max_index] > 0.9:
                found_match_max = True
            if found_match_max:
                found_sources_anitcorrelated[-1].append(found_sources_in[i][j])
                correlation_list.append(correlation_list_of_one_signal[max_index])
                found_sources_not_anitcorrelated[i][j] = None
                found_sources_not_anitcorrelated[i][max_index] = None
                number_of_matched_signals += 1
    print('time to match', time.time()-start)
    for i in range(len(found_sources_not_anitcorrelated)):
        found_sources_not_anitcorrelated[i] = list(filter(None, found_sources_not_anitcorrelated[i]))
    found_sources_not_anitcorrelated_flat = np.concatenate(found_sources_not_anitcorrelated)
    found_sources_anticorrelated_flat = np.concatenate(found_sources_anitcorrelated)

    # index = np.searchsorted(found_sources_in_flat_frequency, 0.003688)
    # index_low = np.searchsorted(pGB_injected_flat['Frequency'], 0.003688)
    # index_frequencies = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.003688)
    # pGB_injected_flat[index_low]
    # found_sources_in_flat[index]
    # corr = correlation_match(found_sources_in_flat[index-2],found_sources_in_flat[index-1])

    found_sources_not_anitcorrelated2 = deepcopy(found_sources_in)
    correlation_list2 = []
    for i in range(int(len(found_sources_anticorrelated_flat)/2)):
        index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0], found_sources_anticorrelated_flat[i*2]['Frequency'])-1
        for j in range(len(found_sources_in[index_of_interest_to_plot])):
            found_match = False
            correlation_list_of_one_signal = []
            for k in range(len(found_sources_in[index_of_interest_to_plot])):
                found_second_dict = {}
                found_dict = {}
                for parameter in parameters:
                    found_second_dict[parameter] = found_sources_in[index_of_interest_to_plot][k][parameter]
                    found_dict[parameter] = found_sources_in[index_of_interest_to_plot][j][parameter]
                correlation = correlation_match(found_second_dict,found_dict)
                correlation_list_of_one_signal.append(correlation)
                if k > 19:
                    print('k')
                    break
            if 0 == len(correlation_list_of_one_signal):
                break
            max_index = np.argmin(correlation_list_of_one_signal)
            if correlation_list_of_one_signal[max_index] < threshold_overlap:
                found_match = True
            if found_match:
                print('found anti',index_of_interest_to_plot,j,max_index)
                correlation_list2.append(correlation_list_of_one_signal[max_index])
                found_sources_not_anitcorrelated2[index_of_interest_to_plot][j] = None
                found_sources_not_anitcorrelated2[index_of_interest_to_plot][max_index] = None
    for i in range(len(found_sources_not_anitcorrelated2)):
        found_sources_not_anitcorrelated2[i] = list(filter(None, found_sources_not_anitcorrelated2[i]))

    ## determine index of a specific frequency
    for j in range(int(len(found_sources_anticorrelated_flat)/2)):
        # if j != 0:
        #     continue
        index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0], found_sources_anticorrelated_flat[j*2]['Frequency'])-1
        #### plot strains
        # for i in range(len(frequencies_search)):
        #     if i != index_of_interest_to_plot:
        #         continue
        #     search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i][1])
        #     found_extended = found_sources_not_anitcorrelated2[i]
        #     if len(pGB_injected_SNR_sorted_overlap[i]) > 20:
        #         search1.plot(found_sources_in=found_extended, pGB_injected= pGB_injected_SNR_sorted_overlap[i][:20], saving_label =SAVEPATH+'/strain added'+ str(int(np.round(frequencies_search[i][0]*10**8))) +save_name+'anitcorrelation.png') 
        #     else:
        #         search1.plot(found_sources_in=found_extended, pGB_injected= pGB_injected_SNR_sorted_overlap[i], saving_label =SAVEPATH+'/strain added'+ str(int(np.round(frequencies_search[i][0]*10**8))) +save_name+'anitcorrelation.png') 
        #         # search1.plot(found_sources_in=found_sources_mp_best[i], pGB_injected=pGB_injected[i][:10], pGB_injected_matched= matched_extended, saving_label =SAVEPATH+'/strain added'+ str(int(np.round(frequencies_search[i][0]*10**8))) +save_name+'in.png') 

    found_sources_not_anitcorrelated2_array = []
    for i in range(len(found_sources_not_anitcorrelated2)):
        found_sources_not_anitcorrelated2_array.append(np.asarray(found_sources_not_anitcorrelated2[i]))

    found_sources_anticorrelated_flat2 = found_sources_anticorrelated_flat[np.array(correlation_list) < threshold_overlap]

    # np.save(SAVEPATH+'/found_sources_not_anticorrelated_'+save_name+'.npy', found_sources_not_anitcorrelated2)
    found_sources_in = found_sources_not_anitcorrelated2
    fn = SAVEPATH+'/found_sources_not_anticorrelated_'+save_name+'.pkl'
    pickle.dump(found_sources_in, open(fn, "wb"))
    fn = SAVEPATH+'/found_sources_anticorrelated_'+save_name+'.pkl'
    pickle.dump(found_sources_anticorrelated_flat2, open(fn, "wb"))

found_sources_in_flat = np.concatenate(found_sources_in)

found_sources_in_flat_array = {attribute: np.asarray([x[attribute] for x in found_sources_in_flat]) for attribute in found_sources_in_flat[0].keys()}
found_sources_in_flat_df = pd.DataFrame(found_sources_in_flat_array)


found_sources_in_flat_df_high_SNR = found_sources_in_flat_df[found_sources_in_flat_df['IntrinsicSNR'] > 12]
# #### parallel
# input = []
# index = []
# for i in range(len(found_sources_in)):
# # for i in range(16):
#     if len(found_sources_in[i]) > 0:
#         index.append(i)
#         input.append((found_sources_in[i],frequencies_search[i][0], frequencies_search[i][1]))
# start = time.time()
# pool = mp.Pool(16)
# SNR_intrinsic = pool.starmap(get_SNR, input)
# pool.close()
# pool.join()
# print('time to calculate SNR for', len(frequencies_search), 'windows: ', time.time()-start)
# for i in range(len(SNR_intrinsic)):
#     for j in range(len(SNR_intrinsic[i])):
#         if len(SNR_intrinsic[i]) > 0:
#             found_sources_in[index[i]][j]['IntrinsicSNR'] = SNR_intrinsic[i][j]
# np.save(SAVEPATH+'/found_sources_not_anitcorrelatedsecond' +save_name+'.npy', found_sources_in)

if version == '1':
    for i in range(len(found_sources_in)):
        for j in range(len(found_sources_in[i])):
            found_sources_in[i][j]['InitialPhase'] = -(found_sources_in[i][j]['InitialPhase'])
    for i in range(len(pGB_injected_SNR_sorted_overlap)):
        for j in range(len(pGB_injected_SNR_sorted_overlap[i])):
            pGB_injected_SNR_sorted_overlap[i][j]['InitialPhase'] = -(pGB_injected_SNR_sorted_overlap[i][j]['InitialPhase'])

############## match
def match_function(found_sources_in, pGB_injected_not_matched):
    found_sources_not_matched = deepcopy(found_sources_in)
    pGB_injected_matched = []
    found_sources_matched = []
    match_list = []
    match_list2 = []
    match_list3 = []
    pGB_best_list = []
    match_best_list = []
    for j in range(len(found_sources_in)):
        match_list2.append(j)
        found_match = False
        match_list_one_found_signal = []
        found_dict = {}
        for parameter in parameters:
            found_dict[parameter] = found_sources_in[j][parameter]
        for k in range(len(pGB_injected_not_matched)):
            pGB_injected_dict = {}
            for parameter in parameters:
                pGB_injected_dict[parameter] = pGB_injected_not_matched[k][parameter]
            # correlation = l2_norm_match(pGB_injected_dict,found_dict)
            correlation = correlation_match(pGB_injected_dict,found_dict)
            # SNR_scaled = SNR_match_scaled(pGB_injected_dict,found_dict)
            # SNR_not_scaled = SNR_match(pGB_injected_dict,found_dict)
            # correlation, amplitude_factor, cross_correlation = SNR_match_amplitude_condsiered(pGB_injected_dict,found_dict)
            match_list_one_found_signal.append(correlation)
            if k > 39:
                break
        if 0 == len(match_list_one_found_signal):
            print('0')
            match_list.append(np.nan)
            continue
        try:
            best_index = np.nanargmax(match_list_one_found_signal)
            # best_index = np.nanargmin(match_list_one_found_signal)
        except:
            print('all NAN:', match_list_one_found_signal, found_sources_in[0], pGB_injected_not_matched, found_sources_in)
            break
        if match_list_one_found_signal[best_index] > 0.9:
            found_match = True
        pGB_best_list.append(pGB_injected_not_matched[best_index])
        if found_match:
            pGB_injected_matched.append(pGB_injected_not_matched[best_index])
            found_sources_matched.append(found_sources_in[j])
            # pGB_injected_not_matched = np.delete(pGB_injected_not_matched, best_index)
            # pGB_injected_not_matched[best_index] = None
            found_sources_not_matched[j] = None
        match_list.append(match_list_one_found_signal[best_index])
        match_list3.append(j)
        match_best_list.append(found_sources_in[j])
    return found_sources_in, pGB_injected_not_matched, match_list, pGB_best_list, match_best_list, found_sources_not_matched, pGB_injected_matched, found_sources_matched, match_list2, match_list3

do_match_parallelized = False
if do_match_parallelized:
    print('match parallelized')
    pGB_injected_matched = []
    found_sources_matched = []
    match_list = []
    match_list2 = []
    match_list3 = []
    pGB_best_list = []
    match_best_list = []
    amplitude_factor = []
    pGB_injected_not_matched = deepcopy(pGB_injected_SNR_sorted_overlap)
    # pGB_injected_not_matched = deepcopy(found_sources_in2)
    found_sources_not_matched = deepcopy(found_sources_in)
    number_of_matched_signals = 0
    input = []
    for i in range(len(found_sources_in)):
        input.append((found_sources_in[i], pGB_injected_not_matched[i]))
    # match_function(*input[1000])
    
    counter=0
    start = time.time()
    pool = mp.Pool(16)
    matches = pool.starmap(match_function, input)
    pool.close()
    pool.join()
    print('time to match', time.time()-start)

    for i in range(len(matches)):
        pGB_injected_matched.append([])
        found_sources_matched.append([])
        match_list.append([])
        match_list2.append([])
        match_list3.append([])
        pGB_best_list.append([])
        match_best_list.append([])
        amplitude_factor.append([])
        found_sources_in[i], pGB_injected_not_matched[i], match_list[i],  pGB_best_list[i], match_best_list[i], found_sources_not_matched[i], pGB_injected_matched[i], found_sources_matched[i], match_list2[i], match_list3[i] = matches[i]

    found_sources_matched_flat = np.concatenate(found_sources_matched)
    for i in range(len(found_sources_not_matched)):
        try:
            found_sources_not_matched[i] = list(filter(None, found_sources_not_matched[i]))
        except:
            pass

do_match_sequential = False
if do_match_sequential:
    pGB_injected_matched = []
    found_sources_matched = []
    pGB_injected_not_matched = deepcopy(pGB_injected_SNR_sorted_overlap)
    found_sources_not_matched = deepcopy(found_sources_in)
    number_of_matched_signals = 0
    match_list = []
    start = time.time()
    percentage = 0
    for i in range(len(found_sources_in)):
    # for i in range(3300, 3400):
        # if i != index_of_interest_to_plot:
        #     continue
        pGB_injected_matched.append([])
        found_sources_matched.append([])
        # if i < 3310 or i > 3350:
        #     continue
        try:
            if i%int(len(found_sources_in)/100) == 0:
                percentage += 1
                print(percentage,'%')
        except:
            pass
        for j in range(len(found_sources_in[i])):
            found_match = False
            match_list_one_found_signal = []
            match_list_one_found_signal_original = []
            match_list_one_found_signal_amplitude = []
            found_dict = {}
            for parameter in parameters:
                found_dict[parameter] = found_sources_in[i][j][parameter]
            for k in range(len(pGB_injected_not_matched[i])):
                pGB_injected_dict = {}
                for parameter in parameters:
                    pGB_injected_dict[parameter] = pGB_injected_not_matched[i][k][parameter]
                l2_norm = l2_norm_match(pGB_injected_dict,found_dict)
                SNR_scaled = SNR_match_scaled(pGB_injected_dict,found_dict)
                correlation_apmlitude, amplitude_factor, cross_correlation = SNR_match_amplitude_condsiered(pGB_injected_dict,found_dict)
                match_list_one_found_signal.append(SNR_scaled)
                match_list_one_found_signal_original.append(np.array(cross_correlation))
                match_list_one_found_signal_amplitude.append(np.array(correlation_apmlitude))
                if k > 39:
                    break
            if 0 == len(match_list_one_found_signal):
                break
            best_index = np.nanargmin(match_list_one_found_signal)
            if match_list_one_found_signal[best_index] < 1:
                found_match = True
            if found_match:
                pGB_injected_matched[-1].append(pGB_injected_not_matched[i][best_index])
                found_sources_matched[-1].append(found_sources_in[i][j])
                pGB_injected_not_matched[i] = np.delete(pGB_injected_not_matched[i], best_index)
                match_list.append(match_list_one_found_signal[best_index])
                found_sources_not_matched[i][j] = None
                number_of_matched_signals += 1
    print('time to match', time.time()-start)
    
    for i in range(len(found_sources_not_matched)):
        found_sources_not_matched[i] = list(filter(None, found_sources_not_matched[i]))

# pGB_injected_not_matched_with_overlap_of_windows = deepcopy(pGB_injected_not_matched)
# # pGB_injected_not_matched_with_overlap_of_windows2 = deepcopy(pGB_injected_not_matched_with_overlap_of_windows)
# pGB_injected_not_matched = []
# for i in range(len(pGB_injected_not_matched_with_overlap_of_windows)):
#     pGB_injected_not_matched.append([])
#     list_count = 0
#     for j in range(len(pGB_injected_not_matched_with_overlap_of_windows[i])):
#         if pGB_injected_not_matched_with_overlap_of_windows[i][j]['Frequency'] > frequencies_search[i][0] and pGB_injected_not_matched_with_overlap_of_windows[i][j]['Frequency'] < frequencies_search[i][1]:
#             pGB_injected_not_matched[i].append(pGB_injected_not_matched_with_overlap_of_windows[i][j])
#             list_count += 1
#         if list_count > 19:
#             break

# pGB_injected_flat = []
# for i in range(len(pGB_injected)):
#     for j in range(len(pGB_injected[i])):
#         if pGB_injected[i][j]['Frequency'] > frequencies_search[i][0] and pGB_injected[i][j]['Frequency'] < frequencies_search[i][1]:
#             pGB_injected_flat.append(pGB_injected[i][j])
# number_of_injected_signals = len(pGB_injected_flat)

# for i in range(len(pGB_injected_matched)):
#     # if i != 5:
#     #     continue
#     if len(pGB_injected_matched[i]) > 0:
#         search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i][1])
#     for j in range(len(pGB_injected_matched[i])):
#         pGB_dict = {}
#         for parameter in parameters:
#             pGB_dict[parameter] = pGB_injected_matched[i][j][parameter]
#         pGB_injected_matched[i][j]['IntrinsicSNR'] = search1.intrinsic_SNR([pGB_dict])

# for i in range(len(pGB_injected_not_matched)):
#     # if i != 5:
#     #     continue
#     if len(pGB_injected_not_matched[i]) > 0:
#         search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i][1])
#     for j in range(len(pGB_injected_not_matched[i])):
#         pGB_dict = {}
#         for parameter in parameters:
#             pGB_dict[parameter] = pGB_injected_not_matched[i][j][parameter]
#         pGB_injected_not_matched[i][j]['IntrinsicSNR'] = search1.intrinsic_SNR([pGB_dict])

# count = 0
# for i in range(len(found_sources_in)):
#     count += 1
#     print(count)
#     if len(found_sources_in[i]) > 0:
#         search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i][1])
#     for j in range(len(found_sources_in[i])):
#         pGB_dict = {}
#         for parameter in parameters:
#             pGB_dict[parameter] = found_sources_in[i][j][parameter]
#         found_sources_in[i][j]['IntrinsicSNR'] = search1.intrinsic_SNR([pGB_dict])

# for i in range(len(found_sources_matched)):
#     if len(found_sources_matched[i]) > 0:
#         search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i][1])
#     for j in range(len(found_sources_matched[i])):
#         pGB_dict = {}
#         for parameter in parameters:
#             pGB_dict[parameter] = found_sources_matched[i][j][parameter]
#         found_sources_matched[i][j]['IntrinsicSNR'] = search1.intrinsic_SNR([pGB_dict])

# for i in range(len(found_sources_not_matched)):
#     if len(found_sources_not_matched[i]) > 0:
#         search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i][1])
#     for j in range(len(found_sources_not_matched[i])):
#         pGB_dict = {}
#         for parameter in parameters:
#             pGB_dict[parameter] = found_sources_not_matched[i][j][parameter]
#         found_sources_not_matched[i][j]['IntrinsicSNR'] = search1.intrinsic_SNR([pGB_dict])     


# end_string = '_SNR_scaled_03_injected_snr'+str(injected_SNR)+'_comparison_'+seed2
# end_string = '_SNR_scaled_03_injected_snr'+str(injected_SNR)
end_string = '_correlation_09_injected_snr'+str(injected_SNR)
# end_string = ''


try:
    found_sources_matched_array = []
    for i in range(len(found_sources_matched)):
        found_sources_matched_array.append(np.asarray(found_sources_matched[i]))
    # found_sources_matched_array = np.asarray(found_sources_matched_array)
    found_sources_not_matched_array = []
    for i in range(len(found_sources_not_matched)):
        found_sources_not_matched_array.append(np.asarray(found_sources_not_matched[i]))
    # found_sources_not_matched_array = np.asarray(found_sources_not_matched_array)
    # pickle.dump(found_sources_matched_array, open(SAVEPATH+'/found_sources_matched_' +save_name+end_string+'.npy', "wb"))
    # pickle.dump(found_sources_in, open(SAVEPATH+'/found_sources_' +save_name+end_string+'.npy', "wb"))
    pickle.dump(found_sources_in, open(SAVEPATH+'/evaluation/found_sources_' +save_name+end_string+'.npy', "wb"))
    pickle.dump(found_sources_matched_array, open(SAVEPATH+'/evaluation/found_sources_matched_' +save_name+end_string+'.npy', "wb"))
    pickle.dump(found_sources_not_matched_array, open(SAVEPATH+'/evaluation/found_sources_not_matched_' +save_name+end_string+'.npy', "wb"))
    pickle.dump(pGB_injected_not_matched, open(SAVEPATH+'/evaluation/injected_not_matched_windows_' +save_name+end_string+'.npy', "wb"))
    pickle.dump(pGB_injected_matched, open(SAVEPATH+'/evaluation/injected_matched_windows_' +save_name+end_string+'.npy', "wb"))
    pickle.dump(match_list, open(SAVEPATH+'/evaluation/match_list_' +save_name+end_string+'.npy', "wb"))
    pickle.dump(pGB_best_list, open(SAVEPATH+'/evaluation/pGB_best_list_' +save_name+end_string+'.npy', "wb"))
    pickle.dump(match_best_list, open(SAVEPATH+'/evaluation/match_best_list_' +save_name+end_string+'.npy', "wb"))
except:
    found_sources_in = pickle.load(open(fn, 'rb'))
    found_sources_matched = pickle.load(open(SAVEPATH+'/evaluation/found_sources_matched_' +save_name+end_string+'.npy', 'rb'))
    found_sources_not_matched = pickle.load(open(SAVEPATH+'/evaluation/found_sources_not_matched_' +save_name+end_string+'.npy', 'rb'))
    pGB_injected_not_matched = pickle.load(open(SAVEPATH+'/evaluation/injected_not_matched_windows_' +save_name+end_string+'.npy', 'rb'))
    pGB_injected_matched = pickle.load(open(SAVEPATH+'/evaluation/injected_matched_windows_' +save_name+end_string+'.npy', 'rb'))
    match_list = pickle.load(open(SAVEPATH+'/evaluation/match_list_' +save_name+end_string+'.npy', 'rb'))
    pGB_best_list = pickle.load(open(SAVEPATH+'/evaluation/pGB_best_list_' +save_name+end_string+'.npy', 'rb'))
    match_best_list = pickle.load(open(SAVEPATH+'/evaluation/match_best_list_' +save_name+end_string+'.npy', 'rb'))

match_list_flat = np.concatenate(match_list)
# match_list2_flat = np.concatenate(match_list2)
# match_list3_flat = np.concatenate(match_list3)
found_sources_in_flat = np.concatenate(found_sources_in)

# match_list_flat = np.concatenate(match_list)
# print(len(match_list_flat))

number_of_injected_signals = 0
for i in range(len(pGB_injected)):
    number_of_injected_signals += len(pGB_injected[i])
number_of_injected_signals_SNR_high = 0
for i in range(len(pGB_injected)):
    number_of_injected_signals_SNR_high += len(pGB_injected[i][pGB_injected[i]['IntrinsicSNR']>10])
# number_of_injected_signals_SNR_high2 = len(pGB_injected_flat[pGB_injected_flat['IntrinsicSNR']>10])
number_of_found_signals = 0
for i in range(len(found_sources_in)):
    number_of_found_signals += len(found_sources_in[i])

### 32, 128, 187, 222, 235, 257, 268, 274
### 128, 222
number_of_found_signals_not_matched = 0
number_of_injected_signals_not_matched = 0
for i in range(len(found_sources_not_matched)):
    number_of_found_signals_not_matched += len(found_sources_not_matched[i])
    number_of_injected_signals_not_matched += len(pGB_injected_not_matched[i])
number_of_matched_signals = len(np.concatenate(found_sources_matched))
print(number_of_matched_signals ,'matched signals out of', number_of_injected_signals , 'injected signals and',number_of_found_signals, 'found signals')
print('sensitivity = matched signals/injected signals:', np.round(number_of_matched_signals/number_of_injected_signals,2))
print('number_of_injected_signals_SNR_high:', np.round(number_of_injected_signals_SNR_high,2))
print('matched signals/found signals:', np.round(number_of_matched_signals/number_of_found_signals,2))
print('number_of_found_signals_not_matched:', np.round(number_of_found_signals_not_matched,2))
print('number_of_injected_signals_not_matched:', np.round(number_of_injected_signals_not_matched,2))

if version == '1':
    for i in range(len(found_sources_matched)):
        for j in range(len(found_sources_matched[i])):
            a = deepcopy(found_sources_matched[i][j])
            a['InitialPhase'] = -(a['InitialPhase'])
            found_sources_matched[i][j] = deepcopy(a)
    for i in range(len(found_sources_not_matched)):
        for j in range(len(found_sources_not_matched[i])):
            a = deepcopy(found_sources_not_matched[i][j])
            a['InitialPhase'] = -(a['InitialPhase'])
            found_sources_not_matched[i][j] = deepcopy(a)
    for i in range(len(pGB_injected_matched)):
        for j in range(len(pGB_injected_matched[i])):
            a = deepcopy(pGB_injected_matched[i][j])
            a['InitialPhase'] = -(a['InitialPhase'])
            pGB_injected_matched[i][j] = deepcopy(a)

found_sources_in_flat = np.concatenate(found_sources_in)
found_sources_in_flat_array = {attribute: np.asarray([x[attribute] for x in found_sources_in_flat]) for attribute in found_sources_in_flat[0].keys()}
found_sources_in_flat_df = pd.DataFrame(found_sources_in_flat_array)
found_sources_matched_flat = np.concatenate(found_sources_matched)
found_sources_matched_flat_array = {attribute: np.asarray([x[attribute] for x in found_sources_matched_flat]) for attribute in found_sources_matched_flat[0].keys()}
found_sources_matched_flat_df = pd.DataFrame(found_sources_matched_flat_array)
found_sources_not_matched_flat = np.concatenate(found_sources_not_matched)
found_sources_not_matched_flat_array = {attribute: np.asarray([x[attribute] for x in found_sources_not_matched_flat]) for attribute in found_sources_not_matched_flat[0].keys()}
found_sources_not_matched_flat_df = pd.DataFrame(found_sources_not_matched_flat_array)

pGB_injected_reduced = []
for i in range(len(pGB_injected)):
    try:
        pGB_injected_reduced.append(pGB_injected[i].iloc[:40])
    except:
        pGB_injected_reduced.append(pGB_injected[i])
pGB_injected_flat_reduced = np.concatenate(pGB_injected_reduced)
pGB_injected_flat_reduced_df = pd.DataFrame(np.asarray(pGB_injected_flat_reduced))
pGB_injected_flat_reduced_df = pGB_injected_flat_reduced_df.sort_values(by='Frequency')
# pGB_injected_flat_highSNR_df = pGB_injected_flat_reduced_df[pGB_injected_flat_reduced_df['IntrinsicSNR']>10]

pGB_injected_matched_flat = []
for i in range(len(pGB_injected_matched)):
    for j in range(len(pGB_injected_matched[i])):
        pGB_injected_matched_flat.append(pGB_injected_matched[i][j])

pGB_injected_not_matched_flat = []
for i in range(len(pGB_injected_not_matched)):
    for j in range(len(pGB_injected_not_matched[i])):
        pGB_injected_not_matched_flat.append(pGB_injected_not_matched[i][j])
# pGB_injected_matched_flat = np.asarray(pGB_injected_matched_flat)

pGB_injected_matched_flat_df = pd.DataFrame(np.asarray(pGB_injected_matched_flat))
pGB_injected_matched_flat_df = pGB_injected_matched_flat_df.sort_values(by= 'Frequency')
pGB_injected_not_matched_flat_df = pd.DataFrame(np.asarray(pGB_injected_not_matched_flat))
pGB_injected_not_matched_flat_df = pGB_injected_not_matched_flat_df.sort_values(by= 'Frequency')

# pGB_injected_not_matched_flat_df = pGB_injected_flat_reduced_df.append(pGB_injected_matched_flat_df)
# pGB_injected_not_matched_flat_df = pGB_injected_not_matched_flat_df.drop_duplicates(keep=False)
# amplitude_considered_
found_sources_in_flat_df.to_pickle(SAVEPATH+'/evaluation/found_sources_' +save_name+end_string+'_df')
found_sources_matched_flat_df.to_pickle(SAVEPATH+'/evaluation/found_sources_matched_' +save_name+end_string+'_df')
found_sources_not_matched_flat_df.to_pickle(SAVEPATH+'/evaluation/found_sources_not_matched_' +save_name+end_string+'_df')
pGB_injected_matched_flat_df.to_pickle(SAVEPATH+'/evaluation/injected_matched_windows_' +save_name+end_string+'_df')
pGB_injected_not_matched_flat_df.to_pickle(SAVEPATH+'/evaluation/injected_not_matched_windows_' +save_name+end_string+'_df')

# plt.figure()
# plt.plot([1,2,3], [1,2,3])
# plt.xlabel(r'$f$ (mHz)')
# plt.show()

# found_sources_in_flat = np.concatenate(found_sources_in)
# found_sources_in_flat_array = {attribute: np.asarray([x[attribute] for x in found_sources_in_flat]) for attribute in found_sources_in_flat[0].keys()}
# found_sources_in_flat_df = pd.DataFrame(found_sources_in_flat_array)


found_sources_in = []
for i in range(len(frequencies_search)):
    lower_index = np.searchsorted(found_sources_in_flat_frequency,frequencies_search[i][0])
    higher_index = np.searchsorted(found_sources_in_flat_frequency,frequencies_search[i][1])
    if i == 102:
        print(lower_index, higher_index)
    found_sources_in.append(found_sources_in_flat[lower_index:higher_index])


found_sources_in_SNR_sorted = []
for i in range(len(found_sources_in)):
    listSNR = []
    for j in range(len(found_sources_in[i])):
        listSNR.append(-found_sources_in[i][j]['IntrinsicSNR'])
    indexesSNR = np.argsort(listSNR)
    found_sources_in_SNR_sorted.append(np.array(found_sources_in[i])[indexesSNR])
found_sources_in = deepcopy(found_sources_in_SNR_sorted)



### determine index of a specific frequency
# index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.00264612)
# index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.002084612)
# index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.003988)+1
# index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.005208333333333333)+3
# index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.0008748)-1
# index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.002114)-3
index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.001149978)
index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.00371719)-2
index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.0236)-1
index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.0214)-1
# index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.00360246099898)-2
# index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.003302)-3
# index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.002026)-2
# index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.00399)+99
index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.003508)-2
# index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.005661)-2
#plot strains
number_of_windows = 3
for i in range(len(frequencies_search)):
    if i != index_of_interest_to_plot:
        continue
    search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i+number_of_windows-1][1], update_noise=False)
    vertical_lines = []
    for j in range(number_of_windows):
        vertical_lines.append(frequencies_search[i+j][0])
        vertical_lines.append(frequencies_search[i+j][1])
    found_extended = np.concatenate(match_best_list[i:i+number_of_windows])
    # found_extended = np.concatenate(found_sources_matched[i:i+number_of_windows])
    # found_not_matched_extended = np.concatenate(found_sources_not_matched[i:i+number_of_windows])

    # chains = []
    # for j in range(len(found_extended)):
    #     chains.append(pd.read_csv(SAVEPATH+'Chains_gpu/frequency'+str(int(np.round(found_extended[j]['Frequency']*10**12)))+'pHz'+save_name+'.csv'))
        


    # matched_extended = np.concatenate(pGB_injected_matched[i:i+number_of_windows])
    # if len(pGB_injected_SNR[i]) > 0:
    pGB_injected_dict_list = []
    matched_extended = []
    not_matched_extended = []
    pGB_injected_not_matched_flat_df_in_window = pGB_injected_not_matched_flat_df[pGB_injected_not_matched_flat_df['Frequency'] >= frequencies_search[i][0]]
    pGB_injected_not_matched_flat_df_in_window = pGB_injected_not_matched_flat_df_in_window[pGB_injected_not_matched_flat_df_in_window['Frequency'] <= frequencies_search[i+number_of_windows-1][1]]
    for j in range(len(pGB_injected_not_matched_flat_df_in_window)):
        pGB_injected_dict_list.append({})
        for parameter in parameters:
            pGB_injected_dict_list[-1][parameter] = pGB_injected_not_matched_flat_df_in_window[parameter].iloc[j]
        if j > 100:
            break
    for j in range(number_of_windows):
        for k in range(len(pGB_best_list[i+j])):
        # for k in range(len(pGB_injected_matched[i+j])):
            matched_extended.append({})
            for parameter in parameters:
                matched_extended[-1][parameter] = pGB_best_list[i+j][k][parameter]
                # matched_extended[-1][parameter] = pGB_injected_matched[i+j][k][parameter]
    save_name_path = SAVEPATH+'/strain added Amplitude'+ str(int(np.round(frequencies_search[i][0]*10**8))) +save_name+str(int(len(matched_extended)))+'.png'
    pGB_injected_dict_list = pGB_injected_dict_list[::2]
    if len(pGB_injected_dict_list) > 60:
        search1.plot(found_sources_in=[], found_sources_not_matched = [], pGB_injected= [],  pGB_injected_matched= pGB_injected_dict_list, vertical_lines= vertical_lines, saving_label =save_name_path) 
    else:
        search1.plotA(found_sources_in=found_extended, pGB_injected= pGB_injected_dict_list,  pGB_injected_matched= matched_extended, vertical_lines= vertical_lines, saving_label =save_name_path) 
        # search1.plot(found_sources_in=found_sources_mp_best[i], pGB_injected=pGB_injected[i][:10], pGB_injected_matched= matched_extended, saving_label =SAVEPATH+'/strain added'+ str(int(np.round(frequencies_search[i][0]*10**8))) +save_name+'in.png') 
    plt.show(block=True)


print(correlation_match(pGB_injected_dict_list[0],found_not_matched_extended[0]))
print(search1.SNR([pGB_injected_dict_list[0]]))
print(search1.SNR(pGB_injected_dict_list[:5]))
print(search1.SNR(found_extended[:5]))
print(search1.SNR(matched_extended[:5]))
print(search1.SNR([found_extended[2]]))
print(search1.SNR([found_not_matched_extended[0]]))
print(search1.SNR([found_not_matched_extended[1]]))
print(search1.loglikelihood([found_not_matched_extended[0]]))
print(search1.loglikelihood([found_not_matched_extended[1]]))
print(search1.SNR([found_not_matched_extended[1],found_not_matched_extended[0]]))
print(search1.SNR([pGB_injected_dict_list[1],pGB_injected_dict_list[2]]))
print(search1.SNR([found_not_matched_extended[0],found_not_matched_extended[1],found_extended[0],found_extended[1],found_extended[2],found_extended[3]]))
for j in range(len(found_sources_mp_all[index_of_interest_to_plot])):
    print(search1.SNR([found_sources_mp_all[index_of_interest_to_plot][j]]))
for j in range(len(found_sources_in[index_of_interest_to_plot])):
    print(search1.SNR([found_sources_in[index_of_interest_to_plot][j]]))

found_sources_mp_all_df = pd.DataFrame(found_sources_mp_all[index_of_interest_to_plot])
found_sources_best_df = pd.DataFrame(found_sources_mp_best[index_of_interest_to_plot])
found_sources_mp_all_df2 = found_sources_mp_all_df.append(found_sources_best_df)
found_sources_mp_all_df2 = found_sources_mp_all_df2.drop_duplicates(subset=['Frequency'], keep=False)
found_sources_out = found_sources_mp_best[index_of_interest_to_plot][-2:]


######### check pipeline subtraction
start_index_shift = 0
found_sources_in_optimize = []
found_sources_out_optimize = []
found_sources = []
### subtract neighbors
tdi_fs_neighbors_subtracted = deepcopy(tdi_fs)
found_sources_to_subtract = np.concatenate([found_sources_mp[start_index_shift+index_of_interest_to_plot-1][3],found_sources_mp[start_index_shift+index_of_interest_to_plot+1][3]])
for j in range(len(found_sources_to_subtract)):
    Xs_subtracted, Ys_subtracted, Zs_subtracted = GB.get_fd_tdixyz(template=found_sources_to_subtract[j], oversample=4)
    source_subtracted = dict({"X": Xs_subtracted, "Y": Ys_subtracted, "Z": Zs_subtracted})
    index_low = np.searchsorted(tdi_fs_neighbors_subtracted["X"].f, Xs_subtracted.f[0])
    index_high = index_low+len(Xs_subtracted)
    for k in ["X", "Y", "Z"]:
        tdi_fs_neighbors_subtracted[k].data[index_low:index_high] = tdi_fs_neighbors_subtracted[k].data[index_low:index_high] - source_subtracted[k].data
tdi_fs_subtracted = deepcopy(tdi_fs_neighbors_subtracted)
plot_subraction = True
if plot_subraction:
    search1 = Search(tdi_fs,Tobs, frequencies_search[index_of_interest_to_plot][0], frequencies_search[index_of_interest_to_plot][1])
    search1.plot(second_data= tdi_fs_subtracted, found_sources_not_matched= found_sources_mp[start_index_shift+index_of_interest_to_plot][3])
# for i in range(len(found_sources_in[index_of_interest_to_plot])):
for i in range(5):
    search_out_subtracted = Search(tdi_fs_subtracted, Tobs, frequencies_search[index_of_interest_to_plot][0], frequencies_search[index_of_interest_to_plot][1])

    lower_frequency = frequencies_search[index_of_interest_to_plot][0]
    upper_frequency = frequencies_search[index_of_interest_to_plot][1]
    # create two sets of found sources. found_sources_in with signals inside the boundary and founce_sources_out with outside sources
    found_sources_in_optimize = []
    found_sources_out_optimize = []
    for j in range(len(found_sources)):
        if found_sources[j]['Frequency'] > lower_frequency and found_sources[j]['Frequency'] < upper_frequency:
            found_sources_in_optimize.append(found_sources[j])
        else:
            found_sources_out_optimize.append(found_sources[j])

    found_sources_out_optimize = [] # ignore signals outside
    try:
        print('injected',search_out_subtracted.SNR([pGB_injected_dict_list[i]]))
        print('injected original',search1.SNR([pGB_injected_dict_list[i]]))
    except:
        pass
    SNR_list = []
    for j in range(3):
        if found_sources_mp_all[start_index_shift+index_of_interest_to_plot][i*3+j]['Amplitude'] < 0:
            print('neg amplitude', found_sources_mp_all[start_index_shift+index_of_interest_to_plot][i*3+j]['Amplitude'] )
            found_sources_mp_all[start_index_shift+index_of_interest_to_plot][i*3+j]['Amplitude'] *= -1
            print('neg amplitude', found_sources_mp_all[start_index_shift+index_of_interest_to_plot][i*3+j]['Amplitude'] )
        SNR_list.append(float(search_out_subtracted.SNR([found_sources_mp_all[start_index_shift+index_of_interest_to_plot][i*3+j]])))
    max_index = np.nanargmax(SNR_list)
    found_sources_in_optimize.append(found_sources_mp_all[start_index_shift+index_of_interest_to_plot][i*3+max_index]) ## should exclude outsiders

    A_optimized = search1.calculate_Amplitude([found_sources_in_optimize[-1]])
    found_sources_in_optimize[-1]['Amplitude'] *= A_optimized.values
    print('found',search_out_subtracted.SNR([found_sources_in_optimize[-1]]))
    print('found log l',search_out_subtracted.loglikelihood([found_sources_in_optimize[-1]]))
    print('found original',search1.SNR([found_sources_in_optimize[-1]]))

    tdi_fs_subtracted = deepcopy(tdi_fs_neighbors_subtracted)
    for j in range(len(found_sources_out_optimize)):
        Xs_subtracted, Ys_subtracted, Zs_subtracted = GB.get_fd_tdixyz(template=found_sources_out_optimize[j], oversample=4)
        source_subtracted = dict({"X": Xs_subtracted, "Y": Ys_subtracted, "Z": Zs_subtracted})
        index_low = np.searchsorted(tdi_fs_subtracted["X"].f, Xs_subtracted.f[0])
        index_high = index_low+len(Xs_subtracted)
        for k in ["X", "Y", "Z"]:
            tdi_fs_subtracted[k].data[index_low:index_high] = tdi_fs_subtracted[k].data[index_low:index_high] - source_subtracted[k].data

    tdi_fs_subtracted = deepcopy(tdi_fs_neighbors_subtracted)
    search_out_subtracted = Search(tdi_fs_subtracted, Tobs, frequencies_search[index_of_interest_to_plot][0], frequencies_search[index_of_interest_to_plot][1])

    print('injected all',search_out_subtracted.SNR(pGB_injected_dict_list[:i+1]))
    print('injected all original',search1.SNR(pGB_injected_dict_list[:i+1]))
    print('found all',search_out_subtracted.SNR(found_sources_in_optimize))
    print('found all log l',search_out_subtracted.loglikelihood(found_sources_in_optimize))
    print('found all original',search1.SNR(found_sources_in_optimize))
    total_boundaries = deepcopy(search1.boundaries)
    amplitudes = []
    for k in range(len(found_sources_in_optimize)):
        amplitudes.append(found_sources_in_optimize[k]['Amplitude'])
    total_boundaries['Amplitude'] = [np.min(amplitudes),np.max(amplitudes)]
    amplitudes_length = search1.boundaries['Amplitude'][1] - search1.boundaries['Amplitude'][0]
    total_boundaries['Amplitude'] = [np.log10(total_boundaries['Amplitude'][0]), np.log10(total_boundaries['Amplitude'][1])]
    total_boundaries['Amplitude'] = [total_boundaries['Amplitude'][0] - amplitudes_length/10,total_boundaries['Amplitude'][1] + amplitudes_length/10,]
                
    start = time.time()
    found_sources_in_optimize = search_out_subtracted.optimize([found_sources_in_optimize], boundaries= total_boundaries)
    print('found optimized',search_out_subtracted.SNR(found_sources_in_optimize))
    print('global optimization time', time.time()-start)

    found_sources = found_sources_in_optimize + found_sources_out_optimize
    # found_sources = [pGB_injected_dict_list[0]]
    #subtract the found sources from original
    found_sources = pGB_injected_dict_list
    tdi_fs_subtracted = deepcopy(tdi_fs_neighbors_subtracted)
    for j in range(len(found_sources)):
        Xs_subtracted, Ys_subtracted, Zs_subtracted = GB.get_fd_tdixyz(template=found_sources[j], oversample=4)
        source_subtracted = dict({"X": Xs_subtracted, "Y": Ys_subtracted, "Z": Zs_subtracted})
        index_low = np.searchsorted(tdi_fs_subtracted["X"].f, Xs_subtracted.f[0])
        index_high = index_low+len(Xs_subtracted)
        for k in ["X", "Y", "Z"]:
            tdi_fs_subtracted[k].data[index_low:index_high] = tdi_fs_subtracted[k].data[index_low:index_high] - source_subtracted[k].data
    search_out_subtracted = Search(tdi_fs_subtracted, Tobs, frequencies_search[index_of_interest_to_plot][0], frequencies_search[index_of_interest_to_plot][1])
    plot_subraction = True
    if plot_subraction:
        search1 = Search(tdi_fs,Tobs, frequencies_search[index_of_interest_to_plot][0], frequencies_search[index_of_interest_to_plot][1])
        search1.plot(second_data= tdi_fs_subtracted, found_sources_not_matched = found_sources)

##### final global optimization
def tdi_subtraction(tdi_fs,found_sources_to_subtract):
    #subtract the found sources from original
    tdi_fs_subtracted2 = deepcopy(tdi_fs)
    for i in range(len(found_sources_to_subtract)):
        Xs_subtracted, Ys_subtracted, Zs_subtracted = GB.get_fd_tdixyz(template=found_sources_to_subtract[i], oversample=4)
        source_subtracted = dict({"X": Xs_subtracted, "Y": Ys_subtracted, "Z": Zs_subtracted})
        index_low = np.searchsorted(tdi_fs_subtracted2["X"].f, Xs_subtracted.f[0])
        index_high = index_low+len(Xs_subtracted)
        for k in ["X", "Y", "Z"]:
            tdi_fs_subtracted2[k].data[index_low:index_high] -= source_subtracted[k].data
    return tdi_fs_subtracted2

i = deepcopy(index_of_interest_to_plot)
found_sources_to_subtract = np.concatenate([found_sources_in[i-1],found_sources_in[i+1]])
tdi_fs_subtracted = tdi_subtraction(tdi_fs,found_sources_to_subtract)
plot_subraction = True
if plot_subraction:
    search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i][1])
    search1.plot(second_data= tdi_fs_subtracted)

correlation_list2 = []
for i in range(len(frequencies_search)):
    if i != index_of_interest_to_plot:
        continue        
    for j in range(len(found_sources_in[i])):
            found_match = False
            # if j != 1:
            #     continue
            # print('i', i, 'j',j)
            correlation_list_of_one_signal = []
            for k in range(len(pGB_injected_SNR_sorted_overlap[i])):
                pGB_injected_dict = {}
                found_dict = {}
                for parameter in parameters:
                    pGB_injected_dict[parameter] = pGB_injected_SNR_sorted_overlap[i].iloc[k][parameter]
                    found_dict[parameter] = found_sources_in[i][j][parameter]
                # print('SNR', correlation_match(pGB_injected_not_matched[i][k],found_sources_in[i][j]),'parameter comparison:',pGB_injected_not_matched[i][k]['EclipticLatitude'],found_sources_in[i][j]['EclipticLatitude'],eclipticlongitude, found_sources_in[i][j]['EclipticLongitude'])
                correlation = correlation_match(pGB_injected_dict_list[0],found_dict)
                correlation_list_of_one_signal.append(correlation)
                if k > 19:
                    break
            correlation_list2.append(correlation_list_of_one_signal)

plt.imshow(correlation_list2)

#### plot SNR - frequency
markersize = 3
alpha = 0.5
parameter_to_plot = 'IntrinsicSNR'
fig = plt.figure()
# plt.plot(pGB_injected_flat_df['Frequency']*10**3,pGB_injected_flat_df['IntrinsicSNR'], '.', color= colors[0], label = 'Injected', markersize= markersize, alpha = alpha)
# plt.plot(pGB_injected_matched_flat_df['Frequency']*10**3,pGB_injected_matched_flat_df['IntrinsicSNR'], '.', color= colors[1], label = 'Injected matched', markersize= markersize, alpha = alpha)
# plt.plot(pGB_injected_flat_df_high_SNR['Frequency']*10**3,pGB_injected_flat_df_high_SNR['IntrinsicSNR'],'.', color= colors[1], markersize= markersize, label = 'Injected SNR > 10', alpha = alpha)
plt.plot(found_sources_matched_flat_df['Frequency']*10**3,found_sources_matched_flat_df['IntrinsicSNR'],'g.', label = 'Found matched', markersize= markersize, alpha = alpha)
# plt.plot(pGB_injected_not_matched_flat_df['Frequency']*10**3,pGB_injected_not_matched_flat_df['IntrinsicSNR'], '+', color = 'r', label = 'Injected not matched', markersize= markersize, alpha = alpha)
plt.plot(found_sources_not_matched_flat_df['Frequency']*10**3,found_sources_not_matched_flat_df['IntrinsicSNR'],'+',color= 'blue',  markerfacecolor='None', markersize= markersize, label = 'Found not matched', alpha = alpha)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('f (mHz)')
plt.xlim(0.3,100)
plt.ylim(0.1,2000)
if parameter_to_plot == 'IntrinsicSNR':
    plt.ylabel('Intrinsic SNR')
else:
    plt.ylabel(parameter_to_plot)    
plt.legend(markerscale=4, loc = 'lower right')
plt.savefig(SAVEPATH+'/Evaluation/'+parameter_to_plot+save_name+'injected_not_matched_found_matched_found_not_matched',dpi=300,bbox_inches='tight')
plt.show()

### plot match histo gramm - frequency
pGB_injected_matched_frequencies = []
for i in range(len(pGB_injected_matched)):
    for j in range(len(pGB_injected_matched[i])):
        pGB_injected_matched_frequencies.append(pGB_injected_matched[i][j]['Frequency']*10**3)
pGB_injected_not_matched_frequencies = []
for i in range(len(pGB_injected_not_matched)):
    for j in range(len(pGB_injected_not_matched[i])):
        pGB_injected_not_matched_frequencies.append(pGB_injected_not_matched[i][j]['Frequency']*10**3)
fig = plt.figure()
plt.hist(pGB_injected_matched_frequencies, 20, color = 'green')
plt.hist(pGB_injected_not_matched_frequencies, 20, color = 'red')
plt.xlabel('f (mHz)')
plt.legend()
plt.savefig(SAVEPATH+'/Evaluation/_SNR_histo'+save_name,dpi=300,bbox_inches='tight')
plt.show()
