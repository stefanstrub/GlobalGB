#%%
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
from getdist import plots, MCSamples
import scipy
from scipy.optimize import differential_evolution
import numpy as np
import xarray as xr
# from getdist import plots, MCSamples
import time
from copy import deepcopy
import multiprocessing as mp
import pandas as pd
import os
import h5py
# from KDEpy import FFTKDE
import sys
sys.path.append('/cluster/home/sstrub/Repositories/LDC/lib/lib64/python3.8/site-packages/ldc-0.1-py3.8-linux-x86_64.egg')

from ldc.lisa.noise import get_noise_model
from ldc.lisa.noise import AnalyticNoise
from ldc.common.series import TimeSeries
import ldc.waveform.fastGB as fastGB
from ldc.common.tools import window
# from ldc.waveform.fastGB import fastGB
# import ldc.waveform.fastGB as fastGB+

# from ldc.common.tools import compute_tdi_snr

from fastkde import fastKDE
# from sklearn.metrics import mean_squared_error
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
# from chainconsumer import ChainConsumer


try:
    import cupy as xp

    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu_available = False

from gbgpu.gbgpu import GBGPU

from gbgpu.utils.constants import *

from sources import *

import subprocess as sp
import os

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

gb_gpu = GBGPU(use_gpu=gpu_available)

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

def pGB_dict_to_gpu_input(pGB):
    params = []
    for i in range(len(pGB)):
        params.append(
            [pGB[i]['Amplitude'],
            pGB[i]['Frequency'],
            pGB[i]['FrequencyDerivative'],
            0,
            pGB[i]['InitialPhase'],
            pGB[i]['Inclination'],
            pGB[i]['Polarization'],
            pGB[i]['EclipticLongitude'],
            pGB[i]['EclipticLatitude']]
        )
    params = np.asanyarray(params)
    params = np.swapaxes(params, 0,1)
    return params

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

dataset = 'Sangria'
VGB = False
add_gaps = False
zero_gaps = False
zero_glitches = False
fill_gaps = False
if dataset == 'Radler':
    DATAPATH = grandparent+"/LDC/Radler/data"
    SAVEPATH = grandparent+"/LDC/pictures/LDC1-4/"
    tdi2 = False
elif dataset == 'Sangria':
    DATAPATH = grandparent+"/LDC/Sangria/data"
    SAVEPATH = grandparent+"/LDC/pictures/Sangria/"
    MBHBPATH = grandparent+"/LDC/MBHB/"
    tdi2 = False
elif dataset == 'Spritz':
    DATAPATH = grandparent+"/LDC/Spritz/data"
    SAVEPATH = grandparent+"/LDC/Spritz/"
    tdi2 = True

if dataset == 'Radler':
    data_fn = DATAPATH + "/LDC1-4_GB_v2.hdf5"
    # data_fn = DATAPATH + "/LDC1-3_VGB_v2_FD_noiseless.hdf5"
    if VGB:
        data_fn = DATAPATH + "/LDC1-3_VGB_v2.hdf5"
elif dataset == 'Sangria':
    data_fn = DATAPATH + "/LDC2_sangria_training_v2.h5"
elif dataset == 'Spritz':
    data_fn = DATAPATH + "/LDC2_spritz_vgb_training_v2.h5"
fid = h5py.File(data_fn)

reduction = 1
mbhbs_removed = False
seed = 1

# get TDI 
if dataset == 'Radler':
    td = np.array(fid["H5LISA/PreProcess/TDIdata"])
    td = np.rec.fromarrays(list(td.T), names=["t", "X", "Y", "Z"])
    dt = float(np.array(fid['H5LISA/GWSources/GalBinaries']['Cadence']))
    Tobs = float(int(np.array(fid['H5LISA/GWSources/GalBinaries']['ObservationDuration']))/reduction)
    names = np.array(fid['H5LISA/GWSources/GalBinaries'])
    params = [fid['H5LISA/GWSources/GalBinaries'][k] for k in names]
    reduced_names = []
    i = 0
    for p in params:
        i += 1
        if p.shape:
            reduced_names.append(names[i-1])
    params = [np.array(p) for p in params if p.shape]
    cat = np.rec.fromarrays(params, names=list(reduced_names))
elif dataset == 'Sangria':
    td = fid["obs/tdi"][()]
    td = np.rec.fromarrays(list(td.T), names=["t", "X", "Y", "Z"])
    td = td['t']
    dt = td["t"][1]-td["t"][0]
    
    td_mbhb = fid["sky/mbhb/tdi"][()]
    # cat_mbhb = fid["sky/mbhb/cat"]
    td_mbhb  = np.rec.fromarrays(list(td_mbhb .T), names=["t", "X", "Y", "Z"])
    td_mbhb  = td_mbhb ['t']

    Tobs = float(int(np.array(fid['obs/config/t_max']))/reduction)
    if mbhbs_removed:
        for k in ["X", "Y", "Z"]:
            td[k] = td[k] - td_mbhb[k]
            # td_injected[k] -= td_injected[k]
    else:
        td_original = deepcopy(td)
        # if reduction == 2:
        #     # wave = pickle.load(open(MBHBPATH+dataset+"_mbhbh_found_6months.pkl", "rb"))
        #     wave = pickle.load(open(MBHBPATH+dataset+'_mbhbh_found_6months_seed'+str(seed)+'.pkl', "rb"))
        #     # wave = pickle.load(open(MBHBPATH+dataset+"_mbhbh_found_6months.pkl", "rb"))
        # else:
        #     wave = pickle.load(open(MBHBPATH+dataset+'_mbhbh_found_12months_seed'+str(seed)+'.pkl', "rb"))
        wave = pickle.load(open(MBHBPATH+'Sangria_mbhbh_found_49w_seed'+str(seed)+'.pkl', "rb"))
        for i, k in enumerate(["X", "Y", "Z"]):
            # td[k] = td_mbhb[k]
            td[k] -= wave[k] 

elif dataset == 'Spritz':
    names = fid["sky/cat"].dtype.names
    params = [np.array(fid["sky/cat"][k]).squeeze() for k in names]
    cat = np.rec.fromarrays(params, names=list(names))
    indexes = np.argsort(cat['Frequency'])
    cat = cat[indexes]
    # print(cat_vgb)
    td = fid["obs/tdi"][()]
    td = np.rec.fromarrays(list(td.T), names=["t", "X", "Y", "Z"])
    td = td['t']
    dt = td["t"][1]-td["t"][0]
    Tobs = float(td['t'][-1]/reduction)

# Build timeseries and frequencyseries object for X,Y,Z
tdi_ts = dict([(k, TimeSeries(td[k][:int(len(td[k][:])/reduction)], dt=dt, t0=td.t[0])) for k in ["X", "Y", "Z"]])
# tdi_ts_o = deepcopy(tdi_ts)
# tdi_fs_o = xr.Dataset(dict([(k, tdi_ts_o[k].ts.fft(win=window)) for k in ["X", "Y", "Z"]]))
GB = fastGB.FastGB(delta_t=dt, T=Tobs)  # in seconds

plt.figure()
plt.plot(tdi_ts['X'].t, tdi_ts['X'].values)
plt.plot(tdi_ts['Y'].t, tdi_ts['Y'].values)
plt.plot(tdi_ts['Z'].t, tdi_ts['Z'].values)
plt.show()

if add_gaps:
    DATAPATH_spritz = grandparent+"/LDC/Spritz/data"
    data_fn_spritz = DATAPATH_spritz + "/LDC2_spritz_vgb_training_v2.h5"
    fid_spritz = h5py.File(data_fn_spritz)

    td_obs = fid_spritz["obs/tdi"][()]
    td_obs = np.rec.fromarrays(list(td_obs.T), names=["t", "X", "Y", "Z"])
    td_obs = td_obs['t']
    td_clean = fid_spritz["clean/tdi"][()]
    td_clean = np.rec.fromarrays(list(td_clean.T), names=["t", "X", "Y", "Z"])
    td_clean = td_clean['t']
    td_galaxy = fid_spritz["gal/tdi"][()]
    td_galaxy = np.rec.fromarrays(list(td_galaxy.T), names=["t", "X", "Y", "Z"])
    td_galaxy = td_galaxy['t']

    tdi = fid_spritz["obs/tdi"][()].squeeze()
    tdi_nf = fid_spritz["noisefree/tdi"][()].squeeze()
    dt_spritz = tdi['t'][1]-tdi['t'][0]

    # Build timeseries and frequencyseries object for X,Y,Z
    tdi_ts = dict([(k, TimeSeries(td[k][:int(len(td[k][:])/reduction)], dt=dt, t0=td.t[0])) for k in ["X", "Y", "Z"]])
    tdi_ts_obs = dict([(k, TimeSeries(td_obs[k][:int(len(td_obs[k][:])/reduction)], dt=dt_spritz, t0=td_obs.t[0])) for k in ["X", "Y", "Z"]])
    tdi_ts_clean = dict([(k, TimeSeries(td_clean[k][:int(len(td_clean[k][:])/reduction)], dt=dt_spritz, t0=td_clean.t[0])) for k in ["X", "Y", "Z"]])
    tdi_ts_galaxy = dict([(k, TimeSeries(td_galaxy[k][:int(len(td_galaxy[k][:])/reduction)], dt=dt_spritz, t0=td_galaxy.t[0])) for k in ["X", "Y", "Z"]])

    tdi_ts_glitches = deepcopy(tdi_ts_obs)
    for k in ["X", "Y", "Z"]:
        tdi_ts_glitches[k].values = tdi_ts_obs[k].values - tdi_ts_clean[k].values - tdi_ts_galaxy[k].values


    ## add gaps to tdi
    tdi_ts_with_glitches = deepcopy(tdi_ts)
    if dataset != 'Spritz':
        for k in ["X", "Y", "Z"]:
            tdi_ts_with_glitches[k].values = tdi_ts[k].values + tdi_ts_glitches[k].values - tdi_ts_glitches[k].values
    if dataset == 'Spritz':
        for k in ["X", "Y", "Z"]:
            # tdi_ts_with_glitches[k].values = tdi_ts_clean[k].values + tdi_ts_glitches[k].values - tdi_ts_glitches[k].values + tdi_ts_galaxy[k].values
            tdi_ts_with_glitches[k].values = tdi_ts_clean[k].values + tdi_ts_galaxy[k].values ### no gaps
            # tdi_ts_with_glitches[k].values = tdi_ts[k].values - tdi_ts_glitches[k].values
    tdi_ts = deepcopy(tdi_ts_with_glitches)

if zero_gaps:
    gaps = {}
    for k in ["X", "Y", "Z"]:
        gap = np.isnan(tdi_ts_with_glitches[k])
        tdi_ts_with_glitches[k][gap] = 0
        gaps[k] = tdi_ts_with_glitches[k] == 0
        # gaps = np.isnan(tdi_ts_with_glitches[k])
        tdi_ts_with_glitches[k][gaps[k]] = 0

if zero_glitches:
    for k in ["X", "Y", "Z"]:
        mad = scipy.stats.median_abs_deviation(tdi_ts_with_glitches[k])
        peaks, properties = scipy.signal.find_peaks(np.abs(tdi_ts_with_glitches[k]), height=10*mad, threshold=None, distance=1)
        # Turning glitches into gaps
        for pk in peaks:
            tdi_ts_with_glitches[k][pk-10:pk+10] = 0.0

    for k in ["X", "Y", "Z"]:
        tdi_ts_with_glitches[k][:300] = 0



if fill_gaps:
    for k in ["X", "Y", "Z"]:
        groups = []
        gaps = tdi_ts_with_glitches[k] == 0
        gaps = tdi_ts_with_glitches[k][gaps]
        start_points = []
        if gaps.t[0].values == tdi_ts_with_glitches[k].t[0].values:
            start_points = [0]
        differences = gaps.t[1:].values - gaps.t[:-1].values
        jumps = differences > 15
        end_points = list(gaps[:-1][jumps].t.values) + list([gaps.t[-1].values])
        start_points = list(start_points) + list(gaps[1:][jumps].t.values)

        for i in range(len(start_points)):
            index_start = int((start_points[i]-tdi_ts_with_glitches[k].t[0].values)/dt)
            index_end = int((end_points[i]-tdi_ts_with_glitches[k].t[0])/dt)
            length_gap = len(tdi_ts_with_glitches[k][index_start:index_end])+1
            tdi_ts_with_glitches[k][index_start:index_start+length_gap] = tdi_ts_with_glitches[k][index_end+1:index_end+1+length_gap].values
    tdi_ts = deepcopy(tdi_ts_with_glitches)

tdi_fs = xr.Dataset(dict([(k, tdi_ts[k].ts.fft(win=window)) for k in ["X", "Y", "Z"]]))

noise_model = "SciRDv1"
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


def tdi_subtraction(tdi_fs,found_sources_mp_subtract, ratio = 1):
    #subtract the found sources from original
    tdi_fs_subtracted2 = deepcopy(tdi_fs)
    for i in range(len(found_sources_mp_subtract)):
        # for j in range(len(found_sources_to_subtract[i])):
            Xs_subtracted, Ys_subtracted, Zs_subtracted = GB.get_fd_tdixyz(template=found_sources_mp_subtract[i], oversample=4, tdi2=tdi2)
            source_subtracted = dict({"X": Xs_subtracted, "Y": Ys_subtracted, "Z": Zs_subtracted})
            index_low = np.searchsorted(tdi_fs_subtracted2["X"].f, Xs_subtracted.f[0])
            try:
                if np.abs(Xs_subtracted.f[0] - tdi_fs_subtracted2["X"].f[index_low-1]) < np.abs(Xs_subtracted.f[0] - tdi_fs_subtracted2["X"].f[index_low]):
                # if tdi_fs_subtracted2["X"].f[index_low] > Xs_subtracted.f[0]:
                    index_low = index_low-1
            except:
                pass
            index_high = index_low+len(Xs_subtracted)
            for k in ["X", "Y", "Z"]:
                tdi_fs_subtracted2[k].data[index_low:index_high] -= source_subtracted[k].data * ratio
    return tdi_fs_subtracted2

load_category = False
if load_category:
    try:
        cat = np.load(SAVEPATH+'cat_sorted.npy', allow_pickle = True)
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

        cat = np.rec.fromarrays(params_dgb, names=list(names_dgb))
        indexes = np.argsort(cat['Frequency'])
        cat = cat[indexes]
        np.save(SAVEPATH+'cat_sorted.npy',cat)

# LDC1-4 #####################################
frequencies = []
frequencies_even = []
frequencies_odd = []
# search_range = [0.00398, 0.0041]
# search_range = [0.0039885, 0.0040205]
# search_range = [0.0039935, 0.0039965]
f_Nyquist = 1/dt/2
search_range = [0.0003, f_Nyquist]
# search_range = [0.000299, f_Nyquist]
# search_range = [0.0003, 0.03]
# search_range = [0.0019935, 0.0020135]
# search_range = [0.0029935, 0.0030135]
# window_length = 1*10**-7 # Hz

frequencies = create_frequency_windows(search_range, Tobs)


frequencies_half_shifted = []
for i in range(len(frequencies)-1):
    frequencies_half_shifted.append([(frequencies[i][1]-frequencies[i][0])/2 +frequencies[i][0],(frequencies[i+1][1]-frequencies[i+1][0])/2 +frequencies[i+1][0]])
# frequencies = frequencies_half_shifted
frequencies_even = frequencies[::2]
frequencies_odd = frequencies[1::2]

##### plot number of signals per frequency window
# frequencies_search = frequencies[::10]
# pGB_injected = []
# for j in range(len(frequencies_search)):
#     padding = (frequencies_search[j][1] - frequencies_search[j][0])/2 *0
#     index_low = np.searchsorted(cat_sorted['Frequency'], frequencies_search[j][0]-padding)
#     index_high = np.searchsorted(cat_sorted['Frequency'], frequencies_search[j][1]+padding)
#     if cat_sorted['Frequency'][index_high] < frequencies_search[j][1]:
#         index_high -= 1
#     indexesA = np.argsort(-cat_sorted[index_low:index_high]['Amplitude'])
#     pGB_injected_window = []
#     pGB_stacked = {}
#     for parameter in parameters:
#         pGB_stacked[parameter] = cat_sorted[parameter][index_low:index_high][indexesA]
#     for i in range(len(cat_sorted['Amplitude'][index_low:index_high])):
#         pGBs = {}
#         for parameter in parameters:
#             pGBs[parameter] = pGB_stacked[parameter][i]
#         pGB_injected_window.append(pGBs)
#     pGB_injected.append(pGB_injected_window)

# counts = np.zeros(len(pGB_injected))
# for i in range(len(pGB_injected)):
#     counts[i] = len(pGB_injected[i])

# frequencies_search = np.asarray(frequencies_search)
# figure = plt.figure()
# plt.loglog(frequencies_search[:,1],counts, '.')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Number of signals')
# plt.show()
# figure = plt.figure()
# plt.loglog(frequencies_search[:,1],frequencies_search[:,1]-frequencies_search[:,0],  linewidth= 4, label= 'Frequency window width')
# plt.loglog(frequencies_search[:,1],np.ones(len(frequencies_search[:,1]))*4*32*10**-9*2, label= 'LISA rotation')
# plt.loglog(frequencies_search[:,1],frequencies_search[:,1]*3* 10**-4, label= 'Doppler modulation')
# plt.loglog(frequencies_search[:,1],frequency_derivative(frequencies_search[:,1],2)*Tobs, label= '$\dot{f}_{max} \cdot T_{obs}$')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Frequency window witdh [Hz]')
# plt.ylim(bottom=(frequencies_search[0,1]-frequencies_search[0,0])/10**1)
# plt.legend()
# plt.show()


# save_name = 'not_anticorrelatedLDC1-4_2_optimized_second'
save_name = 'Sangria_12m_filled_anticorrelated'
save_name = 'not_anticorrelated_original_Sangria_12m_mbhb_SNR9_seed1'
save_name = 'not_anticorrelated_original_Sangria_52w_mbhb_SNR9_seed1'
# save_name = 'Spritz_clean_galaxy'
# save_name = 'Radler_24m'
# if add_gaps:
#     save_name = save_name + '_gaps'
# if VGB:
#     save_name = save_name + '_VGB'
# save_name = dataset + '_VGB_gaps'
# save_name = 'Sangria_1_dynamic_noise'
# save_name = 'not_anticorrelatedRadler_half_dynamic_noise'
# for i in range(65):
frequencies_search = frequencies
frequencies_search_full = deepcopy(frequencies_search)
# batch_index = int(sys.argv[1])

index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.00399)+99

batch_index = 308
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.003977)
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.00399)
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.00404)
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.00264612)
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.007977)
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.016308)-1
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.00545)-1
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.001373)-1
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.02355)-1
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.01488)-1
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], cat[-1]['Frequency'])-5
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.0004)-1
batch_size = 10
start_index = batch_size*batch_index
# print('batch',batch_index, start_index)
# frequencies_search = frequencies_search[start_index:start_index+batch_size]

# print(i, frequencies_search[0])
### highest + padding has to be less than f Nyqist
while frequencies_search[-1][1] + (frequencies_search[-1][1] - frequencies_search[-1][0])/2 > f_Nyquist:
    frequencies_search = frequencies_search[:-1]
# frequencies_search = frequencies_search[70:80]
# frequencies_search = frequencies_search[25:]

if dataset == 'Radler' and VGB:
    frequencies_search = []
    for i in range(len(cat)):
        start_index = np.searchsorted(np.asarray(frequencies)[:,0], cat[i]['Frequency'])-1
        frequencies_search.append(frequencies[start_index])


do_print = False
if do_print:
    found_sources_mp = np.load(SAVEPATH+'found_sources_' +save_name+'.npy', allow_pickle = True)
    # found_sources_mp = np.load(SAVEPATH+'found_sources' +save_name+'optimized.npy', allow_pickle = True)

    found_sources_in_flat = []
    for i in range(len(found_sources_mp)):
        for j in range(len(found_sources_mp[i])):
            found_sources_in_flat.append(found_sources_mp[i][j])
    found_sources_in_flat = np.asarray(found_sources_in_flat)
    found_sources_in_flat_array = {attribute: np.asarray([x[attribute] for x in found_sources_in_flat]) for attribute in found_sources_in_flat[0].keys()}
    found_sources_in_flat_df = pd.DataFrame(found_sources_in_flat_array)
    found_sources_in_flat_df = found_sources_in_flat_df.sort_values('Frequency')
    found_sources_in = []
    for i in range(len(frequencies_search)):
        found_sources_in.append(found_sources_in_flat_df[(found_sources_in_flat_df['Frequency'] > frequencies_search[i][0]) & (found_sources_in_flat_df['Frequency'] < frequencies_search[i][1])].to_dict(orient='records'))


    # found_sources_in_flat = []
    # found_sources_in_flat_frequency = []
    # number_of_found_flat = 0
    # for i in range(len(found_sources_mp)):
    #     for j in range(len(found_sources_mp[i][3])):
    #         found_sources_in_flat.append(found_sources_mp[i][3][j])
    #         found_sources_in_flat_frequency.append(found_sources_in_flat[-1]['Frequency'])
    #         number_of_found_flat += 1
    # found_sources_in_flat_frequency = np.asarray(found_sources_in_flat_frequency)
    # found_sources_in_flat = np.asarray(found_sources_in_flat)
    # indexes_in = np.argsort(found_sources_in_flat_frequency)
    # found_sources_in_flat_frequency = found_sources_in_flat_frequency[indexes_in]
    # found_sources_in_flat = found_sources_in_flat[indexes_in]

    # found_sources_in = []
    # for i in range(len(frequencies_search)):
    #     lower_index = np.searchsorted(found_sources_in_flat_frequency,frequencies_search[i][0])
    #     higher_index = np.searchsorted(found_sources_in_flat_frequency,frequencies_search[i][1])
    #     if i == 102:
    #         print(lower_index, higher_index)
    #     found_sources_in.append(found_sources_in_flat[lower_index:higher_index])

    pGB_injected = []
    for j in range(len(frequencies_search)):
        padding = (frequencies_search[j][1] - frequencies_search[j][0])/2 *0
        index_low = np.searchsorted(cat['Frequency'], frequencies_search[j][0]-padding)
        try:
            if np.abs(frequencies_search[j][0]-padding - cat['Frequency'][index_low-1]) < np.abs(frequencies_search[j][0]-padding - cat['Frequency'][index_low]):
                index_low = index_low-1
        except:
            pass
        index_high = np.searchsorted(cat['Frequency'], frequencies_search[j][1]+padding)
        try:
            if cat['Frequency'][index_high] < frequencies_search[j][1]:
                index_high -= 1
        except:
            pass
        indexesA = np.argsort(-cat[index_low:index_high]['Amplitude'])
        pGB_injected_window = []
        pGB_stacked = {}
        for parameter in parameters:
            pGB_stacked[parameter] = cat[parameter][index_low:index_high][indexesA]
        for i in range(len(cat['Amplitude'][index_low:index_high])):
            pGBs = {}
            for parameter in parameters:
                pGBs[parameter] = pGB_stacked[parameter][i]
            pGB_injected_window.append(pGBs)
        pGB_injected.append(pGB_injected_window)



def hamiltonian_monte_carlo(n_samples, negative_log_prob, grad_log_prob, initial_position, path_len=0.1, step_size=0.5):
    """Run Hamiltonian Monte Carlo sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples to return
    negative_log_prob : callable
        The negative log probability to sample from
    initial_position : np.array
        A place to start sampling from.
    path_len : float
        How long each integration path is. Smaller is faster and more correlated.
    step_size : float
        How long each integration step is. Smaller is slower and more accurate.

    Returns
    -------
    np.array
        Array of length `n_samples`.
    """
    # autograd magic
    dVdq = grad_log_prob

    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = scipy.stats.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]
    for p0 in momentum.rvs(size=size):
        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog(
            samples[-1],
            p0,
            dVdq,
            path_len=path_len,
            step_size=step_size,
        )

        # Check Metropolis acceptance criterion
        start_log_p = negative_log_prob(samples[-1]) - np.sum(momentum.logpdf(p0))
        new_log_p = negative_log_prob(q_new) - np.sum(momentum.logpdf(p_new))
        if np.log(np.random.rand()) < start_log_p - new_log_p:
            samples.append(q_new)
        else:
            samples.append(np.copy(samples[-1]))

    return np.array(samples[1:])

def leapfrog(q, p, dVdq, path_len, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo.

    Parameters
    ----------
    q : np.floatX
        Initial position
    p : np.floatX
        Initial momentum
    dVdq : callable
        Gradient of the velocity
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be

    Returns
    -------
    q, p : np.floatX, np.floatX
        New position and momentum
    """
    q, p = np.copy(q), np.copy(p)

    p -= step_size * dVdq(q) / 2  # half step
    for _ in range(int(path_len / step_size) - 1):
        q += step_size * p  # whole step
        p -= step_size * dVdq(q)  # whole step
    q += step_size * p  # whole step
    p -= step_size * dVdq(q) / 2  # half step

    # momentum flip at end
    return q, -p

class Posterior_computer():
    def __init__(self, tdi_fs, Tobs, frequencies, maxpGB, noise=None, use_gpu=False) -> None:
        self.tdi_fs = tdi_fs
        self.Tobs = Tobs
        self.frequencies = frequencies
        self.maxpGB = maxpGB
        self.use_gpu = use_gpu
        self.search1 = Search(self.tdi_fs,self.Tobs, self.frequencies[0], self.frequencies[1], noise=noise, gb_gpu=gb_gpu, use_gpu=use_gpu)

    def reduce_boundaries(self, plot_confidance = False):
        fisher_information_matrix = self.search1.fisher_information(self.maxpGB)
        FIM = np.zeros((len(parameters),len(parameters)))
        for i,parameter1 in enumerate(parameters):
            for j,parameter2 in enumerate(parameters):
                FIM[i,j] = fisher_information_matrix[parameter1][parameter2]
        covariance_matrix = scipy.linalg.inv(FIM)
        maxpGB01 = scaleto01(self.maxpGB, self.search1.boundaries, self.search1.parameters, self.search1.parameters_log_uniform)

        # lambda_, v = scipy.linalg.eig(covariance_matrix)
        # transf = np.zeros((len(lambda_),len(lambda_)))
        # for i in range(len(lambda_)):
            # transf[i] = v[i] * np.sqrt(lambda_[i])
        # covariance2 = v @ np.diag(lambda_) @ scipy.linalg.inv(v)
        # transf = v @ np.diag(np.sqrt(lambda_))
        # scalematrix = np.max(np.abs(transf), axis=1)
        scalematrix = np.sqrt(np.diag(covariance_matrix))
        # print('scalematrix', scalematrix)
        maxpGB01_low = {}
        maxpGB01_high = {}
        self.boundaries_reduced = {}
        sigma_multiplier = 4
        for parameter in parameters:
            maxpGB01_low[parameter] = maxpGB01[parameter] - scalematrix[parameters.index(parameter)] * sigma_multiplier 
            maxpGB01_high[parameter] = maxpGB01[parameter] + scalematrix[parameters.index(parameter)] * sigma_multiplier 
            if parameter in [ 'Amplitude', 'Inclination']:
                maxpGB01_low[parameter] = maxpGB01[parameter] - scalematrix[parameters.index(parameter)] * sigma_multiplier
                maxpGB01_high[parameter] = maxpGB01[parameter] + scalematrix[parameters.index(parameter)] * sigma_multiplier
            if parameter in [ 'InitialPhase', 'Polarization']:
                maxpGB01_low[parameter] = maxpGB01[parameter] - 0.001
                maxpGB01_high[parameter] = maxpGB01[parameter] + 0.001
            if parameter in [ 'Frequency']:
                mulitplier = 1
                if np.abs(self.search1.boundaries['Frequency'][1] - self.search1.boundaries['Frequency'][0])  > 1e-4:
                    mulitplier = 0.2
                maxpGB01_low[parameter] = maxpGB01[parameter] - scalematrix[parameters.index(parameter)]  * mulitplier
                maxpGB01_high[parameter] = maxpGB01[parameter] + scalematrix[parameters.index(parameter)] * mulitplier
            if parameter == 'FrequencyDerivative':
                # print('scale',scalematrix[parameters.index(parameter)])
                # if scalematrix[parameters.index(parameter)] > 0.07:
                print('scale', scalematrix[parameters.index(parameter)], 'fd')
                range_fd = 1
                # if self.search1.boundaries['FrequencyDerivative'][1] > 1e-14:
                #     range_fd = 0.02
                # if self.search1.boundaries['FrequencyDerivative'][1] > 1e-17:
                #     range_fd = 0.4
                maxpGB01_low[parameter] = maxpGB01[parameter] - range_fd
                maxpGB01_high[parameter] = maxpGB01[parameter] + range_fd
            if maxpGB01_low[parameter] > maxpGB01_high[parameter]:
                placeholder = deepcopy(maxpGB01_low[parameter])
                maxpGB01_low[parameter] = deepcopy(maxpGB01_high[parameter])
                maxpGB01_high[parameter] = deepcopy(placeholder)
            if maxpGB01_low[parameter] < 0:
                maxpGB01_low[parameter] = 0
            if maxpGB01_high[parameter] > 1:
                maxpGB01_high[parameter] = 1
            maxpGB_fisher_low = (maxpGB01_low[parameter] * (self.search1.boundaries[parameter][1] - self.search1.boundaries[parameter][0])) + self.search1.boundaries[parameter][0]
            maxpGB_fisher_high = (maxpGB01_high[parameter] * (self.search1.boundaries[parameter][1] - self.search1.boundaries[parameter][0])) + self.search1.boundaries[parameter][0]
            self.boundaries_reduced[parameter] = [maxpGB_fisher_low, maxpGB_fisher_high]
        # self.boundaries_reduced = deepcopy(boundaries_reduced_fisher)

        # correct Frequency Derivative
        # split_fd = -17
        # if self.boundaries_reduced['FrequencyDerivative'][1] < split_fd+1:
        #     self.boundaries_reduced['FrequencyDerivative'][0] = -18.5
        #     self.boundaries_reduced['FrequencyDerivative'][1] = -16
        # elif self.boundaries_reduced['FrequencyDerivative'][0] < split_fd+0.5:
        #     self.boundaries_reduced['FrequencyDerivative'][0] = -18.5
        # correct Inclination and Amplitude
        # split_inclination = 0.95
        # if self.boundaries_reduced['Inclination'][1] > split_inclination or self.boundaries_reduced['Inclination'][0] < -split_inclination:
        #     if self.boundaries_reduced['Inclination'][1] > split_inclination and self.boundaries_reduced['Inclination'][0] < -split_inclination:
        #         if maxpGB01['Inclination'] > 0.5:
        #             self.boundaries_reduced['Inclination'][0] = 0.5
        #             self.boundaries_reduced['Inclination'][1] = 1
        #         else:
        #             self.boundaries_reduced['Inclination'][0] = -1
        #             self.boundaries_reduced['Inclination'][1] = -0.5
        #     if self.boundaries_reduced['Inclination'][1] > split_inclination:
        #         self.boundaries_reduced['Inclination'][0] = 0.5
        #         self.boundaries_reduced['Inclination'][1] = 1
        #     if self.boundaries_reduced['Inclination'][0] < -split_inclination:
        #         self.boundaries_reduced['Inclination'][0] = -1
        #         self.boundaries_reduced['Inclination'][1] = -0.5
        #     parameter = 'Amplitude'
        #     maxpGB01_low[parameter]  = maxpGB01[parameter] - 0.08
        #     maxpGB01_high[parameter]  = maxpGB01[parameter] + 0.08
        #     if maxpGB01_low[parameter] > maxpGB01_high[parameter]:
        #         placeholder = deepcopy(maxpGB01_low[parameter])
        #         maxpGB01_low[parameter] = deepcopy(maxpGB01_high[parameter])
        #         maxpGB01_high[parameter] = deepcopy(placeholder)
        #     if maxpGB01_low[parameter] < 0:
        #         maxpGB01_low[parameter] = 0
        #     if maxpGB01_high[parameter] > 1:
        #         maxpGB01_high[parameter] = 1
        #     maxpGB_fisher_low = (maxpGB01_low[parameter] * (self.search1.boundaries[parameter][1] - self.search1.boundaries[parameter][0])) + self.search1.boundaries[parameter][0]
        #     maxpGB_fisher_high = (maxpGB01_high[parameter] * (self.search1.boundaries[parameter][1] - self.search1.boundaries[parameter][0])) + self.search1.boundaries[parameter][0]
        #     self.boundaries_reduced[parameter] = [maxpGB_fisher_low, maxpGB_fisher_high]
        # print('boundaries reduced', self.boundaries_reduced)
    
    def reduce_boundaries_from_kde(self, resolution=10**6, proposal=None):
        test_x_m, probability = self.get_samples(resolution=10**6, proposal=proposal)
        min_parameters = np.min(test_x_m, axis=0)
        max_parameters = np.max(test_x_m, axis=0)
        self.boundaries_reduced_kde = deepcopy(self.boundaries_reduced)
        for i, parameter in enumerate(parameters):
            if parameter in ['InitialPhase', 'Polarization']:
                continue
            length = self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0] 
            self.boundaries_reduced_kde[parameter][0] = min_parameters[i]*length + self.boundaries_reduced[parameter][0]
            self.boundaries_reduced_kde[parameter][1] = max_parameters[i]*length + self.boundaries_reduced[parameter][0]
        # self.boundaries_reduced = self.boundaries_reduced_kde
        return self.boundaries_reduced_kde
    
    def reduce_boundaries_from_samples(self, samples):
        min_parameters = np.min(samples, axis=0)
        max_parameters = np.max(samples, axis=0)
        self.boundaries_reduced_samples = deepcopy(self.boundaries_reduced)
        for i, parameter in enumerate(parameters):
            if parameter in ['InitialPhase', 'Polarization']:
                continue
            length = self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0] 
            self.boundaries_reduced_samples[parameter][0] = min_parameters[i]*length + self.boundaries_reduced[parameter][0]
            self.boundaries_reduced_samples[parameter][1] = max_parameters[i]*length + self.boundaries_reduced[parameter][0]
        return self.boundaries_reduced_samples
    
    def train_model(self):
        rmse = 2
        train_size = 0
        test_size = 500
        added_trainig_size = 1000
        j = 0
        samples = np.random.rand(6000,8)
        samples[0] = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
        samples[test_size] = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
        while rmse > 0.6 and j < 5:
            j += 1
            train_size += added_trainig_size
            if j == 1:
                resolution = train_size + test_size
            else:
                resolution = added_trainig_size
            start = time.time()
            samples_likelihood = np.zeros(resolution)
            samples_likelihood2 = np.zeros(resolution)
            incl = np.zeros(resolution)
            for i in range(resolution):
                samples_p = scaletooriginal(samples[i+j*added_trainig_size], self.boundaries_reduced, self.search1.parameters, self.search1.parameters_log_uniform)
                incl[i] = samples_p['Inclination']
                samples_likelihood[i] = self.search1.loglikelihood([samples_p])
            print('sample time of', resolution, 'samples ',time.time() - start)

            samples_flat = np.zeros((resolution  , len(parameters)))
            i = 0
            boundary_ratio = 1# (boundaries_reduced1['FrequencyDerivative'][1]-boundaries_reduced1['FrequencyDerivative'][0])/(boundaries_reduced2['FrequencyDerivative'][1]-boundaries_reduced1['FrequencyDerivative'][0])
            for parameter in parameters:
                if parameter == 'FrequencyDerivative':
                    samples_flat[:, i] = samples[j*added_trainig_size:j*added_trainig_size+resolution,parameters.index(parameter)]*boundary_ratio
                else:
                    samples_flat[:, i] = samples[j*added_trainig_size:j*added_trainig_size+resolution,parameters.index(parameter)]
                i += 1
            # samples_flat = samples_flat*2-1
            if j == 1:
                train_y = samples_likelihood[test_size:]
                test_y = samples_likelihood[:test_size]
                train_x = samples_flat[test_size:]
                test_x = samples_flat[:test_size]
            else:
                train_y = np.append(train_y, samples_likelihood)
                train_x = np.append(train_x, samples_flat,axis=0)

            self.mu = np.mean(train_y)
            self.sigma = np.std(train_y)
            train_y_normalized = (train_y - self.mu) / self.sigma
            kernel = RBF(length_scale=[1,2,5,1,1,1,1,1],length_scale_bounds=[(0.1,10),(0.1,10),(0.1,10),(0.1,10),(0.1,10),(0.1,10),(0.1,100),(0.1,100)])
            start = time.time()
            self.gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(train_x, train_y_normalized)
            print('train',time.time() - start)
            start = time.time()
            observed_pred_sk = self.gpr.predict(test_x)
            print('eval time of ', test_size, 'samples: ',time.time() - start)
            observed_pred_sk_scaled = observed_pred_sk*self.sigma + self.mu
            rmse = np.sqrt(mean_squared_error(test_y,observed_pred_sk_scaled))
            print("RMSE ",rmse,'with training size', len(train_y))
            if rmse > 30:
                print('high RMSE')

            # fig = plt.figure(figsize=(15,6))
            # plt.scatter(train_x[:,0], train_x[:,5], c=train_y, cmap='gray')
            # plt.show()
            if rmse > 5 and j == 5:
                j = 0
                samples = np.random.rand(6000,8)
                samples[0] = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
                samples[test_size] = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
                train_size = 0

    def evaluate(self, x):
        partial_length = 1*10**3
        # start = time.time()
        observed_pred_mean = np.zeros(len(x))
        observed_pred_sk = np.zeros(len(x))
        for n in range(int(len(x)/partial_length)):
            observed_pred_sk[n*partial_length:(n+1)*partial_length] = self.gpr.predict(x[(n)*partial_length:(n+1)*partial_length])
        try:
            observed_pred_sk[int(len(x)/partial_length)*partial_length:] = self.gpr.predict(x[int(len(x)/partial_length)*partial_length:])
        except:
            pass
        observed_pred_sk = np.asarray(observed_pred_sk)
        observed_pred_sk = observed_pred_sk.reshape(len(x))
        observed_pred_mean[:len(x)] = observed_pred_sk[:len(x)]*self.sigma + self.mu
        # print('eval time', time.time()-start)
        return observed_pred_mean
    
    def sampler(self, resolution=1000, path_len=0.01, step_size=0.01):
        x0 = np.asarray([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
        samples = hamiltonian_monte_carlo(n_samples=resolution, negative_log_prob=self.logp_func, grad_log_prob=self.dlogp_func, initial_position= x0, path_len=0.1, step_size=0.01)
        return samples

    def sample_dist(self, data0, data1, numPoints, resolution):
        data = np.array([data0, data1]).T
        grid, mypdf = FFTKDE(bw=0.01, kernel='gaussian').fit(data).evaluate(numPoints)
        axes = np.unique(grid[:, 0]), np.unique(grid[:, 1])
        mypdf = mypdf.reshape(numPoints, numPoints).T
        mypdf = np.clip(mypdf, a_min=0, a_max=None)

        dist = Distribution(mypdf, transform=lambda i:i/len(mypdf[0,:])*(axes[0][-1]-axes[0][0])+axes[0][0])
        data, pdfs = dist(resolution)
        return data, pdfs
    
    def sample_dist_fastkde(self, data0, data1, numPoints, resolution):
        ax = np.linspace(-0.15,1.15, numPoints)
        mypdf,axes = fastKDE.pdf(data0, data1, axes=[ax,ax])
        while np.min(mypdf) < 0:
            ax_expand = (np.random.rand(1)*0.2+0.1)[0]
            ax = np.linspace(-ax_expand,1+ax_expand,numPoints)
            mypdf,axes = fastKDE.pdf(data0, data1, axes=[ax,ax])

        dist = Distribution(mypdf, transform=lambda i:i/len(mypdf[0,:])*(axes[0][-1]-axes[0][0])+axes[0][0])
        data, pdfs = dist(resolution)
        return data, pdfs

    def get_samples(self, resolution=1000, proposal= None):
        numPoints = 2**5+1
        if proposal is None:
            test_x_m = np.random.uniform(size=(resolution,len(parameters)))
            probability = np.ones(resolution)
        else:
            probability = np.ones(resolution)
            test_x_m = np.zeros((resolution,len(parameters)))
            # [test_x_m[:,2],test_x_m[:,1]], pdfs = self.sample_dist(proposal[:,1],proposal[:,2], numPoints, resolution)
            # probability *= pdfs

            # [test_x_m[:,5],test_x_m[:,0]], pdfs = self.sample_dist(proposal[:,0],proposal[:,5], numPoints, resolution)
            # probability *= pdfs

            # [test_x_m[:,4],test_x_m[:,3]], pdfs = self.sample_dist(proposal[:,3],proposal[:,4], numPoints, resolution)
            # probability *= pdfs

            # plt.figure()
            # plt.hist2d(test_x_m[:,5],test_x_m[:,0])
            # plt.show()

            [test_x_m[:,2],test_x_m[:,1]], pdfs = self.sample_dist_fastkde(proposal[:,1],proposal[:,2], numPoints, resolution)
            probability *= pdfs

            [test_x_m[:,5],test_x_m[:,0]], pdfs = self.sample_dist_fastkde(proposal[:,0],proposal[:,5], numPoints, resolution)
            probability *= pdfs

            [test_x_m[:,4],test_x_m[:,3]], pdfs = self.sample_dist_fastkde(proposal[:,3],proposal[:,4], numPoints, resolution)
            probability *= pdfs

            # data = np.array([proposal[:,0],proposal[:,5]]).T
            # grid, mypdf = FFTKDE(bw=0.1, kernel='gaussian').fit(data).evaluate(numPoints)
            # axes = np.unique(grid[:, 0]), np.unique(grid[:, 1])
            # mypdf = mypdf.reshape(numPoints, numPoints).T
            # mypdf = np.clip(mypdf, a_min=0, a_max=None)
            # dist = Distribution(mypdf, transform=lambda i:i/len(mypdf[0,:])*(axes[0][-1]-axes[0][0])+axes[0][0])
            # data, pdfs = dist(resolution)
            # test_x_m[:,0] = data[1]
            # test_x_m[:,5] = data[0]
            # plt.figure()
            # plt.scatter(np.log10(test_x_m[:10000,0]),np.arccos(test_x_m[:10000,5]* (boundaries_reduced['Inclination'][1] - boundaries_reduced['Inclination'][0]) + boundaries_reduced['Inclination'][0]), c=pdfs[:10000])
            # probability *= pdfs

            ### fastkde
            # ax = np.linspace(-0.15,1.15,numPoints)
            # mypdf,axes = fastKDE.pdf(proposal[:,1],proposal[:,2], axes=[ax,ax])
            # #a pdf can not be negative
            # while np.min(mypdf) < 0:
            #     ax_expand = (np.random.rand(1)*0.2+0.1)[0]
            #     ax = np.linspace(-ax_expand,1+ax_expand,numPoints)
            #     mypdf,axes = fastKDE.pdf(data[0], data[1], axes=[ax,ax])

            # dist = Distribution(mypdf, transform=lambda i:i/len(mypdf[0,:])*(axes[0][-1]-axes[0][0])+axes[0][0])
            # data, pdfs = dist(resolution)
            # test_x_m[:,1] = data[1]
            # test_x_m[:,2] = data[0]
            # probability *= pdfs

            # ax = np.linspace(-0.15,1.15,numPoints)
            # mypdf,axes = fastKDE.pdf(proposal[:,0],proposal[:,5], axes=[ax,ax])
            # #a pdf can not be negative
            # while np.min(mypdf) < 0:
            #     ax_expand = (np.random.rand(1)*0.2+0.1)[0]
            #     ax = np.linspace(-ax_expand,1+ax_expand,numPoints)
            #     mypdf,axes = fastKDE.pdf(data[0], data[1], axes=[ax,ax])
            # dist = Distribution(mypdf, transform=lambda i:i/len(mypdf[0,:])*(axes[0][-1]-axes[0][0])+axes[0][0])
            # data, pdfs = dist(resolution)
            # test_x_m[:,0] = data[1]
            # test_x_m[:,5] = data[0]
            # probability *= pdfs

            # ax = np.linspace(-0.15,1.15,numPoints)
            # mypdf,axes = fastKDE.pdf(proposal[:,3],proposal[:,4], axes=[ax,ax])
            # #a pdf can not be negative
            # while np.min(mypdf) < 0:
            #     ax_expand = (np.random.rand(1)*0.2+0.1)[0]
            #     ax = np.linspace(-ax_expand,1+ax_expand,numPoints)
            #     mypdf,axes = fastKDE.pdf(data[0], data[1], axes=[ax,ax])
            # dist = Distribution(mypdf, transform=lambda i:i/len(mypdf[0,:])*(axes[0][-1]-axes[0][0])+axes[0][0])
            # data, pdfs = dist(resolution)
            # test_x_m[:,3] = data[1]
            # test_x_m[:,4] = data[0]
            # probability *= pdfs




            # mypdf,axes = fastKDE.pdf(proposal[:,3],proposal[:,4], axes=[ax,ax])
            # while not(np.all(mypdf>=0)):
            #     proposal = self.calculate_posterior(resolution = self.previous_resolution, proposal= self.previous_mcmc_samples, temperature= self.previous_T)
            #     mypdf,axes = fastKDE.pdf(proposal[:,3],proposal[:,4], axes=[ax,ax])
            # data = np.array([proposal[:,3],proposal[:,4]]).T
            # grid, mypdf = FFTKDE(bw=0.1, kernel='gaussian').fit(data).evaluate(numPoints)
            # axes = np.unique(grid[:, 0]), np.unique(grid[:, 1])
            # mypdf = mypdf.reshape(numPoints, numPoints).T
            # mypdf = np.clip(mypdf, a_min=0, a_max=None)
            # dist = Distribution(mypdf, transform=lambda i:i/len(mypdf[0,:])*(axes[0][-1]-axes[0][0])+axes[0][0])
            # data, pdfs = dist(resolution)
            # test_x_m[:,3] = data[1]
            # test_x_m[:,4] = data[0]
            # probability *= pdfs

            test_x_m[:,6] = np.random.uniform(size=resolution)
            test_x_m[:,7] = np.random.uniform(size=resolution)

            for n in range(8):
                index = np.where(test_x_m[:,n] > 0)
                test_x_m = test_x_m[index]
                probability = probability[index]
                index = np.where(test_x_m[:,n] < 1)
                test_x_m = test_x_m[index]
                probability = probability[index]
        return test_x_m, probability

    def calculate_posterior(self,resolution = 1*10**6, proposal= None, temperature = 1, pGB_true=None):
        test_x_m, probability = self.get_samples(resolution, proposal)
        start = time.time()

        # fig =  corner.corner(test_x_m,  bins=40, hist_kwargs={'density':True, 'lw':3}, plot_datapoints=False, fill_contours=False,  show_titles=True, \
        #                 color='#348ABD',  truth_color='k', use_math_test=True, \
        #                  levels=[0.9], title_kwargs={"fontsize": 12})
        # plt.show()

        start = time.time()
        if self.use_gpu:
            # test_x_m[0] = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
            test_x_m_scaled = scaletooriginal_array(test_x_m, self.boundaries_reduced, self.search1.parameters, self.search1.parameters_log_uniform)
            test_x_m_scaled = np.swapaxes(test_x_m_scaled, 0,1)

            params = np.array(
                [test_x_m_scaled[parameters.index('Amplitude')],
                test_x_m_scaled[parameters.index('Frequency')],
                test_x_m_scaled[parameters.index('FrequencyDerivative')],
                np.zeros_like(test_x_m_scaled[parameters.index('FrequencyDerivative')]),
                test_x_m_scaled[parameters.index('InitialPhase')],
                test_x_m_scaled[parameters.index('Inclination')],
                test_x_m_scaled[parameters.index('Polarization')],
                test_x_m_scaled[parameters.index('EclipticLongitude')],
                test_x_m_scaled[parameters.index('EclipticLatitude')]]
            )
            # for i, parameter in enumerate(['Amplitude', 'Frequency', 'FrequencyDerivative', 'InitialPhase', 'Inclination', 'Polarization', 'EclipticLongitude', 'EclipticLatitude']):
            #     j = 0
            #     if i > 2:
            #         j = 1
            #     params[i+j,0] = self.search1.pGBs[parameter]
            # params = np.array(
            #     [amp_in, f0_in, fdot_in, fddot_in, phi0_in, iota_in, psi_in, lam_in, beta_sky_in,]
            # )

            # params[1,0] = 0.00443235
            observed_pred_mean = self.search1.loglikelihood_gpu(params)
        else:
            observed_pred_mean = self.evaluate(test_x_m)

        # print('time loglikelihood for ', resolution, ' signals: ', time.time()-start)
        flatsamples = np.zeros(len(test_x_m))
        flatsamplesparameters = np.zeros((len(test_x_m),len(parameters)+1))
        i = 0
        flatsamples[:] = observed_pred_mean
        # flatsamplesparameters[:,1:] = test_x_m
        flatsamplesparameters[:,0] = observed_pred_mean

        maxindx = np.unravel_index(flatsamplesparameters[:,0].argmax(), flatsamplesparameters[:,0].shape)
        max_parameters = flatsamplesparameters[maxindx[0],1:]
        # max_loglike = flatsamplesparameters[:,0].max()
        maxpGBpredicted = scaletooriginal(max_parameters, self.boundaries_reduced, self.search1.parameters, self.search1.parameters_log_uniform)
        # print(self.search1.loglikelihood_gpu_signal(pGB_dict_to_gpu_input([maxpGBpredicted])), self.search1.loglikelihood_gpu_signal(pGB_dict_to_gpu_input([self.maxpGB])))
        # print(self.search1.loglikelihood_gpu(pGB_dict_to_gpu_input([maxpGBpredicted])), self.search1.loglikelihood_gpu(pGB_dict_to_gpu_input([self.maxpGB])))
        # print(self.search1.loglikelihood([maxpGBpredicted]), self.search1.loglikelihood([self.maxpGB]))
        if self.search1.loglikelihood([maxpGBpredicted]) > self.search1.loglikelihood([self.maxpGB]):
            maxpGB = maxpGBpredicted
            # print('better maxpGB is found',maxpGB,self.search1.loglikelihood([maxpGBpredicted]), self.search1.loglikelihood([self.maxpGB]), self.maxpGB)
        
        best_value = np.max(observed_pred_mean)
        # print("pred", max_loglike, "true", self.search1.loglikelihood([scaletooriginal(max_parameters, self.boundaries_reduced)]), "max", self.search1.loglikelihood([self.maxpGB]), self.maxpGB)

        # np.random.shuffle(flatsamplesparameters)
        start = time.time()
        # normalizer = np.sum(np.exp(flatsamplesparameters[:,0]-best_value))
        flatsamples_normalized = np.exp(flatsamplesparameters[:,0]-best_value)
        # flatsamples_normalized = flatsamplesparameters[:,0]
        mcmc_samples = []
        mcmc_samples.append(flatsamplesparameters[0,1:])
        previous_p = flatsamples_normalized[0]
        if previous_p == 0:
            previous_p == 1e-300
        current_parameters = flatsamplesparameters[0,1:]
        # probability = scipy.stats.multivariate_normal.pdf(flatsamplesparameters[:,1:], mean=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5], cov=[std_scale,std_scale,std_scale,std_scale,std_scale,std_scale,std_scale,std_scale])
        
        previous_probability = probability[0]
        accepted = 0
        for i in range(len(flatsamples_normalized)-1):
            if ((flatsamples_normalized[i+1] / previous_p) * (previous_probability/probability[i+1]))**(1/temperature) > np.random.uniform():
                previous_p = flatsamples_normalized[i+1]
                previous_probability = probability[i+1]
                current_parameters = flatsamplesparameters[i+1,1:]
                mcmc_samples.append(current_parameters)
                accepted += 1
            else:
                mcmc_samples.append(current_parameters)
        mcmc_samples = np.asarray(mcmc_samples)
        # print('time MHMC', time.time()-start)
        print('acceptance rate',np.round(accepted/len(probability)*100),'%')

        return mcmc_samples
    

    def predict(self,x,k=0):
        #x of shape (m)
        
        #returns the gp predictions where f is the true function and
        #df, ddf, If, IIf are its first and second derivate respectively antiderivates
        #the outputs are the predictions f_p,df_p,ddf_p,If_p,IIf_p where
        #f(x) = f_p(x), df(x) = df_p(x), ddf(x) = ddf_p(x), If(x) = If_p(x) + C1, 
        #IIf(x) = IIf_p(x) + C1*x + C2 with some constants C1,C2
        #set k = 0 for the normal prediction, K = 1,2 for the first or second derivates
        #and k = -1,-2 for the first or second antiderivates
    
        # x = x.reshape(-1,1)
    
        X = x - self.gpr.X_train_
        l = self.gpr.kernel_.length_scale
        A = self.gpr.alpha_

        K_trans = self.gpr.kernel_(x, self.gpr.X_train_)
        y_mean = K_trans @ self.gpr.alpha_

        f = np.prod(np.exp(-(X)**2 / (2*l**2)), axis=1)
        df = f * (-X / l ** 2).T
        
        if k == 0: 
            return f @ A
        elif k == 1: 
            return df @ A
        else:
            raise Exception('Unknown parameter k: {}'.format(k))

    def gradient(self,x):
        step_length = 0.01
        grad = np.ones_like(x)
        for i in range(len(x)):
            x_predict = np.copy(x)
            x_predict[i] = x[i]-step_length/2
            if x_predict[i] < 0:
                x_predict[i] = 0
            low = self.gpr.predict([x_predict])
            x_predict[i] = x[i]+step_length/2
            if x_predict[i] > 1:
                x_predict[i] = 1
            high = self.gpr.predict([x_predict])
            grad[i] = (high-low)/step_length
        return grad

    def predict2(self, x):
        # gets 'l' used in denominator of expected value of gradient for RBF kernel 
        k2_l = self.gpr.kernel_.length_scale

        # not necessary to do predict, but now y_pred has correct shape
        y_pred, sigma = self.gpr.predict(np.asanyarray(x).reshape(1,-1),  return_std=True)

        # allocate array to store gradient
        y_pred_grad = 0.0*y_pred

        # set of points where gradient is to be queried
        # x = np.atleast_2d(np.linspace(-5, 0.8, 1000)).T
        
        X = self.gpr.X_train_
        x_star = x

        # eval_gradient can't be true when eval site doesn't match X
        # this gives standard RBF kernel evaluations
        k_val= self.gpr.kernel_(X, np.atleast_2d(x_star), eval_gradient=False).ravel()

        # x_i - x_star / l^2
        x_diff_over_l_sq = ((X-x_star)/np.power(k2_l,2)).ravel()

        # pair-wise multiply
        intermediate_result = np.multiply(k_val, x_diff_over_l_sq)

        # dot product intermediate_result with the alphas
        final_result = np.dot(intermediate_result, self.gpr.alpha_)

        # store gradient at this point
        y_pred_grad[key] = final_result
        return y_pred_grad

    def logp_func(self,x):
        return -(self.evaluate([x])[0] * self.sigma + self.mu)

    def dlogp_func(self,x, loc=0, scale=1):
        return -(self.predict(x,k=1) * self.sigma)

    def plot_corner(self, mcmc_samples, pGB = {}, save_figure = False, save_chain = False, number_of_signal = 0, parameter_titles = False, rescaled = False):
        start = time.time()
        if not(rescaled):
            mcmc_samples_rescaled = scaletooriginal_array(mcmc_samples, self.boundaries_reduced, self.search1.parameters, self.search1.parameters_log_uniform)
            # mcmc_samples_rescaled = np.zeros(np.shape(mcmc_samples))
            # i = 0
            # for parameter in parameters:
            #     if parameter in ["EclipticLatitude"]:
            #         mcmc_samples_rescaled[:,i] = np.arcsin((mcmc_samples[:,parameters.index(parameter)] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
            #     elif parameter in ["Inclination"]:
            #         mcmc_samples_rescaled[:,i] = np.arccos((mcmc_samples[:,parameters.index(parameter)] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
            #     elif parameter in parameters_log_uniform:
            #         mcmc_samples_rescaled[:,i] = 10**((mcmc_samples[:,parameters.index(parameter)] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
            #     else:
            #         mcmc_samples_rescaled[:,i] = (mcmc_samples[:,parameters.index(parameter)] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0]
            #     i += 1
            # print('time rescale', time.time()-start)
        else:
            mcmc_samples_rescaled = mcmc_samples
        
        mcmc_samples_rescaled[:,parameters.index('Frequency')] *= 10**3 
        save_frequency = self.maxpGB['Frequency']

        if save_chain:
            df = pd.DataFrame(data=mcmc_samples_rescaled, columns=parameters)
            for parameter in ['Amplitude','FrequencyDerivative','EclipticLatitude','EclipticLongitude','Inclination','InitialPhase','Polarization']:
                df[parameter] = df[parameter].astype('float32')
            df.to_csv(SAVEPATH+'Chains_gpu/frequency'+str(int(np.round(save_frequency*10**12)))+'pHz'+save_name+'.csv',index=False)
        mcmc_samples_rescaled[:,parameters.index('Frequency')] /= 10**3 
        # start = time.time()
        # df = pd.DataFrame(data=mcmc_samples_rescaled, columns=parameters)
        # df.to_csv('/home/stefan/Repositories/ldc1_evaluation_data/submission/Stefan_LDC14/GW'+str(int(np.round(maxpGB['Frequency']*10**8)))+'.csv',index=False)
        # print('saving time', time.time()-start)

        if save_figure:
            # print('full time', time.time()-first_start)
            datS = np.zeros(np.shape(mcmc_samples))
            datS[:,0] = mcmc_samples_rescaled[:,2]
            datS[:,1] = np.sin(mcmc_samples_rescaled[:,1])
            datS[:,2] = mcmc_samples_rescaled[:,3]*10**3
            datS[:,3] = mcmc_samples_rescaled[:,4]
            datS[:,4] = np.cos(mcmc_samples_rescaled[:,5])
            datS[:,5] = np.log10(mcmc_samples_rescaled[:,0])
            datS[:,6] = mcmc_samples_rescaled[:,6]
            datS[:,7] = mcmc_samples_rescaled[:,7]
            lbls = [r'\lambda', r'\sin \beta', 'f$ $($mHz$)', r'\dot{f}$ $ ($Hz/s$)', r'\cos \iota', r'A', r'\phi', r'\Phi']

            if pGB:
                tr_s = np.zeros(len(parameters))
                i = 0
                for parameter in ['EclipticLongitude','EclipticLatitude','Frequency','FrequencyDerivative','Inclination','Amplitude','InitialPhase','Polarization']:
                    if parameter in parameters_log_uniform:
                        tr_s[i] = np.log10(pGB[parameter])
                    elif parameter in ['Frequency']:
                        tr_s[i] = pGB[parameter]*10**3
                    elif parameter in ['Inclination']:
                        tr_s[i] = np.cos(pGB[parameter])
                    elif parameter in ['EclipticLatitude']:
                        tr_s[i] = np.sin(pGB[parameter])
                    else:
                        tr_s[i] = pGB[parameter]
                    i += 1
                if tr_s[0] > np.pi:
                    tr_s[0] -= 2*np.pi
            maxvalues = np.zeros(len(parameters))
            i = 0
            for parameter in ['EclipticLongitude','EclipticLatitude','Frequency','FrequencyDerivative','Inclination','Amplitude','InitialPhase','Polarization']:
                if parameter in parameters_log_uniform:
                    maxvalues[i] = np.log10(self.maxpGB[parameter])
                elif parameter in ['Frequency']:
                    maxvalues[i] = self.maxpGB[parameter]*10**3
                elif parameter in ['Inclination']:
                    maxvalues[i] = np.cos(self.maxpGB[parameter])
                elif parameter in ['EclipticLatitude']:
                    maxvalues[i] = np.sin(self.maxpGB[parameter])
                else:
                    maxvalues[i] = self.maxpGB[parameter]
                i += 1

            # rng = []
            # for i in range(len(lbls)):
            #     minrange = min(datS[:,i].min(), tr_s[i])
            #     maxrange = max(datS[:,i].max(), tr_s[i])
            #     range_width = np.abs(maxrange - minrange)
            #     oner = ( minrange- range_width/10, maxrange + range_width/10)
            #     rng.append(oner)
            # Get the getdist MCSamples objects for the samples, specifying same parameter
            # names and labels; if not specified weights are assumed to all be unity
            ndim = 6
            names = ['EclipticLongitude','EclipticLatitude','Frequency','FrequencyDerivative','Inclination','Amplitude']
            labels =  lbls[:6]
            samples = MCSamples(samples=datS[:,:6],names = names, labels = labels)

            g = plots.get_subplot_plotter(subplot_size=0.9)
            samples.updateSettings({'contours': [0.68, 0.95]})
            g.settings.num_plot_contours = 2
            if parameter_titles:
                g.triangle_plot([samples], shaded=True, title_limit=2)
            else:
                g.triangle_plot([samples], shaded=True)
            
            #markers vertical
            for i in range(ndim):
                for ax in g.subplots[i:,i]:
                    if pGB:
                        xlim = ax.get_xlim()
                        ax.set_xlim(np.min([xlim[0], tr_s[i]]),np.max([xlim[1], tr_s[i]]))
                        xlim = ax.get_xlim()
                        if xlim[0] + (xlim[1]-xlim[0])*0.1 > tr_s[i]:
                            ax.set_xlim(xlim[0] - (xlim[1]-xlim[0])*0.1, xlim[1])
                        if xlim[1] - (xlim[1]-xlim[0])*0.1 < tr_s[i]:
                            ax.set_xlim(xlim[0], xlim[1] + (xlim[1]-xlim[0])*0.1)
                        ax.axvline(tr_s[i], color='red', lw = 1)
                    ax.axvline(maxvalues[i], color='green', ls='--', lw = 1)
                # i += 1
            #markers horizontal
            for i in range(ndim):
                for ax in g.subplots[i,:i]:
                    if pGB:
                        ylim = ax.get_ylim()
                        ax.set_ylim(np.min([ylim[0], tr_s[i]]),np.max([ylim[1], tr_s[i]]))
                        ylim = ax.get_ylim()
                        if ylim[0] + (ylim[1]-ylim[0])*0.1 > tr_s[i]:
                            ax.set_ylim(ylim[0] - (ylim[1]-ylim[0])*0.1, ylim[1])
                        if ylim[1] - (ylim[1]-ylim[0])*0.1 < tr_s[i]:
                            ax.set_ylim(ylim[0], ylim[1] + (ylim[1]-ylim[0])*0.1)
                        ax.axhline(tr_s[i], color='red', lw = 1)
                    ax.axhline(maxvalues[i], color='green', ls='--', lw = 1)
                # i += 1
            g.export(SAVEPATH+'Posteriors_gpu/frequency'+ str(int(np.round(save_frequency*10**9)))+save_name+str(parameter_titles)+'.png')
            # plt.show()

def compute_posterior(tdi_fs, Tobs, frequencies, maxpGB, pGB_true = [], number_of_signal = 0, noise=None):
    start = time.time()
    posterior1 = Posterior_computer(tdi_fs, Tobs, frequencies, maxpGB, noise=noise, use_gpu=gpu_available)
    print('time to initialize: ', time.time()-start)
    print(posterior1.search1.loglikelihood([maxpGB]))
    # posterior1.search1.update_noise(pGB=maxpGB)
    # print(posterior1.search1.loglikelihood([maxpGB]))
    # print('SNR found',posterior1.search1.SNR([maxpGB]))
    # posterior1.search1.update_noise(pGB=maxpGB)
    # print('SNR found full noise',posterior1.search1.SNR([maxpGB]))
    # if pGB_true:
    #     print('SNR injected',posterior1.search1.SNR([pGB_true]))
    # posterior1.search1.plot(found_sources_in=[maxpGB])
    start = time.time()
    posterior1.reduce_boundaries()
    print('time to reduce boundaries: ', time.time()-start)
    start = time.time()
    # posterior1.train_model()
    # print('GPU memory free', get_gpu_memory())
    # mcmc_samples = posterior1.calculate_posterior(resolution = 1*10**5, temperature= 10)
    temperature = [15, 10, 5, 3, 2, 1]
    resolution = [10**4, 10**4, 10**4, 10**4, 10**4, 10**4]
    posterior_round = 0
    mcmc_samples = posterior1.calculate_posterior(resolution = resolution[posterior_round], temperature= temperature[posterior_round])
    posterior_round += 1
    # posterior1.boundaries_reduced = posterior1.reduce_boundaries_from_samples(mcmc_samples)

    mcmc_samples = posterior1.calculate_posterior(resolution = resolution[posterior_round], proposal= mcmc_samples, temperature= temperature[posterior_round])
    posterior_round += 1
    
    mcmc_samples = posterior1.calculate_posterior(resolution = resolution[posterior_round], proposal= mcmc_samples, temperature= temperature[posterior_round])
    posterior_round += 1

    mcmc_samples = posterior1.calculate_posterior(resolution = resolution[posterior_round], proposal= mcmc_samples, temperature= temperature[posterior_round])
    posterior_round += 1

    mcmc_samples = posterior1.calculate_posterior(resolution = resolution[posterior_round], proposal= mcmc_samples, temperature= temperature[posterior_round])
    posterior_round += 1

    mcmc_samples_final = posterior1.calculate_posterior(resolution = resolution[posterior_round], proposal= mcmc_samples, temperature= temperature[posterior_round])
    # posterior1.plot_corner(mcmc_samples_final, pGB_true, save_figure= True)
    print('time to compute posterior: ', time.time()-start)
    start = time.time()
    posterior1.plot_corner(mcmc_samples_final, pGB_true, save_figure= False, save_chain= True, number_of_signal = 0, parameter_titles = False)
    # kde_samples, probability = posterior1.get_samples(resolution = 1*10**4, proposal= mcmc_samples)

    # plt.figure()
    # plt.scatter(kde_samples[:100000,0],np.arccos(kde_samples[:100000,5]* (posterior1.boundaries_reduced['Inclination'][1] - posterior1.boundaries_reduced['Inclination'][0]) + posterior1.boundaries_reduced['Inclination'][0]), c=probability[:100000])
    # plt.show()

    # posterior1.plot_corner(kde_samples, pGB_true, save_figure= True, save_chain= True, number_of_signal = 0, parameter_titles = False)
    # print('time to save posterior: ', time.time()-start)
    return mcmc_samples


print('GPU memory free', get_gpu_memory())

# SAVEPATH_sangria = grandparent+"/LDC/pictures/Sangria/"
# end_string = '_SNR_scaled_03_injected_snr5'
# found_sources = np.load(SAVEPATH+'/found_sources_matched' +save_name+end_string+'.npy', allow_pickle=True)
# found_sources_not_matched = np.load(SAVEPATH+'/found_sources_not_matched' +save_name+'.npy', allow_pickle=True)
# pGB_injected_not_matched = np.load(SAVEPATH+'/injected_not_matched_windows' +save_name+'.npy', allow_pickle=True)
# pGB_injected_matched = np.load(SAVEPATH+'/injected_matched_windows' +save_name+'.npy', allow_pickle=True)

# found_sources_in_flat = np.load(SAVEPATH+'/found_sources_'+save_name+'_flat.pkl', allow_pickle=True)
# found_sources = np.load(SAVEPATH+'/found_sources_'+save_name+'.pkl', allow_pickle=True)
# found_sources_in_flat = []
# for i in range(len(found_sources)):
#     for j in range(len(found_sources[i])):
#         found_sources_in_flat.append(found_sources[i][j])

# found_sources = np.load(SAVEPATH+'/found_sources_'+save_name+'.npy', allow_pickle=True)
# for i in range(len(found_sources)):
#     for j in range(len(found_sources[i][3])):
#         found_sources_in_flat.append(found_sources[i][3][j])


# found_sources = np.load(SAVEPATH+'/found_sources_'+save_name+'.pkl', allow_pickle=True)
# found_sources_in_flat = np.concatenate(found_sources)
# pickle.dump(found_sources_in_flat, open(SAVEPATH+'found_sources_' +save_name+'_flat.pkl', 'wb'))

found_sources_in_flat = pickle.load(open(SAVEPATH+'found_sources_' +save_name+'_flat.pkl', 'rb'))
# found_sources_in_flat = np.load(SAVEPATH+'found_sources_' +save_name+'_flat.npy', allow_pickle = True)

found_sources_in_flat = np.asarray(found_sources_in_flat)
found_sources_in_flat_array = {attribute: np.asarray([x[attribute] for x in found_sources_in_flat]) for attribute in found_sources_in_flat[0].keys()}
found_sources_in_flat_df = pd.DataFrame(found_sources_in_flat_array)
found_sources_in_flat_df = found_sources_in_flat_df.sort_values('Frequency')
found_sources_in = []
for i in range(len(frequencies_search)):
    found_sources_in.append(found_sources_in_flat_df[(found_sources_in_flat_df['Frequency'] > frequencies_search[i][0]) & (found_sources_in_flat_df['Frequency'] < frequencies_search[i][1])].to_dict(orient='records'))
# found_sources_matched =found_sources_in

# pGB_injected = pGB_injected_matched

# number_of_signals = 0
# for i in range(len(found_sources_in)):
#     number_of_signals += len(found_sources_in[i])
# number_of_signals2 = 0
# for i in range(len(found_sources_not_matched)):
#     number_of_signals2 += len(found_sources_not_matched[i])

# parameter = 'EclipticLongitude'
# for i in range(len(pGB_injected)):
#     for j in range(len(pGB_injected[i])):
#         if pGB_injected[i][j][parameter] > np.pi:
#             pGB_injected[i][j][parameter] -= 2*np.pi

better_pGB = {'Amplitude': 3.7080776510756e-23, 'EclipticLatitude': -0.0864329194471405, 'EclipticLongitude': -1.5608489415225566, 'Frequency': 0.011097063538503463, 'FrequencyDerivative': 3.795997584356877e-15, 'Inclination': 1.3544536642993756, 'InitialPhase': 3.802341846303522, 'Polarization': 3.0450807858161113}
start = time.time()
number_of_signals = 0
frequencies_found = []
# LDC1-4 ####################
posterior_calculation_input = []
# batch = int(sys.argv[1])

# lower_window = 7210

# plt.figure()
# plt.loglog(np.array(frequencies)[:,1],frequency_derivative(np.array(frequencies)[:,1], M_chirp_upper_boundary))
# plt.show()

do_subtract = True
if do_subtract:
    start = time.time()
    # found_sources_flat = found_sources_in_flat

    # found_sources_flat = []
    # for i in range(len(found_sources_mp_subtract)):
    #     for j in range(len(found_sources_mp_subtract[i])):
    #         found_sources_flat.append(found_sources_mp_subtract[i][j])
    found_sources_flat = np.asarray(found_sources_in_flat)
    found_sources_flat_array = {attribute: np.asarray([x[attribute] for x in found_sources_flat]) for attribute in found_sources_flat[0].keys()}
    found_sources_flat_df = pd.DataFrame(found_sources_flat_array)
    found_sources_flat_df = found_sources_flat_df.sort_values('Frequency')
    # for i in range(len(frequencies_search_full)):
    #     found_sources_flat_df = found_sources_flat_df[(found_sources_flat_df['Frequency']< frequencies_search_full[i][0]) | (found_sources_flat_df['Frequency']> frequencies_search_full[i][1])]
    # found_sources_flat_df = found_sources_flat_df.sort_values('Frequency')
    found_sources_out_flat = found_sources_flat_df.to_dict(orient='records')
    tdi_fs_residual = tdi_subtraction(tdi_fs,found_sources_out_flat)
    tdi_fs_partial_residual = tdi_subtraction(tdi_fs,found_sources_out_flat, ratio = 0.7)

    print('subtraction time', time.time()-start)
    plot_subtraction = False
    if plot_subtraction:
        i = 4000
        i = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.0186)+1
        lower_frequency = frequencies_search[i][0]
        upper_frequency = frequencies_search[i][1]
        search1 = Search(tdi_fs,Tobs, lower_frequency, upper_frequency)
        search1.plot(second_data= tdi_fs_residual)#, found_sources_in=found_sources_out_flat)
        # search1.plot(second_data= tdi_fs_residual, found_sources_in=found_sources_mp_o[start_index][0])
        
    # tdi_fs = deepcopy(tdi_fs_residual)

tdi_fs['A'] = (tdi_fs["Z"] - tdi_fs["X"])/np.sqrt(2.0)

tdi_fs_residual['A'] = (tdi_fs_residual["Z"] - tdi_fs_residual["X"])/np.sqrt(2.0)



def get_noise_from_frequency_domain(tdi_fs, number_of_windows=500):
    tdi_ts = xr.Dataset(dict([(k, tdi_fs[k].ts.ifft()) for k in ["X", "Y", "Z"]]))
    tdi_ts["E"] = deepcopy(tdi_ts["X"])
    tdi_ts["A"] = (tdi_ts["Z"] - tdi_ts["X"])/np.sqrt(2.0)
    tdi_ts["E"].values = (tdi_ts["Z"] - 2.0*tdi_ts["Y"] + tdi_ts["X"])/np.sqrt(6.0)
    tdi_ts["T"] = (tdi_ts["Z"] + tdi_ts["Y"] + tdi_ts["X"])/np.sqrt(3.0)
    f, psdA =  scipy.signal.welch(tdi_ts["A"], fs=1.0/dt, nperseg=len(tdi_ts["X"])/number_of_windows)#, average='mean', window= 'boxcar')
    f, psdE =  scipy.signal.welch(tdi_ts["E"], fs=1.0/dt, nperseg=len(tdi_ts["X"])/number_of_windows)
    # f2, psdE2 =  scipy.signal.welch(tdi_ts["E"], fs=1.0/dt, nperseg=len(tdi_ts["X"]), scaling='spectrum')
    f, psdT =  scipy.signal.welch(tdi_ts["T"], fs=1.0/dt, nperseg=len(tdi_ts["X"])/number_of_windows)
    return f, psdA, psdE, psdT


def smooth_psd(psd, f):
    smoothed = median_windows(psd, 30)
    smoothed[:40] = psd[:40]
    index_cut = np.searchsorted(f, 0.0008)  # 0.0008 for 1,2 years
    index_cut_lower = np.searchsorted(f, 3*10**-4)
    psd_fit = np.ones_like(smoothed)
    psd_fit_low = scipy.signal.savgol_filter(smoothed, 10, 1)
    psd_fit_high = scipy.signal.savgol_filter(smoothed, 70, 1) # 70 for 1,2 years
    psd_fit[:index_cut] = psd_fit_low[:index_cut] 
    psd_fit[index_cut:] = psd_fit_high[index_cut:] 
    psd_fit[:index_cut_lower] = smoothed[:index_cut_lower]
    return psd_fit, smoothed


number_of_windows = 500
start = time.time()
f, psdA, psdE, psdT = get_noise_from_frequency_domain(tdi_fs, number_of_windows=number_of_windows)
print(time.time()-start)
# psdA = np.interp(tdi_fs.f, f, psdA)
# psdE = np.interp(tdi_fs.f, f, psdE)
# psdT = np.interp(tdi_fs.f, f, psdT)

f_partial_residual, psdA_partial, psdE_partial, psdT_partial = get_noise_from_frequency_domain(tdi_fs_partial_residual, number_of_windows=number_of_windows)
# psdA_partial_residual = np.interp(tdi_fs.f, f_partial_residual, psdA_partial_residual)
# psdE_partial_residual = np.interp(tdi_fs.f, f_partial_residual, psdE_partial_residual)
# psdT_partial_residual = np.interp(tdi_fs.f, f_partial_residual, psdT_partial_residual)


f_res, psdA_residual, psdE_residual, psdT_residual = get_noise_from_frequency_domain(tdi_fs_residual, number_of_windows=number_of_windows)
psdA_welch = deepcopy(psdA_residual)
# psdA_residual = np.interp(tdi_fs.f, f_res, psdA_residual)
# psdE_residual = np.interp(tdi_fs.f, f_res, psdE_residual)
# psdT_residual = np.interp(tdi_fs.f, f_res, psdT_residual)

# i = 4000
# lower_frequency = frequencies_search[i][0]
# upper_frequency = frequencies_search[i][1]
# search1 = Search(tdi_fs_residual,Tobs, lower_frequency, upper_frequency)

### mean in each frequency window
# noise_means = deepcopy(search1.SA_full_f)
# noise_means2 = []
# frequencies_means = []
# for i in range(len(frequencies)):
#     padding = (frequencies[i][1] - frequencies[i][0])/2
#     lower_index = np.searchsorted(search1.DAf_full_f.f, frequencies[i][0]-padding)
#     upper_index = np.searchsorted(search1.DAf_full_f.f, frequencies[i][1]+padding)
#     lower_index_inner = np.searchsorted(search1.DAf_full_f.f, frequencies[i][0])
#     upper_index_inner = np.searchsorted(search1.DAf_full_f.f, frequencies[i][1])
#     # indexes = np.logical_and(tdi_fs['X'].f >= frequencies[i][0]-padding, tdi_fs['X'].f < frequencies[i][1]+padding) 
#     DAf = search1.DAf_full_f[lower_index:upper_index]
#     noise_means2.append(np.median((np.abs(DAf).data)**2)/(Tobs)*2)
#     frequencies_means.append(search1.DAf_full_f.f[lower_index+int((upper_index-lower_index)/2)].data)
#     noise_means[lower_index_inner:upper_index_inner] = np.ones_like(search1.DAf_full_f[lower_index_inner:upper_index_inner])*np.median((np.abs(DAf).data)**2)/(Tobs)*2
    # noise_means[lower_index:upper_index] = ((np.abs(DAf).data))**2/(Tobs)*2




# psdA_fit = median_windows(psdA,20)
# psdE_fit = median_windows(psdE,20)
# psdT_fit = median_windows(psdT,20)
psdA_partial, smoothedA = smooth_psd(psdA_partial, f_partial_residual)
psdE_partial, smoothedE = smooth_psd(psdE_partial, f_partial_residual)
psdT_partial, smoothedT = smooth_psd(psdT_partial, f_partial_residual)

psdA_residual, smoothedA = smooth_psd(psdA_residual, f_res)
psdE_residual, smoothedE = smooth_psd(psdE_residual, f_res)
psdT_residual, smoothedT = smooth_psd(psdT_residual, f_res)

# psdA_fit = smooth_psd(noise_means, frequencies_means)

# psdA_fit = np.interp(tdi_fs.f, f_res, psdA_fit)
# psdE_fit = np.interp(tdi_fs.f, f_res, psdE_fit)
# psdT_fit = np.interp(tdi_fs.f, f_res, psdT_fit)

psdA_partial = scipy.interpolate.InterpolatedUnivariateSpline(f_partial_residual, psdA_partial)(tdi_fs.f)
psdE_partial = scipy.interpolate.InterpolatedUnivariateSpline(f_partial_residual, psdE_partial)(tdi_fs.f)
psdT_partial = scipy.interpolate.InterpolatedUnivariateSpline(f_partial_residual, psdT_partial)(tdi_fs.f)

psdA_residual = scipy.interpolate.InterpolatedUnivariateSpline(f_res, psdA_residual)(tdi_fs.f)
psdE_residual = scipy.interpolate.InterpolatedUnivariateSpline(f_res, psdE_residual)(tdi_fs.f)
psdT_residual = scipy.interpolate.InterpolatedUnivariateSpline(f_res, psdT_residual)(tdi_fs.f)


# t = np.linspace(f_res[8], f_res[-2], 50)
# # t = np.append( f_res[1:10], t)
# spline = scipy.interpolate.LSQUnivariateSpline(f_res, smoothed_A, t)
# spline.set_smoothing_factor(10**5)



SA_full_f = Nmodel.psd(freq=tdi_fs.f, option="A")
df = tdi_fs['A'].f[1] - tdi_fs['A'].f[0]
fs = 1/dt

N = len(tdi_ts['X'])
tdi_fsX = np.fft.rfft(tdi_ts['X'].data)[0:N//2+1]/fs

if dataset == 'Radler':
    noise_model = "SciRDv1"
if dataset == 'Sangria':
    noise_model = "sangria"
Nmodel = get_noise_model(noise_model, f_res[1:])
Sn = Nmodel.psd(freq=f_res[1:], option="X")
SA = Nmodel.psd(freq=f_res[1:], option="A")
SE = Nmodel.psd(freq=f_res[1:], option="E")
ST = Nmodel.psd(freq=f_res[1:], option="T")

ldc_noise = AnalyticNoise(f_res[1:], model="sangria", wd=1)
Nmodel = get_noise_model(noise_model, f_res[1:])
SA = Nmodel.psd(freq=f_res[1:], option="A")
SAa = ldc_noise.psd(f_res[1:], option='A')


f_partial_residual1, psdA_residual1, psdE_residual1, psdT_residual1 = get_noise_from_frequency_domain(tdi_fs_residual, number_of_windows=1)

plt.figure()
# plt.plot(xnew, np.sin(xnew), '-.', label='sin(x)')
# plt.plot(xnew, BSpline(*tck)(xnew), '-', label='s=0')
# plt.loglog(f_res, psdA)   
plt.loglog(tdi_fs.f, np.abs(tdi_fs['A']/dt)**2 /(fs*len(tdi_ts['X']))*2 ,color='grey', zorder =0, label=r'PSD $d$')     
plt.loglog(tdi_fs.f, np.abs(tdi_fs_residual['A']/dt)**2 /(fs*len(tdi_ts['X']))*2 ,color='darkgrey', zorder =1, label=r'PSD $d_\mathrm{residual}$')  
plt.loglog(tdi_fs.f, psdA_partial, zorder=4.5, label=r'$S_{A \mathrm{,partial}}$', linewidth=3)   
plt.loglog(f_res, psdA_welch, label=r'$S_{A \mathrm{,welch}}$', linewidth = 3)  
# plt.loglog(f_partial_residual1, psdA_residual1, label='welch2', linewidth = 3)  
# plt.loglog(f_res, psdA_partial)   
# plt.loglog(f_res, smoothed_A)     
# plt.loglog(tdi_fs.f, spline(tdi_fs.f))
# plt.loglog(frequencies_means, psd_residual, zorder=5)    
# plt.loglog(f_res[1:], SA, zorder=5)   
plt.loglog(f_res, smoothedA, color='purple', zorder=4, label=r'$S_{A \mathrm{,median}}$')   
plt.loglog(tdi_fs.f, psdA_residual,  zorder=5, label=r'$S_{A \mathrm{,residual}}$', linewidth=3)   
# plt.loglog(f_res[1:], SAa, 'k--', zorder=5, label='instrument')   
plt.loglog(f_res[1:], SA, 'k--', zorder=5, label=r'$S_{A \mathrm{,instrument}}$')   
# plt.loglog(tdi_fs.f, noise_means, zorder=3)   
# plt.loglog(tdi_fs.f, tdi_fs['X'],'.', zorder =1)   
# plt.loglog(tdi_fs.f, tdi_fsX,'.', zorder =1)    
# plt.loglog(tdi_fs.f, (psdA_residual-np.abs(tdi_fs_residual['A'])**2 /(Tobs)*2)/psdA_residual,'.')    
# plt.loglog(f_res[peaks], psdA_residual[peaks],'x', color='red')     
# plt.loglog(f_res[lower_index_res:], y_pred)  
plt.xlabel(r'$f$ (Hz)')
plt.ylabel(r'PSD (1/Hz)')
plt.xlim(0.0001,0.1)
plt.ylim(10**-43,10**-37)   
plt.legend(loc='upper left')
plt.show()


# psdA_residual = (np.abs(search1.DAf_full_f)).data**2*2/(dt*len(search1.tdi_fs['X']))
# psdE_residual = (np.abs(search1.DEf_full_f)).data**2/(dt*len(search1.tdi_fs['X']))*2
# psdT_residual = (np.abs(search1.DTf_full_f)).data**2/(dt*len(search1.tdi_fs['X']))*2


SA = scipy.interpolate.InterpolatedUnivariateSpline(f_res[1:], SA)(tdi_fs.f)
SE = scipy.interpolate.InterpolatedUnivariateSpline(f_res[1:], SE)(tdi_fs.f)
ST = scipy.interpolate.InterpolatedUnivariateSpline(f_res[1:], ST)(tdi_fs.f)

noise_data = {'A': psdA, 'E': psdE, 'T': psdT}
noise_fit = {'A': psdA_partial, 'E': psdE_partial, 'T': psdT_partial}
noise_residual = {'A': psdA_residual, 'E': psdE_residual, 'T': psdT_residual}
noise_instrument = {'A': SA, 'E': SE, 'T': ST}
# noise_partial_residual = {'A': psdA_partial_residual, 'E': psdE_partial_residual, 'T': psdT_partial_residual}

noise_residual['f'] = tdi_fs.f
noise_df = pd.DataFrame(noise_residual)
noise_df.to_csv(SAVEPATH+'ETH_'+save_name+'_noise.csv')

noise = noise_fit
# search1 = Search(tdi_fs_residual,Tobs, lower_frequency, upper_frequency, noise=noise)

# whitened_data = np.abs(tdi_fs_residual['A']/dt)**2 /(fs*len(tdi_ts['X']))*2 / noise['A']/dt
# whitened_data = tdi_fs_residual['A']*2/dt / np.sqrt(noise['A']*fs*len(tdi_ts['X']))

# psdA_residual_interpol = scipy.interpolate.InterpolatedUnivariateSpline(f_res, psdA_residual)(tdi_fs.f)
whitened_data = tdi_fs_residual['A']*2/dt / np.sqrt(psdA_residual*fs*len(tdi_ts['X']))
# whitened_data = search1.DAf_full_f / np.real(search1.DAf_full_f)

lower_index = np.searchsorted(tdi_fs['X'].f, 0.0003)
upper_index = np.searchsorted(tdi_fs['X'].f, 0.1)
print(np.mean(np.real(whitened_data[lower_index:upper_index])))
print(np.std(np.real(whitened_data[lower_index:upper_index])))
print(np.std(np.imag(whitened_data[lower_index:upper_index])))

# plt.figure()
# plt.hist(np.real(whitened_data[lower_index:upper_index]), bins=2000, density=True)
# plt.xlim(-4,4)
# plt.show()



i = 3000
# search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i][1], noise=noise)
plt.figure()
# plt.loglog(tdi_fs_residual.f, search1.SA_full_f, 'k')
# plt.loglog(tdi_fs_residual.f, psdA)
# plt.loglog(tdi_fs_residual.f, psdA_residual*(dt*len(search1.tdi_fs['X'])), zorder = 5)
# plt.loglog(tdi_fs_residual.f,  np.sqrt(psdA_residual), zorder = 5)
# plt.loglog(tdi_fs_residual.f,  np.sqrt(psdE_residual), zorder = 5)
# plt.loglog(tdi_fs_residual.f,  np.sqrt(noise_means), zorder = 5)
# plt.loglog(tdi_fs_residual.f, psdA_partial_residual)
plt.semilogx(tdi_fs_residual.f[100:], np.real(whitened_data[100:]))
# plt.loglog(tdi_fs.f, np.sqrt(psdA_interp),'-x')
# plt.loglog(f2, np.sqrt(psdA_residual2))
# plt.loglog(tdi_fs.f, np.abs(tdi_fs['A']))
# plt.semilogx(tdi_fs.f, np.abs(search1.DAf_full_f),'k')
# search1.update_noise()
# plt.loglog(tdi_fs.f, np.sqrt(search1.SA_full_f),'r')
# plt.loglog(tdi_fs_residual.f, np.abs(tdi_fs_residual['A']))
plt.xlim(0.0003, 0.1)
plt.show()


tdi_fs_original = deepcopy(tdi_fs)
dataX_original = tdi_fs["X"]
dataY_original = tdi_fs["Y"]
dataZ_original = tdi_fs["Z"]
DAf_original = (dataZ_original - dataX_original)/np.sqrt(2.0)

# cat_index = 0
# for index in range(len(found_sources_in)):
#     for j in range(len(found_sources_in[index])):
#         search1 = Search(tdi_fs,Tobs, frequencies_search[index][0], frequencies_search[index][1], tdi2=tdi2, gb_gpu=gb_gpu, use_gpu=gpu_available)
#         # search1.plot(pGB_injected=[pGB_injected])
#         print(search1.SNR([found_sources_in_flat[cat_index]]))
#         print(search1.loglikelihood([found_sources_in_flat[cat_index]]))
#         # print(search1.loglikelihood_gpu([found_sources_in_flat[cat_index]]))
#         print(found_sources_in_flat[cat_index])
#         cat_index += 1

start_all = time.time()
for i in range(len(found_sources_in)):
    if i < 4000:
        continue
    # if i%10 != 0:
    #     continue
    # if i in [0,len(found_sources_in)-1]:
    #     continue
    for j in range(len(found_sources_in[i])):
        start = time.time()
        # if j < 3 and i == 7213:
        #     continue

        print(i)
        #subtract the found sources of neighbours and own window from original except the signal itself
        tdi_fs_added = deepcopy(tdi_fs_residual)
        # tdi_fs_added_part = deepcopy(tdi_fs)

        Xs_added, Ys_added, Zs_added = GB.get_fd_tdixyz(template=found_sources_in[i][j], oversample=4, tdi2=tdi2)
        source_added = dict({"X": Xs_added, "Y": Ys_added, "Z": Zs_added})
        index_low = np.searchsorted(tdi_fs_added["X"].f, Xs_added.f[0])
        try:
            if np.abs(tdi_fs_added["X"].f[index_low-1] - Xs_added.f[0]) < np.abs(tdi_fs_added["X"].f[index_low] - Xs_added.f[0]):
            # if tdi_fs_added["X"].f[index_low] > Xs_added.f[0]:
                index_low = index_low-1
        except:
            pass
        index_high = index_low+len(Xs_added)
        for k in ["X", "Y", "Z"]:
            tdi_fs_added[k].data[index_low:index_high] = tdi_fs_added[k].data[index_low:index_high] + source_added[k].data
            # tdi_fs_added_part[k].data[index_low:index_high] = tdi_fs_added_part[k].data[index_low:index_high] - source_added[k].data * 1

        plot_addition = False
        if plot_addition:
            # pGB_true = []
            # for k in range(len(pGB_injected[i])):
            #     pGB_true.append(pGB_injected[i][k].to_dict())
            # pGB_true_not_matched = []
            # for k in range(len(pGB_injected_not_matched[i])):
            #     pGB_true_not_matched.append(pGB_injected_not_matched[i].iloc[k].to_dict())
            search1 = Search(tdi_fs_residual,Tobs, frequencies_search[i][0], frequencies_search[i][1], noise=noise, tdi2=tdi2)
            search1.plot(second_data= tdi_fs_added, found_sources_in=found_sources_in[i])
            # print(search1.loglikelihood([found_sources_in[i][j]]))
            # search1.update_noise(pGB=found_sources_in[i][j])
            # print(search1.loglikelihood([found_sources_in[i][j]]))
            # search1.plot(second_data= tdi_fs_added, found_sources_in=found_sources_in[i])

        number_of_signals += 1
        frequencies_found.append(found_sources_in[i][j]['Frequency'])
        compute_posterior(tdi_fs_added, Tobs, frequencies_search[i], found_sources_in[i][j], noise=noise)#, pGB_injected_matched[i][j].to_dict())
        print('time for one signal posterior', time.time() - start)
print('time to search ', number_of_signals, 'signals: ', time.time()-start_all, (time.time()-start_all)/number_of_signals, 's per signal')
# start = time.time()

print('time to search ', len(posterior_calculation_input), 'signals: ', time.time()-start)

