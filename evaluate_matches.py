from re import A
# from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.font_manager
import scipy as sp
from scipy.optimize import differential_evolution
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np
import xarray as xr
import time
from copy import deepcopy
import multiprocessing as mp
import pandas as pd
import polars as pl
import os
import h5py
import sys
sys.path.append('/cluster/home/sstrub/Repositories/LDC/lib/lib64/python3.8/site-packages/ldc-0.1-py3.8-linux-x86_64.egg')

from astropy import units as u
import astropy.coordinates as coord

import jax
from jaxgb.params import GBObject, PARAM_NAMES
from jaxgb.jaxgb import JaxGB
from jaxgb.tdi import to_tdi_combination
import jax.numpy as jnp

import json
from globalGB.search_utils_GB import GBConfig
from DataLoader.data_loader import LISADataLoader

from globalGB.search_utils_GB import PARAM_NAMES, PARAM_INDICES, frequency_derivative_mojito_lower, frequency_derivative_mojito_upper
from globalGB.search_utils_GB import frequency_derivative_tyson_lower, frequency_derivative_tyson_upper, frequency_derivative
jax.config.update('jax_default_device', jax.devices('cpu')[0])
jax.config.update("jax_enable_x64", True)

# get current directory
path = os.getcwd()

# parent directory
parent = os.path.dirname(path)
# grandparent directory
grandparent = os.path.dirname(parent)

# Dataset selection: 'Radler', 'Sangria', or 'Mojito'
data_set = 'Mojito'
reduction = 1
dt = 10.0  # Default time step
seed = 1   # Default seed
SNR_threshold = 9
tdi_generation = 2.0
injected_SNR = 5


save_name = f'{data_set}_SNR_threshold_{SNR_threshold}_seed{seed}'
end_string = '_scaled_error_injected_snr'+str(injected_SNR)

channel_combination = 'AET'

with open('globalGB/GB_search_config.json', 'r') as f:
    config = json.load(f)
    config = GBConfig(config)
loader = LISADataLoader(config=config)
data_fn = config.data_path
loader.load(data_fn, dt=dt, channel_combination=channel_combination)
loader._load_mojito_wdwd_catalog()
parameters_injected = pl.DataFrame(loader.catalog_wdwd, schema=PARAM_NAMES)

t0 = loader.t0
save_name_injected = 'Mojito_WDWD'
max_number_of_injected_signals_per_window = 100

t_init = 97729089.327664 
fn = config.save_path+'/pGB_injected_' +save_name_injected+'.h5'
with h5py.File(fn, 'r') as f:
    pGB_injected = f['injected_sources'][:]
# gbo = GBObject.from_jaxgb_params(jnp.array(pGB_injected), t_init=t_init)
# pGB_injected = np.array(gbo.to_jaxgb_array(t0=t0))
pGB_injected_df = pl.DataFrame(pGB_injected, schema=PARAM_NAMES)
pGB_injected_df = pGB_injected_df.sort('Frequency')

with h5py.File(config.save_path+'/match_results_'+config.data_set+'_SNR_threshold_'+str(int(config.snr_threshold))+'_seed'+str(config.seed)+'_'+config.match_criteria+'.h5', 'r') as f:
        found_sources = np.array(f['found_sources'][:])
        injected_sources = np.array(f['injected_sources'][:])
        match_values = np.array(f['match_values'][:])
        found_sources_matched = np.array(f['found_sources_matched'][:])
        injected_sources_matched = np.array(f['injected_sources_matched'][:])
        match_values_matched = np.array(f['match_values_matched'][:])
        found_sources_not_matched = np.array(f['found_sources_unmatched'][:])
        injected_sources_not_matched = np.array(f['injected_sources_unmatched'][:])
        match_values_not_matched = np.array(f['match_values_unmatched'][:])

number_of_injected_signals = len(pGB_injected)
number_of_found_signals = len(found_sources_matched) + len(found_sources_not_matched)

number_of_matched_signals = len(found_sources_matched)
number_of_not_matched_signals = len(found_sources_not_matched)
print(number_of_matched_signals ,'matched signals out of', number_of_injected_signals , 'injected signals and',number_of_found_signals, 'found signals')
print('matched signals/found signals:', np.round(number_of_matched_signals/number_of_found_signals,2))

injected_sources_matched_df = pl.DataFrame(injected_sources_matched, schema=PARAM_NAMES)
injected_sources_matched_df = injected_sources_matched_df.sort(by= 'Frequency')
injected_sources_not_matched_df = pl.DataFrame(injected_sources_not_matched, schema=PARAM_NAMES)
injected_sources_not_matched_df = injected_sources_not_matched_df.sort(by= 'Frequency')

found_sources_matched_df = pl.DataFrame(found_sources_matched, schema=PARAM_NAMES)
found_sources_matched_df = found_sources_matched_df.sort(by= 'Frequency')
found_sources_not_matched_df = pl.DataFrame(found_sources_not_matched, schema=PARAM_NAMES)
found_sources_not_matched_df = found_sources_not_matched_df.sort(by= 'Frequency')

# remove injected_sources_matched_df signls from pGB_injected_df
injected_sources_not_matched_df = pl.concat([injected_sources_matched_df, pGB_injected_df])
print(len(injected_sources_not_matched_df))
injected_sources_not_matched_df = injected_sources_not_matched_df.unique(keep='none')
print(len(injected_sources_not_matched_df))
injected_sources_not_matched = np.array(injected_sources_not_matched_df)


labels = {'RightAscension': r'$\lambda$'+' (rad)', 'Declination': r'$\beta$'+' (rad)','Frequency': r'$f / f_\mathrm{true}$','FrequencyDerivative': r'$\dot{f}$ $ ($Hz/s$)$','Inclination': r'$\iota$'+' (rad)','Amplitude': r'$ \mathcal{A}$', 'Polarization': r'$\psi$'+' (rad)', 'InitialPhase': r'$\phi_0$'+' (rad)'}




#### plot amplitude - frequency
markersize = 3
alpha = 0.6
# parameter_to_plot = 'IntrinsicSNR'
# parameter_x = 'Frequency'
# parameter_y = 'FrequencyDerivative'
parameter_x = 'Frequency'
parameter_y = 'Amplitude'
fig = plt.figure(figsize=(10,6))
# plt.plot(found_sources_matches[:,PARAM_INDICES[parameter_x]],found_sources_matches[:,PARAM_INDICES[parameter_y]],'.', markersize= markersize, alpha = alpha, zorder = 5, label = 'Found '+ str(len(found_sources_matches))+' signals')
plt.plot(found_sources_matched[:,PARAM_INDICES[parameter_x]],found_sources_matched[:,PARAM_INDICES[parameter_y]],'.', markersize= markersize, alpha = alpha, zorder = 1, label = 'Recovered matched '+ str(len(injected_sources_matched))+' signals')
plt.plot(found_sources_not_matched[:,PARAM_INDICES[parameter_x]],found_sources_not_matched[:,PARAM_INDICES[parameter_y]],'o',  markerfacecolor='None', markersize= markersize, zorder = 2, label = 'Recovered not matched '+ str(len(found_sources_not_matched))+' signals', alpha = alpha)
# plt.plot(pGB_injectced_flat_df['Frequency']*10**3,pGB_injected_flat_df[parameter_y], '.', color= colors[0], label = 'Injected', markersize= markersize, alpha = alpha)
# plt.plot(injected_sources_matched[:,PARAM_INDICES['Frequency']]*10**3,injected_sources_matched[:,PARAM_INDICES[parameter_y]], '.', label = 'Injected matched', markersize= markersize, alpha = alpha, zorder = 0)
# plt.plot(pGB_injected_flat_df_high_SNR['Frequency']*10**3,pGB_injected_flat_df_high_SNR[parameter_y],'.', color= colors[1], markersize= markersize, label = 'Injected SNR > 10', alpha = alpha)
# plt.plot(found_sources_matched_list[data_set][parameter_x],found_sources_matched_list[data_set][parameter_y],'.', label = r'$\tilde\theta_{\mathrm{recovered, } \delta < 0.3}$', markersize= markersize, alpha = alpha, zorder = 5)
# plt.plot(found_sources_not_matched_list[data_set][parameter_x],found_sources_not_matched_list[data_set][parameter_y],'o',  markerfacecolor='None', markersize= markersize, label = r'$\tilde\theta_{\mathrm{recovered, } \delta > 0.3}$', alpha = alpha)
plt.plot(injected_sources_not_matched[:,PARAM_INDICES['Frequency']],injected_sources_not_matched[:,PARAM_INDICES[parameter_y]], '+', label = 'Injected not recovered', markersize= markersize, alpha = alpha, zorder = 0)
# plt.plot(f,np.sqrt(SA), color = 'black', label = 'Sangria', linewidth=2)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$f$ (Hz)')
plt.xlim(0.0003,0.03)
plt.ylim(3*10**-24,None)
if parameter_y == 'IntrinsicSNR':
    plt.ylabel('Intrinsic SNR')
else:
    plt.ylabel(labels[parameter_y])    
plt.legend(markerscale=4, loc = 'upper right')
plt.grid(True)
# plt.savefig(config.save_path+'/Evaluation/'+parameter_y+save_name+'injected_not_matched_found_matched_found_not_matched'+end_string,dpi=300,bbox_inches='tight')
plt.show(block=True)

parameters_to_plot = injected_sources_not_matched_df.sort('Frequency')
chandrasekhar_limit = 1.4
M_chirp_upper_boundary = (chandrasekhar_limit**2)**(3/5)/(2*chandrasekhar_limit)**(1/5)
mask_negative_frequency_derivative = parameters_to_plot[:,PARAM_INDICES['FrequencyDerivative']] < 0
parameters_to_plot_negative_frequency_derivative = parameters_to_plot.filter(mask_negative_frequency_derivative)
parameters_to_plot_positive_frequency_derivative = parameters_to_plot.filter(~mask_negative_frequency_derivative)
plt.figure()
plt.plot(loader.tdi_fs['freq'],np.abs(frequency_derivative_mojito_lower(loader.tdi_fs['freq'])),'k')
plt.plot(loader.tdi_fs['freq'],np.abs(frequency_derivative_tyson_lower(loader.tdi_fs['freq'])),'k:')
# plt.plot(loader.tdi_fs['freq'],np.abs(frequency_derivative(loader.tdi_fs['freq'], M_chirp_upper_boundary)),'r')
plt.plot(loader.tdi_fs['freq'],np.abs(frequency_derivative_mojito_upper(loader.tdi_fs['freq'])),'r')
plt.plot(parameters_to_plot_negative_frequency_derivative[:,PARAM_INDICES['Frequency']],np.abs(parameters_to_plot_negative_frequency_derivative[:,PARAM_INDICES['FrequencyDerivative']]), '+', label = 'Injected not recovered', markersize= markersize, alpha = alpha, zorder = 0)
plt.plot(parameters_to_plot_positive_frequency_derivative[:,PARAM_INDICES['Frequency']],parameters_to_plot_positive_frequency_derivative[:,PARAM_INDICES['FrequencyDerivative']], 'o', label = 'Injected not recovered', markersize= markersize, alpha = alpha, zorder = 0)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$f$ (Hz)')
plt.xlim(0.0003,0.03)
plt.ylim(10**-26,None)
plt.show(block=True)

#### plot
markersize = 3
# parameter_to_plot = 'IntrinsicSNR'
parameter_x = 'RightAscension'
parameter_y = 'Declination'
fig = plt.figure()
# plt.plot(pGB_injectced_flat_df['Frequency']*10**3,pGB_injected_flat_df[parameter_y], '.', color= colors[0], label = 'Injected', markersize= markersize, alpha = alpha)
# plt.plot(pGB_injected_matched_flat_df['Frequency']*10**3,pGB_injected_matched_flat_df[parameter_y], '.', color= colors[1], label = 'Injected matched', markersize= markersize, alpha = alpha)
# plt.plot(pGB_injected_flat_df_high_SNR['Frequency']*10**3,pGB_injected_flat_df_high_SNR[parameter_y],'.', color= colors[1], markersize= markersize, label = 'Injected SNR > 10', alpha = alpha)
plt.plot(found_sources_matched[:,PARAM_INDICES[parameter_x]],found_sources_matched[:,PARAM_INDICES[parameter_y]],'.', label = r'$\tilde\theta_{\mathrm{recovered, } \delta < 0.3}$', markersize= markersize, alpha = alpha, zorder = 5)
plt.plot(found_sources_not_matched[:,PARAM_INDICES[parameter_x]],found_sources_not_matched[:,PARAM_INDICES[parameter_y]],'o',  markerfacecolor='None', markersize= markersize, label = r'$\tilde\theta_{\mathrm{recovered, } \delta > 0.3}$', alpha = alpha)
# plt.plot(pGB_injected_not_matched[:,PARAM_INDICES['Frequency']]*10**3,pGB_injected_not_matched[:,PARAM_INDICES[parameter_y]], '+', label = r'$\tilde\theta_{\mathrm{injected, } \delta > 0.3}$', markersize= markersize, alpha = alpha, zorder = 1)
# plt.plot(f,SA*10**20, color = 'black', label = 'Sangria', linewidth=2)
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('$f$ (Hz)')
plt.xlabel(labels[parameter_x])
plt.xlim(0,2*np.pi)
plt.ylim(-np.pi/2,np.pi/2)
if parameter_y == 'IntrinsicSNR':
    plt.ylabel('Intrinsic SNR')
else:
    plt.ylabel(labels[parameter_y])    
# plt.legend(markerscale=4, loc = 'upper right')
plt.legend(markerscale=4, loc = 'lower left')
# plt.savefig(SAVEPATH+'/Evaluation/'+parameter_x+parameter_y+save_name+'injected_not_matched_found_matched_found_not_matched'+end_string,dpi=300,bbox_inches='tight')
plt.show()



def get_errors(pGB_injected_matched_df, found_sources_matched_df):
    error = {}
    for parameter in PARAM_NAMES:
        error[parameter] = []
        # found_sources_matched_flat_df[parameter+'Error'] = []
        for i in range(len(pGB_injected_matched_df)):
            if parameter == 'RightAscension':
                if pGB_injected_matched_df[parameter][i] > np.pi:
                    pGB_injected_matched_df[parameter][i] -= 2*np.pi
            if parameter in ['RightAscension', 'Declination', 'Inclination', 'InitialPhase', 'Polarization']:
                error[parameter].append(np.abs(np.arcsin(np.sin(pGB_injected_matched_df[parameter][i] - found_sources_matched_df[parameter][i]))))
            elif parameter == 'Amplitude':
                # error[parameter].append(np.log10(pGB_injected_matched_flat_df[parameter][i]) - np.log10(found_sources_matched_flat_df[parameter][i]))
                error[parameter].append(np.abs(pGB_injected_matched_df[parameter][i] - found_sources_matched_df[parameter][i])/pGB_injected_matched_df[parameter][i])
            # found_sources_matched_flat_df[parameter+'Error'].append(error[parameter][i])
            else:
                error[parameter].append(pGB_injected_matched_df[parameter][i] - found_sources_matched_df[parameter][i])
        found_sources_matched_df[parameter+'Error'] = error[parameter]
    return found_sources_matched_df

found_sources_matched_df = get_errors(pGB_injected_matched_df, found_sources_matched_df)
pGB_injected_list = pd.concat([pGB_injected_matched, pGB_injected_not_matched])
pGB_injected_high_SNR_list = pGB_injected_list[pGB_injected_list[:,PARAM_INDICES['IntrinsicSNR']] > 10]
match_flat_list = np.concatenate(match_list)
found_sources_matched['Match'] = match_flat_list

parameter_x = 'EclipticLongitude'
parameter_y = 'EclipticLatitude'
fig = plt.figure()
plt.plot(found_sources_matched[:,PARAM_INDICES[parameter_x]],found_sources_matched[:,PARAM_INDICES[parameter_y]],'.', markersize= markersize, alpha = alpha, zorder = 5)
plt.show()

X = found_sources_list[data_set][parameter_x]
Y = found_sources_list[data_set][parameter_y]
Z = found_sources_list[data_set]['Match']

fig, ax = plt.subplots(1,1, figsize=fig_size)
im = ax.scatter(X,Y,c=Z, norm= LogNorm(vmin=0.001, vmax=1), s=3)
fig.colorbar(im, ax=ax, label='$\delta$')
ax.set_xlim(0,2*np.pi)
ax.set_ylim(-np.pi/2,np.pi/2)
ax.set_xlabel(labels[parameter_x])
ax.set_ylabel(labels[parameter_y])
plt.savefig(SAVEPATH+'/Evaluation/'+parameter_x+parameter_y+'match'+save_names[data_set]+end_string)
plt.show()

##### error histogram

boundaries = {
    "Declination": [-1.0, 1.0],
    "RightAscension": [-np.pi, np.pi],
    "Frequency": [0.0003,0.033333],
    "Inclination": [-1.0, 1.0],
    "InitialPhase": [0.0, 2.0 * np.pi],
    "Polarization": [0.0, 1.0 * np.pi],
}

boundaries['GalactocentricLongitude'] = [0,360]
boundaries['GalactocentricLatitude'] = [-90,90]
# boundaries['Frequency'] = [frequencies_search[0][0],frequencies_search[-1][1]]
boundaries['IntrinsicSNR'] = [np.min(found_sources_matched['IntrinsicSNR']),np.max(found_sources_matched['IntrinsicSNR'])]
boundaries['FrequencyDerivative'] = [np.min(found_sources_matched['FrequencyDerivative']),np.max(found_sources_matched['FrequencyDerivative'])]
boundaries['Amplitude'] = [np.min(found_sources_matched['Amplitude']),np.max(found_sources_matched['Amplitude'])]
# parameter_x = 'GalacticLongitude'
# parameter_y = 'GalacticLatitude'
# parameter_x = 'RightAscension'
# parameter_y = 'EclipticLatitude'
x_scale = 'log'
parameter_x = 'Frequency'
# parameter_y = 'IntrinsicSNR'
parameter_y = 'Amplitude'
n_bins = 50
# x_coordinates = []
# y_coordinates = []
# for i in range(n_bins+1):
#     length = (boundaries[parameter_x][1] - boundaries[parameter_x][0])/n_bins
#     x_coordinates.append(boundaries[parameter_x][0]+length*i)
#     length = (boundaries[parameter_y][1] - boundaries[parameter_y][0])/n_bins
#     y_coordinates.append(boundaries[parameter_y][0]+length*i)

parameter_to_plot = 'Skylocation'
parameter_to_plot = 'RightAscension'
def get_2d_hist(sources, get_errors=True, parameter_to_plot='Declination'):
    error = []
    std = []
    count = []
    sources_parameter_x_sorted = sources.sort_values(by=parameter_x)
    bin_boundaries_x = np.linspace(boundaries[parameter_x][0], boundaries[parameter_x][1],n_bins+1)
    if parameter_x == 'RightAscension':
        bin_boundaries_x = np.linspace(0, 2*boundaries[parameter_x][1],n_bins+1)
    bin_boundaries_y = np.linspace(boundaries[parameter_y][0], boundaries[parameter_y][1],n_bins+1)
    # bin_boundaries_x = np.log10(bin_boundaries_x)
    # bin_boundaries_y = np.log10(bin_boundaries_y)
    if parameter_x == 'Frequency':
        bin_boundaries_x = np.logspace(np.log10(boundaries[parameter_x][0]), np.log10(boundaries[parameter_x][1]),n_bins+1)
        bin_boundaries_y = np.logspace(np.log10(boundaries[parameter_y][0]), np.log10(boundaries[parameter_y][1]),n_bins+1)
    for i in range(n_bins):
        error.append([])
        std.append([])
        count.append([])
        # length = (boundaries[parameter_x][1] - boundaries[parameter_x][0])/n_bins
        start_index = np.searchsorted(sources_parameter_x_sorted[parameter_x],bin_boundaries_x[i], side='left')
        end_index = np.searchsorted(sources_parameter_x_sorted[parameter_x],bin_boundaries_x[i+1], side='left')
        section = sources_parameter_x_sorted[start_index:end_index]
        for j in range(n_bins):
            section_parameter_y_sorted = section.sort_values(by=parameter_y)
            # length = (boundaries[parameter_y][1] - boundaries[parameter_y][0])/n_bins
            start_index = np.searchsorted(section_parameter_y_sorted[parameter_y],bin_boundaries_y[j], side='left')
            end_index = np.searchsorted(section_parameter_y_sorted[parameter_y],bin_boundaries_y[j+1], side='left')
            field = section[start_index:end_index]
            change_to_deg = False
            if get_errors == True:
                if parameter_to_plot+'Error' == 'SkylocationError':
                    change_to_deg = False
                if change_to_deg:
                    field[parameter_to_plot+'Error'] *= 180/np.pi
                if parameter_to_plot in ['Frequency']:
                    try:
                        error[-1][-1]=(np.mean(np.abs(field[parameter_to_plot+'Error']/field[parameter_to_plot])))
                    except:
                        pass
                elif parameter_to_plot in ['Match']:
                    error[-1].append(np.mean(np.abs(field[parameter_to_plot])))
                else:
                    error[-1].append(np.mean(np.abs(field[parameter_to_plot+'Error'])))
                if parameter_to_plot in ['Match']:
                    std[-1].append(np.std(field[parameter_to_plot]))
                else:
                    std[-1].append(np.std(field[parameter_to_plot+'Error']))
            count[-1].append(len(field['Frequency']))
    for i in range(len(count)):
        for j in range(len(count[i])):
            if count[i][j] == 0:
                count[i][j] = np.nan
    return error, std, count, bin_boundaries_x, bin_boundaries_y

# data_set =0
error, std, count_not_matched, bin_boundaries_x, bin_boundaries_y = get_2d_hist(found_sources_not_matched_df, get_errors=False, parameter_to_plot=parameter_to_plot)
error, std, count, bin_boundaries_x, bin_boundaries_y = get_2d_hist(found_sources_matched_df, get_errors=True, parameter_to_plot=parameter_to_plot)

for i in range(len(count_not_matched)):
    for j in range(len(count_not_matched[i])):
        if np.isnan(count_not_matched[i][j]):
            count_not_matched[i][j] = 0

count_zeros = deepcopy(count)
for i in range(len(count_zeros)):
    for j in range(len(count_zeros[i])):
        if np.isnan(count_zeros[i][j]):
            count_zeros[i][j] = 0

match_ratio = np.asarray(count_zeros)/(np.asarray(count_zeros)+np.asarray(count_not_matched))

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
fig.set_size_inches(8,10)
im = ax0.pcolormesh(bin_boundaries_x,bin_boundaries_y, np.array(error).T)
# im.set_clim(0,5)
fig.colorbar(im, ax=ax0)
ax0.set_title('mean '+parameter_to_plot)
im1 = ax1.pcolormesh(bin_boundaries_x,bin_boundaries_y, np.array(std).T)
# im1.set_clim(0,5)
fig.colorbar(im1, ax=ax1)
ax1.set_title('standard deviation')
im2 = ax2.pcolormesh(bin_boundaries_x,bin_boundaries_y, np.array(count).T)
fig.colorbar(im2, ax=ax2)
ax2.set_title('number of signals')
ax0.set_yscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax0.set_xscale('log')
ax1.set_xscale('log')
ax2.set_xscale('log')
ax2.set_xlabel(parameter_x)
ax0.set_ylabel(parameter_y)
ax1.set_ylabel(parameter_y)
ax2.set_ylabel(parameter_y)
fig.tight_layout()
plt.savefig(SAVEPATH+'/Evaluation/'+parameter_x+parameter_y+parameter_to_plot+save_names[data_set]+end_string)
plt.show()




fig, ax0 = plt.subplots(nrows=1, figsize=(10,6))
# fig.set_size_inches(6,6)
im = ax0.pcolormesh(bin_boundaries_x,bin_boundaries_y, np.array(match_ratio).T)
# im.set_clim(0,5)
cbar = fig.colorbar(im, ax=ax0)
# cbar.ax.set_ylabel('# of contacts', rotation=270, )
ax0.set_title('match ratio')
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.set_ylabel(parameter_y)
ax0.set_xlabel(parameter_x)
fig.tight_layout()
plt.savefig(SAVEPATH+'/Evaluation/'+parameter_x+parameter_y+'match_rate'+save_names[data_set]+end_string)
plt.show()

##### plot SNR to error
# for parameter in  parameters +['Skylocation']:
# for parameter in  ['SkylocationError']:
x_parameter = 'Frequency'
# x_parameter = 'IntrinsicSNR'
data_set = 0
for parameter in  ['Frequency']:
    # x_parameter = parameter
    markersize = 4
    alpha = 0.5
    fig = plt.figure()
    # plt.plot(found_sources_matched_list[data_set][parameter+'Error']/found_sources_matched_list[data_set][parameter],found_sources_matched_list[data_set][y_parameter],'.', label = 'found', markersize=markersize)
    # if parameter in ['Frequency', 'FrequencyDerivative']:
    #     plt.scatter(found_sources_matched_list[data_set][x_parameter],found_sources_matched_list[data_set][parameter+'Error']/found_sources_matched_list[data_set]['Frequency'], c=found_sources_matched_list[data_set]['Frequency'])
    # else:
    plt.plot(found_sources_matched_list[data_set][x_parameter],found_sources_matched_list[data_set][parameter+'Error'],'.', alpha =0.4, markersize=6)
        # plt.scatter(found_sources_matched_list[data_set][x_parameter],found_sources_matched_list[data_set][parameter], c=found_sources_matched_list[data_set]['Frequency'])
    plt.ylabel(parameter+' Error')
    if parameter in ['Frequency', 'FrequencyDerivative']:
        plt.ylabel('Relative '+parameter+' Error')
    plt.xlabel(x_parameter)
    plt.xscale('log')
    plt.yscale('log')
    # plt.legend(markerscale=4, loc = 'upper right')
    plt.savefig(SAVEPATH+'/Evaluation/SNRto'+parameter+'Error'+save_names[data_set]+end_string)
    plt.show()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
blue = deepcopy(colors[0])
colors[0] = deepcopy(colors[1])
colors[1] = deepcopy(blue)
colors[2] = '#00008B' 
# colors[3] = '#BB5500' 

linestyle = ['solid', 'dashed', 'solid', 'solid']
# labels_plot = ['ETH 1 yr', 'MM 1 yr', 'ETH 0.5 yr', 'MM 0.5 yr']
# labels_plot = ['LDC1 0.5 yr', 'LDC1 1 yr', 'LDC2 1 yr', 'LDC1 2 yr']
# labels_plot = ['LDC1 2 yr', 'LDC2 1 yr', 'LDC1 1 yr', 'LDC1 0.5 yr']
labels_plot = ['Radler 2 yr', 'Sangria 1 yr', 'Radler 1 yr', 'Radler 0.5 yr']
line_width = 3
custom_lines = [plt.Line2D([0], [0], color=colors[0], lw=line_width, linestyle=linestyle[0]),
                plt.Line2D([0], [0], color=colors[1], lw=line_width, linestyle=linestyle[1]),
                plt.Line2D([0], [0], color=colors[2], lw=line_width, linestyle=linestyle[2]),
                plt.Line2D([0], [0], color=colors[3], lw=line_width, linestyle=linestyle[3])]


labels = {'EclipticLongitude': r'$\lambda$'+' (rad)', 'EclipticLatitude': r'$\beta$'+' (rad)','Frequency': '$f$ $($mHz$)$','FrequencyDerivative': r'$\dot{f}$ $ ($Hz/s$)$','Inclination': r'$\iota$'+' (rad)','Amplitude': r'A', 'Polarization': r'$\psi$'+' (rad)', 'InitialPhase': r'$\phi_0$'+' (rad)'}
labels = {'EclipticLongitude': r'$\lambda$'+' (rad)', 'EclipticLatitude': r'$\beta$'+' (rad)','Frequency': r'$f / f_\mathrm{true}$','FrequencyDerivative': r'$\dot{f}$ $ ($Hz/s$)$','Inclination': r'$\iota$'+' (rad)','Amplitude': r'$ \mathcal{A} / \mathcal{A}_\mathrm{true}$', 'Polarization': r'$\psi$'+' (rad)', 'InitialPhase': r'$\phi_0$'+' (rad)'}

parameter_to_plot = 'Match'
parameter_x = 'EclipticLongitude'
parameter_y = 'EclipticLatitude'
fig, ax = plt.subplots(1, 1, figsize=np.array(fig_size), constrained_layout=True)
error, std, count, bin_boundaries_x, bin_boundaries_y = get_2d_hist(found_sources_list[data_set], get_errors=True, parameter_to_plot=parameter_to_plot)
im = ax.pcolormesh(bin_boundaries_x,bin_boundaries_y, np.array(error).T)
fig.colorbar(im, ax=ax, label=parameter_to_plot)
plt.show()

### histogram
# for parameter in parameters: 
# for parameter in  ['Amplitude']:
    # fig = plt.figure()
parameter_order = [
    "RightAscension",
    "Frequency",
    "Amplitude",
    "InitialPhase",
    "Declination",
    "FrequencyDerivative",
    "Inclination",
    "Polarization",
]

parameter_order_reduced = [
    "EclipticLatitude",
    "EclipticLongitude",
    "Frequency",
    "FrequencyDerivative",
    "Amplitude",
    "Inclination",
]

# fig, axs = plt.subplots(3, 2, figsize=np.array(fig_size)*[1.8,6], constrained_layout=True)
# parameter_x = 'Frequency'
# parameter_y = 'Amplitude'
# for ax, parameter in zip(axs.flat, parameter_order_reduced):
#     parameter_to_plot = parameter
#     # error, std, count_not_matched, bin_boundaries_x, bin_boundaries_y = get_2d_hist(found_sources_not_matched_list[data_set], get_errors=False)
#     error, std, count, bin_boundaries_x, bin_boundaries_y = get_2d_hist(found_sources_matched_list[data_set], get_errors=True, parameter_to_plot=parameter_to_plot)
#     im = ax.pcolormesh(bin_boundaries_x,bin_boundaries_y, np.array(error).T)
#     fig.colorbar(im, ax=ax, label='$\Delta$'+labels[parameter])
#     # ax.set_title(parameter_to_plot)
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     # ax.set_xlabel(parameter_x)
#     # ax.set_ylabel(parameter_y)
# for i in range(3):
#     axs[i,0].set_ylabel(parameter_y)
# # for i in range(4):
# # for i in range(4):
# # axs[1,0].set_ylabel(parameter_y)
# # for i in range(4):
# axs[-1,0].set_xlabel(parameter_x)
# axs[-1,1].set_xlabel(parameter_x)
# # fig.tight_layout()
# plt.savefig(SAVEPATH+'/Evaluation/'+parameter_x+parameter_y+parameter_to_plot+save_names[data_set]+end_string)
# plt.show()

line_width = 3
n_bins = 30
fig, axs = plt.subplots(2, 4, figsize=(12,5), constrained_layout=True)
for ax, parameter in zip(axs.flat, parameter_order):
    if parameter == 'Skylocation':
        ax.hist(np.abs(found_sources_matched_df[parameter+'Error']), bins= np.logspace(-2,2, n_bins))
    elif parameter == 'Frequency':
        ax.hist(np.abs(found_sources_matched_df[parameter+'Error']/pGB_injected_matched_df[parameter]), bins= np.logspace(-8,-3.5, n_bins), log=False, density=False, histtype='step', linewidth=line_width)
    elif parameter == 'FrequencyDerivative':
        ax.hist(np.abs(found_sources_matched_df[parameter+'Error']), bins=np.logspace(-19,-13, n_bins), density=False, histtype='step', linewidth=line_width)
    elif parameter == 'Amplitude':
        ax.hist(found_sources_matched_df[parameter+'Error'], bins=np.logspace(-4,1,n_bins), density=False, histtype='step', linewidth=line_width)
    elif parameter == 'Inclination':
        ax.hist(found_sources_matched_df[parameter+'Error'], bins=np.logspace(-5,np.log10(np.pi/2), n_bins), density=False, histtype='step', linewidth=line_width)
    elif parameter in ['RightAscension', 'Declination']:
        print(parameter)
        ax.hist(found_sources_matched_df[parameter+'Error'], bins=np.logspace(-5,np.log10(np.pi/2), n_bins), log= False, density=False, histtype='step', linewidth=line_width)
    else:
        ax.hist(found_sources_matched_df[parameter+'Error'], bins=n_bins, density=False, histtype='step', linewidth=line_width)
    ax.set_xlabel('$\Delta$'+parameter)
    if parameter == 'Skylocation':
        ax.set_xlabel('$\Delta$'+labels[parameter]+' (deg)')
        ax.set_xscale('log')
    if parameter == 'Inclination':
        ax.set_xlabel('$\Delta$'+labels[parameter])
        ax.set_xlim(10**-5,np.pi/2)
        ax.set_xscale('log')
    if parameter == 'Amplitude':
        ax.set_xlabel(r'$\Delta \mathcal{A} / \mathcal{A}_{true}$')
        ax.set_xscale('log')
        ax.set_xlim(10**-4,10)
    if parameter == 'FrequencyDerivative':
        ax.set_xscale('log')
        ax.set_xlabel('$\Delta$'+labels[parameter])
        ax.set_xlim(10**-19,10**-13)
    if parameter == 'Frequency':
        ax.set_xscale('log')
        ax.set_xlim(10**-8,10**-4)
        # ax.ylim(0,10**3)
        ax.set_xlabel(r'$\Delta f / f_{true}$')
    if parameter in [ 'InitialPhase', 'Polarization']:
        ax.set_xlabel('$\Delta$'+labels[parameter])
        ax.set_xlim(0,np.pi/2)
    if parameter in ['RightAscension', 'Declination']:
        ax.set_xlabel('$\Delta$'+labels[parameter])
        ax.set_xlim(10**-5,np.pi/2)
        ax.set_xscale('log')
    # ax.legend(custom_lines, [label1, label2, label3], loc='best')
    ax.grid(True)
    # ax.legend(loc='upper right')
# ax.yscale('log')
# axs[0,1].legend(custom_lines, [label1, label2, label3], loc='best')
# axs[1,1].legend(custom_lines, [label1, label2, label3], loc='best')
# axs[0,-1].legend(custom_lines, labels_plot, loc='upper right')
# axs[1,0].legend(custom_lines, [label1, label2, label3], loc='upper left')
# axs[0,2].legend(custom_lines, [label1, label2, label3], loc='upper left')
# axs[1,2].legend(custom_lines, [label1, label2, label3], loc='upper right')
axs[0,0].set_ylabel('Count')
axs[1,0].set_ylabel('Count')
# plt.savefig(SAVEPATH+'/Evaluation/error_histogram'+save_names[0]+save_names[1]+save_names[2]+save_names[3]+end_string+'linear',dpi=300,bbox_inches='tight')
plt.show()

plus = [2.5,2.05,1.85,2.35]
cross = [2.2,1.9,2.47,2.19,2.46,1.9]
mid = 2.18
factor = 0.4/(2.5-2.18)

cmap = plt.get_cmap("seismic")
for i in range(len(plus)):
    print(cmap((cross[i]-mid)*factor+0.5))

plt.figure()
m = plt.plot([0,1],[0,1], color= cmap((plus[2]-mid)*factor+0.5), label = 'ETH 2 yr')
norm = mpl.colors.Normalize(vmin=0.1,vmax=0.9)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])   
# plt.legend(markerscale=4, loc = 'upper right')
plt.tight_layout()
# cbar= plt.colorbar(sm, ticks=weeks_plot, label='Observation time (weeks)', boundaries=np.arange(np.min(weeks_plot)-increment/2,np.max(weeks_plot+1)+increment/2,increment), ax=axs.ravel().tolist())
cbar= plt.colorbar(sm,  label='Contracted                                          Stretched', orientation='horizontal', ticks=[])
plt.tight_layout()
plt.show()


parameter_order = [
    "EclipticLatitude",
    "EclipticLongitude",
    "Frequency",
    "FrequencyDerivative",
    "Amplitude",
    "Inclination",
    "Polarization",
    "InitialPhase",
]

n_bins = 30
fig, axs = plt.subplots(4, 2, figsize=np.array(fig_size)*[1.5,1.5], constrained_layout=True)
for ax, parameter in zip(axs.flat, parameter_order):
    for i in range(4):
        if parameter == 'Skylocation':
            ax.hist(np.abs(found_sources_matched_list[i][parameter+'Error']), bins= np.logspace(-2,2, n_bins))
        elif parameter == 'Frequency':
            ax.hist(np.abs(found_sources_matched_list[i][parameter+'Error']/pGB_injected_matched_list[i][parameter]), bins= np.logspace(-8,-3.5, n_bins), log=False, density=False, histtype='step', linestyle=linestyle[i], color=colors[i], linewidth=line_width)
        elif parameter == 'FrequencyDerivative':
            ax.hist(np.abs(found_sources_matched_list[i][parameter+'Error']), bins=np.logspace(-19,-13, n_bins), density=False, histtype='step', linestyle=linestyle[i], color=colors[i], linewidth=line_width)
        elif parameter == 'Amplitude':
            ax.hist(found_sources_matched_list[i][parameter+'Error'], bins=np.logspace(-4,1,n_bins), density=False, histtype='step', linestyle=linestyle[i], color=colors[i], linewidth=line_width)
        elif parameter == 'Inclination':
            ax.hist(found_sources_matched_list[i][parameter+'Error'], bins=np.logspace(-5,np.log10(np.pi/2), n_bins), density=False, histtype='step', linestyle=linestyle[i], color=colors[i], linewidth=line_width)
        elif parameter in ['EclipticLatitude', 'EclipticLongitude']:
            ax.hist(found_sources_matched_list[i][parameter+'Error'], bins=np.logspace(-5,np.log10(np.pi/2), n_bins), log= False, density=False, histtype='step', linestyle=linestyle[i], color=colors[i], linewidth=line_width)
        else:
            ax.hist(found_sources_matched_list[i][parameter+'Error'], bins=n_bins, density=False, histtype='step', linestyle=linestyle[i], color=colors[i], linewidth=line_width)
    ax.set_xlabel('$\Delta$'+parameter)
    if parameter == 'Skylocation':
        ax.set_xlabel('$\Delta$'+labels[parameter]+' (deg)')
        ax.set_xscale('log')
    if parameter == 'Inclination':
        ax.set_xlabel('$\Delta$'+labels[parameter])
        ax.set_xlim(10**-5,np.pi/2)
        ax.set_xscale('log')
    if parameter == 'Amplitude':
        ax.set_xlabel(r'$\Delta \mathcal{A} / \mathcal{A}_{true}$')
        ax.set_xscale('log')
        ax.set_xlim(10**-4,10)
    if parameter == 'FrequencyDerivative':
        ax.set_xscale('log')
        ax.set_xlabel('$\Delta$'+labels[parameter])
        ax.set_xlim(10**-19,10**-13)
    if parameter == 'Frequency':
        ax.set_xscale('log')
        ax.set_xlim(10**-8,10**-4)
        # ax.ylim(0,10**3)
        ax.set_xlabel(r'$\Delta f / f_{true}$')
    if parameter in [ 'InitialPhase', 'Polarization']:
        ax.set_xlabel('$\Delta$'+labels[parameter])
        ax.set_xlim(0,np.pi/2)
    if parameter in ['EclipticLongitude', 'EclipticLatitude']:
        ax.set_xlabel('$\Delta$'+labels[parameter])
        ax.set_xlim(10**-5,np.pi/2)
        ax.set_xscale('log')
    # ax.legend(custom_lines, [label1, label2, label3], loc='best')
    ax.grid(True)
    # ax.legend(loc='upper right')
# ax.yscale('log')
# axs[0,1].legend(custom_lines, [label1, label2, label3], loc='best')
# axs[1,1].legend(custom_lines, [label1, label2, label3], loc='best')
axs[-1,-1].legend(custom_lines, labels_plot, loc='upper right')
# axs[1,0].legend(custom_lines, [label1, label2, label3], loc='upper left')
# axs[0,2].legend(custom_lines, [label1, label2, label3], loc='upper left')
# axs[1,2].legend(custom_lines, [label1, label2, label3], loc='upper right')
axs[0,0].set_ylabel('Count')
axs[1,0].set_ylabel('Count')
axs[2,0].set_ylabel('Count')
axs[3,0].set_ylabel('Count')
plt.savefig(SAVEPATH+'/Evaluation/error_histogram4'+save_names[0]+save_names[1]+save_names[2]+save_names[3]+end_string+'linear',dpi=300,bbox_inches='tight')
plt.show()

rcParams.update(plot_parameter_big)

upper_bound = 0.5
fig, ax1 = plt.subplots(1,1, figsize=fig_size)
for i in range(4):
    ax1.hist(match_flat_list[i], bins= np.linspace(0,upper_bound,n_bins), histtype='step', linestyle=linestyle[i], linewidth=2, color=colors[i])
    # ax2.hist(match_flat_list[i], bins= np.linspace(0,upper_bound,n_bins), histtype='step', linestyle=linestyle[i], linewidth=2, color=colors[i])
plt.xlabel('$\delta$')
plt.ylabel('Count')
# plt.yscale('log')
plt.legend(custom_lines, labels_plot, loc='upper right')
# plt.ylim(0,2500)
plt.xlim(0,upper_bound)
plt.grid(True)
plt.savefig(SAVEPATH+'/Evaluation/correlation_comparison'+save_names[0]+save_names[1]+save_names[2]+save_names[3]+end_string,dpi=300,bbox_inches='tight')
plt.show()


fig, [ax1, ax2] = plt.subplots(2,1, figsize=np.array(fig_size)*[1,1.5])
for i in range(4):
    olap = match_flat_list[i]
    # olap_high = olap[np.array(olap)<1]
    olap_high = olap[np.array(olap)>0]
    olap_high.sort()
    # olap_high = olap_high[:-1]
    ax1.plot(olap_high, np.arange(len(olap_high))/len(olap_high), label=labels_plot[i], linestyle=linestyle[i], linewidth=2, color=colors[i])

    olap = match_flat_list[i]
    # olap_high = olap[np.array(olap)<1]
    olap_high = olap[np.array(olap)>0]
    olap_high.sort()
    olap_high = olap_high[::-1]
    cumsum = np.cumsum(np.ones_like(olap_high))
    ax2.plot(olap_high, cumsum, label=labels_plot[i], linestyle=linestyle[i], linewidth=2, color=colors[i])
    # axs.hist(olap_high, histtype=u'step', density=True)
ax1.grid(True)
# ax1.set_ylabel('fraction of counts > $\delta$')
ax1.set_ylabel('fraction of counts < $\mathcal{O}$')
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax2.grid(True)
# ax2.set_xlabel('$\delta$')
# ax2.set_ylabel('total counts < $\delta$')
ax2.set_xlabel('$\mathcal{O}$')
ax2.set_ylabel('total counts > $\mathcal{O}$')
ax2.set_xlim(0,1)
# ax2.set_ylim(0,1)
ax1.legend()
plt.savefig(SAVEPATH+'/Evaluation/CDF'+save_names[0]+save_names[1]+save_names[2]+save_names[3]+end_string,dpi=300,bbox_inches='tight')
plt.tight_layout()
plt.show()

upper_boundx = 0.5
upper_boundy = 50
n_bins = 100
vmax = [20,20,10,10]
for i in range(4):
    fig = plt.figure(figsize=fig_size, constrained_layout=True)
    plt.hist2d(match_flat_list[i], match_best_list[i]['IntrinsicSNR'], bins= (np.linspace(0,upper_boundx,n_bins),np.linspace(5,upper_boundy,n_bins)), vmin=0)
    plt.xlabel('$\delta$')
    plt.ylabel('SNR')
    plt.title(labels_plot[i])
    plt.colorbar( )
    plt.xlim(0,upper_boundx)
    plt.ylim(5,upper_boundy)
    plt.grid(True)
    plt.savefig(SAVEPATH+'/Evaluation/match_SNR_comparison'+save_names[i]+end_string,dpi=300,bbox_inches='tight')
    plt.show()



fig = plt.figure(figsize=fig_size, constrained_layout=True)
plt.hist(found_sources_matched_list[data_set]['Frequency'], bins= np.linspace(0,0.03,n_bins), histtype='step', linestyle=linestyle[i], linewidth=2, log=False)
# plt.hist(found_sources_matched_list[data_set]2['Frequency'], bins= np.linspace(0,0.03,n_bins), histtype='step', linewidth=2, log=False)
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.legend(custom_lines,  labels_plot, loc='upper right')
plt.ylim(0,1000)
plt.xlim(0,0.03)
plt.grid(True)
plt.savefig(SAVEPATH+'/Evaluation/frequency_hist_comparison'+save_name+save_name2+end_string,dpi=300,bbox_inches='tight')
plt.show()

found_sources_df_list = [None]*4
for i in range(4):
    found_sources_df_list[i] = pd.concat([found_sources_matched_list[i], found_sources_not_matched_list[i]])
# found_sources_df2 = pd.concat([found_sources_matched_list[data_set]2, found_sources_not_matched_flat_df2])
# found_sources_df3 = pd.concat([found_sources_matched_list[data_set]3, found_sources_not_matched_flat_df3])
# found_sources_df4 = pd.concat([found_sources_matched_list[data_set]4, found_sources_not_matched_flat_df4])

# fig = plt.figure(figsize=fig_size)
# counts3, bins, bars = plt.hist(found_sources_df3['Frequency'], bins= np.linspace(0,0.03,n_bins), histtype='step', linewidth=2, log=False)
# counts, bins, bars = plt.hist(found_sources_df['Frequency'], bins= np.linspace(0,0.03,n_bins), histtype='step', linestyle=linestyle[i], linewidth=2.5, log=False)
# counts2, bins, bars = plt.hist(found_sources_df2['Frequency'], bins= np.linspace(0,0.03,n_bins), histtype='step', linewidth=2, log=False)


# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=fig_size )
# # fig = plt.figure(figsize=fig_size, constrained_layout=True)
# counts_matched, bins, bars = ax1.hist(found_sources_matched_list[data_set]['Frequency']*1000, bins= np.linspace(0,30,n_bins), histtype='step', linestyle=linestyle[i], linewidth=3, log=False)
# counts_matched2, bins, bars = ax1.hist(found_sources_matched_flat_df2['Frequency']*1000, bins= np.linspace(0,30,n_bins), histtype='step', linestyle=linestyle[i], linewidth=2.5, log=False)
# counts_matched3, bins, bars = ax1.hist(found_sources_matched_flat_df3['Frequency']*1000, bins= np.linspace(0,30,n_bins), histtype='step', linewidth=2, log=False)
# match_rate = counts_matched / counts
# match_rate2 = counts_matched2 / counts2
# match_rate3 = counts_matched3 / counts3
# # match_rate = np.nan_to_num(match_rate, nan=0.0)True)
# fig = plt.figure(figsize=fig_size)
# counts, bins, bars = plt.hist(found_sources_df['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linestyle=linestyle[i], linewidth=3, log=False, color=colors[0])
# counts2, bins, bars = plt.hist(found_sources_df2['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linestyle=linestyle[i], linewidth=2.5, log=False, color=colors[1])
# counts3, bins, bars = plt.hist(found_sources_df3['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linewidth=2, log=False, color=colors[2])
# counts4, bins, bars = plt.hist(found_sources_df4['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linewidth=2, log=False, color=colors[3])
# plt.xscale('log')
# plt.show()

n_bins = 30
match_rate_list = [None]*4
marker_list = ['.','o','P','P']
markersize_list = [20,7,8,7]
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=[9,9])
for i in range(4):
    counts, bins, bars = ax1.hist(found_sources_df_list[i]['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linestyle=linestyle[i], linewidth=3, log=False, color=colors[i])
    # counts2, bins, bars = ax1.hist(found_sources_df2['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linestyle=linestyle[i], linewidth=2.5, log=False, color=colors[1])
    # counts3, bins, bars = ax1.hist(found_sources_df3['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linewidth=2, log=False, color=colors[2])
    # counts4, bins, bars = ax1.hist(found_sources_df4['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linewidth=2, log=False, color=colors[3])
    # fig = plt.figure(figsize=fig_size, constrained_layout=True)
    # counts, bins, bars = plt.hist(found_sources_df['Frequency'], bins= np.linspace(0,0.03,n_bins), histtype='step', linestyle=linestyle[i], linewidth=2.5, log=False)
    # counts2, bins, bars = plt.hist(found_sources_df2['Frequency'], bins= np.linspace(0,0.03,n_bins), histtype='step', linewidth=2, log=False)
    counts_matched, bins, bars = ax2.hist(found_sources_matched_list[i]['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linestyle=linestyle[i], linewidth=3, log=False, color=colors[i])
    # counts_matched2, bins, bars = ax2.hist(found_sources_matched_flat_df2['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linestyle=linestyle[i], linewidth=2.5, log=False, color=colors[1])
    # counts_matched3, bins, bars = ax2.hist(found_sources_matched_flat_df3['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linewidth=2, log=False, color=colors[2])
    # counts_matched4, bins, bars = ax2.hist(found_sources_matched_flat_df4['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linewidth=2, log=False, color=colors[3])
    match_rate_list[i] = counts_matched / counts
# match_rate2 = counts_matched2 / counts2
# match_rate3 = counts_matched3 / counts3
# match_rate4 = counts_matched4 / counts4
# match_rate = np.nan_to_num(match_rate, nan=0.0)
# match_rate2 = np.nan_to_num(match_rate2, nan=0.0)
bins = np.logspace(np.log10(0.3),np.log10(30),n_bins)
center_bins = np.logspace(np.log10((bins[1]+bins[0])/2),np.log10((bins[-2]+bins[-1])/2),n_bins-1)
ax3.set_xlabel('Frequency (mHz)')
ax1.set_ylabel('Submitted Signals')
ax2.set_ylabel('Matched Signals')
ax1.legend(custom_lines,  labels_plot, loc='upper right')
ax2.legend(custom_lines,  labels_plot, loc='upper right')
ax1.set_ylim(0,2500)
ax2.set_ylim(0,2500)
# ax3.set_ylim(0,1)
# ax1.set_xlim(0,30)
ax1.set_xscale('log')
ax2.set_xscale('log')
ax3.set_xscale('log')
ax1.grid(True)
ax2.grid(True)
for i in range(4):
    ax3.plot(center_bins, match_rate_list[i], marker_list[i],markersize= markersize_list[i],label=labels_plot[i], color=colors[i])
# ax3.plot(center_bins,match_rate3, '.', markersize=20, label=label3, color=colors[2])
# ax3.plot(center_bins,match_rate4, 'P', markersize=7, label=label4, color=colors[3])
# ax3.plot(center_bins,match_rate, 'o', markersize=8, label=label1 ,markerfacecolor='None', color=colors[0])
# ax3.plot(center_bins,match_rate2, 'P', markersize=7, label=label2, markerfacecolor='None', color=colors[1])
ax3.set_ylabel('Match Rate')
ax3.grid(True)
ax3.legend(loc='lower right')
plt.savefig(SAVEPATH+'/Evaluation/frequency_hist_all_found_comparison3'+save_names[0]+save_names[1]+save_names[2]+save_names[3]+end_string,dpi=300,bbox_inches='tight')
plt.show()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, constrained_layout=True, figsize=[9,12])
for i in range(4):
    counts_injected, bins, bars = ax1.hist(pGB_injected_high_SNR_list[i]['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linestyle=linestyle[i], linewidth=3, log=False, color=colors[i])
    counts, bins, bars = ax2.hist(found_sources_df_list[i]['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linestyle=linestyle[i], linewidth=3, log=False, color=colors[i])
    # counts2, bins, bars = ax1.hist(found_sources_df2['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linestyle=linestyle[i], linewidth=2.5, log=False, color=colors[1])
    # counts3, bins, bars = ax1.hist(found_sources_df3['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linewidth=2, log=False, color=colors[2])
    # counts4, bins, bars = ax1.hist(found_sources_df4['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linewidth=2, log=False, color=colors[3])
    # fig = plt.figure(figsize=fig_size, constrained_layout=True)
    # counts, bins, bars = plt.hist(found_sources_df['Frequency'], bins= np.linspace(0,0.03,n_bins), histtype='step', linestyle=linestyle[i], linewidth=2.5, log=False)
    # counts2, bins, bars = plt.hist(found_sources_df2['Frequency'], bins= np.linspace(0,0.03,n_bins), histtype='step', linewidth=2, log=False)
    counts_matched, bins, bars = ax3.hist(found_sources_matched_list[i]['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linestyle=linestyle[i], linewidth=3, log=False, color=colors[i])
    # counts_matched2, bins, bars = ax2.hist(found_sources_matched_flat_df2['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linestyle=linestyle[i], linewidth=2.5, log=False, color=colors[1])
    # counts_matched3, bins, bars = ax2.hist(found_sources_matched_flat_df3['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linewidth=2, log=False, color=colors[2])
    # counts_matched4, bins, bars = ax2.hist(found_sources_matched_flat_df4['Frequency']*1000, bins= np.logspace(np.log10(0.3),np.log10(30),n_bins), histtype='step', linewidth=2, log=False, color=colors[3])
    match_rate_list[i] = counts_matched / counts
# match_rate2 = counts_matched2 / counts2
# match_rate3 = counts_matched3 / counts3
# match_rate4 = counts_matched4 / counts4
# match_rate = np.nan_to_num(match_rate, nan=0.0)
# match_rate2 = np.nan_to_num(match_rate2, nan=0.0)
bins = np.logspace(np.log10(0.3),np.log10(30),n_bins)
center_bins = np.logspace(np.log10((bins[1]+bins[0])/2),np.log10((bins[-2]+bins[-1])/2),n_bins-1)
ax4.set_xlabel('Frequency (mHz)')
ax1.set_ylabel('Injected SNR > 10')
ax2.set_ylabel('Submitted')
ax3.set_ylabel('Matched')
ax1.legend(custom_lines,  labels_plot, loc='upper right')
# ax2.legend(custom_lines,  labels_plot, loc='upper right')
ax1.set_ylim(0,2500)
ax2.set_ylim(0,2500)
ax3.set_ylim(0,2500)
# ax3.set_ylim(0,1)
# ax1.set_xlim(0,30)
ax1.set_xscale('log')
ax2.set_xscale('log')
ax3.set_xscale('log')
ax4.set_xscale('log')
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
for i in range(4):
    ax4.plot(center_bins, match_rate_list[i], marker_list[i],markersize= markersize_list[i],label=labels_plot[i], color=colors[i])
# ax3.plot(center_bins,match_rate3, '.', markersize=20, label=label3, color=colors[2])
# ax3.plot(center_bins,match_rate4, 'P', markersize=7, label=label4, color=colors[3])
# ax3.plot(center_bins,match_rate, 'o', markersize=8, label=label1 ,markerfacecolor='None', color=colors[0])
# ax3.plot(center_bins,match_rate2, 'P', markersize=7, label=label2, markerfacecolor='None', color=colors[1])
ax4.set_ylabel('Match Rate')
ax4.grid(True)
ax4.legend(loc='lower right')
plt.savefig(SAVEPATH+'/Evaluation/frequency_hist_all_found_comparison4'+save_names[0]+save_names[1]+save_names[2]+save_names[3]+end_string,dpi=300,bbox_inches='tight')
plt.show()


def get_distance(amplitude, frequency, frequency_derivative):
    c = 3*10**8
    distance = 2/(96/5*np.pi**(8/3)*frequency**(11/3))*frequency_derivative*np.pi*(2/3)*frequency**(2/3)/amplitude*c /3.086e+19 #to kpc
    return distance

def get_distance_for_dataframe(dataframe):
    dataframe['Distance'] = np.zeros(len(dataframe['Amplitude']))
    postitive_fd_mask = dataframe['FrequencyDerivative'] >= 0
    distance = get_distance(dataframe['Amplitude'][postitive_fd_mask], dataframe['Frequency'][postitive_fd_mask],  dataframe['FrequencyDerivative'][postitive_fd_mask])
    dataframe['Distance'][postitive_fd_mask] = distance
    return dataframe

def get_galactic_coordinates(dataframe):
    coordinates_found = coord.SkyCoord(dataframe['EclipticLongitude'], dataframe['EclipticLatitude'], dataframe['Distance'], unit='rad', frame='barycentricmeanecliptic')
    dataframe['GalacticLongitude'] = coordinates_found.galactic.l.value
    dataframe['GalacticLatitude'] = coordinates_found.galactic.b.value
    dataframe['GalacticDistance'] = coordinates_found.galactic.distance.value
    return dataframe

def get_galactocentric_coordinates(dataframe):
    # dataframe = dataframe[dataframe['FrequencyDerivative'] >= 0]
    #### get Galactocentric coordinates with distance
    # coordinates_found = coord.SkyCoord(dataframe['EclipticLongitude']* u.rad, dataframe['EclipticLatitude']* u.rad, dataframe['Distance'] * u.kpc, frame='barycentricmeanecliptic')
    coordinates_found = coord.SkyCoord(np.array(dataframe['EclipticLongitude'])* u.rad, np.array(dataframe['EclipticLatitude'])* u.rad, np.array(dataframe['Distance'])* u.kpc, frame='barycentricmeanecliptic')
    coordinates_found = coordinates_found.transform_to(coord.Galactocentric) 
    dataframe['GalactocentricX'] = coordinates_found.galactocentric.x.value
    dataframe['GalactocentricY'] = coordinates_found.galactocentric.y.value
    dataframe['GalactocentricZ'] = coordinates_found.galactocentric.z.value
    return dataframe['GalactocentricX'], dataframe['GalactocentricY'], dataframe['GalactocentricZ'] 

# found_sources_matched_flat_df['Amplitude'] = pGB_injected_matched_flat_df['Amplitude']
# found_sources_matched_flat_df['FrequencyDerivative'] = pGB_injected_matched_flat_df['FrequencyDerivative']
found_sources_matched_positive_fd_list = []
found_sources_not_matched_positive_fd_list = []
pGB_injected_matched_positive_fd_list = []
pGB_injected_positive_fd_high_SNR_list = []
for i in range(4):
    found_sources_matched_list[i] = get_distance_for_dataframe(found_sources_matched_list[i])
    found_sources_not_matched_list[i] = get_distance_for_dataframe(found_sources_not_matched_list[i])
    pGB_injected_matched_list[i] = get_distance_for_dataframe(pGB_injected_matched_list[i])
    pGB_injected_high_SNR_list[i] = get_distance_for_dataframe(pGB_injected_high_SNR_list[i])

    found_sources_matched_list[i] = get_galactic_coordinates(found_sources_matched_list[i])
    found_sources_not_matched_list[i] = get_galactic_coordinates(found_sources_not_matched_list[i])
    pGB_injected_matched_list[i] = get_galactic_coordinates(pGB_injected_matched_list[i])
    pGB_injected_high_SNR_list[i] = get_galactic_coordinates(pGB_injected_high_SNR_list[i])

    postitive_fd_mask= found_sources_matched_list[i]['FrequencyDerivative']>0
    found_sources_matched_positive_fd_list.append(found_sources_matched_list[i][postitive_fd_mask])
    found_sources_not_matched_positive_fd_list.append(found_sources_not_matched_list[i][found_sources_not_matched_list[i]['FrequencyDerivative']>0])
    pGB_injected_matched_positive_fd_list.append(pGB_injected_matched_list[i][pGB_injected_matched_list[i]['FrequencyDerivative']>0])
    pGB_injected_positive_fd_high_SNR_list.append(pGB_injected_high_SNR_list[i][pGB_injected_high_SNR_list[i]['FrequencyDerivative']>0])

    found_sources_matched_positive_fd_list[i]['GalactocentricX'], found_sources_matched_positive_fd_list[i]['GalactocentricY'], found_sources_matched_positive_fd_list[i]['GalactocentricZ']  = get_galactocentric_coordinates(found_sources_matched_positive_fd_list[i])
    found_sources_not_matched_positive_fd_list[i]['GalactocentricX'], found_sources_not_matched_positive_fd_list[i]['GalactocentricY'], found_sources_not_matched_positive_fd_list[i]['GalactocentricZ']  = get_galactocentric_coordinates(found_sources_not_matched_positive_fd_list[i])
    pGB_injected_matched_positive_fd_list[i]['GalactocentricX'], pGB_injected_matched_positive_fd_list[i]['GalactocentricY'], pGB_injected_matched_positive_fd_list[i]['GalactocentricZ']  = get_galactocentric_coordinates(pGB_injected_matched_positive_fd_list[i])
    pGB_injected_positive_fd_high_SNR_list[i]['GalactocentricX'], pGB_injected_positive_fd_high_SNR_list[i]['GalactocentricY'], pGB_injected_positive_fd_high_SNR_list[i]['GalactocentricZ']  = get_galactocentric_coordinates(pGB_injected_positive_fd_high_SNR_list[i])

pGB_injected = pGB_injected_list[0]
pGB_injected = get_distance_for_dataframe(pGB_injected)
pGB_injected_positive_fd = pGB_injected[pGB_injected['FrequencyDerivative']>0]
pGB_injected_positive_fd['GalactocentricX'], pGB_injected_positive_fd['GalactocentricY'], pGB_injected_positive_fd['GalactocentricZ']  = get_galactocentric_coordinates(pGB_injected_positive_fd)

# dataframe = pGB_injected_positive_fd_high_SNR_list[0]
# # dataframe = dataframe[dataframe['FrequencyDerivative'] >= 0]
# #### get Galactocentric coordinates with distance
# coordinates_found = coord.SkyCoord(np.array(dataframe['EclipticLongitude'])* u.rad, np.array(dataframe['EclipticLatitude'])* u.rad, np.array(dataframe['Distance'])* u.kpc, frame='barycentricmeanecliptic')
# coordinates_found = coordinates_found.transform_to(coord.Galactocentric) 
# dataframe['GalactocentricX'] = np.array(coordinates_found.galactocentric.x.value)
# dataframe['GalactocentricY'] = coordinates_found.galactocentric.y.value
# dataframe['GalactocentricZ'] = coordinates_found.galactocentric.z.value


sun_c = coord.SkyCoord([1]* u.rad, [1]* u.rad, distance=[0]* u.kpc, 
                   frame='barycentricmeanecliptic')
sun_c = sun_c.transform_to(coord.Galactocentric)
sun = {}
sun['GalactocentricX'] = sun_c.galactocentric.x.value
sun['GalactocentricY'] = sun_c.galactocentric.y.value
sun['GalactocentricZ'] = sun_c.galactocentric.z.value

##### plot distance
markersize = 5
alpha = 0.5
parameter_x = 'GalacticLongitude'
fig = plt.figure()
plt.plot(found_sources_matched_positive_fd_list[data_set][parameter_x],found_sources_matched_positive_fd_list[data_set]['Distance'],'.', label = 'found', markersize=1, alpha=0.5)
# postitive_fd_mask = found_sources_not_matched_positive_fd_list[data_set]['FrequencyDerivative'] >= 0
# plt.plot(found_sources_not_matched_positive_fd_list[data_set][parameter_x],found_sources_not_matched_positive_fd_list[data_set]['Distance'],'r.', label = 'Injected', markersize= 1, zorder=1)
plt.plot(pGB_injected_matched_positive_fd_list[data_set][parameter_x],pGB_injected_matched_positive_fd_list[data_set]['Distance'],'o', color='g', label = 'Injected', markersize= 1,alpha=0.5, zorder=1)
# postitive_fd_mask = pGB_injected_not_matched_positive_fd_list[data_set]['FrequencyDerivative'] >= 0
# plt.plot(pGB_injected_not_matched_positive_fd_list[data_set][parameter_x],pGB_injected_not_matched_positive_fd_list[data_set]['Distance'],'+', label = 'not matched', color = 'r', markersize=2, zorder= 1)
# postitive_fd_mask = pGB_injected_flat_highSNR_df['FrequencyDerivative'] >= 0
# plt.plot(pGB_injected_flat_highSNR_df[parameter_x][postitive_fd_mask],pGB_injected_flat_highSNR_df['Distance'][postitive_fd_mask],'+', label = 'injected SNR>10', color = 'r', markersize=2, zorder= 4)
plt.xlabel(parameter_x)
plt.ylabel('Distance [kpc]')
plt.ylim(0,40)
plt.legend(markerscale=4, loc = 'upper right')
plt.savefig(SAVEPATH+'/galacticLatitudeDistance'+save_names[data_set])
plt.show()

##### plot galactocentric coordinates 2d
markersize = 4
alpha = 0.3
fig = plt.figure()
ax = plt.axes()

# from matplotlib.image import NonUniformImage
H, xedges, yedges = np.histogram2d(pGB_injected_positive_fd['GalactocentricX'],pGB_injected_positive_fd['GalactocentricY'], bins=(100, 100))
# im = NonUniformImage(ax, interpolation='bilinear')
# xcenters = (xedges[:-1] + xedges[1:]) / 2
# ycenters = (yedges[:-1] + yedges[1:]) / 2
# im.set_data(xcenters, ycenters, H)
# ax.images.append(im)
X, Y = np.meshgrid(xedges, yedges)
ax.pcolormesh(X, Y, H)
# postitive_fd_mask = found_sources_matched_positive_fd_list[data_set]['FrequencyDerivative'] >= 0
# ax.plot(found_sources_matched_positive_fd_list[data_set]['GalactocentricX'],found_sources_matched_positive_fd_list[data_set]['GalactocentricY'],'.', markersize=markersize, alpha=alpha)
ax.plot(sun['GalactocentricX'],sun['GalactocentricY'],'.',c='red', markersize=7, alpha=1, zorder = 2)
# postitive_fd_mask = found_sources_not_matched_list[data_set]['FrequencyDerivative'] >= 0
# ax.plot(found_sources_not_matched_list[data_set]['GalacticLatitude'][postitive_fd_mask],found_sources_not_matched_list[data_set]['Distance'][postitive_fd_mask],'r.', label = 'Injected', markersize= 1, zorder=1)
# postitive_fd_mask = pGB_injected_matched_list[data_set]['FrequencyDerivative'] >= 0
ax.plot(pGB_injected_matched_positive_fd_list[data_set]['GalactocentricX'],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricY'],'.', color='g', label = 'Injected', markersize=markersize, alpha=alpha, zorder=1)
# postitive_fd_mask = pGB_injected_not_matched_positive_fd_list[data_set]['FrequencyDerivative'] >= 0
# ax.plot(pGB_injected_positive_fd_high_SNR_list[data_set]['GalactocentricX'],pGB_injected_positive_fd_high_SNR_list[data_set]['GalactocentricY'],'.', label = 'not matched', color = 'grey', markersize=2, zorder= -1, alpha=alpha)
# ax.plot(pGB_injected_positive_fd['GalactocentricX'],pGB_injected_positive_fd['GalactocentricY'],'.', label = 'not matched', color = 'grey', markersize=1, zorder= -1, alpha=0.01)
# postitive_fd_mask = pGB_injected_flat_highSNR_df['FrequencyDerivative'] >= 0
# ax.plot(pGB_injected_flat_highSNR_df['GalacticLatitude'][postitive_fd_mask],pGB_injected_flat_highSNR_df['Distance'][postitive_fd_mask],'+', label = 'injected SNR>10', color = 'r', markersize=2, zorder= 4)
ax.set_xlabel('Galactocentric X [kpc]')
ax.set_ylabel('Galactocentric Y [kpc]')
plt.tight_layout()
plt.xlim(-15,15)
plt.ylim(-10,10)
# plt.legend(markerscale=4, loc = 'upper right')
plt.savefig(SAVEPATH+'/galactocentric'+save_names[data_set])
plt.show()


from matplotlib.image import NonUniformImage
H, xedges, yedges = np.histogram2d(pGB_injected_positive_fd['GalactocentricX'],pGB_injected_positive_fd['GalactocentricY'], bins=(50, 50))
fig = plt.figure()
ax = fig.add_subplot(111, title='NonUniformImage: interpolated', aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
# H[H == 0] = 1
im = NonUniformImage(ax, norm=LogNorm())#, cmap=plt.cm.gray)
xcenters = (xedges[:-1] + xedges[1:]) / 2
ycenters = (yedges[:-1] + yedges[1:]) / 2
im.set_data(xcenters, ycenters, H)
ax.images.append(im)
plt.colorbar(im)
plt.xlim(-25,25)
plt.ylim(-25,25)
plt.show()

fig = plt.figure()
H, xedges, yedges, _ = plt.hist2d(pGB_injected_positive_fd['GalactocentricX'],pGB_injected_positive_fd['GalactocentricY'],bins=(100, 100), density=False, range=[[-20,20],[-20,20]], cmap=plt.cm.gray)
# H[H == 0] = 1
# H = H * (np.abs(xedges[1]-xedges[0])**2)
plt.xlim(-20,20)
plt.ylim(-20,20)
plt.xlabel('Galactocentric X [kpc]')
plt.ylabel('Galactocentric Y [kpc]')
plt.colorbar(label = r'GB density $[1/ \mathrm{kpc}^2]$')
plt.show()

# fig = plt.figure()
# # H[H == 0] = 1
# plt.imshow(H, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], norm=LogNorm(), cmap=plt.cm.gray)
# plt.colorbar()
# plt.show()

fig, (ax1) = plt.subplots(1, 1, sharex=False, constrained_layout=True, figsize=np.array([6,4.5])*1.5)
# H[H == 0] = 1
im = ax1.pcolormesh(xedges, yedges, H.T, norm=LogNorm(), cmap=plt.cm.gray)
ax1.plot(sun['GalactocentricX'],sun['GalactocentricY'],'.',c='red', markersize=4, alpha=1, zorder = 2,  label='Sun')
ax1.set_xlabel('Galactocentric X [kpc]')
ax1.set_ylabel('Galactocentric Y [kpc]')
fig.colorbar(im, ax=ax1, label=r'Injected GB density $[1/ \mathrm{kpc}^2]$')
plt.legend()
plt.savefig(SAVEPATH+'/injected_galactocentric_2D_'+save_names[data_set])
plt.show()


fig = plt.figure()
H, xedges, yedges, _ = plt.hist2d(pGB_injected_matched_positive_fd_list[data_set]['GalactocentricX'],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricY'],bins=(100, 100), density=False, range=[[-20,20],[-20,20]], cmap=plt.cm.gray)
plt.xlim(-20,20)
plt.ylim(-20,20)
plt.xlabel('Galactocentric X [kpc]')
plt.ylabel('Galactocentric Y [kpc]')
plt.colorbar(label = r'GB density $[1/ \mathrm{kpc}^2]$')
plt.legend()
plt.show()

fig, (ax1) = plt.subplots(1, 1, sharex=False, constrained_layout=True, figsize=np.array([6,4.5])*1.5)
# H[H == 0] = 1
im = ax1.pcolormesh(xedges, yedges, H.T, norm=LogNorm(), cmap=plt.cm.gray)
ax1.plot(sun['GalactocentricX'],sun['GalactocentricY'],'.',c='red', markersize=4, alpha=1, zorder = 2,  label='Sun')
ax1.set_xlabel('Galactocentric X [kpc]')
ax1.set_ylabel('Galactocentric Y [kpc]')
fig.colorbar(im, ax=ax1, label=r'Recovered GB density $[1/ \mathrm{kpc}^2]$')
plt.legend()
plt.savefig(SAVEPATH+'/recovered_galactocentric_2D_'+save_names[data_set])
plt.show()

fig = plt.figure()
H, xedges, yedges, _ = plt.hist2d(pGB_injected_matched_positive_fd_list[data_set]['GalactocentricX'],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricY'], bins=(100, 100), range=[[-20,20],[-20,20]], cmap=plt.cm.gray, label='recovered GBs')
H[H == 0] = 1
plt.plot(sun['GalactocentricX'],sun['GalactocentricY'],'.',c='red', markersize=4, alpha=1, zorder = 2,  label='sun')
plt.xlim(-20,20)
plt.ylim(-20,20)
plt.xlabel('Galactocentric X [kpc]')
plt.ylabel('Galactocentric Y [kpc]')
plt.colorbar(label = r'GB density $[1/ \mathrm{kpc}^2]$')
plt.legend()
plt.show()



##### plot galactocentric coordinates 3d
rcParams['axes.labelpad'] = 20
markersize = 5
alpha = 0.5
fig = plt.figure(figsize=[9,9])
ax = plt.axes(projection='3d')
# postitive_fd_mask = found_sources_matched_positive_fd_list[data_set]['FrequencyDerivative'] >= 0
# ax.scatter(found_sources_matched_positive_fd_list[data_set]['GalactocentricX'],found_sources_matched_positive_fd_list[data_set]['GalactocentricY'],found_sources_matched_positive_fd_list[data_set]['GalactocentricZ'],marker='.')
# postitive_fd_mask = found_sources_not_matched_list[data_set]['FrequencyDerivative'] >= 0
# ax.plot(found_sources_not_matched_list[data_set]['GalacticLatitude'][postitive_fd_mask],found_sources_not_matched_list[data_set]['Distance'][postitive_fd_mask],'r.', label = 'Injected', markersize= 1, zorder=1)
# postitive_fd_mask = pGB_injected_matched_list[data_set]['FrequencyDerivative'] >= 0
ax.plot(pGB_injected_matched_positive_fd_list[data_set]['GalactocentricX'],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricY'],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricZ'], '.', markersize=0.5)
# postitive_fd_mask = pGB_injected_not_matched_list[data_set]['FrequencyDerivative'] >= 0
# ax.plot(pGB_injected_not_matched_list[data_set]['GalacticLatitude'][postitive_fd_mask],pGB_injected_not_matched_list[data_set]['Distance'][postitive_fd_mask],'+', label = 'not matched', color = 'r', markersize=2, zorder= 1)
# postitive_fd_mask = pGB_injected_flat_highSNR_df['FrequencyDerivative'] >= 0
# ax.plot(pGB_injected_flat_highSNR_df['GalacticLatitude'][postitive_fd_mask],pGB_injected_flat_highSNR_df['Distance'][postitive_fd_mask],'+', label = 'injected SNR>10', color = 'r', markersize=2, zorder= 4)
plt.plot(sun['GalactocentricX'],sun['GalactocentricY'],'.',c='red', markersize=6, alpha=1, zorder = 2,  label='Sun')
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
# ax.set_zlim(-1,1)
# start, end = ax.get_zlim()
# ax.zaxis.set_ticks(np.arange(start, end+0.5, 0.5))
ax.set_zlim(-15,15)
ax.set_xlim(-15,15)
ax.set_ylim(-15,15)
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.savefig(SAVEPATH+'/galactocentric_3D_'+save_names[data_set])
plt.show()

##### plot galactocentric coordinates 3d
rcParams['axes.labelpad'] = 10
markersize = 5
alpha = 0.5
fig = plt.figure(figsize=[9,9])
ax = plt.axes(projection='3d')
# postitive_fd_mask = found_sources_matched_positive_fd_list[data_set]['FrequencyDerivative'] >= 0
# ax.scatter(found_sources_matched_positive_fd_list[data_set]['GalactocentricX'],found_sources_matched_positive_fd_list[data_set]['GalactocentricY'],found_sources_matched_positive_fd_list[data_set]['GalactocentricZ'],marker='.')
# postitive_fd_mask = found_sources_not_matched_list[data_set]['FrequencyDerivative'] >= 0
# ax.plot(found_sources_not_matched_list[data_set]['GalacticLatitude'][postitive_fd_mask],found_sources_not_matched_list[data_set]['Distance'][postitive_fd_mask],'r.', label = 'Injected', markersize= 1, zorder=1)
# postitive_fd_mask = pGB_injected_matched_list[data_set]['FrequencyDerivative'] >= 0
ax.plot(pGB_injected_matched_positive_fd_list[data_set]['GalactocentricX'],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricY'],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricZ'], '.', markersize=0.5)
# postitive_fd_mask = pGB_injected_not_matched_list[data_set]['FrequencyDerivative'] >= 0
# ax.plot(pGB_injected_not_matched_list[data_set]['GalacticLatitude'][postitive_fd_mask],pGB_injected_not_matched_list[data_set]['Distance'][postitive_fd_mask],'+', label = 'not matched', color = 'r', markersize=2, zorder= 1)
# postitive_fd_mask = pGB_injected_flat_highSNR_df['FrequencyDerivative'] >= 0
# ax.plot(pGB_injected_flat_highSNR_df['GalacticLatitude'][postitive_fd_mask],pGB_injected_flat_highSNR_df['Distance'][postitive_fd_mask],'+', label = 'injected SNR>10', color = 'r', markersize=2, zorder= 4)
plt.plot(sun['GalactocentricX'],sun['GalactocentricY'],'.',c='red', markersize=6, alpha=1, zorder = 2,  label='Sun')
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
# ax.set_zlim(-1,1)
# start, end = ax.get_zlim()
# ax.zaxis.set_ticks(np.arange(start, end+0.5, 0.5))
ax.set_zlim(-15,15)
ax.set_xlim(-15,15)
ax.set_ylim(-15,15)
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.savefig(SAVEPATH+'/galactocentric_3D_'+save_names[data_set])
plt.show()



# generate some points of a 3D Gaussian
# num_signals = 1000
# points = np.array([pGB_injected_matched_positive_fd_list[data_set]['GalactocentricX'][:num_signals],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricY'][:num_signals],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricZ'][:num_signals]])
points = np.array([pGB_injected_matched_positive_fd_list[data_set]['GalactocentricX'],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricY'],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricZ']])

space_limit = 15
# do kernel density estimation to get smooth estimate of distribution
# make grid of points
x, y, z = np.mgrid[-space_limit:space_limit:100j, -space_limit:space_limit:100j, -space_limit:space_limit:100j]
kernel = sp.stats.gaussian_kde(points)
positions = np.vstack((x.ravel(), y.ravel(), z.ravel()))
density = np.reshape(kernel(positions).T, x.shape)

# now density is 100x100x100 ndarray

fig = plt.figure(figsize=[9,9])
# plot points
point_limit = 15
ax = plt.subplot(projection='3d')
# ax.plot(pGB_injected_matched_positive_fd_list[data_set]['GalactocentricX'],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricY'],pGB_injected_matched_positive_fd_list[data_set]['GalactocentricZ'], '.', markersize=0.5)
maskx = points[0,:] > -point_limit
points_reduced_part = points[:,points[0,:] > -point_limit]
points_reduced_part = points_reduced_part[:,points_reduced_part[0,:] < point_limit]
points_reduced_part = points_reduced_part[:,points_reduced_part[1,:] > -point_limit]
points_reduced_part = points_reduced_part[:,points_reduced_part[1,:] < point_limit]
points_reduced_part = points_reduced_part[:,points_reduced_part[2,:] > -point_limit]
points_reduced_part = points_reduced_part[:,points_reduced_part[2,:] < point_limit]

ax.plot(points[0], points[1], points[2], '.', markersize=0.5, zorder=10)
# ax.plot(points_reduced_part[0], points_reduced_part[1], points_reduced_part[2], '.', markersize=0.5, zorder=10)
plt.plot(sun['GalactocentricX'],sun['GalactocentricY'],'.',c='red', markersize=6, alpha=1, zorder =11,  label='Sun')
ax.set_xlabel('Galactocentric X [kpc]')
ax.set_ylabel('Galactocentric Y [kpc]')
ax.set_zlabel('Galactocentric Z [kpc]')

levels = [0.1,0.3,0.5,0.7,0.9,1]
# plot projection of density onto x-axis
plotdat = np.sum(density, axis=0)
plotdat = plotdat / np.max(plotdat)
ploty, plotz = np.mgrid[-space_limit:space_limit:100j, -space_limit:space_limit:100j]
ax.contour(plotdat, ploty, plotz, levels, offset=space_limit, zdir='x')
# plot projection of density onto y-axis
plotdat = np.sum(density, axis=1)
plotdat = plotdat / np.max(plotdat)
plotx, plotz = np.mgrid[-space_limit:space_limit:100j, -space_limit:space_limit:100j]
ax.contour(plotx, plotdat, plotz, levels, offset=space_limit, zdir='y')
# plot projection of density onto z-axis
plotdat = np.sum(density, axis=2)
plotdat = plotdat / np.max(plotdat)
plotx, ploty = np.mgrid[-space_limit:space_limit:100j, -space_limit:space_limit:100j]
ax.contour(plotx, ploty, plotdat, levels, offset=-space_limit, zdir='z')

ax.set_xlim((-space_limit, space_limit))
ax.set_ylim((-space_limit, space_limit))
ax.set_zlim((-space_limit, space_limit))

start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end+5, 5))
start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(start, end+5, 5))
start, end = ax.get_zlim()
ax.zaxis.set_ticks(np.arange(start, end+5, 5))

plt.tight_layout()
plt.savefig(SAVEPATH+'/galactocentric_3D_contour_'+save_names[data_set])
plt.show()