from re import A
# from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
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
import os
import h5py
import sys
sys.path.append('/cluster/home/sstrub/Repositories/LDC/lib/lib64/python3.8/site-packages/ldc-0.1-py3.8-linux-x86_64.egg')

from astropy import units as u
import astropy.coordinates as coord


from ldc.lisa.noise import get_noise_model
from ldc.lisa.noise import AnalyticNoise

from Search import Search
from sources import *


# customized settings
plot_parameter = {  # 'backend': 'ps',
    "font.family" :'DeJavu Serif',
    "font.serif" : ["Computer Modern Serif"],
    "mathtext.fontset": "cm",
}


# customized settings
plot_parameter_big = {  # 'backend': 'ps',
    "font.family" :'DeJavu Serif',
    "font.serif": "Times",
    # "font.serif" : ["Computer Modern Serif"],
    "font.size": 20,
    "mathtext.fontset": "cm",
    "axes.labelsize": "medium",
    "axes.titlesize": "medium",
    "legend.fontsize": "medium",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "grid.color": "k",
    "grid.linestyle": ":",
    "grid.linewidth": 1,
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

# get current directory
path = os.getcwd()
 
# parent directory
parent = os.path.dirname(path)
# grandparent directory
grandparent = os.path.dirname(parent)
Radler = True
if Radler:
    DATAPATH = grandparent+"/LDC/Radler/data/"
    SAVEPATH = grandparent+"/LDC/pictures/LDC1-4/"
    SAVEPATH = grandparent+"/LDC/Radler/LDC1-4_evaluation/"
else:
    DATAPATH = grandparent+"/LDC/Sangria/data/"
    SAVEPATH = grandparent+"/LDC/pictures/Sangria/"

SAVEPATH_sangria = grandparent+"/LDC/pictures/Sangria/"
# duration = '7864320'
duration = '15728640'


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

labels = {'EclipticLongitude': r'$\lambda$'+' (rad)', 'EclipticLatitude': r'$\beta$'+' (rad)','Frequency': r'$f / f_\mathrm{true}$','FrequencyDerivative': r'$\dot{f}$ $ ($Hz/s$)$','Inclination': r'$\iota$'+' (rad)','Amplitude': r'$ \mathcal{A}$', 'Polarization': r'$\psi$'+' (rad)', 'InitialPhase': r'$\phi_0$'+' (rad)'}

# end_string = '_SNR_scaled_03_injected_snr5'
end_string = '_SNR_scaled_03_injected_snr5_comparison_'
# end_string = '_correlation_09_injected_snr5'
# end_string = 'correlation'
def load_files(save_path, save_name):
    found_sources_flat_df = pd.read_pickle(save_path+'found_sources_' +save_name+'_df')
    for i in range(len(found_sources_flat_df)):
        if found_sources_flat_df['EclipticLongitude'][i] < 0:
            found_sources_flat_df['EclipticLongitude'][i] += 2*np.pi
    found_sources_matched_flat_df = pd.read_pickle(save_path+'/found_sources_matched_' +save_name+'_df')
    for i in range(len(found_sources_matched_flat_df)):
        if found_sources_matched_flat_df['EclipticLongitude'][i] < 0:
            found_sources_matched_flat_df['EclipticLongitude'][i] += 2*np.pi
    found_sources_not_matched_flat_df = pd.read_pickle(save_path+'/found_sources_not_matched_' +save_name+'_df')
    for i in range(len(found_sources_not_matched_flat_df)):
        if found_sources_not_matched_flat_df['EclipticLongitude'][i] < 0:
            found_sources_not_matched_flat_df['EclipticLongitude'][i] += 2*np.pi
    other_seed_matched_flat_df = pd.read_pickle(save_path+'/injected_matched_windows_' +save_name+'_df')
    other_seed_not_matched_flat_df = pd.read_pickle(save_path+'/injected_not_matched_windows_' +save_name+'_df')
    match_list = np.load(save_path+'match_list_' +save_name+'.npy', allow_pickle=True)
    # correlation_list = np.load(save_path+'match_list_' +save_name+'_correlation_09_injected_snr5.npy', allow_pickle=True)
    # correlation_list = np.load(save_path+'match_list_' +save_name+'_SNR_scaled_03_injected_snr5.npy', allow_pickle=True)
    pGB_best_list = np.load(save_path+'pGB_best_list_' +save_name+'.npy', allow_pickle=True)
    match_best_list = np.load(save_path+'match_best_list_' +save_name+'.npy', allow_pickle=True)

    pGB_best_list_flat = []
    for i in range(len(pGB_best_list)):
        for j in range(len(pGB_best_list[i])):
            pGB_best_list_flat.append(pGB_best_list[i][j])
    pGB_best_list_df = pd.DataFrame(np.asarray(pGB_best_list_flat))

    match_best_list_flat = np.concatenate(match_best_list)
    match_best_list_flat_array = {attribute: np.asarray([x[attribute] for x in match_best_list_flat]) for attribute in match_best_list_flat[0].keys()}
    match_best_list_flat_df = pd.DataFrame(match_best_list_flat_array)

    number_of_found_signals_not_matched = len(found_sources_not_matched_flat_df)
    number_of_matched_signals = len(found_sources_matched_flat_df)
    number_of_found_signals = number_of_matched_signals + number_of_found_signals_not_matched
    number_of_seed_matched_signals = len(other_seed_matched_flat_df) + len(other_seed_not_matched_flat_df)

    print(number_of_matched_signals ,'matched signals out of', number_of_found_signals, 'found signals '+save_name)
    print('matched signals/found signals:', np.round(number_of_matched_signals/number_of_found_signals,2))
    print('number of seed_matched signals:', np.round(number_of_seed_matched_signals))
    return found_sources_flat_df, found_sources_matched_flat_df, found_sources_not_matched_flat_df, other_seed_matched_flat_df, other_seed_not_matched_flat_df, match_list, pGB_best_list_df, match_best_list_flat_df

found_sources_matched_df_list = []
found_sources_not_matched_df_list = []
other_seed_matched_df_list = []
other_seed_not_matched_df_list = []

def create_df_per_seed(seed):
    save_names  = []
    SAVEPATHS = []
    for i in range(1,11):
        if i == seed:
            continue
        save_names.append('original_Sangria_6m_mbhb_SNR9_seed'+str(seed)+'_SNR_scaled_03_injected_snr5_comparison_'+str(i))
        SAVEPATHS.append(SAVEPATH_sangria)

    found_sources_list = []
    found_sources_matched_list = []
    found_sources_not_matched_list = []
    other_seed_matched_list = []
    other_seed_not_matched_list = []
    match_list = []
    pGB_best_list = []
    match_best_list = []
    found_sources_df = pd.DataFrame()
    found_sources_matched_df = pd.DataFrame()
    found_sources_not_matched_df = pd.DataFrame()
    other_seed_matched_df = pd.DataFrame()
    other_seed_not_matched_df = pd.DataFrame()
    match_df = pd.DataFrame()
    correlation_df = pd.DataFrame()
    pGB_best_df = pd.DataFrame()
    match_best_df = pd.DataFrame()
    for i, save_name in enumerate(save_names):
        found_sources_flat_df, found_sources_matched_flat_df, found_sources_not_matched_flat_df, other_seed_matched_flat_df, other_seed_not_matched_flat_df, match, pGB_best, match_best = load_files(SAVEPATHS[i], save_name)
        found_sources_list.append(found_sources_flat_df)
        seed_number = int(save_name[-1])
        if seed_number == 0:
            seed_number = 10
        found_sources_matched_flat_df['Seed'] = seed_number
        found_sources_not_matched_flat_df['Seed'] = seed_number
        other_seed_matched_flat_df['Seed'] = seed_number
        other_seed_not_matched_flat_df['Seed'] = seed_number
        found_sources_matched_list.append(found_sources_matched_flat_df)
        found_sources_not_matched_list.append(found_sources_not_matched_flat_df)
        other_seed_matched_list.append(other_seed_matched_flat_df)
        other_seed_not_matched_list.append(other_seed_not_matched_flat_df)
        match_list.append(match)
        pGB_best_list.append(pGB_best)
        match_best_list.append(match_best)

    found_sources_df = pd.concat(found_sources_list).reset_index()
    found_sources_matched_df = pd.concat(found_sources_matched_list).reset_index()
    found_sources_not_matched_df = pd.concat(found_sources_not_matched_list).reset_index()
    other_seed_matched_df = pd.concat(other_seed_matched_list).reset_index()
    other_seed_not_matched_df = pd.concat(other_seed_not_matched_list).reset_index()
    pGB_best_df = pd.concat(pGB_best_list).reset_index()
    match_best_df = pd.concat(match_best_list).reset_index()

    found_sources_matched_df_list.append(found_sources_matched_df)
    found_sources_not_matched_df_list.append(found_sources_not_matched_df)
    other_seed_matched_df_list.append(other_seed_matched_df)
    other_seed_not_matched_df_list.append(other_seed_not_matched_df)

# create_df_per_seed(1)
# create_df_per_seed(2)
# create_df_per_seed(3)
# create_df_per_seed(4)
# create_df_per_seed(5)
# create_df_per_seed(6)
# create_df_per_seed(7)
# create_df_per_seed(8)
# create_df_per_seed(9)
# create_df_per_seed(10)

# pickle.dump(found_sources_matched_df_list, open(SAVEPATH_sangria+'found_sources_matched_df_list_index', 'wb'))
# pickle.dump(found_sources_not_matched_df_list, open(SAVEPATH_sangria+'found_sources_not_matched_df_list_index', 'wb'))
# pickle.dump(other_seed_matched_df_list, open(SAVEPATH_sangria+'other_seed_matched_df_list_index', 'wb'))
# pickle.dump(other_seed_not_matched_df_list, open(SAVEPATH_sangria+'other_seed_not_matched_df_list_index', 'wb'))

# for i in range(len(found_sources_matched_df_list)):
#     found_sources_matched_df_list[i] = found_sources_matched_df_list[i].drop(['index'], axis=1)
#     found_sources_not_matched_df_list[i] = found_sources_not_matched_df_list[i].drop(['index'], axis=1)
#     other_seed_matched_df_list[i] = other_seed_matched_df_list[i].drop(['index'], axis=1)
#     other_seed_not_matched_df_list[i] = other_seed_not_matched_df_list[i].drop(['index'], axis=1)

# pickle.dump(found_sources_matched_df_list, open(SAVEPATH_sangria+'found_sources_matched_df_list', 'wb'))
# pickle.dump(found_sources_not_matched_df_list, open(SAVEPATH_sangria+'found_sources_not_matched_df_list', 'wb'))
# pickle.dump(other_seed_matched_df_list, open(SAVEPATH_sangria+'other_seed_matched_df_list', 'wb'))
# pickle.dump(other_seed_not_matched_df_list, open(SAVEPATH_sangria+'other_seed_not_matched_df_list', 'wb'))

found_sources_matched_df_list = pickle.load(open(SAVEPATH_sangria+'found_sources_matched_df_list_index', 'rb'))
found_sources_not_matched_df_list = pickle.load(open(SAVEPATH_sangria+'found_sources_not_matched_df_list_index', 'rb'))
other_seed_matched_df_list = pickle.load(open(SAVEPATH_sangria+'other_seed_matched_df_list_index', 'rb'))
other_seed_not_matched_df_list = pickle.load(open(SAVEPATH_sangria+'other_seed_not_matched_df_list_index', 'rb'))


seed = 3
counts = found_sources_matched_df_list[seed-1]['Frequency'].value_counts().reset_index(name='counts')
counts_counts = counts['counts'].value_counts().reset_index(name='counts_counts')
print(counts_counts)
number_of_sets = counts_counts['counts_counts']
print(counts_counts['counts_counts'].sum())
print(np.array(number_of_sets))

found_sources_matched_df_no_seed = found_sources_matched_df_list[seed-1].drop(['Seed'], axis=1)
counts_full = found_sources_matched_df_no_seed.value_counts().reset_index(name='counts')

found_sources_not_matched_df_no_seed = found_sources_not_matched_df_list[seed-1].drop(['Seed'], axis=1)
found_sources_not_matched_df_no_seed.drop_duplicates(keep=False,inplace=True)
counts_full_zero = found_sources_not_matched_df_no_seed.value_counts().reset_index(name='counts')

# plt.figure()
# plt.hist(counts_full_zero['counts'], bins=100)
# plt.xlabel('Number of matches')
# plt.ylabel('Number of signals')
# plt.yscale('log')
# # plt.savefig(SAVEPATH_sangria+'number_of_matches_per_signal_seed'+str(seed)+'.pdf', bbox_inches='tight')
# plt.show()


# cmap = plt.get_cmap("plasma")
# cmapr = plt.get_cmap("plasma_r")
# ticks = np.arange(1,10,1)
# plt.figure()
# for i in ticks:
#     if i == 10:
#         pass
#     j = 10-i
#     index = counts_full['counts'] == j
#     plt.plot(counts_full[index]['Frequency'], counts_full[index]['Amplitude'], '.', label=str(j), color= cmapr((j-1)/9))   
# # plt.plot(found_sources_not_matched_df_no_seed['Frequency'], found_sources_not_matched_df_no_seed['Amplitude'], '.', label=str(j), color= cmapr(0))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$f$ $ ($Hz$)$')
# plt.ylabel(r'$\mathcal{A}$')
# norm = mpl.colors.Normalize(vmin=1,vmax=10)
# sm = plt.cm.ScalarMappable(cmap=cmapr, norm=norm)
# sm.set_array([])
# cbar= plt.colorbar(sm, ticks=ticks, label=r'$N$', boundaries=np.arange(np.min(ticks)-1/2,np.max(ticks+1)+1/2,1))
# # cbar.ax.invert_yaxis()
# # plt.legend()
# plt.grid()
# plt.savefig(SAVEPATH_sangria+'number_of_matches_per_signal_seed'+str(seed)+'_frequency_amplitude.png', bbox_inches='tight')
# plt.show()

# 




found_sources_matched_df_extended_list = deepcopy(found_sources_matched_df_list)
groups = deepcopy(found_sources_matched_df_list[0])
for frequency_of_signal in groups['Frequency'].unique():
    groups = groups.drop(groups[groups['Frequency'] == frequency_of_signal].index[1:]) 
groups = groups.drop(['Seed'], axis=1)
groups['Seed'] = [int(1) for _ in range(len(groups))] 
groups['Group'] = [j for j in range(len(groups))]  # add seed column with value 1

# unify all dataframes
for seed in range(2, len(found_sources_matched_df_list)):
    print('seed', seed)
    for i in range(len(groups)):
        same_frequency = other_seed_matched_df_list[seed-1][other_seed_matched_df_list[seed-1]['Frequency'] == groups['Frequency'].iloc[i]]
        same_frequency_and_seed = same_frequency[same_frequency['Seed'] == groups['Seed'].iloc[i]]
        new_entry = found_sources_matched_df_list[seed-1].iloc[list(same_frequency_and_seed.index)]
        new_entry = new_entry.drop(['Seed'], axis=1)
        new_entry['Seed'] = [int(seed) for _ in range(len(new_entry))]  # add seed column with value seed
        new_entry['Group'] = groups['Group'].iloc[i]  # add group column
        # check if the entry is already in the groups dataframe
        # if len(new_entry) > 0
        # if len(groups[groups['Frequency'] == new_entry['Frequency'].iloc[0]][groups['Seed'] == new_entry['Seed'].iloc[0]]) > 0:
        #     # print('entry already in groups')
        #     continue
        # else:
        groups = pd.concat([groups, new_entry], ignore_index=True)
        # remove douplicates
    print('douplicates', groups[groups.duplicated(subset=['Frequency', 'Seed'],keep=False)])
    groups = groups.drop_duplicates(subset=['Frequency', 'Seed', 'Group'], keep='first')
    print('douplicates', groups[groups.duplicated(subset=['Frequency', 'Seed'])])
print(groups)
groups2 = groups.copy()

print('douplicates', len(groups[groups.duplicated(subset=['Frequency', 'Seed'],keep=False)]))

groups_no_doup = groups.drop_duplicates(subset=['Frequency', 'Seed'], keep='first')
counts = groups_no_doup['Group'].value_counts().reset_index(name='counts')


counts_counts = counts['counts'].value_counts().reset_index(name='counts_counts')

# TODO: check for the other seeds. 

# # give the entries with the same inex and different frequency
# for i in range(len(found_sources_matched_df_list[seed-1])):
#     frequencies_len = len(found_sources_matched_df_list[seed-1][found_sources_matched_df_list[seed-1]['Frequency'] == found_sources_matched_df_list[seed-1]['Frequency'].iloc[i]])
#     indexes_len = len(found_sources_matched_df_list[seed-1][found_sources_matched_df_list[seed-1]['index'] == found_sources_matched_df_list[seed-1]['index'].iloc[i]])
#     if frequencies_len != indexes_len:
#         print('not equal frequencies and indexes', frequencies_len, indexes_len)

# for seed in range(1,11):
#     counts = found_sources_matched_df_list[seed-1]['Frequency'].value_counts().reset_index(name='counts')
#     indexes = found_sources_matched_df_list[seed-1]['index'].value_counts().reset_index(name='counts')
#     counts_counts = counts['counts'].value_counts().reset_index(name='counts_counts')
#     print(counts_counts)
#     number_of_sets = counts_counts['counts_counts']
#     number_of_extended_sets = 0
#     i = number_of_sets.iloc[0]-100
#     while i in range(number_of_sets.iloc[0]-100, len(counts)):
#         # counts = found_sources_matched_df_list[seed-1]['Frequency'].value_counts().reset_index(name='counts')
#         # print(counts.iloc[i])
#         found_sources_matched_df_extended_list[seed-1] = found_sources_matched_df_extended_list[seed-1].reset_index(drop=True)
#         other_seed_matched_df_list[seed-1] = other_seed_matched_df_list[seed-1].reset_index(drop=True)
#         print(i)
#         frequency_of_signal = counts['Frequency'].iloc[i]
#         # print(found_sources_matched_df_extended_list[seed-1][found_sources_matched_df_extended_list[seed-1]['Frequency'] == frequency_of_signal])
#         seed_pairs = other_seed_matched_df_list[seed-1][found_sources_matched_df_extended_list[seed-1]['Frequency'] == frequency_of_signal]['Seed'].to_list()
#         frequency_pair = other_seed_matched_df_list[seed-1][found_sources_matched_df_extended_list[seed-1]['Frequency'] == frequency_of_signal]['Frequency'].to_list()
#         for j in range(len(seed_pairs)):
#             breaker = False
#             # print('seed', seed_pairs[j])
#             matching_seeds = found_sources_matched_df_extended_list[seed_pairs[j]-1][found_sources_matched_df_extended_list[seed_pairs[j]-1]['Frequency'] == frequency_pair[j]]['Seed']
#             # if len(matching_seeds.to_list()) > len(seed_pairs): # extend the group
#                 # print('extend the group')
#             matches_list = found_sources_matched_df_extended_list[seed_pairs[j]-1][found_sources_matched_df_extended_list[seed_pairs[j]-1]['Frequency'] == frequency_pair[j]]
#             injected_list = other_seed_matched_df_list[seed_pairs[j]-1][found_sources_matched_df_extended_list[seed_pairs[j]-1]['Frequency'] == frequency_pair[j]]
#             for seed_extend in matching_seeds.to_list():
#                 if seed_extend not in seed_pairs + [seed]:
#                     number_of_extended_sets += 1
#                     print('seed extend', seed_extend, 'of seed', seed, 'due to seed', seed_pairs[j])
#                     # print(injected_list)
#                     frequency_injected = float(injected_list[matches_list['Seed'] == seed_extend]['Frequency'])
#                     signal_to_extend = found_sources_not_matched_df_list[seed-1][found_sources_not_matched_df_list[seed-1]['Frequency'] == frequency_of_signal]
#                     # print(signal_to_extend)
#                     # print(found_sources_matched_df_extended_list[seed-1][found_sources_matched_df_extended_list[seed-1]['Frequency'] == frequency_of_signal])
#                     found_sources_matched_df_extended_list[seed-1] = pd.concat([found_sources_matched_df_extended_list[seed-1], signal_to_extend])
#                     # found_sources_matched_df_extended_list[seed-1] = pd.concat([found_sources_matched_df_extended_list[seed-1], signal_to_extend])
#                     # print(found_sources_matched_df_extended_list[seed-1][found_sources_matched_df_extended_list[seed-1]['Frequency'] == frequency_of_signal])
#                     # remove from not matched
#                     found_sources_not_matched_df_list[seed-1] = found_sources_not_matched_df_list[seed-1].drop(found_sources_not_matched_df_list[seed-1][found_sources_not_matched_df_list[seed-1]['Frequency'] == frequency_of_signal].index)
#                     # print('injected')
#                     signal_to_extend = injected_list[injected_list['Seed'] == seed_extend]
#                     # signal_to_extend = other_seed_not_matched_df_list[seed-1][other_seed_not_matched_df_list[seed-1]['Frequency'] == frequency_injected]
#                     # print('before',other_seed_matched_df_list[seed-1][other_seed_matched_df_list[seed-1]['Frequency'] == frequency_injected])
#                     other_seed_matched_df_list[seed-1] = pd.concat([other_seed_matched_df_list[seed-1], signal_to_extend])
#                     # print(other_seed_matched_df_list[seed-1][other_seed_matched_df_list[seed-1]['Frequency'] == frequency_injected])
#                     # remove from not matched
#                     other_seed_not_matched_df_list[seed-1] = other_seed_not_matched_df_list[seed-1].drop(found_sources_not_matched_df_list[seed-1][found_sources_not_matched_df_list[seed-1]['Frequency'] == frequency_of_signal].index)


#                     breaker = True
#                     break
#                     # print(found_sources_not_matched_df_list[seed-1][found_sources_not_matched_df_list[seed-1]['Frequency'] == frequency_pair[j]])
#                 # print(found_sources_matched_df_extended_list[seed_pairs[j]-1][found_sources_matched_df_extended_list[seed_pairs[j]-1]['Frequency'] == frequency_pair[j]])
#             if breaker:
#                 i -= 1
#                 break
#         i += 1
#             # print(found_sources_not_matched_df_list[seed_pairs[j]-1][found_sources_not_matched_df_list[seed_pairs[j]-1]['Frequency'] == frequency_pair[j]])

# counts = found_sources_matched_df_list[seed-1]['Frequency'].value_counts().reset_index(name='counts')
# counts_counts = counts['counts'].value_counts().reset_index(name='counts_counts')
# print(counts_counts)
# number_of_sets = counts_counts['counts_counts']
# print(counts_counts['counts_counts'].sum())

# counts = found_sources_matched_df_extended_list[seed-1]['Frequency'].value_counts().reset_index(name='counts')
# counts_counts = counts['counts'].value_counts().reset_index(name='counts_counts')
# print(counts_counts)
# number_of_sets = counts_counts['counts_counts']
# print(counts_counts['counts_counts'].sum())


found_sources_matched_df_list[0][found_sources_matched_df_list[0]['Frequency'] == found_sources_matched_df_list[0]['Frequency'].iloc[0]]


print('end')