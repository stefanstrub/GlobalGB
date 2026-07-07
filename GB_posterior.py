import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import h5py
import jax
import jax.numpy as jnp
import lisaorbits
import numpy as np
import polars as pl

import corner
import matplotlib.pyplot as plt
from jaxgb.jaxgb import JaxGB
from jaxgb.params import GBObject

from globalGB.search_utils_GB import GB_pe, PARAM_NAMES, GBConfig, GB_Searcher
from globalGB.config import load_config
from NoiseEstimate.noise_estimate import *
from DataLoader.data_loader import LISADataLoader
from globalGB.GB_runner import GBSearchRunner

def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Search for Galactic binaries in Mojito data.")
    parser.add_argument("which_run", type=str, choices=["even1st", "even", "odd"], help="Window set to analyze.")
    parser.add_argument("batch_index", type=str, help="Batch index of frequency windows to process.")
    return parser.parse_args(argv)

# group found_sources by overlapping gw signals
def get_significant_frequency_range(source, fgb, threshold=0.1):
    """Compute the frequency range where signal amplitude > threshold * max."""
    signal = fgb.get_tdi(jnp.array(source))
    kmin = fgb.get_kmin(source[0])
    freqs = fgb.get_frequency_grid(jnp.array([kmin])).squeeze()
    max_amp = np.max(np.abs(signal[0]))
    significant_indices = np.where(np.abs(signal[0]) > max_amp * threshold)[0]
    if len(significant_indices) == 0:
        return (freqs[0], freqs[-1])
    return (freqs[significant_indices[0]], freqs[significant_indices[-1]])

from globalGB.grouping import merge_ranges, ranges_overlap
def main(argv=None):
    args = parse_args(argv)
    batch_index = int(args.batch_index)
    config = GBConfig(load_config())

    runner = GBSearchRunner(
        batch_index=0,
        which_run=args.which_run,
        config=config,
    )

    runner.load_data()
    runner.prepare_frequency_windows()
    savepath = runner.savepath
    with h5py.File(runner.savepath+f'/found_signals_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5', 'r') as f:
        found_sources = f['recovered_sources'][:]
    found_sources_df = pl.DataFrame(found_sources, schema=PARAM_NAMES)
    found_sources_df = found_sources_df.sort('Frequency')

    fgb = JaxGB(
    orbits=runner.waveform_args["orbits"],
    t_obs=runner.waveform_args["Tobs"],
    t0=runner.waveform_args["t0"],
    n=2**10,
    )

    try:
        with h5py.File(savepath + f'/grouped_found_sources_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5', 'r') as f:
            grouped_found_sources = [{'frequency_range': f[f'group_{i}']['frequency_range'][:], 'sources': f[f'group_{i}']['sources'][:]} for i in range(f.attrs['n_groups'])]
        print(f"Loaded {len(grouped_found_sources)} groups from {savepath + f'/grouped_found_sources_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5'}")
    except:
        print(f"No grouped found sources found at {savepath + f'/grouped_found_sources_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5'}")


        # Compute significant frequency range for each source
        ungrouped_found_sources = found_sources_df.to_numpy()
        source_ranges = []
        for i, source in enumerate(ungrouped_found_sources):
            freq_range = get_significant_frequency_range(source, fgb)
            source_ranges.append(freq_range)
            print(f"Source {i}/{len(ungrouped_found_sources)} f0={source[0]*1e3:.4f} mHz, significant range: [{freq_range[0]*1e3:.4f}, {freq_range[1]*1e3:.4f}] mHz")

        # Group sources with overlapping significant frequency ranges
        groups = []  # List of (group_freq_range, [sources])
        
        for i, source in enumerate(ungrouped_found_sources):
            print(f"Processing source {i}/{len(ungrouped_found_sources)}")
            source_range = source_ranges[i]
            merged = False
            
            for group in groups[-10:]:
                group_range, group_sources = group
                if ranges_overlap(group_range, source_range):
                    # Merge into this group
                    group[0] = merge_ranges(group_range, source_range)
                    group_sources.append(source)
                    merged = True
                    break
            
            if not merged:
                # Start a new group
                groups.append([source_range, [source]])

        # Convert to final format: list of numpy arrays
        grouped_found_sources_list = []
        for group_range, group_sources in groups:
            grouped_found_sources_list.append({
                'frequency_range': group_range,
                'sources': np.array(group_sources)
            })
            print(f"Group: {len(group_sources)} sources in range [{group_range[0]*1e3:.4f}, {group_range[1]*1e3:.4f}] mHz")

        print(f"\nTotal groups: {len(grouped_found_sources_list)}")

        # Save grouped found sources to h5 file (each group as separate dataset)
        groups_fn = savepath + f'/grouped_found_sources_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5'
        with h5py.File(groups_fn, 'w') as f:
            f.attrs['n_groups'] = len(grouped_found_sources_list)
            for i, group in enumerate(grouped_found_sources_list):
                grp = f.create_group(f'group_{i}')
                grp.create_dataset('frequency_range', data=group['frequency_range'])
                grp.create_dataset('sources', data=group['sources'])
        print(f"Saved {len(grouped_found_sources_list)} groups to {groups_fn}")

        with h5py.File(savepath + f'/grouped_found_sources_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5', 'r') as f:
            grouped_found_sources = [{'frequency_range': f[f'group_{i}']['frequency_range'][:], 'sources': f[f'group_{i}']['sources'][:]} for i in range(f.attrs['n_groups'])]
        print(f"Loaded {len(grouped_found_sources)} groups from {savepath + f'/grouped_found_sources_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5'}")

    # create dictionary with length of groups as keys and number of groups with same length as values
    group_lengths = {i: len(group['sources']) for i, group in enumerate(grouped_found_sources)}
    group_lengths_values = list(group_lengths.values())
    group_lengths_values_unique = np.unique(group_lengths_values)
    group_lengths_values_unique_counts = {int(i): int(group_lengths_values.count(i)) for i in group_lengths_values_unique}
    print(f"Group lengths: {group_lengths_values_unique_counts}")

    # index_with_most_sources = np.argmax(group_lengths_values)
    # group_with_most_sources = grouped_found_sources[index_with_most_sources]
    # print(f"Group with most sources: {index_with_most_sources}")
    # print(f"Frequency range: [{group_with_most_sources['frequency_range'][0]*1e3:.5f}, {group_with_most_sources['frequency_range'][1]*1e3:.5f}] mHz")
    # print(f"Number of sources: {len(group_with_most_sources['sources'])}")
    # print(f"{'='*60}")

    # plt.figure()
    # # plt.plot(runner.freq, np.abs(runner.tdi_fs['A']), label='A data')
    # for source in grouped_found_sources[index_with_most_sources]['sources']:
    #     As, Es, Ts = fgb.get_tdi(jnp.array(source))
    #     freq = fgb.get_frequency_grid(jnp.array([fgb.get_kmin(source[0])])).squeeze()
    #     plt.semilogy(freq, np.abs(As), label=f'Source {source[0]*1e3:.5f} mHz')
    # for source in grouped_found_sources[index_with_most_sources+1]['sources']:
    #     As, Es, Ts = fgb.get_tdi(jnp.array(source))
    #     freq = fgb.get_frequency_grid(jnp.array([fgb.get_kmin(source[0])])).squeeze()
    #     plt.semilogy(freq, np.abs(As), '--', label=f'Source {source[0]*1e3:.5f} mHz')
    # for source in grouped_found_sources[index_with_most_sources-1]['sources']:
    #     As, Es, Ts = fgb.get_tdi(jnp.array(source))
    #     freq = fgb.get_frequency_grid(jnp.array([fgb.get_kmin(source[0])])).squeeze()
    #     plt.semilogy(freq, np.abs(As), '--', label=f'Source {source[0]*1e3:.5f} mHz')
    # plt.xlabel('Frequency (mHz)')
    # plt.ylabel('|TDI A|')
    # plt.legend()
    # plt.title(f'Group {index_with_most_sources} with most sources')
    # plt.show(block=True)

    start_index = batch_index*config.batch_size
    groups_to_process = grouped_found_sources[start_index:start_index+config.batch_size]

    frequency_range = groups_to_process[1]['frequency_range']
    loader = LISADataLoader(config=config)
    loader._load_mojito_wdwd_catalog()
    cat_df = pl.DataFrame(loader.catalog_wdwd, schema=PARAM_NAMES)
    cat_df = cat_df.sort('Frequency')
    injected_sources = np.array(cat_df[-2])
    # transform injected sources to the correct t0
    t_init = 97729089.327664 
    gbo = GBObject.from_jaxgb_params(jnp.array(injected_sources), t_init=t_init)
    injected_sources_t0 = np.array(gbo.to_jaxgb_array(t0=runner.t0))
    # get snr of injected sources
    search = GB_Searcher(
        runner.tdi_fs, runner.Tobs,
        frequency_range[0], frequency_range[1],
        runner.waveform_args, dt=config.dt,
        channel_combination=config.channel_combination
    )
    # search.plot(injected_sources_t0)
    # snr = []
    # loglikelihood = []
    # for source in injected_sources_t0:
    #     snr.append(search.SNR(source))
    #     loglikelihood.append(search.loglikelihood(source))
    # snr = np.array(snr)
    # loglikelihood = np.array(loglikelihood)
    # print(f"Injected sources SNR: {snr}")
    # print(f"Injected sources loglikelihood: {loglikelihood}")
    # print(f"{'='*60}")

    # Run MCMC for each group of overlapping sources
    for i, group in enumerate(groups_to_process):
        frequency_range = group['frequency_range']
        initial_parameters = group['sources']
        # if len(initial_parameters) == 1:
        #     window_size = frequency_range[1] - frequency_range[0]
        #     frequency_range_extended = [frequency_range[0]-(window_size*2), frequency_range[1]+(window_size*2)]
        print(f"Processing group {i}/{len(groups_to_process)}")
        print(f"Frequency range: [{frequency_range[0]*1e3:.4f}, {frequency_range[1]*1e3:.4f}] mHz")
        print(f"Number of sources: {len(initial_parameters)}")
        print(f"Found sources SNR: {search.SNR(initial_parameters)}")
        print(f"Found sources loglikelihood: {search.loglikelihood(initial_parameters)}")
        print(f"{'='*60}")
        
        gb_pe = GB_pe(
            runner.tdi_fs, 
            initial_parameters, 
            runner.Tobs, 
            frequency_range[0], 
            frequency_range[1], 
            runner.waveform_args, 
            config.dt, 
            channel_combination=config.channel_combination
        )
        start_time = time.time()
        chains, ensemble = gb_pe.eryn_mcmc_GB(nsteps=3000, burn=0, ntemps=4, nwalkers=4, nleaves_max=len(initial_parameters)+1)
        # chains, acceptance_fraction = gb_pe.MH_mcmc_GB(nsteps=20000, burn=1000, ntemps=1, nwalkers=5)
        # chains, n_signals_chain, acceptance_fraction = gb_pe.RJMCMC_GB(nsteps=10000, burn=1000, birth_weight=0, death_weight=0, ntemps=1, nwalkers=1, n_max=len(initial_parameters))
        # chains = np.concatenate(chains, axis=0)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        chains_nan_rows_removed = []
        for leaf in range(chains.shape[1]):
            chains_nan_rows_removed.append(chains[~np.isnan(chains[:,leaf,:]).all(axis=1)][:,leaf,:])

        # transform found sources to the correct t0
        t_init = 97729089.327664
        chains_t_init = []
        # shift found sources to the same initial time
        for leaf in range(len(chains_nan_rows_removed)):
            chains_t_init_leaf = GBObject.from_jaxgb_params(jnp.array(chains_nan_rows_removed[leaf]), t_init=runner.t0).to_jaxgb_array(t0=t_init)
            chains_t_init.append(np.array(chains_t_init_leaf))

        initial_parameters_t_init = GBObject.from_jaxgb_params(jnp.array(initial_parameters), t_init=runner.t0).to_jaxgb_array(t0=t_init)
        
        time_stamp = time.strftime('%Y-%m-%dT%H-%M-%S') 
        chains_fn = savepath + f'/CD1Lrun2_Umbrella_v1_GB_posteriordir/CD1Lrun2_Umbrella_v1_GB_posteriors{len(chains_t_init)}_{i+start_index}.h5'
        os.makedirs(os.path.dirname(chains_fn), exist_ok=True)
        with h5py.File(chains_fn, 'w') as f:
            g = f.create_group("chains")
            g.attrs["n_leaves"] = len(chains_t_init)
            for leaf, arr in enumerate(chains_t_init):
                g.create_dataset(f"leaf_{leaf}", data=np.asarray(arr), compression="gzip", shuffle=True)

            f.create_dataset('initial_parameters', data=initial_parameters_t_init)
            f.attrs['parameter_names'] = PARAM_NAMES
            f.attrs['frequency_range_min'] = frequency_range[0]
            f.attrs['frequency_range_max'] = frequency_range[1]
            f.attrs['t0'] = runner.t0
            f.attrs['t_init'] = t_init
            f.attrs['time_taken'] = np.round((end_time - start_time), 0)
            f.attrs['time_stamp'] = time_stamp
        print(f"Saved chains to {chains_fn}")

        # free memory
        del chains, chains_nan_rows_removed, chains_t_init, initial_parameters_t_init, gb_pe


    # load chains
    group_index = 1171 # 2431
    group = grouped_found_sources[group_index]
    frequency_range = group['frequency_range']
    chains_fn = savepath + f'/CD1Lrun2_Umbrella_v1_GB_posteriordir/CD1Lrun2_Umbrella_v1_GB_posteriors2_{group_index}.h5'
    with h5py.File(chains_fn, 'r') as f:
        g = f["chains"]
        chains_t_init = [g[k][:] for k in sorted(g.keys(), key=lambda s: int(s.split("_")[1]))]
        initial_parameters = f['initial_parameters'][:]
        frequency_range_min = f.attrs['frequency_range_min']
        frequency_range_max = f.attrs['frequency_range_max']
        t0 = f.attrs['t0']
        t_init = f.attrs['t_init']
        time_taken = f.attrs['time_taken']
        parameter_names = f.attrs['parameter_names']
    
    
    # plot the chains
    chains_plot = chains_t_init[0]
    fig = plt.figure()
    for i in range(chains_plot.shape[1]):
        plt.plot(chains_plot[:,i]-chains_plot[0,i], label=PARAM_NAMES[i])
    plt.legend()
    plt.show(block=True)

    # swap Frequency and Amplitude
    chains_plot_swapped = chains_plot[:, [2, 0, 1, 3, 4, 5, 6, 7]]
    new_param_names = [
    "Amplitude",
    "Frequency",
    "FrequencyDerivative",
    "RightAscension",
    "Declination",
    "Polarization",
    "Inclination",
    "InitialPhase",
    ]
    new_param_labels = [r'A', r'f', r'$\dot{f}$', r'$\alpha$', r'$\delta$', r'$\psi$', r'$\iota$', r'$\phi_0$']
    fig = corner.corner(chains_plot_swapped, labels=new_param_labels, truths=injected_sources[0, [2, 0, 1, 3, 4, 5, 6, 7]], smooth=True, smooth1d=True)
    # corner.corner(chains_nan_rows_removed1, color='r', labels=PARAM_NAMES, fig=fig)
    # plt.tight_layout()
    plt.show(block=True)
    
if __name__ == "__main__":
    main(argv=sys.argv[1:])
