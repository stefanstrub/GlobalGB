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

from jaxgb.jaxgb import JaxGB

from globalGB.search_utils_GB import GB_pe, PARAM_NAMES
from NoiseEstimate.noise_estimate import *
from DataLoader.data_loader import LISADataLoader
from globalGB.GB_runner import GBSearchRunner

def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Search for Galactic binaries in Mojito data.")
    parser.add_argument("batch_index", type=int, help="Batch index of frequency windows to process.")
    parser.add_argument("which_run", type=str, choices=["even1st", "even", "odd"], help="Window set to analyze.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open('globalGB/GB_search_config.json', 'r') as f:
        config = json.load(f)
    runner = GBSearchRunner(
        batch_index=args.batch_index,
        which_run=args.which_run,
        config=config,
    )

    runner.load_data()
    runner.prepare_frequency_windows()
    savepath = runner.savepath
    found_sources = np.load(savepath+'/found_signals_noisy_Mojito_SNR_threshold_9_seed1.npy')
    found_sources_df = pl.DataFrame(found_sources, schema=PARAM_NAMES)
    frequency_range = runner.frequencies_search[0]
    initial_parameters = found_sources_df.filter((pl.col('Frequency') > frequency_range[0]) & (pl.col('Frequency') < frequency_range[1])).to_numpy()
    gb_pe = GB_pe(runner.tdi_fs, initial_parameters, runner.Tobs, frequency_range[0], frequency_range[1], runner.waveform_args, config['dt'], channel_combination=config['channel_combination'])
    chains, ensemble = gb_pe.mcmc_GB(nsteps=100, burn=0, ntemps=4, nwalkers=32)
    np.save(savepath+'/chains_t0_Mojito_SNR_threshold_9_seed1.npy', chains)
    # np.save(savepath+'/ensemble_t0_Mojito_SNR_threshold_9_seed1.npy', ensemble)


    

if __name__ == "__main__":
    main(argv=sys.argv[1:])
