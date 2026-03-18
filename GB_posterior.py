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
    with h5py.File(runner.savepath+f'/found_signals_Mojito_SNR_threshold_{int(config['snr_threshold'])}_seed{config['seed']}.h5', 'r') as f:
        found_sources = f['recovered_sources'][:]
    found_sources_df = pl.DataFrame(found_sources, schema=PARAM_NAMES)
    frequency_range = runner.frequencies_search[0]
    initial_parameters = found_sources_df.filter((pl.col('Frequency') > frequency_range[0]) & (pl.col('Frequency') < frequency_range[1])).to_numpy()
    gb_pe = GB_pe(runner.tdi_fs, initial_parameters, runner.Tobs, frequency_range[0], frequency_range[1], runner.waveform_args, config['dt'], channel_combination=config['channel_combination'])
    chains, ensemble = gb_pe.mcmc_GB(nsteps=100, burn=0, ntemps=4, nwalkers=32)
    chains_fn = savepath + f'/chains/chains_t0_Mojito_SNR_threshold_{int(config["snr_threshold"])}_seed{config["seed"]}.h5'
    with h5py.File(chains_fn, 'w') as f:
        f.create_dataset('chains', data=chains)


    

if __name__ == "__main__":
    main(argv=sys.argv[1:])
