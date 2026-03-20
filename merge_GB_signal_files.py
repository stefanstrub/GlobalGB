"""
Utility script to merge per-window GB search outputs into a single file.

This script is primarily kept as an example / helper and is not part of the
core library API.  It scans a directory containing the per-batch GB search
results (HDF5), concatenates all recovered sources into a single array, sorts
them by frequency and writes the result as a single HDF5 file.

The current implementation is tailored to the Mojito configuration with
``SNR_threshold=9`` and ``seed=1`` and expects the same directory structure
produced by :class:`globalGB.GB_runner.GBSearchRunner`.
"""

from __future__ import annotations

import os
import sys
from os import listdir
from os.path import isfile, join
from typing import List, Tuple
import json
import h5py
import numpy as np

from globalGB.search_utils_GB import PARAM_INDICES


def load_sources_from_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single per-batch HDF5 result file.

    Returns
    -------
    sources : np.ndarray
        2-D array of recovered source parameters (may be empty).
    wall_times : np.ndarray
        1-D array of per-window wall-clock times.
    number_of_evaluations : np.ndarray
        1-D array of per-window number of function evaluations.
    """
    with h5py.File(path, "r") as f:
        sources = f["recovered_sources"][:]
        # get wall times from the file
        wall_times = f["wall_times"][:] if "wall_times" in f else np.array([])
        number_of_evaluations = f["number_of_evaluations"][:] if "number_of_evaluations" in f else np.array([])
    return sources, wall_times, number_of_evaluations


def flatten_found_sources(
    raw_sources: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Concatenate per-batch recovered sources into a single 2-D array.

    Parameters
    ----------
    raw_sources
        List of ``(sources, wall_times, number_of_evaluations)`` tuples as returned by
        :func:`load_sources_from_file`.
    """
    all_sources: List[np.ndarray] = []
    total_time = 0.0
    total_number_of_evaluations = 0
    for sources, wall_times, number_of_evaluations in raw_sources:
        if sources.size > 0:
            all_sources.append(sources)
        if wall_times.size > 0:
            total_time += np.sum(wall_times)
        if number_of_evaluations.size > 0:
            total_number_of_evaluations += np.sum(number_of_evaluations)
    if not all_sources:
        return np.empty((0, len(PARAM_INDICES)))
    return np.concatenate(all_sources), total_time, total_number_of_evaluations


def sort_by_frequency(found_sources: np.ndarray) -> np.ndarray:
    """
    Sort the flattened sources by their GW frequency.

    Parameters
    ----------
    found_sources
        2D array of sources in the GB parameter convention.
    """
    if found_sources.size == 0:
        return found_sources

    freqs = found_sources[:, PARAM_INDICES["Frequency"]]
    order = np.argsort(freqs)
    return found_sources[order]


def main(argv: list[str] | None = None) -> None:
    """
    Entry point for the merge script.

    Usage
    -----
    The script expects two positional arguments:

    - ``data_set`` (currently unused, kept for future generalisation), and
    - ``which_run``: one of ``"even1st"``, ``"even"``, ``"odd"``, ``"global"``.
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 1:
        raise SystemExit(
            "Usage: python merge_GB_signal_files.py <data_set> <which_run> (one of 'even1st', 'even', 'odd', 'global')"
        )

    which_run = str(argv[0])

    with open('globalGB/GB_search_config.json', 'r') as f:
        config = json.load(f)
    base_found_dir = config["save_path"]

    # Paths are currently hard-coded for Mojito, SNR threshold 9 and seed 1.
    directory_name = f"/found_signals_{config['data_set']}_SNR_threshold_{int(config['snr_threshold'])}_{which_run}_seed{config['seed']}"
    if which_run in ["global"]:
        directory_name = f"/found_signals_{config['data_set']}_SNR_threshold_{int(config['snr_threshold'])}_seed{config['seed']}"
    else:
        directory_name = f"/found_signals_{config['data_set']}_SNR_threshold_{int(config['snr_threshold'])}_{which_run}_seed{config['seed']}"

    input_dir = base_found_dir + directory_name
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    file_names = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    if not file_names:
        raise RuntimeError(f"No files found in input directory: {input_dir}")

    raw_sources = [
        load_sources_from_file(os.path.join(input_dir, fname)) for fname in file_names
    ]

    flat_sources, wall_times, number_of_evaluations = flatten_found_sources(raw_sources)
    flat_sources_sorted = sort_by_frequency(flat_sources)

    print("number of recovered sources:", len(flat_sources_sorted))
    print("total wall time:", np.round(wall_times/60, 1), "minutes")
    print("total number of function evaluations:", number_of_evaluations)

    output_base = base_found_dir
    if which_run in ["odd", "even"]:
        output_base = base_found_dir + f"/found_signals_{config['data_set']}_SNR_threshold_{int(config['snr_threshold'])}_global_seed{config['seed']}"
    os.makedirs(output_base, exist_ok=True)

    # save as hdf5 file
    output_path = output_base + file_name + ".h5"
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('recovered_sources', data=flat_sources_sorted)

    print(f"Saved merged catalogue to: {output_path}")


if __name__ == "__main__":
    main()
