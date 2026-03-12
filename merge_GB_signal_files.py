"""
Utility script to merge per-window GB search outputs into a single file.

This script is primarily kept as an example / helper and is not part of the
core library API.  It scans a directory containing the per-batch GB search
results, concatenates all recovered sources into a single array, sorts them
by frequency and writes the result as a NumPy ``.npy`` file.

The current implementation is tailored to the Mojito configuration with
``SNR_threshold=9`` and ``seed=1`` and expects the same directory structure
produced by :class:`globalGB.GB_runner.GBSearchRunner`.
"""

from __future__ import annotations

import os
import sys
import pickle
from os import listdir
from os.path import isfile, join
from typing import List

import numpy as np

from globalGB.search_utils_GB import PARAM_INDICES


def get_project_root() -> str:
    """
    Return the project root two levels above the current working directory.

    This mirrors the convention used elsewhere in the project where the LDC
    data directory is placed at ``<project_root>/LDC``.
    """
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    return os.path.dirname(parent)


def load_sources_from_file(path: str):
    """
    Load a single per-window result file.

    The function first tries to unpickle the file; if that fails it falls back
    to ``numpy.load(..., allow_pickle=True)`` for backwards compatibility.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return np.load(path, allow_pickle=True).tolist()


def flatten_found_sources(raw_sources: List, which_run: str) -> np.ndarray:
    """
    Flatten the nested per-window search outputs into a 2D array.

    Parameters
    ----------
    raw_sources
        List of objects as loaded from the per-window result files.
    which_run
        Run label passed on the command line (``"even1st"``, ``"even"`` or
        ``"odd"``).  For non-empty labels we expect the hierarchical data
        structure produced by :class:`Segment_GB_Searcher`.
    """
    found_sources_mp_unsorted: List[np.ndarray] = []
    times: List[float] = []

    for sources in raw_sources:
        for entry in sources:
            if which_run:
                # `entry` is a tuple-like structure; index 3 holds the array of
                # recovered sources, index 4 the central frequency, index 5 the
                # wall-clock time for this window.
                times.append(entry[5])
                if len(entry[3]) > 0:
                    found_sources_mp_unsorted.append(entry[3])
            else:
                # Backwards-compatibility path where each file already contains
                # a flat list of sources.
                found_sources_mp_unsorted.append(entry)

    if not found_sources_mp_unsorted:
        return np.empty((0, len(PARAM_INDICES)))

    if which_run:
        found_sources_in_flat = np.concatenate(found_sources_mp_unsorted)
    else:
        found_sources_in_flat = np.asarray(found_sources_mp_unsorted)

    # Report total runtime in hours for quick diagnostics.
    if times:
        print("time", np.round(np.sum(times) / 3600.0, 1), "hours")

    return np.asarray(found_sources_in_flat)


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
    - ``which_run``: one of ``"even1st"``, ``"even"``, ``"odd"``.
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 2:
        raise SystemExit(
            "Usage: python merge_GB_signal_files.py <data_set> <which_run>"
        )

    _data_set = str(argv[0])
    which_run = str(argv[1])

    project_root = get_project_root()

    # Paths are currently hard-coded for Mojito, SNR threshold 9 and seed 1.
    base_found_dir = os.path.join(project_root, "LDC", "Mojito", "found_signals", "GB")
    run_suffix = f"{which_run}_" if which_run else ""
    run_name = f"/found_signals_Mojito_SNR_threshold_9_{run_suffix}seed1"
    save_folder = "/found_signals_Mojito_SNR_threshold_9_seed1"

    input_dir = base_found_dir + run_name
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    file_names = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    if not file_names:
        raise RuntimeError(f"No files found in input directory: {input_dir}")

    raw_sources = [
        load_sources_from_file(os.path.join(input_dir, fname)) for fname in file_names
    ]

    flat_sources = flatten_found_sources(raw_sources, which_run=which_run)
    flat_sources_sorted = sort_by_frequency(flat_sources)

    print("number of recovered sources:", len(flat_sources_sorted))

    output_base = base_found_dir + save_folder
    os.makedirs(output_base, exist_ok=True)

    output_path = output_base + run_name + ".npy"
    np.save(output_path, flat_sources_sorted)
    print(f"Saved merged catalogue to: {output_path}")


if __name__ == "__main__":
    main()
