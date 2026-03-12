"""
High-level driver for the Galactic-binary (GB) search.

This module provides the `GBSearchRunner` class, which orchestrates the full
search pipeline for a batch of frequency windows:

- loading and preprocessing LISA TDI data,
- constructing the orbital configuration and waveform arguments,
- defining and batching the frequency windows to scan,
- subtracting already-identified sources in neighbouring windows,
- optionally pruning windows that show no significant change after a first pass,
- running the segment-level search, and
- saving all recovered sources for the processed batch to disk.

The interface is deliberately lightweight so it can be driven either from
the command line (see `GB_search.py`) or from higher-level analysis scripts.
"""

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
import pandas as pd

from jaxgb.jaxgb import JaxGB

from globalGB.search_utils_GB import (
    Segment_GB_Searcher,
    create_frequency_windows,
    tdi_subtraction,
    PARAM_NAMES,
)
from NoiseEstimate.noise_estimate import *
from DataLoader.data_loader import LISADataLoader


jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)

class GBSearchRunner:
    """
    Encapsulates the full GB search pipeline for a batch of frequency windows.

    Parameters
    ----------
    batch_index
        Zero-based index of the frequency-window batch to process. This is used
        to split a potentially very large total search range across many jobs.
    which_run
        Label that selects the subset of windows and the subtraction strategy.
        Allowed values:

        - ``"even1st"`` – first pass over the even-numbered windows,
        - ``"even"``    – second pass over only those even windows that changed
          significantly after the first pass,
        - ``"odd"``     – pass over the odd-numbered windows, subtracting
          sources found in the corresponding even windows.
    config
        Configuration dictionary loaded from ``GB_search_config.json``. It must
        at least contain the keys used in this class (see code for details),
        such as ``file_path``, ``save_path``, ``data_set``, ``dt``,
        ``channel_combination``, ``frequency_range``, ``batch_size``,
        ``signals_per_window``, ``signals_per_window_first_run``,
        ``snr_threshold``, ``seed`` and ``tdi_generation``.
    """

    def __init__(
        self,
        batch_index: int,
        which_run: str,
        config: dict,
    ):
        self.batch_index = int(batch_index)
        self.which_run = str(which_run)
        self.cfg = config

        # Paths relative to the current working directory.  We keep track of
        # the grandparent directory because the LDC data sets are expected to
        # live in ``<grandparent>/LDC``.
        cwd = os.getcwd()
        parent = os.path.dirname(cwd)
        self.grandparent = os.path.dirname(parent)


        # Paths to the input TDI data and the directory where search results
        # for this run should be stored.
        self.datapath = self.cfg["file_path"]
        self.savepath = self.cfg["save_path"]

        # Deterministic numpy RNG for reproducibility of any stochastic steps
        # (in particular the optimisers in `search_utils_GB`).
        np.random.seed(self.cfg["seed"])

        self.tdi_ts = None
        self.tdi_fs = None
        self.Tobs = None
        self.t0 = None
        self.waveform_args = None
        self.frequencies_search: List[Tuple[float, float]] = []
        self.frequencies_search_full: List[Tuple[float, float]] = []
        self.search_range: Tuple[float, float] = (0.0, 0.0)
        self.output_filename: Optional[str] = None

    def load_data(self) -> None:
        """
        Load LISA TDI data and construct the orbital configuration.

        This method populates

        - ``self.tdi_ts`` and ``self.tdi_fs`` with the time- and
          frequency-domain TDI data for the requested channel combination,
        - ``self.Tobs`` and ``self.t0`` with the observation time and start
          time, and
        - ``self.waveform_args`` with the arguments needed by the waveform
          generator (TDI generation scheme, interpolated LISA orbits, etc.).
        """
        loader = LISADataLoader(
            dataset=self.cfg["data_set"],
            base_path=os.path.join(self.grandparent, "LDC"),
        )

        loader.load(
            self.datapath,
            dt=self.cfg["dt"],
            channel_combination=self.cfg["channel_combination"],
        )
        self.tdi_ts = loader.tdi_ts
        self.tdi_fs = loader.tdi_fs
        self.Tobs = loader.Tobs
        self.t0 = loader.t0

        with h5py.File(self.datapath, "r") as fid:
            orbits_data = fid["orbits"]
            sampling = dict(orbits_data["sampling"].attrs)
            t_grid = np.arange(sampling["t0"], sampling["t0"] + sampling["duration"], sampling["dt"])
            spacecraft_positions = np.stack(
                [
                    orbits_data["sc_position_1"][:],
                    orbits_data["sc_position_2"][:],
                    orbits_data["sc_position_3"][:],
                ]
            ).swapaxes(0, 1)
            spacecraft_velocities = np.stack(
                [
                    orbits_data["sc_velocity_1"][:],
                    orbits_data["sc_velocity_2"][:],
                    orbits_data["sc_velocity_3"][:],
                ]
            ).swapaxes(0, 1)

        orbits = lisaorbits.InterpolatedOrbits(t_grid, spacecraft_positions, spacecraft_velocities)

        self.waveform_args = {
            "tdi_generation": self.cfg["tdi_generation"],
            "orbits": orbits,
            "Tobs": self.Tobs,
            "t0": self.t0,
        }

    def prepare_frequency_windows(self) -> None:
        """
        Build and select the list of frequency windows to be processed.

        The full search range is first split into overlapping windows using
        ``create_frequency_windows``.  These are then separated into
        even- and odd-numbered windows.  Depending on ``which_run`` only the
        relevant subset is kept and finally a batch slice corresponding to
        ``batch_index`` is selected.

        The method sets

        - ``self.frequencies_search`` to the windows that will actually be
          searched in this job,
        - ``self.frequencies_search_full`` to the corresponding full list
          before any pruning, and
        - ``self.search_range`` to the overall frequency interval covered
          by this batch.
        """
        frequencies = create_frequency_windows(self.cfg["frequency_range"], self.Tobs)

        frequencies_even = frequencies[::2]
        frequencies_odd = frequencies[1::2]

        if self.which_run in ["even1st", "even"]:
            frequencies_search = frequencies_even
        else:
            frequencies_search = frequencies_odd

        batch_size = self.cfg["batch_size"]
        start_index = batch_size * self.batch_index

        print(
            "batch",
            self.batch_index,
            "batch size",
            batch_size,
            "start index",
            start_index,
            "total number of windows",
            len(frequencies_search)
        )

        frequencies_search = frequencies_search[start_index : start_index + batch_size]

        # Ensure that the uppermost window including its buffer stays inside
        # the global search band.  If necessary, drop the highest windows.
        while (
            frequencies_search[-1][1]
            + (frequencies_search[-1][1] - frequencies_search[-1][0]) / 2
            > self.cfg["frequency_range"][1]
        ):
            frequencies_search = frequencies_search[:-1]
        self.frequencies_search_full = deepcopy(frequencies_search)
        self.frequencies_search = frequencies_search
        self.search_range = (frequencies_search[0][0], frequencies_search[-1][1])
        print(
            "search range",
            f"{np.round(self.search_range[0] * 1e3, 5)} mHz to {np.round(self.search_range[1] * 1e3, 5)} mHz",
        )

    def subtract_neighboring_windows(self) -> None:
        """
        Subtract sources found in the complementary set of frequency windows.

        On alternating passes the search uses previously recovered sources from
        the complementary set of windows (even vs. odd) to subtract power from
        the current windows.  This reduces source confusion at the edges of
        each window and improves convergence of the optimisers.

        Notes
        -----
        - For the very first pass (``which_run == "even1st"``) there are no
          previous results to subtract, so this method is a no-op.
        - The subtraction acts directly on ``self.tdi_fs``.
        """
        if self.which_run in ["even1st"]:
            return

        start = time.time()
        save_directory = f"/found_signals_{self.cfg['data_set']}_SNR_threshold_{int(self.cfg['snr_threshold'])}_seed{self.cfg['seed']}"

        if self.which_run in ["odd"]:
            save_name_previous = (
                f"/found_signals_{self.cfg['data_set']}_SNR_threshold_{int(self.cfg['snr_threshold'])}_even1st_seed{self.cfg['seed']}"
            )
        else:
            save_name_previous = (
                f"/found_signals_{self.cfg['data_set']}_SNR_threshold_{int(self.cfg['snr_threshold'])}_odd_seed{self.cfg['seed']}"
            )

        found_sources_flat = np.load(
            self.savepath + save_directory + save_name_previous + ".npy", allow_pickle=True
        )
        found_sources_flat_df = pd.DataFrame(found_sources_flat, columns=PARAM_NAMES)
        found_sources_outside_flat_df = found_sources_flat_df.sort_values("Frequency")

        fgb = JaxGB(
            orbits=self.waveform_args["orbits"],
            t_obs=self.waveform_args["Tobs"],
            t0=self.waveform_args["t0"],
            n=1024,
        )

        for win in self.frequencies_search_full:
            found_sources_outside_flat_df = found_sources_outside_flat_df[
                (found_sources_outside_flat_df["Frequency"] < win[0])
                | (found_sources_outside_flat_df["Frequency"] > win[1])
            ]

        found_sources_outside_flat_df = found_sources_outside_flat_df.sort_values("Frequency")
        found_sources_outside_flat = jnp.asarray(found_sources_outside_flat_df.values)

        self.tdi_fs = tdi_subtraction(
            self.tdi_fs,
            found_sources_outside_flat,
            fgb,
            self.waveform_args["tdi_generation"],
            self.cfg["channel_combination"],
        )

        print("subtraction time", time.time() - start)

    def remove_even_windows_if_unchanged(self) -> None:
        """
        Down-select even windows that merit a second pass.

        After the initial ``"even1st"`` run, only those even windows are
        re-analysed for ``which_run == "even"`` that either:

        - show significant power between adjacent windows (indicating leakage
          from nearby sources), or
        - contain at least ``max_signals_per_window_first_run`` sources detected
          in the first pass.

        The method updates ``self.frequencies_search`` in place.
        """
        if self.which_run not in ["even"]:
            return

        frequencies_search_reduced: List[Tuple[float, float]] = []
        frequencies_search_skipped: List[Tuple[float, float]] = []

        save_directory = f"/found_signals_{self.cfg['data_set']}_SNR_threshold_{int(self.cfg['snr_threshold'])}_seed{self.cfg['seed']}"
        save_name_previous = (
            f"/found_signals_{self.cfg['data_set']}_SNR_threshold_{int(self.cfg['snr_threshold'])}_even1st_seed{self.cfg['seed']}"
        )

        found_sources_flat = np.load(
            self.savepath + save_directory + save_name_previous + ".npy", allow_pickle=True
        )
        found_sources_flat_df = pd.DataFrame(found_sources_flat, columns=PARAM_NAMES).sort_values("Frequency")
        save_name_previous_odd = (
            f"/found_signals_{self.cfg['data_set']}_SNR_threshold_{int(self.cfg['snr_threshold'])}_odd_seed{self.cfg['seed']}"
        )

        found_sources_flat = np.load(
            self.savepath + save_directory + save_name_previous_odd + ".npy", allow_pickle=True
        )
        found_sources_outside_flat_df = pd.DataFrame(found_sources_flat, columns=PARAM_NAMES).sort_values("Frequency")

        for win in self.frequencies_search:
            found_sources_outside_lower = []
            found_sources_outside_upper = []
            try:
                index = np.searchsorted(np.asarray(self.frequencies_search_full)[:, 0], win[0])
                found_sources_outside_lower = found_sources_outside_flat_df[
                    (found_sources_outside_flat_df["Frequency"] > self.frequencies_search_full[index - 1][1])
                    & (found_sources_outside_flat_df["Frequency"] < self.frequencies_search_full[index][0])
                ]
            except Exception:
                pass
            try:
                index = np.searchsorted(np.asarray(self.frequencies_search_full)[:, 0], win[0])
                found_sources_outside_upper = found_sources_outside_flat_df[
                    (found_sources_outside_flat_df["Frequency"] > self.frequencies_search_full[index][1])
                    & (found_sources_outside_flat_df["Frequency"] < self.frequencies_search_full[index + 1][0])
                ]
            except Exception:
                pass

            if len(found_sources_outside_lower) > 0 or len(found_sources_outside_upper) > 0:
                frequencies_search_reduced.append(win)
            else:
                inside = found_sources_flat_df[
                    (found_sources_flat_df["Frequency"] > win[0])
                    & (found_sources_flat_df["Frequency"] < win[1])
                ]
                if len(inside) > self.cfg["signals_per_window_first_run"] - 1:
                    frequencies_search_reduced.append(win)
                else:
                    frequencies_search_skipped.append(win)

        print("frequencies_search_reduced length", len(frequencies_search_reduced))
        print("frequencies_search_skipped length", len(frequencies_search_skipped))
        self.frequencies_search = frequencies_search_reduced

    def load_initial_guess(self):
        """
        Return an initial guess for the search.

        This hook is kept for future extensions where an external catalogue
        or a coarse pre-search could be used to seed the optimisation.  At the
        moment it simply returns an empty list, meaning that every window is
        searched from scratch.
        """
        return []

    def prepare_output_paths(self) -> None:
        """
        Construct the output file name for the current batch and ensure that
        the corresponding directory exists.

        The file name encodes the batch index, the covered frequency range in
        nHz, and the global search configuration (data set, SNR threshold and
        run label).
        """
        save_name = f"{self.cfg['data_set']}_SNR_threshold_{int(self.cfg['snr_threshold'])}_{self.which_run}"

        directory = os.path.join(
            self.savepath, f"found_signals_{save_name}_seed{self.cfg['seed']}"
        )

        os.makedirs(directory, exist_ok=True)

        fn = os.path.join(
            directory,
            f"found_signals_batch_index_{self.batch_index}_"
            f"{int(np.round(self.search_range[0] * 1e9))}nHz_to"
            f"{int(np.round(self.search_range[1] * 1e9))}nHz_{save_name}.pkl",
        )
        print("output filename", fn)
        self.output_filename = fn

    def run_segment_search(self) -> None:
        """
        Run the segment-level search on all selected windows and save results.

        This method constructs a :class:`Segment_GB_Searcher` instance and
        iterates over all windows in ``self.frequencies_search``.  For each
        window it launches the optimisation in the underlying
        :class:`GB_Searcher`, collects all recovered sources and finally
        serialises the full list of results to ``self.output_filename`` using
        :mod:`pickle`.
        """
        max_signals_per_window = (
            self.cfg["max_signals_per_window_first_run"]
            if self.which_run in ["even1st"]
            else self.cfg["max_signals_per_window"]
        )

        found_sources_sorted_initial = self.load_initial_guess()

        GB_segment_searcher = Segment_GB_Searcher(
            self.tdi_fs,
            self.Tobs,
            waveform_args=self.waveform_args,
            dt=self.cfg["dt"],
            SNR_threshold=self.cfg["snr_threshold"],
            channel_combination=self.cfg["channel_combination"],
            max_signals_per_window=max_signals_per_window,
            found_sources_previous=found_sources_sorted_initial,
        )

        start = time.time()
        found_sources = []
        for f_low, f_high in self.frequencies_search:
            found_sources.append(GB_segment_searcher.search(f_low, f_high))

        print(
            "searched ",
            len(self.frequencies_search),
            " segments in ",
            np.round((time.time() - start) / 60, 1),
            "minutes",
        )

        if self.output_filename is None:
            raise RuntimeError("Output filename not prepared.")
        with open(self.output_filename, "wb") as f:
            import pickle

            print("saving found sources to", self.output_filename)
            pickle.dump(found_sources, f)

    def run(self) -> None:
        self.load_data()
        self.prepare_frequency_windows()
        self.subtract_neighboring_windows()
        self.remove_even_windows_if_unchanged()
        self.prepare_output_paths()
        self.run_segment_search()

