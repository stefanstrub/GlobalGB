"""
Gravitational Wave Binary Signal Matching Pipeline

This module provides tools for matching found gravitational wave signals 
against injected signals from a catalog, computing overlap metrics, and 
analyzing matching results.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from copy import deepcopy
import time
import sys
import os

import numpy as np
import jax
import jax.numpy as jnp
import polars as pl
import h5py
import lisaorbits
import pickle
from matplotlib import pyplot as plt, rcParams
import json

from jaxgb.jaxgb import JaxGB

from globalGB.search_utils_GB import GB_Searcher, create_frequency_windows, max_signal_bandwidth, PARAM_NAMES, PARAM_INDICES, GBConfig
from DataLoader.data_loader import LISADataLoader

# Configure JAX
jax.config.update('jax_default_device', jax.devices('cpu')[0])
jax.config.update("jax_enable_x64", True)




def setup_plotting():
    """Configure matplotlib for publication-quality figures."""
    plot_params = {
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
    rcParams.update(plot_params)
    
    fig_width_pt = 1.5 * 464.0
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    inches_per_pt = 1.0 / 72.27
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * golden_mean
    rcParams.update({"figure.figsize": [fig_width, fig_height]})


class WaveformCalculator:
    """Handles waveform generation and alignment for signal comparison."""
    
    def __init__(self, fgb: JaxGB, tdi_generation: float = 2.0, 
                 channel_combination: str = 'AET'):
        self.fgb = fgb
        self.tdi_generation = tdi_generation
        self.channel_combination = channel_combination
        self._get_tdi_jit = jax.jit(self._get_tdi)
    
    def _get_tdi(self, params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.fgb.get_tdi(
            params, 
            tdi_generation=self.tdi_generation, 
            tdi_combination=self.channel_combination
        )
    
    def get_tdi(self, params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute TDI channels for given parameters (JIT-compiled)."""
        return self._get_tdi_jit(jnp.array(params))
    
    def align_waveforms(self, params_a: np.ndarray, params_b: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                  np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute and align waveforms for two sets of parameters.
        
        Returns:
            Tuple of (A_a, E_a, T_a, A_b, E_b, T_b) aligned arrays
        """
        params_a = jnp.array(params_a)
        params_b = jnp.array(params_b)
        
        A_a, E_a, T_a = self.get_tdi(params_a)
        A_b, E_b, T_b = self.get_tdi(params_b)
        
        f0_a = params_a[PARAM_INDICES['Frequency']]
        f0_b = params_b[PARAM_INDICES['Frequency']]
        kmin_a = self.fgb.get_kmin(f0_a)
        kmin_b = self.fgb.get_kmin(f0_b)
        
        if kmin_a == kmin_b:
            return A_a, E_a, T_a, A_b, E_b, T_b
        
        if kmin_a > kmin_b:
            offset = kmin_a - kmin_b
            A_b_aligned = A_b[offset:]
            E_b_aligned = E_b[offset:]
            T_b_aligned = T_b[offset:]
            A_a_aligned = A_a[:len(A_b_aligned)]
            E_a_aligned = E_a[:len(E_b_aligned)]
            T_a_aligned = T_a[:len(T_b_aligned)]
        else:
            offset = kmin_b - kmin_a
            A_a_aligned = A_a[offset:]
            E_a_aligned = E_a[offset:]
            T_a_aligned = T_a[offset:]
            A_b_aligned = A_b[:len(A_a_aligned)]
            E_b_aligned = E_b[:len(E_a_aligned)]
            T_b_aligned = T_b[:len(T_a_aligned)]
        
        return A_a_aligned, E_a_aligned, T_a_aligned, A_b_aligned, E_b_aligned, T_b_aligned


class SignalMatcher:
    """Computes various match metrics between gravitational wave signals."""
    
    def __init__(self, waveform_calc: WaveformCalculator, Tobs: float):
        self.waveform_calc = waveform_calc
        self.Tobs = Tobs
    
    def correlation(self, params_injected: np.ndarray, params_found: np.ndarray) -> float:
        """Compute correlation between injected and found signals."""
        A_inj, E_inj, _, A_found, E_found, _ = self.waveform_calc.align_waveforms(
            params_injected, params_found
        )
        
        dh = np.sum(np.real(A_inj * np.conjugate(A_found)))
        dh += np.sum(np.real(E_inj * np.conjugate(E_found)))
        
        hh = np.sum(np.absolute(A_found)**2) + np.sum(np.absolute(E_found)**2)
        ss = np.sum(np.absolute(A_inj)**2) + np.sum(np.absolute(E_inj)**2)
        
        return dh / (np.sqrt(ss) * np.sqrt(hh))
    
    def overlap(self, params_injected: np.ndarray, params_found: np.ndarray) -> float:
        """Compute overlap between injected and found signals."""
        A_inj, E_inj, _, A_found, E_found, _ = self.waveform_calc.align_waveforms(
            params_injected, params_found
        )
        
        sh = np.sum(np.absolute(A_inj * np.conjugate(A_found)))
        sh += np.sum(np.absolute(E_inj * np.conjugate(E_found)))
        
        ss = np.sum(np.absolute(A_inj)**2) + np.sum(np.absolute(E_inj)**2)
        hh = np.sum(np.absolute(A_found)**2) + np.sum(np.absolute(E_found)**2)
        
        return sh / np.sqrt(ss * hh)
    
    def scaled_snr_error(self, params_injected: np.ndarray, params_found: np.ndarray) -> float:
        """Compute scaled SNR error between injected and found signals."""
        A_inj, E_inj, _, A_found, E_found, _ = self.waveform_calc.align_waveforms(
            params_injected, params_found
        )
        
        error = np.sum(np.absolute(A_inj - A_found)**2)
        error += np.sum(np.absolute(E_inj - E_found)**2)
        
        ss = np.sum(np.absolute(A_inj)**2) + np.sum(np.absolute(E_inj)**2)
        hh = np.sum(np.absolute(A_found)**2) + np.sum(np.absolute(E_found)**2)
        
        return error / (np.sqrt(ss) * np.sqrt(hh))
    
    def match_signals(self, found_sources: np.ndarray, injected_catalog: pl.DataFrame,
                      use_overlap: bool = True, verbose: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match found signals to injected catalog sources.
        
        Args:
            found_sources: Array of found signal parameters
            injected_catalog: DataFrame of injected catalog parameters
            use_overlap: If True, use overlap metric; otherwise use scaled SNR error
            verbose: Print progress information
            
        Returns:
            Tuple of (found_sources, matched_injected, match_values)
        """
        injected_matches = []
        found_matches = []
        match_values = []
        
        start_time = time.time()
        n_sources = len(found_sources)
        
        for i, found_params in enumerate(found_sources):
            if verbose and i % 100 == 0:
                print(f'Matching: {i}/{n_sources}')
            
            freq = found_params[PARAM_NAMES.index('Frequency')]
            bandwidth = max_signal_bandwidth(freq, self.Tobs)
            freq_range = (freq - bandwidth/4, freq + bandwidth/4)
            
            candidates = injected_catalog.filter(
                (pl.col('Frequency') > freq_range[0]) & 
                (pl.col('Frequency') < freq_range[1])
            )
            
            if len(candidates) == 0:
                if verbose:
                    print(f'No match candidates for source {i} at f={freq:.6f}')
                continue
            
            match_scores = []
            for k in range(len(candidates)):
                candidate_params = np.array(candidates[k])[0]
                if use_overlap:
                    score = self.overlap(candidate_params, found_params)
                else:
                    score = self.scaled_snr_error(candidate_params, found_params)
                match_scores.append(score)
            
            if use_overlap:
                best_idx = int(np.nanargmax(match_scores))
            else:
                best_idx = int(np.nanargmin(match_scores))
            
            injected_matches.append(np.array(candidates[best_idx])[0])
            found_matches.append(found_params)
            match_values.append(match_scores[best_idx])
        
        if verbose:
            print(f'Matching completed in {time.time() - start_time:.1f}s')
        
        return (np.array(found_matches), np.array(injected_matches), 
                np.array(match_values))


@dataclass
class MatchResults:
    """Container for matching results with analysis methods."""
    
    found_sources: np.ndarray
    injected_sources: np.ndarray
    match_values: np.ndarray
    threshold: float
    use_overlap: bool
    
    @property
    def match_mask(self) -> np.ndarray:
        if self.use_overlap:
            return self.match_values > self.threshold
        return self.match_values < self.threshold
    
    @property
    def matched_found(self) -> np.ndarray:
        return self.found_sources[self.match_mask]
    
    @property
    def matched_injected(self) -> np.ndarray:
        return self.injected_sources[self.match_mask]
    
    @property
    def matched_values(self) -> np.ndarray:
        return self.match_values[self.match_mask]
    
    @property
    def unmatched_found(self) -> np.ndarray:
        return self.found_sources[~self.match_mask]
    
    @property
    def unmatched_injected(self) -> np.ndarray:
        return self.injected_sources[~self.match_mask]
    
    @property
    def unmatched_values(self) -> np.ndarray:
        return self.match_values[~self.match_mask]
    
    @property
    def n_matched(self) -> int:
        return int(np.sum(self.match_mask))
    
    @property
    def n_total(self) -> int:
        return len(self.found_sources)
    
    @property
    def match_fraction(self) -> float:
        return self.n_matched / self.n_total if self.n_total > 0 else 0.0
    
    def to_dataframes(self) -> Dict[str, pl.DataFrame]:
        """Convert results to Polars DataFrames."""
        matched_found_df = pl.DataFrame(self.matched_found, schema=PARAM_NAMES)
        matched_found_df = matched_found_df.with_columns(
            pl.lit(self.matched_values).alias('match_score')
        ).sort('Frequency')
        
        matched_injected_df = pl.DataFrame(self.matched_injected, schema=PARAM_NAMES)
        matched_injected_df = matched_injected_df.with_columns(
            pl.lit(self.matched_values).alias('match_score')
        ).sort('Frequency')
        
        unmatched_found_df = pl.DataFrame(self.unmatched_found, schema=PARAM_NAMES)
        unmatched_found_df = unmatched_found_df.with_columns(
            pl.lit(self.unmatched_values).alias('match_score')
        ).sort('Frequency')
        
        unmatched_injected_df = pl.DataFrame(self.unmatched_injected, schema=PARAM_NAMES)
        unmatched_injected_df = unmatched_injected_df.with_columns(
            pl.lit(self.unmatched_values).alias('match_score')
        ).sort('Frequency')
        
        return {
            'matched_found': matched_found_df,
            'matched_injected': matched_injected_df,
            'unmatched_found': unmatched_found_df,
            'unmatched_injected': unmatched_injected_df,
        }
    
    def print_summary(self, n_injected: int):
        """Print matching statistics summary."""
        print(f'\n{"="*60}')
        print('MATCHING RESULTS SUMMARY')
        print(f'{"="*60}')
        print(f'Matched signals: {self.n_matched} / {self.n_total} found')
        print(f'Match fraction: {self.match_fraction:.2%}')
        print(f'Sensitivity (matched/injected): {self.n_matched/n_injected:.2%}')
        print(f'Threshold: {self.threshold} ({"overlap" if self.use_overlap else "SNR error"})')
        print(f'{"="*60}\n')
    
    def save(self, path: str, name: str):
        """Save results to HDF5 file."""
        filepath = Path(path) / f'match_results_{name}.h5'
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('found_sources', data=self.found_sources)
            f.create_dataset('injected_sources', data=self.injected_sources)
            f.create_dataset('match_values', data=self.match_values)
            f.attrs['threshold'] = self.threshold
            f.attrs['use_overlap'] = self.use_overlap
        print(f'Results saved to {filepath}')
    
    @classmethod
    def load(cls, filepath: str) -> 'MatchResults':
        """Load results from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            return cls(
                found_sources=f['found_sources'][:],
                injected_sources=f['injected_sources'][:],
                match_values=f['match_values'][:],
                threshold=f.attrs['threshold'],
                use_overlap=f.attrs['use_overlap'],
            )


class GBMatchingPipeline:
    """Main pipeline for matching gravitational wave binary signals."""
    
    def __init__(self, config: GBConfig):
        self.config = config
        self.loader: Optional[LISADataLoader] = None
        self.fgb: Optional[JaxGB] = None
        self.waveform_calc: Optional[WaveformCalculator] = None
        self.matcher: Optional[SignalMatcher] = None
        self.frequencies: List[Tuple[float, float]] = []
        
    def setup(self, data_fn: Optional[str] = None):
        """Initialize data loader, waveform generator, and matcher."""
        
        # Load data
        self.loader = LISADataLoader(config=self.config)
        data_fn = self.loader.data_path
        
        self.loader.load(data_fn=data_fn)
        self.loader._load_mojito_wdwd_catalog()
        
        # Load orbits and create waveform generator
        with h5py.File(data_fn, 'r') as fid:
            orbits_data = fid['orbits']
            sampling = dict(orbits_data["sampling"].attrs)
            orbits = lisaorbits.InterpolatedOrbits(
                np.arange(sampling["t0"], sampling["t0"] + sampling["duration"], sampling["dt"]),
                np.stack([orbits_data["sc_position_1"][:],
                         orbits_data["sc_position_2"][:],
                         orbits_data["sc_position_3"][:]]).swapaxes(0, 1),
                np.stack([orbits_data["sc_velocity_1"][:],
                         orbits_data["sc_velocity_2"][:],
                         orbits_data["sc_velocity_3"][:]]).swapaxes(0, 1)
            )
        
        self.fgb = JaxGB(
            orbits=orbits,
            t_obs=self.loader.Tobs,
            t0=self.loader.t0,
            n=int(2**10)
        )
        
        self.waveform_calc = WaveformCalculator(
            self.fgb,
            tdi_generation=self.config.tdi_generation,
            channel_combination=self.config.channel_combination
        )
        
        self.matcher = SignalMatcher(self.waveform_calc, self.loader.Tobs)
        
        # Create frequency windows
        search_range = [0.0003, self.config.f_nyquist]
        self.frequencies = create_frequency_windows(search_range, self.loader.Tobs)
        
        # Trim frequencies to valid range
        while (self.frequencies[-1][1] + 
               (self.frequencies[-1][1] - self.frequencies[-1][0])/2 > self.config.f_nyquist):
            self.frequencies = self.frequencies[:-1]
        
        print(f'Setup complete. {len(self.frequencies)} frequency windows.')
    
    def load_found_sources(self) -> np.ndarray:
        """Load found sources from file."""
        fn = os.path.join(
            self.config.found_signals_path,
            f'found_signals_noisy_{self.config.save_name}_flat.npy'
        )
        sources = np.load(fn, allow_pickle=False)
        print(f'Loaded {len(sources)} found sources')
        return sources
    
    def load_injected_catalog(self) -> pl.DataFrame:
        """Load and prepare injected source catalog."""
        cat_array = self.loader.catalog_wdwd
        cat_df = pl.DataFrame(cat_array, schema=PARAM_NAMES)
        cat_df = cat_df.sort('Frequency')
        print(f'Loaded {len(cat_df)} injected sources')
        return cat_df
    
    def prepare_injected_for_matching(self, injected_df: pl.DataFrame, 
                                      save_name: str = 'Mojito_WDWD') -> np.ndarray:
        """Filter injected catalog to relevant frequency windows."""
        cache_fn = os.path.join(
            self.config.save_path, 
            f'pGB_injected_no_SNR_{save_name}.npy'
        )
        
        if os.path.exists(cache_fn):
            pGB_injected = np.load(cache_fn, allow_pickle=True)
            print(f'Loaded cached injected sources: {len(pGB_injected)}')
        else:
            pGB_injected = []
            for j, freq_window in enumerate(self.frequencies):
                if j % 100 == 0:
                    print(f'Processing window {j}/{len(self.frequencies)}')
                filtered = injected_df.filter(
                    (pl.col('Frequency') > freq_window[0]) & 
                    (pl.col('Frequency') < freq_window[1])
                )
                filtered = filtered.sort('Amplitude', descending=True)[:self.config.max_injected_per_window]
                pGB_injected.append(np.array(filtered))
            
            pGB_injected = np.concatenate(pGB_injected)
            np.save(cache_fn, pGB_injected)
            print(f'Saved {len(pGB_injected)} filtered injected sources')
        
        return pGB_injected
    
    def run_matching(self, found_sources: np.ndarray, 
                    injected_catalog: pl.DataFrame) -> MatchResults:
        """Run the matching process."""
        found, injected, values = self.matcher.match_signals(
            found_sources,
            injected_catalog,
            use_overlap=self.use_overlap,
            verbose=True
        )
        
        return MatchResults(
            found_sources=found,
            injected_sources=injected,
            match_values=values,
            threshold=self.config.overlap_threshold,
            use_overlap=self.config.use_overlap,
        )
    
    def plot_frequency_window(self, results: MatchResults, 
                             freq_start: float, n_windows: int = 1):
        """Plot matched and unmatched signals in a frequency window."""
        freq_idx = np.searchsorted(
            np.asarray(self.frequencies)[:, 0], freq_start
        )
        freq_range = (
            self.frequencies[freq_idx][0],
            self.frequencies[freq_idx + n_windows - 1][1]
        )
        
        search = GB_Searcher(
            self.loader.tdi_fs,
            self.loader.Tobs,
            freq_range[0],
            freq_range[1],
            waveform_args={
                'tdi_generation': self.config.tdi_generation,
                'orbits': self.fgb.orbits,
                'Tobs': self.loader.Tobs,
                't0': self.loader.t0
            },
            dt=self.config.dt,
            channel_combination=self.config.channel_combination
        )
        
        dfs = results.to_dataframes()
        mask = (pl.col('Frequency') >= freq_range[0]) & (pl.col('Frequency') <= freq_range[1])
        
        search.plotAE(
            found_matched=jnp.array(dfs['matched_found'].filter(mask)),
            found_not_matched=jnp.array(dfs['unmatched_found'].filter(mask)),
            injected_matched=jnp.array(dfs['matched_injected'].filter(mask)),
            injected_not_matched=jnp.array(dfs['unmatched_injected'].filter(mask)),
        )


def main():
    """Main entry point for the matching pipeline."""
    setup_plotting()
    np.random.seed(42)
    
    # Parse command line arguments
    with open('globalGB/GB_search_config.json', 'r') as f:
        config_dict = json.load(f)
        config = GBConfig(config_dict)

    loader = LISADataLoader(config=config)
    loader.load(
        data_path=config.data_path,
        dt=config.dt,
        channel_combination=config.channel_combination,
    )
    tdi_ts = loader.tdi_ts
    tdi_fs = loader.tdi_fs
    config.Tobs = loader.Tobs
    config.t0 = loader.t0
    
    # Create and run pipeline
    pipeline = GBMatchingPipeline(config)
    pipeline.setup()
    
    # Load data
    found_sources = pipeline.load_found_sources()
    injected_df = pipeline.load_injected_catalog()
    
    # Sort found sources
    found_df = pl.DataFrame(found_sources, schema=PARAM_NAMES).sort('Frequency')
    found_sources = np.array(found_df)
    
    # Check for cached results
    cache_path = os.path.join(config.save_path, 'found_sources.h5')
    
    if os.path.exists(cache_path):
        print('Loading cached matching results...')
        with h5py.File(cache_path, 'r') as f:
            results = MatchResults(
                found_sources=f['found_sources'][:],
                injected_sources=f['injected_sources'][:],
                match_values=f['scaled_error'][:],
                threshold=config.overlap_threshold,
                use_overlap=config.use_overlap,
            )
    else:
        print('Running matching (this may take a while)...')
        results = pipeline.run_matching(found_sources, injected_df)
        results.save(config.save_path, config.save_name)
    
    # Print summary
    results.print_summary(n_injected=len(injected_df))
    
    # Plot example frequency window
    pipeline.plot_frequency_window(results, freq_start=0.004)
    
    return results


if __name__ == '__main__':
    results = main()
