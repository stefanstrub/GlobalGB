"""
Core search utilities for Galactic-binary (GB) analysis.

This module collects the low-level building blocks that are used by the
high-level runner in ``GB_runner.py``:

- definitions of the GB parameter ordering and convenient index maps,
- transformations between physical parameter space and the unit hypercube,
- construction of frequency windows tailored to the expected signal bandwidth,
- the :class:`GB_Searcher` class which provides likelihood and SNR evaluations
  and several optimisation back-ends for a single frequency window,
- the :class:`Segment_GB_Searcher` class which repeatedly applies
  :class:`GB_Searcher` and performs subtraction and global re-optimisation
  within a window, and
- basic plotting helpers for diagnostic visualisation.

Most public functions and classes are documented with NumPy-style docstrings
so they can be used directly in analysis scripts or from interactive sessions.
"""

from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy
import numpy as np
import time
from copy import deepcopy
import jax.numpy as jnp
import jax
from jaxgb.jaxgb import JaxGB

from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from eryn.moves import GaussianMove, StretchMove, CombineMove
import corner

# from ldc.lisa.noise import get_noise_model

from .GB_boundaries import boundaries_dict

# customized settings
plot_parameter = {  # 'backend': 'ps',
    "font.family": "DeJavu Serif",
    "font.serif": "Times",
    "mathtext.fontset": "cm",
    "font.size": 16,
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
fig_size = [fig_width*2, fig_height]
fig_size_squared = [fig_width, fig_width]
rcParams.update({"figure.figsize": fig_size})
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
# Parameter indices for array-based pGB representation
# Order: [Amplitude, Declination, RightAscension, Frequency, FrequencyDerivative, Inclination, InitialPhase, Polarization]
PARAM_NAMES = [
    "Frequency",
    "FrequencyDerivative",
    "Amplitude",
    "RightAscension",
    "Declination",
    "Polarization",
    "Inclination",
    "InitialPhase",
]

PARAM_INDICES = {
    "Amplitude": PARAM_NAMES.index("Amplitude"),
    "Declination": PARAM_NAMES.index("Declination"),
    "RightAscension": PARAM_NAMES.index("RightAscension"),
    "Frequency": PARAM_NAMES.index("Frequency"),
    "FrequencyDerivative": PARAM_NAMES.index("FrequencyDerivative"),
    "Inclination": PARAM_NAMES.index("Inclination"),
    "InitialPhase": PARAM_NAMES.index("InitialPhase"),
    "Polarization": PARAM_NAMES.index("Polarization"),
}


N_PARAMS = len(PARAM_NAMES)
N_PARAMS_NO_AMP = N_PARAMS - 1  # Without Amplitude
PARAM_NAMES_NO_AMP = [p for p in PARAM_NAMES if p != 'Amplitude']
PARAM_LOG_UNIFORM = ['Amplitude']  # Default log-uniform parameters


@dataclass
class GBConfig:
    """Configuration for the GB matching pipeline."""
    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)


def scaletooriginal_jax(previous_max, boundaries):
    """
    JAX version of `scaletooriginal` for differentiable objectives.

    Parameters
    ----------
    previous_max : array-like, shape (8,)
        Parameters scaled to [0, 1] in the PARAM_NAMES order.
    boundaries : array-like, shape (8, 2) or dict
        Parameter boundaries, consistent with the conventions used in `GB_Searcher`.
        - Declination boundaries are on sin(dec)
        - Inclination boundaries are on cos(inc)
        - Amplitude boundaries are on log10(amp)
    """
    previous_max = jnp.asarray(previous_max)
    if isinstance(boundaries, dict):
        bounds_arr = jnp.asarray([boundaries[p] for p in PARAM_NAMES])
    else:
        bounds_arr = jnp.asarray(boundaries)

    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    scaled = previous_max * (upper - lower) + lower

    out = scaled
    out = out.at[PARAM_INDICES["Declination"]].set(
        jnp.arcsin(scaled[PARAM_INDICES["Declination"]])
    )
    out = out.at[PARAM_INDICES["Inclination"]].set(
        jnp.arccos(scaled[PARAM_INDICES["Inclination"]])
    )
    if "Amplitude" in PARAM_LOG_UNIFORM:
        out = out.at[PARAM_INDICES["Amplitude"]].set(
            jnp.power(10.0, scaled[PARAM_INDICES["Amplitude"]])
        )
    return out

def frequency_derivative(f: float, Mc: float) -> float:
    """Computes the frequency derivative based on chirp mass Mc."""
    G, c = 6.674e-11, 3e8
    Mc_s = Mc * 2e30 * G / c**3
    return 96 / (5 * np.pi * Mc_s**2) * (np.pi * Mc_s * f) ** (11 / 3)

def frequency_derivative_tyson_upper(f):
    return 8*10**-7*f**(11/3)

def frequency_derivative_tyson_lower(f):
    return -5*10**-6*f**(13/3)

def scaletooriginal(previous_max, boundaries, parameters=None):
    """Scales the parameters back to their original values.
    previous_max: array of parameters in [0,1] range (shape: (8,) or (n_signals, 8))
    boundaries: array of boundaries (shape: (8, 2)) or dict
    Returns: array of original parameter values
    """
    previous_max = np.atleast_1d(previous_max)
    if previous_max.ndim == 1:
        previous_max = previous_max.reshape(1, -1)
    
    # Convert dict boundaries to array if needed
    if isinstance(boundaries, dict):
        bounds_arr = np.array([boundaries[p] for p in PARAM_NAMES])
    else:
        bounds_arr = np.asarray(boundaries)
    
    maxpGB = np.zeros_like(previous_max)
    log_uniform_indices = [PARAM_INDICES[p] for p in PARAM_LOG_UNIFORM if p in PARAM_INDICES]
    
    for i, param in enumerate(PARAM_NAMES):
        lower, upper = bounds_arr[i]
        scaled = previous_max[:, i] * (upper - lower) + lower
        
        if param == "Declination":
            maxpGB[:, i] = np.arcsin(scaled)
        elif param == "Inclination":
            maxpGB[:, i] = np.arccos(scaled)
        elif i in log_uniform_indices:
            maxpGB[:, i] = 10**scaled
        else:
            maxpGB[:, i] = scaled
    
    return maxpGB.squeeze()


def transform_sample_to_input_space(sample):
    """
    Transform the sample to the input space
    """
    # Track if input is 1D
    was_1d = sample.ndim == 1

    # Convert to 2D for consistent processing
    sample_2d = sample.reshape(1, -1) if was_1d else sample
    sample_copy = np.copy(sample_2d)

    sample_copy[:, 0] = 10 ** sample_copy[:, 0]
    sample_copy[:, 1] = 10 ** sample_copy[:, 1]

    sample_copy[:, 7] = np.arccos(sample_copy[:, 7])
    sample_copy[:, 9] = np.arccos(sample_copy[:, 9])

    # Return in original shape
    return sample_copy.reshape(-1) if was_1d else sample_copy

def transform_input_to_sample_space(input):
    """
    Transform the input to the sample space
    """
    # Store original shape info
    was_1d = input.ndim == 1
    # Reshape to 2D for consistent processing
    input_2d = input.reshape(1, -1) if was_1d else input
    input_copy = np.copy(input_2d)

    # Apply transformations safely
    if input_copy.shape[1] > 0:
        input_copy[:, 0] = np.log10(input_copy[:, 0])
    if input_copy.shape[1] > 1:
        input_copy[:, 1] = np.log10(input_copy[:, 1])
    # Optional alternative:
    # input_copy[:, 1] = np.log10(input_copy[:, 1] / input_copy[:, 0])

    if input_copy.shape[1] > 7:
        input_copy[:, 7] = np.cos(input_copy[:, 7])
    if input_copy.shape[1] > 9:
        input_copy[:, 9] = np.cos(input_copy[:, 9])

    # Return to original shape if input was 1D
    return input_copy.reshape(-1) if was_1d else input_copy

def transform_parameters_to_01(params, boundaries):
    """
    Transform parameters from the prior space to unit cube
    """
    # Track if input is 1D
    was_1d = params.ndim == 1

    # Convert to 2D for consistent processing
    params_2d = params.reshape(1, -1) if was_1d else params
    # Ensure params has the same shape as boundaries along axis 0
    if params_2d.shape[1] != boundaries.shape[0] and params_2d.shape[0] == boundaries.shape[0]:
        params_2d = params_2d.T
    params01 = (params_2d - boundaries[:, 0]) / (boundaries[:, 1] - boundaries[:, 0])
    return params01.reshape(-1) if was_1d else params01


def transform_parameters_from_01(params01, boundaries):
    """
    Transform parameters from the unit cube to the prior space
    """
    # Track if input is 1D
    was_1d = params01.ndim == 1
    # Convert to 2D for consistent processing
    params01_2d = params01.reshape(1, -1) if was_1d else params01
    # Ensure params01 has the same shape as boundaries along axis 0
    if params01_2d.shape[1] != boundaries.shape[0] and params01_2d.shape[0] == boundaries.shape[0]:
        params01_2d = params01_2d.T
    params = params01_2d * (boundaries[:, 1] - boundaries[:, 0]) + boundaries[:, 0]
    return params.reshape(-1) if was_1d else params

def scaleto01(params, boundaries, parameters=None):
    """Scales the parameters to the range [0, 1].
    params: array of original parameter values (shape: (8,) or (n_signals, 8))
    boundaries: array of boundaries (shape: (8, 2)) or dict
    parameters_log_uniform: list of parameters that are log-uniform
    Returns: array of parameters scaled to [0, 1]
    """
    params = np.atleast_1d(params)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    
    # Convert dict boundaries to array if needed
    if isinstance(boundaries, dict):
        bounds_arr = np.array([boundaries[p] for p in PARAM_NAMES])
    else:
        bounds_arr = np.asarray(boundaries)
    
    params01 = np.zeros_like(params)
    log_uniform_indices = [PARAM_INDICES[p] for p in PARAM_LOG_UNIFORM if p in PARAM_INDICES]
    
    for i, param in enumerate(PARAM_NAMES):
        lower, upper = bounds_arr[i]
        
        if param == "Declination":
            transformed = np.sin(params[:, i])
        elif param == "Inclination":
            transformed = np.cos(params[:, i])
        elif i in log_uniform_indices:
            transformed = np.log10(params[:, i])
        else:
            transformed = params[:, i]
        
        params01[:, i] = (transformed - lower) / (upper - lower)
    
    return params01.squeeze()

def reduce_boundaries(maxpGB, boundaries, parameters=None, ratio=0.1):
    """Reduces the boundaries of the parameters.
    maxpGB: array of parameter values (shape: (8,))
    boundaries: array of boundaries (shape: (8, 2)) or dict
    ratio: ratio of the boundaries
    Returns: array of reduced boundaries (shape: (8, 2))
    """
    maxpGB = np.atleast_1d(maxpGB)
    
    # Convert dict boundaries to array if needed
    if isinstance(boundaries, dict):
        bounds_arr = np.array([boundaries[p] for p in PARAM_NAMES])
    else:
        bounds_arr = np.asarray(boundaries)
    
    boundaries_reduced = bounds_arr.copy()
    
    for i, param in enumerate(PARAM_NAMES):
        length = bounds_arr[i, 1] - bounds_arr[i, 0]
        half_width = length * ratio / 2
        
        if param == "Declination":
            center = np.sin(maxpGB[i])
        elif param == "Inclination":
            center = np.cos(maxpGB[i])
        elif param == "Amplitude":
            center = np.log10(maxpGB[i])
        else:
            center = maxpGB[i]
        
        boundaries_reduced[i, 0] = max(center - half_width, bounds_arr[i, 0])
        boundaries_reduced[i, 1] = min(center + half_width, bounds_arr[i, 1])
    
    return boundaries_reduced
    
def max_signal_bandwidth(frequency, Tobs, chandrasekhar_limit=1.4):
    """Computes the maximum signal bandwidth.
    frequency: frequency of the signal
    Tobs: observation time
    chandrasekhar_limit: Chandrasekhar limit
    """
    M_chirp_upper_boundary = (chandrasekhar_limit**2)**(3/5)/(2*chandrasekhar_limit)**(1/5)
    f_smear = frequency *2* 10**-4
    f_deviation = frequency_derivative(frequency, M_chirp_upper_boundary)*Tobs
    # window_length = np.max([f_smear, f_deviation])
    window_length = f_smear + f_deviation
    window_length += 4/31536000*2
    return window_length

def create_frequency_windows(search_range, Tobs, chandrasekhar_limit=1.4):
    """Creates frequency windows.
    search_range: search range
    Tobs: observation time
    chandrasekhar_limit: Chandrasekhar limit
    """
    frequencies = []
    current_frequency = search_range[0]
    while current_frequency < search_range[1]:
        window_length = max_signal_bandwidth(current_frequency, Tobs, chandrasekhar_limit)
        upper_limit = current_frequency+window_length*2
        frequencies.append([current_frequency, upper_limit])
        current_frequency = deepcopy(upper_limit)
    return frequencies

class GB_Searcher:
    """
    Single-window search and likelihood engine for Galactic binaries.

    This class provides a uniform interface to:

    - extract the relevant TDI data for a given frequency window,
    - construct (or accept) a waveform generator,
    - estimate the local noise power spectral density,
    - evaluate log-likelihoods and signal-to-noise ratios (SNRs) for one or
      more GB parameter vectors, and
    - run various optimisation back-ends (differential evolution, SLSQP,
      Langevin-based search, amplitude-only optimisation).

    Parameters
    ----------
    tdi_fs
        TDI data in the frequency domain.  For the default path this is an
        :class:`xarray.Dataset` with at least the channels specified by
        ``channel_combination`` and a coordinate ``'freq'``.
    Tobs
        Observation time in seconds.
    lower_frequency, upper_frequency
        Bounds of the frequency window (in Hz) to be analysed.
    waveform_args
        Dictionary containing the arguments required by the waveform generator,
        typically including keys such as ``'orbits'``, ``'Tobs'``,
        ``'t0'`` and ``'tdi_generation'``.
    dt
        Sampling interval in seconds.
    get_tdi, get_kmin
        Optional callables that override the waveform generation and the
        mapping from starting GW frequency to FFT index.  If not provided,
        suitable JAX-accelerated versions from :mod:`jaxgb` are constructed.
    channel_combination
        String selecting which TDI channels to analyse.  The default
        ``"AET"`` expects the pre-combined A/E/T channels; ``"XYZ"`` will
        construct them from X/Y/Z.
    noise_model
        Optional external noise model.  If ``None`` the noise is estimated
        directly from the data in this window.
    recombination
        Differential evolution recombination parameter passed through to
        :func:`scipy.optimize.differential_evolution`.
    update_noise
        If ``True`` (default), estimate the noise from the data in this window.
    """

    def __init__(
        self,
        tdi_fs,
        Tobs,
        lower_frequency,
        upper_frequency,
        waveform_args,
        dt,
        get_tdi=None,
        get_kmin=None,
        channel_combination="AET",
        noise_model=None,
        recombination=0.75,
        update_noise=True,
    ):
        # Use module-level parameter definitions
        self.intrinsic_parameters = ['Declination', 'RightAscension', 'Frequency', 'FrequencyDerivative']
        self.N_samples = int(Tobs/dt)
        self.channel_combination = channel_combination
        self.tdi_fs = tdi_fs
        self.freq = tdi_fs['freq']
        self.Tobs = Tobs
        self.dt = dt
        self.fs = 1/dt
        self.waveform_args = waveform_args
        self.reduced_frequency_boundaries = None
        self.recombination = recombination
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        chandrasekhar_limit = 1.4
        M_chirp_upper_boundary = (chandrasekhar_limit**2)**(3/5)/(2*chandrasekhar_limit)**(1/5)
        self.padding = max_signal_bandwidth(lower_frequency, Tobs, chandrasekhar_limit)/2
        self.frequency_T_threshold = 19.1*10**-3/2


        self.use_T_component = False
        if self.upper_frequency + self.padding > self.frequency_T_threshold:
            self.use_T_component = True

        frequencyrange =  [lower_frequency-self.padding, upper_frequency+self.padding]
        frequency_boundaries =  [lower_frequency-self.padding/2, upper_frequency+self.padding/2]
            
        self.indexes = np.logical_and(self.freq > frequencyrange[0], self.freq < frequencyrange[1]) 

        if self.channel_combination == 'XYZ':
            self.dataA = (self.tdi_fs['Z'] - self.tdi_fs['X'])/np.sqrt(2.0)[self.indexes]
            self.dataE = (self.tdi_fs['Z'] - 2.0*self.tdi_fs['Y'] + self.tdi_fs['X'])/np.sqrt(6.0)[self.indexes]
            self.dataT = (self.tdi_fs['Z'] + self.tdi_fs['Y'] + self.tdi_fs['X'])/np.sqrt(3.0)[self.indexes]
        else:
            self.dataA = self.tdi_fs['A'][self.indexes]
            self.dataE = self.tdi_fs['E'][self.indexes]
            self.dataT = self.tdi_fs['T'][self.indexes]

        self.freq = self.freq[self.indexes]
        fmin, fmax = self.freq[0], self.freq[-1]

        self.df = self.freq[1] - self.freq[0]
        if noise_model is None:
            update_noise = True
        if update_noise:
            self.update_noise()


        # if get_tdi and get_kmin are not provided, use jaxgb to generate the waveforms
        if get_tdi is not None:
            self.get_tdi = get_tdi
        else:
            number_frequency_bins = len(self.freq)
            # set N to the next power of 2 greater than number_frequency_bins, but maximum 2**11 and minimum 2**6
            self.N = np.min([int(2**(np.ceil(np.log2(number_frequency_bins)))), 2**11])
            self.N = np.max([self.N, 2**6])
            self.fgb = JaxGB(orbits=waveform_args['orbits'],  t_obs=waveform_args['Tobs'], t0=waveform_args['t0'], n=self.N)
            # Create JIT-compiled version
            @jax.jit
            def get_tdi_jit(params, tdi_generation=waveform_args['tdi_generation'], tdi_combination=channel_combination):
                return self.fgb.get_tdi(params, tdi_generation=tdi_generation, tdi_combination=tdi_combination)
            self.get_tdi = get_tdi_jit

        if get_kmin is not None:
            self.get_kmin = get_kmin
        else:
            self.get_kmin = self.fgb.get_kmin

        @jax.jit
        def from_01toSNR_jit(params):
            return self.from01toSNR_jax(params)
        self.from01toSNR = from_01toSNR_jit
            
        f_0 = fmin
        f_transfer = 19.1*10**-3
        snr = 7
        amplitude_lower = 2*snr/(Tobs * np.sin(f_0/ f_transfer)**2/self.SA[0])**0.5
        snr = 1000
        amplitude_upper = 2*snr/(Tobs * np.sin(f_0/ f_transfer)**2/self.SA[0])**0.5
        amplitude = [amplitude_lower, amplitude_upper]
        fd_range = [frequency_derivative_tyson_lower(lower_frequency),frequency_derivative(upper_frequency,M_chirp_upper_boundary)]

        self.boundaries = deepcopy(boundaries_dict)
        if 'Frequency' not in self.boundaries.keys():
            self.boundaries['Frequency'] = frequency_boundaries
        if 'FrequencyDerivative' not in self.boundaries.keys():
            self.boundaries['FrequencyDerivative'] = fd_range
        if 'Amplitude' not in self.boundaries.keys():
            self.boundaries['Amplitude'] =  [np.log10(amplitude[0]),np.log10(amplitude[1])]

        if self.boundaries['FrequencyDerivative'][0] > self.boundaries['FrequencyDerivative'][1]:
            c = self.boundaries['FrequencyDerivative'][0]
            self.boundaries['FrequencyDerivative'][0] = self.boundaries['FrequencyDerivative'][1]
            self.boundaries['FrequencyDerivative'][1] = c
        
        # Convert boundaries dict to array format
        self.boundaries_arr = np.array([self.boundaries[p] for p in PARAM_NAMES])
        self.boundaries_reduced = self.boundaries_arr.copy()
        
        # Initialize random pGBs as array (shape: (8,))
        previous_max = np.random.rand(N_PARAMS)
        self.pGBs = scaletooriginal(previous_max, self.boundaries_arr)


        start = time.time()
        self.from01toSNR(np.array([0.5]*N_PARAMS_NO_AMP))
        print('time from01toSNR', time.time()-start)
        start = time.time()
        self.from01toSNR(np.array([0.5]*N_PARAMS_NO_AMP))
        print('time from01toSNR 2', time.time()-start)


    def update_noise(self, pGB=None):
        """Updates the noise within the frequency window.
        pGB: gravitational wave parameters array (shape: (8,))
        """ 
        if pGB is not None:
            pGB = np.atleast_1d(pGB)
            if pGB.ndim == 1:
                pGB = pGB.reshape(1, -1)
            Xs, Ys, Zs = self.get_tdi(pGB[0])
            Xs_total = self.align_waveform_to_data(Xs, pGB[0])
            Ys_total = self.align_waveform_to_data(Ys, pGB[0])
            Zs_total = self.align_waveform_to_data(Zs, pGB[0])

            Af = (Zs_total - Xs_total)/np.sqrt(2.0)
            Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
            Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
        else:
            Af = 0
            Ef = 0
            Tf = 0
        
        self.SA = np.ones(len(self.dataA))*np.median((np.abs(self.dataA-Af)/self.dt)**2/(self.fs*self.N_samples)*2)
        self.SE = np.ones(len(self.dataE))*np.median((np.abs(self.dataE-Ef)/self.dt)**2/(self.fs*self.N_samples)*2)
        self.ST = np.ones(len(self.dataT))*np.median((np.abs(self.dataT-Tf)/self.dt)**2/(self.fs*self.N_samples)*2)


    def align_waveform_to_data_jax(self, waveform, pGB):
        """Aligns a JAX waveform array to the data frequency grid.
        waveform: JAX array from get_tdi (1D array at specific frequency bins)
        pGB: parameter array to get the starting frequency
        """
        f0 = pGB[PARAM_INDICES['Frequency']]
        kmin = self.get_kmin(f0)

        n_freq = len(self.freq)
        waveform_len = len(waveform)

        # Compute the frequency index offset between data and waveform
        # self.freq[0] corresponds to some frequency, kmin*df is the waveform start
        wf_start_freq = kmin * self.df
        data_start_freq = float(self.freq[0])
        start_idx = jnp.round(jnp.divide(wf_start_freq - data_start_freq, self.df))
        start_idx = start_idx.astype(jnp.int32)
        # Clip to valid range
        wf_start = jnp.maximum(0, -start_idx)
        data_start = jnp.maximum(0, start_idx)
        wf_end = jnp.minimum(waveform_len, n_freq - start_idx)
        data_end = jnp.minimum(n_freq, start_idx + waveform_len)
        aligned = jnp.zeros(n_freq, dtype=jnp.complex128)

        # Use index-based approach compatible with JAX tracing
        # Create index arrays for both source (waveform) and destination (aligned)
        data_indices = jnp.arange(n_freq)
        wf_indices = data_indices - start_idx

        # Create mask for valid indices
        valid_mask = (wf_indices >= 0) & (wf_indices < waveform_len)

        # Clip waveform indices to valid range for safe indexing
        safe_wf_indices = jnp.clip(wf_indices, 0, waveform_len - 1)

        # Use where to conditionally copy values
        aligned = jnp.where(valid_mask, waveform[safe_wf_indices], aligned)
        return aligned

    def align_waveform_to_data(self, waveform, pGB):
        """Aligns a numpy waveform array to the data frequency grid.
        waveform: numpy array from get_tdi (1D array at specific frequency bins)
        pGB: parameter array to get the starting frequency
        """
        f0 = pGB[PARAM_INDICES['Frequency']]
        kmin = self.get_kmin(f0)
        n_freq = len(self.freq)
        waveform_len = len(waveform)
        # Compute the frequency index offset between data and waveform
        # self.freq[0] corresponds to some frequency, kmin*df is the waveform start
        wf_start_freq = kmin * self.df
        data_start_freq = float(self.freq[0])

        # Index in data array where waveform should start
        start_idx = int(np.round((wf_start_freq - data_start_freq) / self.df))

        # Clip to valid range
        wf_start = max(0, -start_idx)
        data_start = max(0, start_idx)
        wf_end = min(waveform_len, n_freq - start_idx)
        data_end = min(n_freq, start_idx + waveform_len)
        aligned = np.zeros(n_freq, dtype=np.complex128)
        if data_start < n_freq and data_end > 0 and wf_start < waveform_len:
            copy_len = min(data_end - data_start, wf_end - wf_start)
            aligned[data_start:data_start + copy_len] = np.asarray(
                waveform[wf_start:wf_start + copy_len]
            )
        return aligned
    
    def scalarproduct(self, a, b):
        """Computes the scalar product of two signals.
        a: signal 1
        b: signal 2
        """
        diff = np.real(a[0] * np.conjugate(b[0])) ** 2 + np.real(a[1] * np.conjugate(b[1])) ** 2 + np.real(a[2] * np.conjugate(b[2])) ** 2
        res = 4*float(np.sum(diff / self.Sn) * self.df)
        return res


    def get_dh_hh(self, pGBs):
        """Computes the differential and harmonic components of the signal.
        pGBs: array of gravitational wave parameters (shape: (n_signals, 8) or (8,))
        """
        pGBs = np.atleast_2d(pGBs)
        
        # Accumulate aligned waveforms
        self.Af = np.zeros(len(self.freq), dtype=np.complex128)
        self.Ef = np.zeros(len(self.freq), dtype=np.complex128)
        self.Tf = np.zeros(len(self.freq), dtype=np.complex128) if self.use_T_component else None
        
        for i in range(len(pGBs)):
            As, Es, Ts = self.get_tdi(pGBs[i])
            self.Af += self.align_waveform_to_data(As, pGBs[i])
            self.Ef += self.align_waveform_to_data(Es, pGBs[i])
            if self.use_T_component:
                self.Tf += self.align_waveform_to_data(Ts, pGBs[i])

        # Compute differential and harmonic components
        dh = np.sum(np.real(self.dataA * np.conj(self.Af)) / self.SA)
        hh = np.sum((np.abs(self.Af)**2) / self.SA)
        dh += np.sum(np.real(self.dataE * np.conj(self.Ef)) / self.SE)
        hh += np.sum((np.abs(self.Ef)**2) / self.SE)
        
        if self.use_T_component:
            dh += np.sum(np.real(self.dataT * np.conj(self.Tf)) / self.ST)
            hh += np.sum(np.abs(self.Tf)**2 / self.ST)

        dh *= 4.0 * self.df
        hh *= 4.0 * self.df
        return dh, hh

    def get_dh_hh_jax(self, pGBs):
        """Computes the differential and harmonic components of the signal using JAX.
        pGBs: array of gravitational wave parameters (shape: (n_signals, 8) or (8,))
        """
        pGBs = jnp.atleast_2d(pGBs)
        
        # Accumulate aligned waveforms
        self.Af = jnp.zeros(len(self.freq), dtype=jnp.complex128)
        self.Ef = jnp.zeros(len(self.freq), dtype=jnp.complex128)
        self.Tf = jnp.zeros(len(self.freq), dtype=jnp.complex128) if self.use_T_component else None
        
        for i in range(len(pGBs)):
            As, Es, Ts = self.get_tdi(pGBs[i])
            self.Af += self.align_waveform_to_data_jax(As, pGBs[i])
            self.Ef += self.align_waveform_to_data_jax(Es, pGBs[i])
            if self.use_T_component:
                self.Tf += self.align_waveform_to_data_jax(Ts, pGBs[i])

        # Compute differential and harmonic components
        dh = jnp.sum(jnp.real(self.dataA * jnp.conj(self.Af)) / self.SA)
        hh = jnp.sum((jnp.abs(self.Af)**2) / self.SA)
        dh += jnp.sum(jnp.real(self.dataE * jnp.conj(self.Ef)) / self.SE)
        hh += jnp.sum((jnp.abs(self.Ef)**2) / self.SE)
        
        if self.use_T_component:
            dh += jnp.sum(jnp.real(self.dataT * jnp.conj(self.Tf)) / self.ST)
            hh += jnp.sum(jnp.abs(self.Tf)**2 / self.ST)

        dh *= 4.0 * self.df
        hh *= 4.0 * self.df
        return dh, hh


    def SNR(self, pGBs):
        """Computes the signal-to-noise ratio.
        pGBs: array of gravitational wave parameters (shape: (n_signals, 8) or (8,))
        """
        dh, hh = self.get_dh_hh(pGBs)
        SNR_values = dh / np.sqrt(hh)

        return SNR_values

    def SNR_jax(self, pGBs):
        """Computes the signal-to-noise ratio using JAX.
        pGBs: array of gravitational wave parameters (shape: (n_signals, 8) or (8,))
        """
        dh, hh = self.get_dh_hh_jax(pGBs)
        SNR_values = dh / jnp.sqrt(hh)

        return SNR_values


    def loglikelihood(self, pGBs):
        """Computes the log-likelihood.
        pGBs: array of gravitational wave parameters (shape: (n_signals, 8) or (8,))
        """
        dh, hh = self.get_dh_hh(pGBs)
        logliks = dh - 0.5 * hh
        return logliks


    def maladynamic_search_from01toSNR_jax(
        self,
        frequency_boundaries=None,
        initial_guess=None,
        n_steps: int = 2000,
        step_size: float = 0.02,
        beta: float = 1.0,
        target_accept: float = 0.57,
        adapt_steps: int = 600,
        adapt_interval: int = 50,
        seed: int = 0,
        max_candidates_history: int = 0,
        optimize_amplitude: bool = True,
    ):
        """
        MALA (Metropolis-adjusted Langevin algorithm) "dynamic search" that uses JAX gradients
        of `from01toSNR_jax` (objective is -SNR) in the 0-1 parameterization (without amplitude).

        This is intended as a stochastic optimizer: it tracks and returns the best (highest SNR)
        state encountered, not a converged posterior sample.

        Parameters
        ----------
        frequency_boundaries : list[float] | None
            Optional [f_low, f_high] to clamp the frequency boundaries for this window.
            If None, uses current `self.boundaries_reduced` if set, else `self.boundaries_arr`.
        initial_guess : array-like | None
            Initial guess in ORIGINAL parameter space (shape (8,)). If None, starts from random.
        n_steps : int
            Number of MALA iterations.
        step_size : float
            Initial Langevin step size (epsilon).
        beta : float
            Inverse-temperature multiplier. Larger beta makes the chain greedier for high SNR.
        target_accept : float
            Target acceptance rate for simple step size adaptation.
        adapt_steps : int
            Number of initial steps during which to adapt step_size.
        adapt_interval : int
            How often (in steps) to update step_size during adaptation.
        seed : int
            PRNG seed.
        max_candidates_history : int
            If > 0, returns a dict with short histories (SNR, accept flags, step sizes).
        optimize_amplitude : bool
            If True, analytically rescales amplitude for the best sample (via `calculate_Amplitude`).

        Returns
        -------
        best_pGB : np.ndarray, shape (8,)
            Best parameters in original scale.
        best_snr : float
            Best SNR value achieved by `self.SNR(best_pGB)`.
        info : dict
            Optional diagnostics (present if max_candidates_history > 0).
        """
        # Prepare boundaries
        if getattr(self, "boundaries_reduced", None) is None:
            self.boundaries_reduced = self.boundaries_arr.copy()
        if frequency_boundaries is not None:
            self.boundaries_reduced = np.asarray(self.boundaries_reduced).copy()
            self.boundaries_reduced[PARAM_INDICES["Frequency"]] = frequency_boundaries

        # Initialize x0 in 0-1 without amplitude
        if initial_guess is not None:
            p01 = scaleto01(np.asarray(initial_guess), self.boundaries_reduced)
            x0 = np.delete(p01, PARAM_INDICES["Amplitude"])
            x0 = np.clip(x0, 1e-6, 1.0 - 1e-6)
        else:
            x0 = np.random.rand(N_PARAMS_NO_AMP)
            x0 = np.clip(x0, 1e-6, 1.0 - 1e-6)

        x = jnp.asarray(x0)
        bounds_arr = jnp.asarray(self.boundaries_reduced)
        amp_idx = int(PARAM_INDICES["Amplitude"])
        mask_no_amp = np.array([i for i in range(N_PARAMS) if i != amp_idx], dtype=int)

        def reflect01(z):
            # Reflect into [0,1] with period 2
            z2 = jnp.mod(z, 2.0)
            return jnp.where(z2 > 1.0, 2.0 - z2, z2)

        def x_no_amp_to_pGB(x_no_amp):
            # x_no_amp: (7,) -> p01_full: (8,)
            p01_full = jnp.zeros((N_PARAMS,))
            p01_full = p01_full.at[mask_no_amp].set(x_no_amp)
            p01_full = p01_full.at[amp_idx].set(0.5)
            return scaletooriginal_jax(p01_full, bounds_arr)

        def snr_of_x(x_no_amp):
            pGB = x_no_amp_to_pGB(x_no_amp)
            return self.SNR_jax(pGB[None, :]).squeeze()

        def logpi(x_no_amp):
            # logpi ~ beta * SNR
            return beta * snr_of_x(x_no_amp)

        grad_logpi = jax.grad(logpi)

        key = jax.random.PRNGKey(seed)
        eps = float(step_size)

        # Track best (in numpy space for downstream non-jax code)
        best_x = x
        best_snr = float(snr_of_x(x))

        history = {
            "snr": [],
            "accepted": [],
            "eps": [],
        }

        def log_q(y, x_curr, g_curr, eps_local):
            mu = x_curr + 0.5 * (eps_local**2) * g_curr
            diff = y - mu
            return -0.5 * jnp.sum((diff / eps_local) ** 2)

        accepted_count = 0
        for t in range(n_steps):
            key, k1, k2 = jax.random.split(key, 3)

            g = grad_logpi(x)
            mu = x + 0.5 * (eps**2) * g
            y = mu + eps * jax.random.normal(k1, shape=x.shape)
            y = reflect01(y)
            y = jnp.clip(y, 1e-6, 1.0 - 1e-6)

            lp_x = logpi(x)
            lp_y = logpi(y)
            g_y = grad_logpi(y)

            log_alpha = (lp_y - lp_x) + (log_q(x, y, g_y, eps) - log_q(y, x, g, eps))
            u = jnp.log(jax.random.uniform(k2, shape=()))
            accept = u < log_alpha
            print('g', g)
            print('x', x)
            print('y', y)
            print('log_alpha', log_alpha)
            print('u', u)
            print('accept', accept)
            x = jnp.where(accept, y, x)
            accepted_count += int(bool(accept))

            snr_val = float(snr_of_x(x))
            print('snr_val', snr_val)
            if snr_val > best_snr:
                best_snr = snr_val
                best_x = x

            history["snr"].append(snr_val)
            history["accepted"].append(bool(accept))
            history["eps"].append(eps)

            # Simple step size adaptation (Robbins-Monro on log-eps)
            if (t + 1) <= adapt_steps and (t + 1) % adapt_interval == 0:
                acc_rate = accepted_count / float(t + 1)
                # conservative adaptation rate
                eta = 0.05
                eps *= float(np.exp(eta * (acc_rate - target_accept)))
                eps = float(np.clip(eps, 1e-4, 0.25))

        best_pGB_jax = x_no_amp_to_pGB(best_x)
        best_pGB = np.asarray(best_pGB_jax)

        if optimize_amplitude:
            try:
                A_factor = self.calculate_Amplitude(best_pGB[None, :])
                best_pGB[PARAM_INDICES["Amplitude"]] *= float(np.asarray(A_factor))
            except Exception:
                # keep amplitude as-is if analytic rescale fails
                pass

        best_snr_final = float(np.asarray(self.SNR(best_pGB[None, :])).squeeze())

        info = {
            "accept_rate": accepted_count / float(max(1, n_steps)),
            "final_step_size": eps,
        }
        if history is not None:
            info["history"] = history

        return best_pGB, best_snr_final, info


    def differential_evolution_search(self, frequency_boundaries, initial_guess=None, number_of_signals=1):
        """Performs a differential evolution search.
        frequency_boundaries: boundaries of the frequency
        initial_guess: initial guess array (shape: (n_signals, 8) or (8,))
        number_of_signals: number of signals
        Returns: array of optimized parameters (shape: (n_signals, 8)), n_function_evals
        """
        bounds = [(0, 1) for _ in range(N_PARAMS_NO_AMP * number_of_signals)]
        
        # Set up reduced boundaries with frequency constraint
        self.boundaries_reduced = self.boundaries_arr.copy()
        self.boundaries_reduced[PARAM_INDICES['Frequency']] = frequency_boundaries
        
        if initial_guess is not None:
            initial_guess = np.atleast_2d(initial_guess)
            initial_guess01 = np.zeros(N_PARAMS_NO_AMP * number_of_signals)
            for signal in range(number_of_signals):
                pGBstart01 = scaleto01(initial_guess[signal], self.boundaries_reduced)
                # Extract non-amplitude parameters (indices 1-7)
                pGBstart01_no_amp = pGBstart01[1:]
                pGBstart01_no_amp = np.clip(pGBstart01_no_amp, 0, 1)
                initial_guess01[signal*N_PARAMS_NO_AMP:(signal+1)*N_PARAMS_NO_AMP] = pGBstart01_no_amp
            start = time.time()
            res = scipy.optimize.differential_evolution(self.from01toSNR, bounds=bounds, strategy='best1exp', popsize=8, tol=1e-8, maxiter=1000, recombination=self.recombination, mutation=(0.5, 1), x0=initial_guess01)
            print('time', time.time()-start)
        else:
            start = time.time()
            res = scipy.optimize.differential_evolution(self.from01toSNR, bounds=bounds, strategy='best1exp', popsize=8, tol=1e-8, maxiter=1000, recombination=self.recombination, mutation=(0.5, 1))
            # res = scipy.optimize.shgo(self.from01toSNR, bounds=bounds)
            print('time', time.time()-start)
        
        # Convert result back to original scale
        maxpGB = np.zeros((number_of_signals, N_PARAMS))
        for signal in range(number_of_signals):
            pGB01 = np.insert(res.x, (signal+1)*PARAM_INDICES['Amplitude'], 0.5)  # Amplitude
            maxpGB[signal] = scaletooriginal(pGB01, self.boundaries_reduced)
        
        print(res)
        print(maxpGB)
        return maxpGB, res.nfev

    def optimize(self, pGBmodes, boundaries=None):
        """
        Optimizes pGB parameters using scipy.optimize.minimize with SLSQP method.
        pGBmodes: array of shape (n_signals, 8)
        boundaries: array of boundaries (shape: (8, 2)) or dict
        Returns: optimized parameters array (shape: (n_signals, 8))
        """
        print(pGBmodes)
        if boundaries is None:
            boundaries = self.boundaries_arr
        elif isinstance(boundaries, dict):
            boundaries = np.array([boundaries[p] for p in PARAM_NAMES])
        
        pGBmodes = np.atleast_2d(pGBmodes)
        if pGBmodes.ndim == 2:
            pGBmodes = pGBmodes.reshape(1, *pGBmodes.shape)
        
        n_modes = pGBmodes.shape[0]
        n_signals = pGBmodes.shape[1]
        bounds = [(0, 1)] * (N_PARAMS * n_signals)
        print(pGBmodes)
        current_best_value = None
        current_maxpGB = None
        
        for i in range(n_modes):
            maxpGB = pGBmodes[i].copy()  # Shape: (n_signals, 8)
            
            for j in range(2):
                # Set up boundaries
                if j > 0:
                    boundaries_reduced = np.array([reduce_boundaries(maxpGB[s], boundaries, ratio=0.4) for s in range(n_signals)])
                else:
                    boundaries_reduced = np.tile(boundaries, (n_signals, 1, 1))
                
                # Scale to [0, 1]
                x = []
                for signal in range(n_signals):
                    pGB01 = scaleto01(maxpGB[signal], boundaries_reduced[signal])
                    x.extend(pGB01)
                x = np.array(x)
                
                res = scipy.optimize.minimize(
                    self.from01tologlikelihood_negative, x, args=(boundaries_reduced,),
                    method='SLSQP', bounds=bounds, tol=1e-5
                )
                
                # Scale back to original
                for signal in range(n_signals):
                    maxpGB[signal] = scaletooriginal(
                        res.x[signal*N_PARAMS:(signal+1)*N_PARAMS],
                        boundaries_reduced[signal]
                    )
            
            best_value = self.loglikelihood(maxpGB)
            if current_best_value is None or best_value > current_best_value:
                current_best_value = best_value
                current_maxpGB = maxpGB.copy()
        
        return current_maxpGB

    def calculate_Amplitude(self, pGBs):
        """Calculates the amplitude of the signal analytically based on the parameters pGBs.
        pGBs: array of parameters (shape: (n_signals, 8) or (8,))
        """
        # pGBs = np.atleast_2d(pGBs)
        
        # # Accumulate aligned waveforms
        # Af = np.zeros(len(self.freq), dtype=np.complex128)
        # Ef = np.zeros(len(self.freq), dtype=np.complex128)
        
        # for i in range(len(pGBs)):
        #     As, Es, Ts = self.get_tdi(pGBs[i])
        #     Af += self.align_waveform_to_data(As, pGBs[i])
        #     Ef += self.align_waveform_to_data(Es, pGBs[i])
            
        # # Compute optimal amplitude scaling
        # dh = np.sum(np.real(self.dataA * np.conj(Af) + self.dataE * np.conj(Ef)) / self.SA)
        # hh = np.sum((np.abs(Af)**2 + np.abs(Ef)**2) / self.SA)
        dh, hh = self.get_dh_hh(pGBs)
        A_factor = dh / hh
        return A_factor


    def optimizeA(self, pGBmodes, boundaries=None):
        """
        Optimizes amplitude of pGBs using scipy.optimize.minimize with trust-constr method.
        pGBmodes: array of shape (n_modes, n_signals, 8) or (n_signals, 8)
        boundaries: array of boundaries (shape: (8, 2)) or dict
        Returns: optimized parameters array (shape: (n_signals, 8))
        """
        if boundaries is None:
            boundaries = self.boundaries_arr
        elif isinstance(boundaries, dict):
            boundaries = np.array([boundaries[p] for p in PARAM_NAMES])
        
        pGBmodes = np.atleast_2d(pGBmodes)
        if pGBmodes.ndim == 2:
            pGBmodes = pGBmodes.reshape(1, *pGBmodes.shape)
        
        n_modes = pGBmodes.shape[0]
        n_signals = pGBmodes.shape[1]
        bounds = [(0, 1)] * n_signals
        
        current_best_value = None
        current_maxpGB = None
        
        for i in range(n_modes):
            maxpGB = pGBmodes[i].copy()
            
            for j in range(2):
                # Use full boundaries
                boundaries_reduced = boundaries.copy()
                
                # Scale to [0, 1]
                pGBs01 = np.array([scaleto01(maxpGB[s], boundaries_reduced) for s in range(n_signals)])
                
                # Extract amplitude values only
                x = pGBs01[:, PARAM_INDICES['Amplitude']]
                self.pGBx = pGBs01.flatten()
                
                res = scipy.optimize.minimize(
                    self.from01tologlikelihooda, x, args=(self.pGBx, boundaries_reduced),
                    method='trust-constr', bounds=bounds, tol=1e-1
                )
                
                # Update amplitude in pGBs01 and scale back
                for signal in range(n_signals):
                    pGB01 = pGBs01[signal].copy()
                    pGB01[PARAM_INDICES['Amplitude']] = res.x[signal]
                    maxpGB[signal] = scaletooriginal(pGB01, boundaries_reduced)
            
            best_value = self.loglikelihood(maxpGB)
            if current_best_value is None or best_value > current_best_value:
                current_best_value = best_value
                current_maxpGB = maxpGB.copy()
        
        print('final optimized loglikelihood', self.loglikelihood(current_maxpGB), current_maxpGB[0, PARAM_INDICES['Frequency']])
        return current_maxpGB

    def from01tologlikelihood(self, pGBs01, boundaries=None):
        """Computes the log-likelihood from the parameters in range [0,1]
        pGBs01: flat array of parameters in [0,1] (length: n_signals * 8)
        """
        if boundaries is None:
            boundaries = self.boundaries_reduced
        pGBs01 = np.asarray(pGBs01)
        # if pGBs01 is not flattened, flatten it
        if pGBs01.ndim != 1:
            pGBs01 = pGBs01.flatten()
        n_signals = len(pGBs01) // N_PARAMS
        pGBs = np.zeros((n_signals, N_PARAMS))
        
        for signal in range(n_signals):
            pGB01 = pGBs01[signal*N_PARAMS:(signal+1)*N_PARAMS]
            # Clip inclination to [0, 1]
            pGB01 = np.clip(pGB01, 0, 1)
            pGBs[signal] = scaletooriginal(pGB01, boundaries[signal])
        
        p = self.loglikelihood(pGBs)
        return p


    def from01tologlikelihood_negative(self, pGBs01, boundaries=None):
        """Computes the log-likelihood from the parameters in range [0,1] with the amplitude and switches sign to negative
        """
        if boundaries is None:
            boundaries = self.boundaries_reduced
        return -self.from01tologlikelihood(pGBs01, boundaries)

    def from01toSNR_numpy(self, pGBs01):
        """Computes the SNR from the parameters in range [0,1] without the amplitude
        pGBs01: flat array of parameters in [0,1] (length: n_signals * 7, no amplitude)
        """
        x = np.asarray(pGBs01)
        n_signals = x.shape[0] // N_PARAMS_NO_AMP
        x = x.reshape((n_signals, N_PARAMS_NO_AMP))

        bounds_arr = np.asarray(self.boundaries_reduced)
        amp_idx = int(PARAM_INDICES["Amplitude"])
        mask_no_amp = np.asarray([i for i in range(N_PARAMS) if i != amp_idx], dtype=np.int32)

        p01_full = np.zeros((n_signals, N_PARAMS))
        p01_full[:, mask_no_amp] = x
        p01_full[:, amp_idx] = 0.5
        pGBs = scaletooriginal(p01_full, bounds_arr)
        
        p = -self.SNR(pGBs) # we want to maximize the SNR
        return p

    def from01toSNR_jax(self, pGBs01):
        """Computes the SNR from the parameters in range [0,1] without the amplitude
        pGBs01: flat array of parameters in [0,1] (length: n_signals * 7, no amplitude)
        """
        x = jnp.asarray(pGBs01)
        n_signals = x.shape[0] // N_PARAMS_NO_AMP
        x = x.reshape((n_signals, N_PARAMS_NO_AMP))

        bounds_arr = jnp.asarray(self.boundaries_reduced)
        amp_idx = int(PARAM_INDICES["Amplitude"])
        mask_no_amp = jnp.asarray([i for i in range(N_PARAMS) if i != amp_idx], dtype=jnp.int32)

        p01_full = jnp.zeros((n_signals, N_PARAMS))
        p01_full = p01_full.at[:, mask_no_amp].set(x)
        p01_full = p01_full.at[:, amp_idx].set(0.5)

        pGBs = jax.vmap(lambda p01: scaletooriginal_jax(p01, bounds_arr))(p01_full)
        return -self.SNR_jax(pGBs)

    def from01tologlikelihooda(self, amplitudes01, pGBx, boundaries_reduced):
        """Computes the log-likelihood optimizing only amplitude
        amplitudes01: array of amplitude values in [0,1] (length: n_signals)
        pGBx: flat array of all parameters in [0,1] (length: n_signals * 8)
        boundaries_reduced: array of boundaries (shape: (8, 2))
        """
        pGBx = np.asarray(pGBx)
        amplitudes01 = np.asarray(amplitudes01)
        n_signals = len(amplitudes01)
        pGBs = np.zeros((n_signals, N_PARAMS))
        
        for signal in range(n_signals):
            pGB01 = pGBx[signal*N_PARAMS:(signal+1)*N_PARAMS].copy()
            pGB01[PARAM_INDICES['Amplitude']] = amplitudes01[signal]
            pGBs[signal] = scaletooriginal(pGB01, boundaries_reduced)
        
        p = -self.loglikelihood(pGBs)
        return p

    def plot(self, pGBs=None, second_data=None):
        """Plots the pGBs in the frequency domain
        pGBs: array of parameters (shape: (n_signals, 8) or (8,))
        """
        plt.figure()
        plt.plot(self.freq, (self.dataA), label='A data')
        if second_data is not(None):
            plt.plot(second_data['A'].f, (second_data['A'].data.real), label='A data second')
        if pGBs is not(None):
            pGBs = np.atleast_2d(pGBs)
            for i in range(len(pGBs)):
                As, Es, Ts = self.get_tdi(pGBs[i])
                As_aligned = self.align_waveform_to_data(As, pGBs[i])
                plt.plot(self.freq, (As_aligned), label=f'A injected {i}')
        plt.xlim(self.lower_frequency-self.padding, self.upper_frequency+self.padding)
        plt.legend(loc='upper right')
        plt.show(block=True)
        
    def plotA_f(self, found_matched, found_not_matched, injected_matched, injected_not_matched):
        """Plots the pGBs in the frequency domain and amplitude and frequency
        found_matched: pl.DataFrame of found sources in the frequency window
        found_not_matched: pl.DataFrame of found sources not matched in the frequency window
        injected_matched: pl.DataFrame of pGBs injected matched in the frequency window
        injected_not_matched: pl.DataFrame of pGBs injected not matched in the frequency window
        """
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=np.array(fig_size)*[1,1])
        freq_plot = self.freq*10**3
        axes[0].plot(freq_plot, np.abs(self.dataA), label='Data', color='black', linewidth=2)
        for i in range(len(found_matched)):
            As, Es, Ts = self.get_tdi(found_matched[i])
            As_aligned = self.align_waveform_to_data(As, found_matched[i])
            if i == 0:
                axes[0].plot(freq_plot, np.abs(As_aligned), '--', label=f'Recovered matched', color=colors[i%7])
                axes[1].plot(found_matched[i][PARAM_INDICES['Frequency']]*10**3, found_matched[i][PARAM_INDICES['Amplitude']], 'o', label=f'Recovered matched', color=colors[i%7])
            else:
                axes[0].plot(freq_plot, np.abs(As_aligned), '--', color=colors[i%7])
                axes[1].plot(found_matched[i][PARAM_INDICES['Frequency']]*10**3, found_matched[i][PARAM_INDICES['Amplitude']], 'o', color=colors[i%7])
        for i in range(len(found_not_matched)):
            As, Es, Ts = self.get_tdi(found_not_matched[i])
            As_aligned = self.align_waveform_to_data(As, found_not_matched[i])
            if i == 0:
                axes[0].plot(freq_plot, np.abs(As_aligned), 'x-', label=f'Recovered not matched', color='gray')
                axes[1].plot(found_not_matched[i][PARAM_INDICES['Frequency']]*10**3, found_not_matched[i][PARAM_INDICES['Amplitude']], 'o', markerfacecolor="None", label=f'Recovered not matched', color='gray', markeredgewidth=2)
            else:
                axes[0].plot(freq_plot, np.abs(As_aligned), 'x-', color='gray')
                axes[1].plot(found_not_matched[i][PARAM_INDICES['Frequency']]*10**3, found_not_matched[i][PARAM_INDICES['Amplitude']], 'o', markerfacecolor="None", color='gray', markeredgewidth=2)
        for i in range(len(injected_matched)):
            As, Es, Ts = self.get_tdi(injected_matched[i])
            As_aligned = self.align_waveform_to_data(As, injected_matched[i])
            if i == 0:
                axes[0].plot(freq_plot, np.abs(As_aligned), label=f'Injected matched', color=colors[i%7], linewidth=5, alpha=0.5)
                axes[1].plot(injected_matched[i][PARAM_INDICES['Frequency']]*10**3, injected_matched[i][PARAM_INDICES['Amplitude']], '+', label=f'Injected matched', color=colors[i%7], markersize=10, markeredgewidth=2)
            else:
                axes[0].plot(freq_plot, np.abs(As_aligned), color=colors[i%7], linewidth=5, alpha=0.5)
                axes[1].plot(injected_matched[i][PARAM_INDICES['Frequency']]*10**3, injected_matched[i][PARAM_INDICES['Amplitude']], '+', color=colors[i%7], markersize=10, markeredgewidth=2)
        for i in range(len(injected_not_matched)):
            As, Es, Ts = self.get_tdi(injected_not_matched[i])
            As_aligned = self.align_waveform_to_data(As, injected_not_matched[i])
            if i == 0:
                axes[0].plot(freq_plot, np.abs(As_aligned), label=f'Injected not recovered', color='gray', linewidth=5, alpha=0.5, zorder=0)
                axes[1].plot(injected_not_matched[i][PARAM_INDICES['Frequency']]*10**3, injected_not_matched[i][PARAM_INDICES['Amplitude']], '+', label=f'Injected not recovered', color='gray', markersize=10, markeredgewidth=2, zorder=0)
            else:
                axes[0].plot(freq_plot, np.abs(As_aligned), color='gray', linewidth=5, alpha=0.5, zorder=0)
                axes[1].plot(injected_not_matched[i][PARAM_INDICES['Frequency']]*10**3, injected_not_matched[i][PARAM_INDICES['Amplitude']], '+', color='gray', markersize=10, markeredgewidth=2, zorder=0)
        axes[0].set_yscale('log')
        axes[0].set_xlim((self.lower_frequency)*10**3, (self.upper_frequency)*10**3)
        axes[0].legend(loc='lower left')
        axes[0].set_ylabel(f'|TDI A|')
        axes[1].set_yscale('log')
        axes[1].set_xlim((self.lower_frequency)*10**3, (self.upper_frequency)*10**3)
        axes[1].legend(loc='lower left')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_xlabel('Frequency (mHz)')
        # plt.ylim(1e-19, 1e-16)
        # plt.legend(False)
        fig.tight_layout()
        plt.show(block=True)
        
    def plotAE(self, found_matched, found_not_matched, injected_matched, injected_not_matched):
        """Plots the pGBs in the frequency domain
        found_matched: pl.DataFrame of found sources in the frequency window
        found_not_matched: pl.DataFrame of found sources not matched in the frequency window
        injected_matched: pl.DataFrame of pGBs injected matched in the frequency window
        injected_not_matched: pl.DataFrame of pGBs injected not matched in the frequency window
        """
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=np.array(fig_size)*[1,1])
        freq_plot = self.freq*10**3
        axes[0].plot(freq_plot, (self.dataA), label='Data', color='black', linewidth=2)
        axes[1].plot(freq_plot, (self.dataE), label='Data', color='black', linewidth=2)
        for i in range(len(found_matched)):
            As, Es, Ts = self.get_tdi(found_matched[i])
            As_aligned = self.align_waveform_to_data(As, found_matched[i])
            Es_aligned = self.align_waveform_to_data(Es, found_matched[i])
            if i == 0:
                axes[0].plot(freq_plot, (As_aligned), '--', label=f'Recovered matched', color=colors[i%7])
                axes[1].plot(freq_plot, (Es_aligned), '--', label=f'Recovered matched', color=colors[i%7])
            else:
                axes[0].plot(freq_plot, (As_aligned), '--', color=colors[i%7])
                axes[1].plot(freq_plot, (Es_aligned), '--', color=colors[i%7])
        for i in range(len(found_not_matched)):
            As, Es, Ts = self.get_tdi(found_not_matched[i])
            As_aligned = self.align_waveform_to_data(As, found_not_matched[i])
            Es_aligned = self.align_waveform_to_data(Es, found_not_matched[i])
            if i == 0:
                axes[0].plot(freq_plot, (As_aligned), 'x-', label=f'Recovered not matched', color='gray')
                axes[1].plot(freq_plot, (Es_aligned), 'x-', label=f'Recovered not matched', color='gray')
            else:
                axes[0].plot(freq_plot, (As_aligned), 'x-', color='gray')
                axes[1].plot(freq_plot, (Es_aligned), 'x-', color='gray')
        for i in range(len(injected_matched)):
            As, Es, Ts = self.get_tdi(injected_matched[i])
            As_aligned = self.align_waveform_to_data(As, injected_matched[i])
            Es_aligned = self.align_waveform_to_data(Es, injected_matched[i])
            if i == 0:
                axes[0].plot(freq_plot, (As_aligned), label=f'Injected matched', color=colors[i%7], linewidth=5, alpha=0.5)
                axes[1].plot(freq_plot, (Es_aligned), label=f'Injected matched', color=colors[i%7], linewidth=5, alpha=0.5)
            else:
                axes[0].plot(freq_plot, (As_aligned), color=colors[i%7], linewidth=5, alpha=0.5)
                axes[1].plot(freq_plot, (Es_aligned), color=colors[i%7], linewidth=5, alpha=0.5)
        for i in range(len(injected_not_matched)):
            As, Es, Ts = self.get_tdi(injected_not_matched[i])
            As_aligned = self.align_waveform_to_data(As, injected_not_matched[i])
            Es_aligned = self.align_waveform_to_data(Es, injected_not_matched[i])
            if i == 0:
                axes[0].plot(freq_plot, (As_aligned), label=f'Injected not recovered', color='gray', linewidth=5, alpha=0.5, zorder=0)
                axes[1].plot(freq_plot, (Es_aligned), label=f'Injected not recovered', color='gray', linewidth=5, alpha=0.5, zorder=0)
            else:
                axes[0].plot(freq_plot, (As_aligned), color='gray', linewidth=5, alpha=0.5, zorder=0)
                axes[1].plot(freq_plot, (Es_aligned), color='gray', linewidth=5, alpha=0.5, zorder=0)

        axes[0].set_xlim((self.lower_frequency)*10**3, (self.upper_frequency)*10**3)
        axes[0].legend(loc='lower left')
        axes[0].set_ylabel(f'TDI A')
        axes[1].set_xlim((self.lower_frequency)*10**3, (self.upper_frequency)*10**3)
        axes[1].legend(loc='lower left')
        axes[1].set_ylabel(f'TDI E')
        axes[1].set_xlabel('Frequency (mHz)')
        # plt.ylim(1e-19, 1e-16)
        # plt.legend(False)
        fig.tight_layout()
        plt.show(block=True)
        
    def plot_time_domain(self, pGBs):
        """Plots the pGBs in the time domain
        pGBs: array of parameters (shape: (n_signals, 8) or (8,))
        """
        pGBs = np.atleast_2d(pGBs)
        plt.figure()
        # Time domain plotting requires inverse FFT - showing frequency domain magnitude instead
        plt.plot(self.freq, np.abs(self.dataX.values), label='X data')
        for i in range(len(pGBs)):
            Xs, Ys, Zs = self.get_tdi(pGBs[i])
            Xs_aligned = self.align_waveform_to_data(Xs, pGBs[i])
            plt.plot(self.freq, np.abs(Xs_aligned), label=f'X injected {i}')
        plt.xlim(self.lower_frequency, self.upper_frequency)
        # plt.legend(loc='upper right')
        plt.legend(False)
        plt.show(block=True)

class Segment_GB_Searcher:
    """
    Segment-level search controller over a single frequency window.

    A :class:`Segment_GB_Searcher` repeatedly instantiates :class:`GB_Searcher`
    for a given frequency window, extracts candidate sources and subtracts
    them from the TDI data until no source above the chosen SNR threshold
    remains.

    Parameters
    ----------
    tdi_fs
        Full TDI data set in the frequency domain.  A shallow copy is made and
        updated as sources are successively subtracted.
    Tobs
        Observation time in seconds.
    signals_per_window
        Maximum number of sources to extract per window before stopping.
    waveform_args
        Dictionary of waveform-generation keyword arguments (see
        :class:`GB_Searcher`).
    dt
        Sampling interval in seconds.
    SNR_threshold
        Minimum SNR required for a source to be accepted.
    channel_combination
        TDI channel combination (see :class:`GB_Searcher`).
    found_sources_previous
        Optional catalogue of sources found in previous runs; used to build
        initial guesses when re-visiting a window.
    subtract_neighbors
        If ``True``, include padding around the window when building initial
        guesses from ``found_sources_previous``.
    """

    def __init__(
        self,
        tdi_fs,
        Tobs,
        max_signals_per_window,
        waveform_args,
        dt,
        SNR_threshold=9,
        channel_combination="AET",
        found_sources_previous=None,
        subtract_neighbors=False,
    ):
        self.tdi_fs = tdi_fs
        self.Tobs = Tobs
        self.max_signals_per_window = max_signals_per_window
        self.found_sources_previous = found_sources_previous
        self.waveform_args = waveform_args
        self.channel_combination = channel_combination
        self.dt = dt
        self.SNR_threshold = SNR_threshold
        self.subtract_neighbors = subtract_neighbors

    def search(self, lower_frequency, upper_frequency):
        found_sources = []
        tdi_fs_search = deepcopy(self.tdi_fs)
        print('start search', np.round(lower_frequency*10**3, 5), 'mHz to', np.round(upper_frequency*10**3, 5), 'mHz')
        start_search = time.time()
        initial_guess = []

        # use found sources from previous search to get initial guess
        if len(self.found_sources_previous) > 0:
            search = GB_Searcher(tdi_fs_search,self.Tobs, lower_frequency, upper_frequency, self.waveform_args, dt=self.dt, channel_combination=self.channel_combination)
            if self.subtract_neighbors:
                padding_of_initial_guess_range = 0
            else:
                padding_of_initial_guess_range = search.padding
            found_sources_previous_in_range = self.found_sources_previous[self.found_sources_previous['Frequency'] > lower_frequency-padding_of_initial_guess_range]
            found_sources_previous_in_range = found_sources_previous_in_range[found_sources_previous_in_range['Frequency'] < upper_frequency+padding_of_initial_guess_range]
            indexesA = np.argsort(-found_sources_previous_in_range['Amplitude'])
            pGB_stacked = {}
            for parameter in PARAM_NAMES:
                pGB_stacked[parameter] = found_sources_previous_in_range[parameter][indexesA]
            for i in range(len(found_sources_previous_in_range['Amplitude'])):
                pGBs = {}
                for parameter in PARAM_NAMES:
                    pGBs[parameter] = pGB_stacked[parameter][i]
                initial_guess.append(pGBs)
            
            ### sort the initial guesses such that the highest SNR guess comes first
            SNR_guesses = []
            for i in range(len(initial_guess)):
                SNR_guesses.append(search.SNR([initial_guess[i]]))
            indexes = np.argsort(SNR_guesses)[::-1]
            initial_guess = [initial_guess[i] for i in indexes]

        found_sources_all = []
        number_of_evaluations_all = []
        found_sources_inside = []
        current_SNR = self.SNR_threshold + 1

        if lower_frequency > 10**-2:
            self.max_signals_per_window = 3
        ind = 0
        while current_SNR > self.SNR_threshold and ind < self.max_signals_per_window:
            ind += 1
            search = GB_Searcher(tdi_fs_search,self.Tobs, lower_frequency, upper_frequency, self.waveform_args, dt=self.dt, channel_combination=self.channel_combination)

            start = time.time()
            if ind <= len(initial_guess):
                search_repetitions = 2
            else:
                search_repetitions = 2
            for i in range(search_repetitions):
                if ind <= len(initial_guess) and i == 0:
                    maxpGBsearch_new, number_of_evaluations =  search.differential_evolution_search(search.boundaries['Frequency'], initial_guess = [initial_guess[ind-1]])
                else:
                    maxpGBsearch_new, number_of_evaluations =  search.differential_evolution_search(search.boundaries['Frequency'])


                found_sources_all.append(maxpGBsearch_new)
                number_of_evaluations_all.append(number_of_evaluations)
                new_SNR = search.SNR(maxpGBsearch_new[0])
                print('which signal per segment', ind,'and repetition:', i, ' SNR of found signal', np.round(new_SNR,3))
                if i == 0:
                    current_SNR = deepcopy(new_SNR)
                    maxpGBsearch = deepcopy(maxpGBsearch_new)
                if new_SNR >= current_SNR:
                    current_SNR = deepcopy(new_SNR)
                    maxpGBsearch = deepcopy(maxpGBsearch_new)
                print('current max SNR', np.round(current_SNR,3))
                found_sources_all[-1] = maxpGBsearch_new
                if current_SNR < self.SNR_threshold-2:
                    break

            if current_SNR < self.SNR_threshold:
                break

            print('to optimize Amplitude', maxpGBsearch)
            for j in range(len(maxpGBsearch)):
                A_optimized = search.calculate_Amplitude(jnp.array(maxpGBsearch[j]))
                if A_optimized > 0:
                    maxpGBsearch[j][PARAM_INDICES['Amplitude']] *= A_optimized
                else:
                    for i in range(30):
                        print('wavegeneration error!')
                    print('Amplitude optimization failed with parameters:', maxpGBsearch[j])
                    print('switch to optimize with scipy minimize trust-constr')
                    maxpGBsearch[j] = search.optimizeA([[maxpGBsearch[j]]])[0]
                    print('Optimized with parameters:', maxpGBsearch[j])
                print('loglikelihood optimized amplitude',search.loglikelihood([maxpGBsearch[j]]))
            print('in range', maxpGBsearch[0][PARAM_INDICES['Frequency']] > lower_frequency and maxpGBsearch[0][PARAM_INDICES['Frequency']] < upper_frequency)
            # new_SNR = search.SNR(maxpGBsearch[0])

            # current_loglikelihood_ratio = search.loglikelihood(maxpGBsearch[0])
            # print('current loglikelihood ratio', current_loglikelihood_ratio)
            current_SNR = search.SNR(maxpGBsearch[0])
            print('current SNR', current_SNR)
            found_sources.append(maxpGBsearch[0])

            # create two sets of found sources. found_sources_inside with signals inside the boundary and founce_sources_out with outside sources
            found_sources_inside = []
            found_sources_outside = []
            for i in range(len(found_sources)):
                freq = found_sources[i][PARAM_INDICES['Frequency']]
                if lower_frequency < freq < upper_frequency:
                    found_sources_inside.append(found_sources[i])
                else:
                    found_sources_outside.append(found_sources[i])

            #global optimization of found sources inside the window
            if len(found_sources_inside) > 0:
                tdi_fs_subtracted = tdi_subtraction(self.tdi_fs, found_sources_outside, search.fgb, self.waveform_args['tdi_generation'], self.channel_combination)
                search_out_subtracted = GB_Searcher(tdi_fs_subtracted,self.Tobs, lower_frequency, upper_frequency, self.waveform_args, dt=self.dt, channel_combination=self.channel_combination)
                total_boundaries = deepcopy(search_out_subtracted.boundaries)

                for i in range(3):
                    start = time.time()
                    found_sources_inside_opt = search_out_subtracted.optimize([found_sources_inside], boundaries= total_boundaries)
                    print('global optimization time', time.time()-start)
                    
                    found_sources_not_anitcorrelated2 = deepcopy(found_sources_inside_opt)
                    correlation_list2 = []
                    found_correlation = False
                    for j in range(len(found_sources_inside_opt)):
                        correlation_list_of_one_signal = []
                        for k in range(len(found_sources_inside_opt)):
                            found_second_dict = {}
                            found_dict = {}
                            for parameter in PARAM_NAMES:
                                found_second_dict[PARAM_INDICES[parameter]] = found_sources_inside_opt[k][PARAM_INDICES[parameter]]
                                found_dict[PARAM_INDICES[parameter]] = found_sources_inside_opt[j][PARAM_INDICES[parameter]]
                            # correlation = correlation_match(found_second_dict, found_dict, GB, noise_model)
                            correlation = 0
                            correlation_list_of_one_signal.append(correlation)
                            if k > 19:
                                print('k')
                                break
                        if 0 == len(correlation_list_of_one_signal):
                            break
                        max_index = np.argmin(correlation_list_of_one_signal)
                        if correlation_list_of_one_signal[max_index] < -0.7:
                            found_correlation = True
                            print('found anti',j,max_index)
                            correlation_list2.append(correlation_list_of_one_signal[max_index])
                            found_sources_not_anitcorrelated2[j] = None
                            found_sources_not_anitcorrelated2[max_index] = None
                    if not(found_correlation):
                        break
                
                if found_correlation:
                    for i in range(10):
                        print('found anti correlated signals')
                #### after optimization a signal inside window could lay outside. Therefore new selection is required
                if not(found_correlation):
                    if search_out_subtracted.loglikelihood(found_sources_inside_opt) > search_out_subtracted.loglikelihood(found_sources_inside):
                        print('new loglikelihood after global optimization', search_out_subtracted.loglikelihood(found_sources_inside_opt), 'old loglikelihood', search_out_subtracted.loglikelihood(found_sources_inside))
                        found_sources_inside = []
                        for i in range(len(found_sources_inside_opt)):
                            if found_sources_inside_opt[i][PARAM_INDICES['Frequency']] > lower_frequency and found_sources_inside_opt[i][PARAM_INDICES['Frequency']] < upper_frequency:
                                found_sources_inside.append(found_sources_inside_opt[i])
                            else:
                                found_sources_outside.append(found_sources_inside_opt[i])
                    else:
                        for i in range(10):
                            print('optimization failed: ', 'new loglikelihood', search_out_subtracted.loglikelihood(found_sources_inside_opt), 'old loglikelihood', search_out_subtracted.loglikelihood(found_sources_inside))

            found_sources = found_sources_inside + found_sources_outside
            #subtract the found sources from the original data set tdi_fs
            tdi_fs_search = tdi_subtraction(self.tdi_fs, found_sources, search.fgb, self.waveform_args['tdi_generation'], self.channel_combination)

        print('search time', time.time()-start_search, 'frequency', lower_frequency, upper_frequency)
        print('found_sources_inside',found_sources_inside)
        return found_sources, found_sources_all, number_of_evaluations_all, found_sources_inside, [lower_frequency, upper_frequency], time.time()-start_search

def tdi_subtraction(tdi_fs,sources_to_subtract, fgb, tdi_generation, channel_combination):
    """
    Subtract a list of sources from a TDI data set.
    Args:
        tdi_fs: TDI data set
        sources_to_subtract: List of sources to subtract
    Returns:
        TDI data set with the sources subtracted
    """

    tdi_fs_subtracted2 = deepcopy(tdi_fs)
    for i in range(len(sources_to_subtract)):
        source_subtracted = fgb.get_tdi(jnp.asarray(sources_to_subtract[i]), tdi_generation=tdi_generation, tdi_combination=channel_combination)
        kmin = fgb.get_kmin(sources_to_subtract[i][0])
        ffreqs = fgb.get_frequency_grid(jnp.array([kmin])).squeeze()
        index_low = np.searchsorted(tdi_fs_subtracted2['freq'], ffreqs[0])
        index_high = index_low+len(source_subtracted[0])
        for j, k in enumerate(channel_combination):
            tdi_fs_subtracted2[k][index_low:index_high] = tdi_fs_subtracted2[k][index_low:index_high] - source_subtracted[j]

    return tdi_fs_subtracted2


class GB_pe:
    """
    Simple parameter-estimation wrapper for Galactic binaries.

    This class provides a convenience interface around :class:`GB_Searcher`
    and the `eryn` sampler to perform Markov-chain Monte Carlo (MCMC)
    parameter estimation starting from an initial point in parameter space.

    The implementation is intentionally conservative: it is aimed at
    post-processing a small number of well-identified sources rather than
    performing large-scale inference.
    """

    def __init__(
        self,
        tdi_fs,
        initial_parameters,
        Tobs,
        lower_frequency,
        upper_frequency,
        waveform_args,
        dt,
        get_tdi=None,
        get_kmin=None,
        channel_combination="AET",
        noise_model=None,
        recombination=0.75,
        update_noise=True,
    ):
        self.tdi_fs = tdi_fs
        self.initial_parameters = initial_parameters
        self.Tobs = Tobs
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.waveform_args = waveform_args
        self.dt = dt


    def mcmc_GB(self, nsteps=5000, burn=500, ntemps=4, nwalkers=32, ):
        """
        Eryn sampler
        """
        branch_names = ["GB"]
        nleaves_max = 10
        priors = {
            "GB": ProbDistContainer(
                {
                    0: uniform_dist(0, 1),  # Amplitude
                    1: uniform_dist(0, 1),  # Declination
                    2: uniform_dist(0, 1),  # Right Ascension
                    3: uniform_dist(0, 1),  # Frequency
                    4: uniform_dist(0, 1),  # Frequency Derivative
                    5: uniform_dist(0, 1),  # Inclination
                    6: uniform_dist(0, 1),  # Initial Phase
                    7: uniform_dist(0, 1),  # Polarization
                }
            )
        }

        search = GB_Searcher(self.tdi_fs, self.Tobs, self.lower_frequency, self.upper_frequency, self.waveform_args, dt=self.dt)
        # injected mean
        mean = scaleto01(self.initial_parameters, search.boundaries)

        ndim = len(mean[0])
        # define covariance matrix
        cov = 1e-2*np.diag(np.ones(ndim))
        # fill kwargs dictionary
        tempering_kwargs=dict(ntemps=ntemps)
        # random prior draw for initial points, when using real signals, start around true

        coords = {"GB": np.zeros((ntemps, nwalkers, nleaves_max, ndim))*0.5}

        # this is the sigma for the multivariate Gaussian that sets starting points
        # We need it to be very small to assume we are passed the search phase
        # we will verify this is with likelihood calculations
        sig1 = 10**-10

        # setup initial walkers to be the correct count (it will spread out)
        for nn in range(nleaves_max):
            if nn >= len(mean):
                # not going to add parameters for these unused leaves
                continue
                
            coords["GB"][:, :, nn] = np.random.multivariate_normal(mean[nn], np.diag(np.ones(len(mean[nn])) * sig1), size=(ntemps, nwalkers)) 

        # make sure to start near the proper setup
        inds = {"GB": np.zeros((ntemps, nwalkers, nleaves_max), dtype=bool)}

        # turn False -> True for any binary in the sampler
        inds['GB'][:, :, :len(mean)] = True

        # for the Gaussian Move
        factor = [10**-9, 10**-6, 10**-6, 10**-6, 10**-6, 10**-6, 10**-6, 10**-6]
        cov = {"GB": np.diag(np.ones(ndim)) * factor}

        moves = GaussianMove(cov)


        function_to_maximize = search.from01tologlikelihood
            
        # ensemble = EnsembleSampler(
        #     nwalkers,
        #     {"GB": ndim},
        #     function_to_maximize,
        #     priors,
        #     nleaves_max=nleaves_max,
        #     branch_names=["GB"],
        #     nbranches=1,
        #     tempering_kwargs=tempering_kwargs,
        # )

        ensemble = EnsembleSampler(
            nwalkers,
            ndim,  
            function_to_maximize,
            priors,
            tempering_kwargs=dict(ntemps=ntemps),
            nbranches=len(branch_names),
            branch_names=branch_names,
            nleaves_max=nleaves_max,
            nleaves_min=0,
            moves=moves,
            rj_moves=True,  # basic generation of new leaves from the prior
        )


        # print('initial points', initial_points)
        print(mean, 'loglikelihood', function_to_maximize(np.array([mean[0]])))
        print(mean, 'loglikelihood', function_to_maximize(np.array([mean[1]])))
        print(mean, 'loglikelihood', function_to_maximize(mean))
        # print( initial_points[0,0,0], 'loglikelihood', function_to_maximize(initial_points[0,0,0]))
        # always check likelihood and prior values
        log_prior = ensemble.compute_log_prior(coords, inds=inds)
        log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        # make sure it is reasonably close to the maximum
        # will not be zero due to noise
        print("Log-likelihood:\n", log_like)
        print("\nLog-prior:\n", log_prior)
                
        # setup starting state
        state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

        thin_by = 1
        ensemble.run_mcmc(state, nsteps, progress=True, burn=burn, thin_by=thin_by)

        chain = ensemble.get_chain(discard=0, thin=1)["GB"][:, 0]
        self.chains_sample = []
        
        for parameters_chain in chain:
            for parameters_walker in parameters_chain:
                self.chains_sample.append([])
                for parameter_01 in parameters_walker:
                    params = scaletooriginal(parameter_01, search.boundaries)
                    self.chains_sample[-1].append(np.copy(params))

        chains = np.array(self.chains_sample)

        fig = corner.corner(chains[:,0,:], labels=PARAM_NAMES)
        plt.show(block=True)

        return chains, ensemble