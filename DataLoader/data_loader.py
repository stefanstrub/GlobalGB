"""
Unified data loader for LISA datasets (Radler, Sangria, Spritz, Mojito, Windowed)
"""

import numpy as np
import h5py
import pickle
import os
from copy import deepcopy
from ldc.common.series import TimeSeries, FrequencySeries, TDI
from ldc.common.tools import window
from MojitoProcessor import load_mojito_l1, process_pipeline


class LISADataLoader:
    """Unified data loader for LISA datasets (Radler, Sangria, Spritz, Mojito, Windowed)"""
    
    SUPPORTED_DATASETS = ['Radler', 'Sangria', 'Spritz', 'Mojito', 'Windowed']
    
    def __init__(self, config: dict):
        """
        Initialize the data loader.
        
        Args:
            dataset: One of 'Radler', 'Sangria', 'Spritz', 'Mojito', 'Windowed'
            config: Configuration dictionary
        """
        if config["data_set"] not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset must be one of {self.SUPPORTED_DATASETS}")

        self.dataset = config["data_set"]
        self.data_path = config["data_path"]
        self.catalog_path = config["catalog_path"]
        self.save_path = config["save_path"]

        # Data attributes (populated after loading)
        self.td = None           # TDI time domain data
        self.tdi_fs = None       # TDI frequency series
        self.tdi_ts = None       # TDI time series dict
        self.dt = None           # Time step
        self.Tobs = None         # Observation time
        self.catalog = None      # Source catalog (legacy, same as catalog_mbhb)
        self.catalog_mbhb = None # MBHB catalog
        self.catalog_wdwd = None # WDWD (galactic binary) catalog
        self.fid = None          # HDF5 file handle
        self.td_mbhb = None      # MBHB TDI data (if available)
        self.freq = None         # Frequency array
        self.CENTRAL_FREQ = 281600000000000.0
        
    
    def load(self, filename: str = None, dt: float = None, 
             Tobs: float = None, weeks: int = None, channel_combination='XYZ', **kwargs):
        """
        Load data from file.
        
        Args:
            filename: Custom filename (optional, auto-detected if None)
            dt: Time step (required for some datasets)
            Tobs: Observation time in seconds
            weeks: Observation time in weeks (alternative to Tobs)
            **kwargs: Dataset-specific options:
                - subtract_mbhb: bool - Subtract MBHB signals (Sangria)
                - subtract_gb: bool - Subtract GB signals (Sangria)
                - s_index: int - Source index for catalog (Mojito)
                
        Returns:
            self for method chaining
        """
        if weeks is not None:
            Tobs = float(weeks * 7 * 24 * 3600)
        
        loader_method = getattr(self, f'_load_{self.dataset.lower()}')
        loader_method(filename, dt, Tobs, channel_combination=channel_combination, **kwargs)
        
        
        return self
        
    
    def _load_radler(self, filename, dt, Tobs, **kwargs):
        """Load Radler dataset"""
        if filename is None:
            filename = self.data_path + "/LDC1-1_MBHB_v2_TD.hdf5"
        
        self.fid = h5py.File(filename)
        td = np.array(self.fid["H5LISA/PreProcess/TDIdata"])
        td = np.rec.fromarrays(list(td.T), names=["t", "X", "Y", "Z"])
        
        self.dt = float(np.array(self.fid['H5LISA/GWSources/MBHB-0']['Cadence']))
        self.td = TDI(dict([(k, TimeSeries(td[k], dt=self.dt, t0=0)) 
                           for k in ["X", "Y", "Z"]]))
        self.Tobs = float(np.array(self.fid['H5LISA/GWSources/MBHB-0']['ObservationDuration']))
        
        # Load catalog
        names = list(self.fid['H5LISA/GWSources/MBHB-0'].keys())
        params = [self.fid['H5LISA/GWSources/MBHB-0'].get(k)[()] for k in names]
        self.catalog = [dict(zip(names, params))]
        
        # Load noiseless data if available
        noiseless_fn = self.data_path + "/LDC1-1_MBHB_v2_TD_noiseless.hdf5"
        if os.path.exists(noiseless_fn):
            fid_noiseless = h5py.File(noiseless_fn)
            td_mbhb = np.array(fid_noiseless["H5LISA/PreProcess/TDIdata"])
            td_mbhb = np.rec.fromarrays(list(td_mbhb.T), names=["t", "X", "Y", "Z"])
            self.td_mbhb = TDI(dict([(k, TimeSeries(td_mbhb[k], dt=self.dt, t0=0)) 
                                    for k in ["X", "Y", "Z"]]))
    
    def _load_sangria(self, filename, dt, Tobs, subtract_mbhb=False, 
                      subtract_gb=False, subtract_vgb=False, subtract_dgb=False,
                      subtract_igb=False, **kwargs):
        """Load Sangria dataset"""
        if filename is None:
            filename = self.data_path + "/LDC2_sangria_training_v2.h5"
        
        self.fid = h5py.File(filename)
        td = self.fid["obs/tdi"][()]
        td = np.rec.fromarrays(list(td.T), names=["t", "X", "Y", "Z"])
        td = td['t']
        
        self.dt = td["t"][1] - td["t"][0]
        
        # Load MBHB TDI
        td_mbhb = self.fid["sky/mbhb/tdi"][()]
        td_mbhb = np.rec.fromarrays(list(td_mbhb.T), names=["t", "X", "Y", "Z"])
        self.td_mbhb = td_mbhb['t']
        
        # Load other source TDIs
        self.td_vgb = None
        self.td_dgb = None
        self.td_igb = None
        
        if "sky/vgb/tdi" in self.fid:
            td_vgb = self.fid["sky/vgb/tdi"][()]
            td_vgb = np.rec.fromarrays(list(td_vgb.T), names=["t", "X", "Y", "Z"])
            self.td_vgb = td_vgb['t']
            
        if "sky/dgb/tdi" in self.fid:
            td_dgb = self.fid["sky/dgb/tdi"][()]
            td_dgb = np.rec.fromarrays(list(td_dgb.T), names=["t", "X", "Y", "Z"])
            self.td_dgb = td_dgb['t']
            
        if "sky/igb/tdi" in self.fid:
            td_igb = self.fid["sky/igb/tdi"][()]
            td_igb = np.rec.fromarrays(list(td_igb.T), names=["t", "X", "Y", "Z"])
            self.td_igb = td_igb['t']
        
        # Load MBHB catalog
        mbhb = self.fid["sky/mbhb/cat"]
        self.catalog = []
        for index in range(len(mbhb)):
            pMBHB = dict(zip(mbhb.dtype.names, mbhb[index]))
            for name in mbhb.dtype.names:
                pMBHB[name] = mbhb[index][name][0]
            self.catalog.append(pMBHB)
        
        # Optional subtraction
        if subtract_mbhb:
            for k in ["X", "Y", "Z"]:
                td[k] = td[k] - self.td_mbhb[k]
                
        if subtract_vgb and self.td_vgb is not None:
            for k in ["X", "Y", "Z"]:
                td[k] = td[k] - self.td_vgb[k]
                
        if subtract_dgb and self.td_dgb is not None:
            for k in ["X", "Y", "Z"]:
                td[k] = td[k] - self.td_dgb[k]
                
        if subtract_igb and self.td_igb is not None:
            for k in ["X", "Y", "Z"]:
                td[k] = td[k] - self.td_igb[k]
        
        # Truncate to Tobs if specified
        if Tobs is not None:
            t_max_index = np.searchsorted(td['t'], Tobs)
            self.td = TDI(dict([(k, TimeSeries(td[k][:t_max_index], dt=self.dt, t0=td.t[0])) 
                               for k in ["X", "Y", "Z"]]))
        else:
            self.td = TDI(dict([(k, TimeSeries(td[k], dt=self.dt, t0=td.t[0])) 
                               for k in ["X", "Y", "Z"]]))
        
        self.Tobs = Tobs or float(td['t'][-1]) + self.dt
    
    def _load_mojito(self, filename=None, dt=None, Tobs=None, channel_combination='AET', **kwargs):
        """Load Mojito dataset"""
        if dt is not None:
            self.dt = dt
        if filename is None:
            filename = self.data_path
        if channel_combination is not None:
            self.channel_combination = channel_combination
        if Tobs is not None:
            self.Tobs = Tobs
        
        self.data = load_mojito_l1(filename)
        self.Tobs_original = float(self.data.duration)
        # ── Pipeline parameters ───────────────────────────────────────────────────────

        # Downsampling parameters
        downsample_kwargs = {
            "target_fs": 1/self.dt,  # Hz — target sampling rate (None = no downsampling).
            "kaiser_window": 31.0,  # Kaiser window beta parameter (higher = more aggressive anti-aliasing)
        }

        # Filter parameters
        filter_kwargs = {
            "highpass_cutoff": 5e-6,  # Hz — high-pass cutoff (always applied)
            "lowpass_cutoff": 0.5 # half of the target sampling rate in Hz
            * downsample_kwargs[
                "target_fs"
            ],  # Hz — low-pass cutoff (set None for high-pass only)
            "order": 2,  # Butterworth filter order
        }

        # Trim parameters
        trim_kwargs = {
            "fraction": 0.02,  # Fraction of post-downsample duration trimmed from each end.
            # Total amount of data remaining is (1 - fraction) * N, for N
            # the number of samples after downsampling.
        }



        # Window parameters
        window_kwargs = {
            "window": "tukey",  # Window type: 'tukey', 'hann', 'hamming', 'blackman'
            "alpha": 0.0125,  # Taper fraction for Tukey window
        }
        # ─────────────────────────────────────────────────────────────────────────────

        processed_segments =process_pipeline(
            self.data,
            downsample_kwargs=downsample_kwargs,
            filter_kwargs=filter_kwargs,
            trim_kwargs=trim_kwargs,
            window_kwargs=window_kwargs,
            channels=self.channel_combination,
        )

        td = processed_segments["segment0"]
        

        

        # Convert from frequency to fractional frequency
        self.CENTRAL_FREQ = self.data.metadata["laser_frequency"]

        self.freq = np.fft.rfftfreq(td.N, d=td.dt)
        self.tdi_ts = {ch: td.data[ch] for ch in td.channels}
        # self.tdi_ts = {ch: TimeSeries(td.data[ch], dt=td.dt, t0=self.data.t0) for ch in td.channels}
        self.tdi_fs = {ch: np.fft.rfft(td.data[ch])*td.dt / self.CENTRAL_FREQ for ch in td.channels}
        self.tdi_fs['freq'] = self.freq

        self.Tobs = float(len(td.data[self.channel_combination[-1]])) * td.dt # Tobs after trimming
        trim_time = (self.Tobs_original - self.Tobs)/2 # Time to trim from the start
        self.t0 = self.data.t0 + trim_time # Start time after trimming
        # self.t0_processed = td.t0
        
    
    def _load_mojito_catalog(self):
        """Load Mojito MBHB and WDWD catalogs"""
        # Load MBHB catalog
        self._load_mojito_mbhb_catalog()
        
        # Load WDWD catalog
        self._load_mojito_wdwd_catalog()
        
        # Set legacy catalog attribute to MBHB
        self.catalog = self.catalog_mbhb
    
    def _load_mojito_mbhb_catalog(self):
        """Load Mojito MBHB catalog"""
        catalog_path = self.catalog_path + "/mbhb_cat_mojito_lite_processed_MT.hdf5"
        
        if not os.path.exists(catalog_path):
            print(f"Warning: MBHB catalog file not found at {catalog_path}")
            self.catalog_mbhb = None
            return
            
        fid_mbhb = h5py.File(catalog_path, "r")
        
        parameters = ['Mass1', 'Mass2', 'Spin1', 'Spin2', 'Distance', 'Phase', 
                     'Inclination', 'EclipticLongitude', 'EclipticLatitude', 
                     'Polarization', 'CoalescenceTime']
        parameters_to_keys = {
            'Mass1': 'PrimaryMassSSBFrame', 
            'Mass2': 'SecondaryMassSSBFrame',
            'Spin1': 'PrimarySpinParameter', 
            'Spin2': 'SecondarySpinParameter',
            'Distance': 'LuminosityDistance', 
            'Phase': 'TrueAnomaly',
            'Inclination': 'InclinationAngle', 
            'EclipticLongitude': 'RightAscension',
            'EclipticLatitude': 'Declination', 
            'Polarization': 'PolarisationAngle',
            'CoalescenceTime': 'TimeCoalescencePhenomTPHMSSBFrame'
        }
        
        parameters_mbhb = {}
        cat_mbhb = []
        for key in fid_mbhb['Binaries'].keys():
            parameters_mbhb[key] = np.array(fid_mbhb['Binaries'][key])
        for parameter in parameters:
            cat_mbhb.append(parameters_mbhb[parameters_to_keys[parameter]])
        self.catalog_mbhb = np.array(cat_mbhb).T
        fid_mbhb.close()
    
    def _load_mojito_wdwd_catalog(self):
        """Load Mojito WDWD (galactic binary) catalog"""
        catalog_path = self.catalog_path + "/wdwd_cat_mojito_lite_processed.hdf5"
        if not os.path.exists(catalog_path):
            print(f"Warning: WDWD catalog file not found at {catalog_path}")
            self.catalog_wdwd = None
            return
            
        fid_wdwd = h5py.File(catalog_path, "r")
        
        # WDWD parameters (galactic binaries have different parameters than MBHB)
        # Common GB parameters: Amplitude, Frequency, FrequencyDerivative, 
        # EclipticLatitude, EclipticLongitude, Inclination, Polarization, InitialPhase


        parameters = [
            "Frequency",
            "FrequencyDerivative",
            "Amplitude",
            "RightAscension",
            "Declination",
            "Polarization",
            "Inclination",
            "InitialPhase",
        ]

        parameters_to_keys = {
            'Frequency': 'GW22FrequencySSBFrame',
            'FrequencyDerivative': 'GW22FrequencyDerivativeSourceFrame',
            'Declination': 'Declination',
            'RightAscension': 'RightAscension',
            'Inclination': 'InclinationAngle',
            'Polarization': 'PolarisationAngle',
            'InitialPhase': 'TrueAnomaly',
            'Amplitude': 'Amplitude'}

        parameters_wdwd = {}
        cat_wdwd = []
        for key in fid_wdwd['Binaries'].keys():
            parameters_wdwd[key] = np.array(fid_wdwd['Binaries'][key])
        for parameter in parameters:
            cat_wdwd.append(parameters_wdwd[parameters_to_keys[parameter]])
        self.catalog_wdwd = np.array(cat_wdwd).T
        
        fid_wdwd.close()
    
    def _load_spritz(self, filename, dt, Tobs, **kwargs):
        """Load Spritz dataset"""
        if filename is None:
            filename = self.data_path + "/LDC2_spritz_vgb_training_v2.h5"
        
        self.fid = h5py.File(filename)
        
        # Load catalog
        names = self.fid["sky/cat"].dtype.names
        cat_vgb = dict(zip(names, [self.fid["sky/cat"][name] for name in names]))
        self.catalog = []
        for i in range(len(cat_vgb['Frequency'])):
            self.catalog.append({})
            for name in names:
                self.catalog[i][name] = cat_vgb[name][i][0]
        
        # Load TDI variants
        td = self.fid["obs/tdi"][()]
        td = np.rec.fromarrays(list(td.T), names=["t", "X", "Y", "Z"])
        td = td['t']
        
        self.dt = td["t"][1] - td["t"][0]
        
        # Load other TDI variants
        self.td_clean = None
        self.td_sky = None
        self.td_noisefree = None
        
        if "clean/tdi" in self.fid:
            td_clean = self.fid["clean/tdi"][()]
            td_clean = np.rec.fromarrays(list(td_clean.T), names=["t", "X", "Y", "Z"])
            self.td_clean = td_clean['t']
            
        if "sky/tdi" in self.fid:
            td_sky = self.fid["sky/tdi"][()]
            td_sky = np.rec.fromarrays(list(td_sky.T), names=["t", "X", "Y", "Z"])
            self.td_sky = td_sky['t']
            
        if "noisefree/tdi" in self.fid:
            td_noisefree = self.fid["noisefree/tdi"][()]
            td_noisefree = np.rec.fromarrays(list(td_noisefree.T), names=["t", "X", "Y", "Z"])
            self.td_noisefree = td_noisefree['t']
        
        # Truncate to Tobs if specified
        if Tobs is not None:
            t_max_index = np.searchsorted(td['t'], Tobs)
            self.td = TDI(dict([(k, TimeSeries(td[k][:t_max_index], dt=self.dt, t0=td.t[0])) 
                               for k in ["X", "Y", "Z"]]))
        else:
            self.td = TDI(dict([(k, TimeSeries(td[k], dt=self.dt, t0=td.t[0])) 
                               for k in ["X", "Y", "Z"]]))
        
        self.Tobs = Tobs or float(td['t'][-1]) + self.dt
    
    def _load_windowed(self, filename, dt, Tobs, **kwargs):
        """Load Windowed dataset (custom text format)"""
        if filename is None:
            filename = self.data_path + '/data.txt'
        
        td = np.loadtxt(filename)
        td = list(td.T)
        self.dt = dt or 15
        Tobs = self.dt * len(td[0])
        td.append(np.arange(0, Tobs, self.dt))
        self.td = np.rec.fromarrays(td, names=["X", "Y", "Z", "t"])
        self.Tobs = Tobs
        
        # Load injected parameters if available
        params_file = self.data_path + '/parameters.txt'
        if os.path.exists(params_file):
            self.catalog = np.loadtxt(params_file)
    
    def to_frequency_domain(self, win=None):
        """
        Convert time domain data to frequency domain.
        
        Args:
            win: Window function (defaults to ldc window)
            
        Returns:
            xarray Dataset with frequency domain TDI
        """
        import xarray as xr
        
        if win is None:
            win = window
            
        self.tdi_ts = dict([(k, TimeSeries(self.td[k], dt=self.dt)) 
                          for k in ["X", "Y", "Z"]])
        self.tdi_fs = xr.Dataset(dict([(k, self.tdi_ts[k].ts.fft(win=win)) 
                                       for k in ["X", "Y", "Z"]]))
        return self.tdi_fs
    
    def to_AET(self):
        """
        Convert XYZ TDI to AET channels.
        
        Returns:
            dict with A, E, T channels
        """
        tdi_aet = {}
        tdi_aet['A'] = (self.td['Z'] - self.td['X']) / np.sqrt(2.0)
        tdi_aet['E'] = (self.td['Z'] - 2.0 * self.td['Y'] + self.td['X']) / np.sqrt(6.0)
        tdi_aet['T'] = (self.td['Z'] + self.td['Y'] + self.td['X']) / np.sqrt(3.0)
        tdi_aet['t'] = np.copy(self.td['X'].t)
        return tdi_aet
    
    def get_frequencies(self):
        """Get frequency array for FFT"""
        if self.freq is None:
            self.freq = np.fft.rfftfreq(len(self.td['X']), d=self.dt)
        return self.freq
    
    def subtract_signal(self, signal_td, channels=["X", "Y", "Z"]):
        """
        Subtract a signal from the TDI data.
        
        Args:
            signal_td: Signal TDI data (dict or TDI object)
            channels: List of channels to subtract from
        """
        for k in channels:
            self.td[k] -= signal_td[k]
    
    def add_signal(self, signal_td, channels=["X", "Y", "Z"]):
        """
        Add a signal to the TDI data.
        
        Args:
            signal_td: Signal TDI data (dict or TDI object)
            channels: List of channels to add to
        """
        for k in channels:
            self.td[k] += signal_td[k]
    
    def get_source_params(self, index: int = 0, catalog_type: str = 'mbhb'):
        """
        Get parameters for a specific source from the catalog.
        
        Args:
            index: Source index in catalog
            catalog_type: 'mbhb' or 'wdwd' (for Mojito dataset)
            
        Returns:
            dict or array with source parameters
        """
        if catalog_type == 'mbhb':
            catalog = self.catalog_mbhb if self.catalog_mbhb is not None else self.catalog
        elif catalog_type == 'wdwd':
            catalog = self.catalog_wdwd
        else:
            catalog = self.catalog
        
        if catalog is None:
            raise ValueError(f"No {catalog_type} catalog loaded")
        
        if isinstance(catalog, list):
            return catalog[index]
        elif isinstance(catalog, np.ndarray):
            return catalog[index]
        elif isinstance(catalog, dict):
            # Return dict with values at index
            return {k: v[index] if hasattr(v, '__getitem__') else v for k, v in catalog.items()}
        else:
            raise ValueError(f"Unknown catalog type: {type(catalog)}")
    
    def close(self):
        """Close any open file handles"""
        if self.fid is not None:
            self.fid.close()
            self.fid = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def get_catalog_size(self, catalog_type: str = 'mbhb'):
        """
        Get the number of sources in a catalog.
        
        Args:
            catalog_type: 'mbhb' or 'wdwd'
            
        Returns:
            Number of sources in the catalog
        """
        if catalog_type == 'mbhb':
            catalog = self.catalog_mbhb if self.catalog_mbhb is not None else self.catalog
        elif catalog_type == 'wdwd':
            catalog = self.catalog_wdwd
        else:
            raise ValueError(f"Unknown catalog_type: {catalog_type}")
        
        if catalog is None:
            return 0
        
        if isinstance(catalog, (list, np.ndarray)):
            return len(catalog)
        elif isinstance(catalog, dict):
            # Return length of first array in dict
            for v in catalog.values():
                if hasattr(v, '__len__'):
                    return len(v)
            return 0
        return 0
    
    def __repr__(self):
        mbhb_size = self.get_catalog_size('mbhb') if (self.catalog_mbhb is not None or self.catalog is not None) else 0
        wdwd_size = self.get_catalog_size('wdwd') if self.catalog_wdwd is not None else 0
        return (f"LISADataLoader(dataset='{self.dataset}', "
                f"Tobs={self.Tobs}, dt={self.dt}, "
                f"data_loaded={self.td is not None}, "
                f"mbhb_sources={mbhb_size}, wdwd_sources={wdwd_size})")
