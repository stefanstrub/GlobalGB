"""
Plot found GB signals together with the data spectrum, the injected signals and
their generated TDI waveforms inside a chosen frequency range.

Example:
    python plot_found_vs_injected.py --fmin 0.0003 --fmax 0.003
    python plot_found_vs_injected.py --fmin 0.0029 --fmax 0.0031 --channel A --output plot.png

Found signals are read from all ``found_signals_batch_*.h5`` files in
``config.save_path/found_signals_*``. Injected signals are read from
``config.save_path/pGB_injected_Mojito_WDWD.h5``. The data spectrum is the
amplitude spectral density of the selected TDI channel of ``config.data_path``.

Waveforms are generated with JaxGB for every signal in the range (using the same
orbits, t0 and frame convention as ``match_GBs.py``). The per-signal waveforms are
saved to an HDF5 file, and the summed waveform amplitude of the injected and of the
found signals is plotted on top of the data.
"""

import argparse
import glob
import os

import jax

jax.config.update("jax_enable_x64", True)

import h5py
import jax.numpy as jnp
import lisaorbits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, NullLocator

from jaxgb.jaxgb import JaxGB
from jaxgb.params import GBObject

from globalGB.config import load_config
from globalGB.search_utils_GB import GBConfig, PARAM_INDICES, PARAM_NAMES
from DataLoader.data_loader import LISADataLoader

# Reference time used for the injected/found catalogues in match_GBs.py.
T_INIT = 97729089.327664


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fmin", type=float, required=True, help="Lower frequency bound in Hz.")
    parser.add_argument("--fmax", type=float, required=True, help="Upper frequency bound in Hz.")
    parser.add_argument("--channel", type=str, default="A", help="TDI channel for the data and waveforms (default: A).")
    parser.add_argument("--found-glob", type=str, default=None,
                        help="Optional glob for found-signal files; default searches config.save_path.")
    parser.add_argument("--waveform-output", type=str, default=None,
                        help="Optional HDF5 path for the generated waveforms; "
                             "default is config.save_path/waveforms_<fmin>_<fmax>_<channel>.h5.")
    parser.add_argument("--output", type=str, default=None, help="Optional file to save the figure instead of showing it.")
    return parser.parse_args(argv)


def load_found_sources(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No found-signal files match: {pattern}")
    sources = []
    for fn in files:
        with h5py.File(fn, "r") as f:
            if "recovered_sources" in f and len(f["recovered_sources"]) > 0:
                sources.append(f["recovered_sources"][:])
    print(f"Loaded found signals from {len(files)} files")
    return np.concatenate(sources) if sources else np.empty((0, len(PARAM_NAMES)))


def load_injected_sources(config):
    fn = os.path.join(config.save_path, "pGB_injected_Mojito_WDWD.h5")
    with h5py.File(fn, "r") as f:
        return f["injected_sources"][:]


def load_data_spectrum(loader, channel):
    if channel not in loader.tdi_fs:
        channel = [k for k in loader.tdi_fs if k != "freq"][0]
        print(f"Requested channel not found, using {channel}")
    freq = np.asarray(loader.tdi_fs["freq"])
    # Single-sided ASD estimate from the one-sided FFT of the full data record.
    t_obs = 1.0 / (freq[1] - freq[0])  # observation time from the frequency grid spacing
    asd = np.sqrt(2.0 * np.abs(loader.tdi_fs[channel]) ** 2 / t_obs)
    return freq, asd, channel


def build_waveform_generator(config, loader):
    """Build the JaxGB waveform generator the same way as match_GBs.GBMatchingPipeline.setup."""
    with h5py.File(config.data_path, "r") as fid:
        orbits_data = fid["orbits"]
        sampling = dict(orbits_data["sampling"].attrs)
        orbits = lisaorbits.InterpolatedOrbits(
            np.arange(sampling["t0"], sampling["t0"] + sampling["duration"], sampling["dt"]),
            np.stack([orbits_data["sc_position_1"][:],
                      orbits_data["sc_position_2"][:],
                      orbits_data["sc_position_3"][:]]).swapaxes(0, 1),
            np.stack([orbits_data["sc_velocity_1"][:],
                      orbits_data["sc_velocity_2"][:],
                      orbits_data["sc_velocity_3"][:]]).swapaxes(0, 1),
        )
    return JaxGB(orbits=orbits, t_obs=loader.Tobs, t0=loader.t0, n=int(2**10))


def shift_found_to_t_init(found, loader):
    """Shift found sources to the injected-catalogue reference time (as in match_GBs.py)."""
    gbo = GBObject.from_jaxgb_params(jnp.array(found), t_init=loader.t0)
    return np.array(gbo.to_jaxgb_array(t0=T_INIT))


def generate_waveforms(fgb, params, channel_index, tdi_generation, channel_combination):
    """
    Generate the TDI frequency-domain waveform of every source.

    Returns a list of dicts with the frequency grid and the complex response of the
    requested channel, plus the full set of channels for saving.
    """
    waveforms = []
    for p in params:
        p = jnp.asarray(p)
        f0 = float(p[PARAM_INDICES["Frequency"]])
        kmin = fgb.get_kmin(f0)
        freqs = np.asarray(fgb.get_frequency_grid(jnp.array([kmin])).squeeze())
        channels = fgb.get_tdi(p, tdi_generation=tdi_generation, tdi_combination=channel_combination)
        channels = [np.asarray(c) for c in channels]
        waveforms.append({
            "freq": freqs,
            "channels": channels,
            "h": channels[channel_index],
        })
    return waveforms


def sum_waveform_amplitude(waveforms, freq_data, fmin, fmax):
    """Sum |h|^2 of all waveforms on the data frequency grid and return sqrt of it."""
    df = freq_data[1] - freq_data[0]
    power = np.zeros(len(freq_data))
    for w in waveforms:
        idx = np.round(w["freq"] / df).astype(int)
        ok = (idx >= 0) & (idx < len(freq_data)) & in_range(w["freq"], fmin, fmax)
        np.add.at(power, idx[ok], np.abs(w["h"][ok]) ** 2)
    return np.sqrt(power)


def save_waveforms(path, groups, channel_combination):
    """Save per-signal waveforms: one group per signal under ``injected/`` and ``found/``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, waveforms in groups.items():
            grp = f.create_group(name)
            for i, w in enumerate(waveforms):
                sub = grp.create_group(str(i))
                sub.create_dataset("freq", data=w["freq"])
                for letter, values in zip(channel_combination, w["channels"]):
                    sub.create_dataset(letter, data=values)
    print(f"Saved waveforms to {path}")


def in_range(values, fmin, fmax):
    return (values >= fmin) & (values <= fmax)


def main(argv=None):
    args = parse_args(argv)
    config = GBConfig(load_config())

    found_pattern = args.found_glob or os.path.join(
        config.save_path, "found_signals_*", "found_signals_batch_index_*.h5"
    )
    found = load_found_sources(found_pattern)
    injected = load_injected_sources(config)

    f_idx = PARAM_INDICES["Frequency"]
    a_idx = PARAM_INDICES["Amplitude"]

    found_sel = found[in_range(found[:, f_idx], args.fmin, args.fmax)] if len(found) else found
    inj_sel = injected[in_range(injected[:, f_idx], args.fmin, args.fmax)]
    print(f"In range [{args.fmin}, {args.fmax}] Hz: {len(found_sel)} found, {len(inj_sel)} injected")

    loader = LISADataLoader(config=config)
    loader.load(config.data_path, dt=config.dt, channel_combination=config.channel_combination)
    freq, asd, channel = load_data_spectrum(loader, args.channel)
    data_mask = in_range(freq, args.fmin, args.fmax) & (freq > 0)

    # Waveform generation. Injected sources are already in the T_INIT frame; found
    # sources are shifted there first, matching the match_GBs pipeline.
    channel_index = config.channel_combination.index(channel)
    fgb = build_waveform_generator(config, loader)
    print("Generating injected waveforms...")
    inj_waves = generate_waveforms(fgb, inj_sel, channel_index,
                                   config.tdi_generation, config.channel_combination)
    print("Generating found waveforms...")
    found_sel_t_init = shift_found_to_t_init(found_sel, loader) if len(found_sel) else found_sel
    found_waves = generate_waveforms(fgb, found_sel_t_init, channel_index,
                                     config.tdi_generation, config.channel_combination)

    wave_output = args.waveform_output or os.path.join(
        config.save_path, f"waveforms_{args.fmin:g}_{args.fmax:g}_{channel}.h5"
    )
    save_waveforms(wave_output, {"injected": inj_waves, "found": found_waves}, config.channel_combination)

    inj_amp = sum_waveform_amplitude(inj_waves, freq, args.fmin, args.fmax)
    found_amp = sum_waveform_amplitude([found_waves[0]], freq, args.fmin, args.fmax)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq[data_mask], asd[data_mask], color="0.6", lw=0.8, label=f"Data (channel {channel})", zorder=0)
    ax.plot(freq[data_mask], inj_amp[data_mask], color="C1", lw=0.8, alpha=0.8,
              label="Injected waveforms (summed)", zorder=1)
    ax.plot(freq[data_mask], found_amp[data_mask], color="C0", lw=0.8, alpha=0.8,
              label="Found waveforms (summed)", zorder=1)
    ax.plot(inj_sel[:, f_idx], inj_sel[:, a_idx], "+", color="C1", ms=8, label=f"Injected ({len(inj_sel)})")
    ax.plot(found_sel[:, f_idx], found_sel[:, a_idx], "o", mfc="none", color="C0", ms=6,
            label=f"Found ({len(found_sel)})")
    ax.set_xlim(args.fmin, args.fmax)
    ax.set_xticks(np.linspace(args.fmin, args.fmax, 5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.5g"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("$f$ (Hz)")
    ax.set_ylabel("Amplitude / ASD / waveform")
    ax.set_title(f"Found vs injected signals, {args.fmin:g} to {args.fmax:g} Hz")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
