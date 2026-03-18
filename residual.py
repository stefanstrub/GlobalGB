import json
from globalGB.GB_runner import GBSearchRunner
from globalGB.search_utils_GB import GBConfig
from jaxgb.jaxgb import JaxGB
import h5py
from globalGB.search_utils_GB import tdi_subtraction
import numpy as np
import matplotlib.pyplot as plt


def main(argv=None):
    with open('globalGB/GB_search_config.json', 'r') as f:
        config = json.load(f)
        config = GBConfig(config)
    runner = GBSearchRunner(
        batch_index=0,
        which_run="even",
        config=config,
    )
    runner.load_data()
    fgb = JaxGB(
    orbits=runner.waveform_args["orbits"],
    t_obs=runner.waveform_args["Tobs"],
    t0=runner.waveform_args["t0"],
    n=1024,
    )
    # load found sources h5 file
    with h5py.File('/home/stefan/LDC/Mojito/found_signals/GB_EMRI/GB/found_signals_Mojito_SNR_threshold_9_seed1.h5', 'r') as f:
        found_sources = f['recovered_sources'][:]
    tdi_fs_residual_EMRI = tdi_subtraction(
    runner.tdi_fs,
    found_sources,
    fgb,
    runner.waveform_args["tdi_generation"],
    runner.cfg.channel_combination)

    # load found sources h5 file
    with h5py.File('/home/stefan/LDC/Mojito/found_signals/GB/found_signals_Mojito_SNR_threshold_9_seed1.h5', 'r') as f:
        found_sources = f['recovered_sources'][:]
    tdi_fs_residual = tdi_subtraction(
    runner.tdi_fs,
    found_sources,
    fgb,
    runner.waveform_args["tdi_generation"],
    runner.cfg.channel_combination)

    channel_combination = [x for x in runner.cfg.channel_combination]
    # create time domain data
    time_domain_residual_EMRI = {ch: np.fft.irfft(tdi_fs_residual_EMRI[ch], axis=0) for ch in channel_combination}
    time_domain_residual = {ch: np.fft.irfft(tdi_fs_residual[ch], axis=0) for ch in channel_combination}
    # plot time domain data
    plt.figure()
    plt.plot(time_domain_residual_EMRI[channel_combination[0]])
    plt.plot(time_domain_residual[channel_combination[0]])
    plt.show(block=True)
    # plot frequency domain data
    plt.figure()
    plt.loglog(tdi_fs_residual_EMRI['freq'], np.abs(runner.tdi_fs[channel_combination[0]]), label='Data')
    plt.loglog(tdi_fs_residual_EMRI['freq'], np.abs(tdi_fs_residual_EMRI[channel_combination[0]]), label='Residual EMRI + GB + Noise')
    plt.loglog(tdi_fs_residual['freq'], np.abs(tdi_fs_residual[channel_combination[0]]), label='Residual GB + Noise')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(10**-4, 10**-1)
    plt.title('Frequency domain data')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show(block=True)


    # save time domain data
    with h5py.File(runner.savepath+f'/time_domain_residual_Mojito_SNR_threshold_{int(runner.cfg.snr_threshold)}_seed{runner.cfg.seed}.h5', 'w') as f:
        f.create_dataset('residual', data=np.stack([time_domain_residual_EMRI[ch] for ch in channel_combination]))
        f.attrs['t0'] = runner.waveform_args["t0"]
        f.attrs['dt'] = runner.cfg.dt
        f.attrs['snr_threshold'] = runner.cfg.snr_threshold
        f.attrs['seed'] = runner.cfg.seed
        f.attrs['channel_combination'] = runner.cfg.channel_combination

    # load time domain residual h5 file
    with h5py.File(runner.savepath+f'/time_domain_residual_Mojito_SNR_threshold_{int(runner.cfg.snr_threshold)}_seed{runner.cfg.seed}.h5', 'r') as f:
        time_domain_residual_loaded = f['residual'][:]
        t0 = f.attrs['t0']
        dt = f.attrs['dt']
        channel_combination = f.attrs['channel_combination']



if __name__ == "__main__":
    main()
