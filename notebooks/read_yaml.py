import numpy as np
import numpy.lib.recfunctions as recf
import pandas as pd
import os
import yaml
import h5py
import xarray as xr
import matplotlib.pyplot as plt

from ldc.common.series import TimeSeries
import ldc.waveform.fastGB as fastGB
try:
    from ldc.common.series import window ### manual install of  ldc
except:
    from ldc.common.tools import window ### pip install of ldc


# get current directory
path = os.getcwd()
# parent directory
parent = os.path.dirname(path)
# grandparent directory
grandparent = os.path.dirname(parent)
datapath = grandparent+"/LDC/Sangria/evaluation/eth/v3/parameters_yaml/"

def download(yml_solution):
    L = yaml.load(open(yml_solution, "r"), yaml.Loader)
    df = pd.DataFrame(L["estimates"])
    names = list(df.columns)
    cat = np.rec.fromarrays([np.nan*np.zeros((len(df)))]*len(names),
                            names=names)
    for i_n, n in enumerate(df.columns):
        v = [df[n][i] for i in range(len(df))]
        v = np.array([float(e) for e in v])
        cat[n] = v[:]
    units = L["units"]
    if units["Frequency"]=='mHz':
        cat["Frequency"] *= 1e-3
        units["Frequency"] = "Hz"
    negative_longitude_mask = cat["EclipticLongitude"] < 0
    cat["EclipticLongitude"][negative_longitude_mask] += 2*np.pi
    idx2 = np.arange(len(df))
    cat = recf.append_fields(cat, ['index'], [idx2], usemask=False)
    return cat, units


yml_solution = str(datapath+"ETH-LDC2-sangria-training-v2-training_GB.yaml")

cat, units = download(yml_solution)
print(cat)
print(units)

seed = 42
weeks = 52
Tobs = float((weeks)*7*24*3600)
SNR_threshold = 9
HM = False


data_set = 'Windowed'
if data_set == 'Radler':
    DATAPATH = grandparent+"/LDC/Radler/data"
    SAVEPATH = grandparent+"/LDC/pictures/LDC1-4/"
if data_set == 'Sangria':
    DATAPATH = grandparent+"/LDC/Sangria/data"
    SAVEPATH = grandparent+"/LDC/pictures/Sangria/"
    MBHBPATH = grandparent+"/LDC/MBHB/"

if data_set == 'Radler':
    sangria_fn = DATAPATH + "/LDC1-4_GB_v2.hdf5"
    # sangria_fn = DATAPATH + "/LDC1-3_VGB_v2_FD_noiseless.hdf5"
    # sangria_fn = DATAPATH + "/LDC1-3_VGB_v2.hdf5"
    fid = h5py.File(sangria_fn)
if data_set == 'Sangria':
    sangria_fn = DATAPATH + "/LDC2_sangria_training_v2.h5"
    fid = h5py.File(sangria_fn)



# get TDI 
if data_set == 'Radler':
    td = np.array(fid["H5LISA/PreProcess/TDIdata"])
    td = np.rec.fromarrays(list(td.T), names=["t", "X", "Y", "Z"])
    dt = float(np.array(fid['H5LISA/GWSources/GalBinaries']['Cadence']))
    # Tobs = float(int(np.array(fid['H5LISA/GWSources/GalBinaries']['ObservationDuration']))/reduction)
if data_set == 'Sangria':
    td = fid["obs/tdi"][()]
    td = np.rec.fromarrays(list(td.T), names=["t", "X", "Y", "Z"])
    td = td['t']
    dt = td["t"][1]-td["t"][0]

# Build timeseries and frequencyseries object for X,Y,Z
t_max_index = np.searchsorted(td['t'], Tobs)
tdi_ts = dict([(k, TimeSeries(td[k][:t_max_index], dt=dt, t0=td.t[0])) for k in ["X", "Y", "Z"]])
# tdi_ts = dict([(k, TimeSeries(td[k][:int(len(td[k][:])/reduction)], dt=dt, t0=td.t[0])) for k in ["X", "Y", "Z"]])
tdi_fs = xr.Dataset(dict([(k, tdi_ts[k].ts.fft(win=window)) for k in ["X", "Y", "Z"]]))
GB = fastGB.FastGB(delta_t=dt, T=Tobs)  # in seconds


plt.figure(figsize=(10, 6))
plt.plot(tdi_ts["X"].t, tdi_ts["X"].x, label="X")
plt.plot(tdi_ts["Y"].t, tdi_ts["Y"].x, label="Y")
plt.plot(tdi_ts["Z"].t, tdi_ts["Z"].x, label="Z")
plt.xlabel("Time [s]")
plt.ylabel("TDI")
plt.legend()
plt.show()
