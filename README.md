# GlobalGB

Pipeline for searching **Galactic Binaries (GBs)** in Mojito/LDC-style LISA data,
merging the per-window results, and matching recovered sources against an
injected catalogue.

```
conda create -n global_gb python=3.12
conda activate global_gb
```

```
pip install -r requirements.txt
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mojito-processor
```

## How to run

The typical workflow is:

- **Search GBs** in many frequency windows (often as an array job): `GB_search.py`
- **Merge** all per-batch outputs into a single catalogue file: `merge_GB_signal_files.py`
- **Match** the merged catalogue against injected sources: `match_GBs.py`
- **MCMC** create a MCMC chain for the posterior distribution: `GB_posterior.py`

All scripts read `globalGB/GB_search_config.json`.

## `globalGB/GB_search_config.json`

Example:

```json
{
  "data_set": "Mojito",
  "dt": 2.5,
  "snr_threshold": 9.0,
  "tdi_generation": 2,
  "max_signals_per_window": 10,
  "max_signals_per_window_first_run": 3,
  "channel_combination": "AET",
  "frequency_range": [0.0003, 0.05],
  "seed": 1,
  "batch_size": 10,
  "data_path": "/path/to/data.h5",
  "save_path": "/path/to/output_dir",
  "catalog_path": "/path/to/catalogues/injections",
  "match_criteria": "overlap",
  "overlap_threshold": 0.9,
  "scaled_error_threshold": 0.3
}
```

Field meaning:

- **`data_set`**: label used in output naming (e.g. `"Mojito"`).
- **`dt`**: sampling time used when loading/processing TDI channels.
- **`snr_threshold`**: SNR threshold for declaring a recovered signal.
- **`tdi_generation`**: TDI generation scheme passed to the waveform generator.
- **`max_signals_per_window`**: maximum sources to recover per window (main runs).
- **`max_signals_per_window_first_run`**: maximum sources per window for the first
  even pass (`which_run=even1st`), typically smaller for speed.
- **`channel_combination`**: TDI channels to use (e.g. `"AET"`).
- **`frequency_range`**: global search band \([f_min, f_max]\) in Hz.
- **`seed`**: RNG seed used by the search (reproducibility).
- **`batch_size`**: number of frequency windows handled by one job (one `batch_index`).
- **`data_path`**: input HDF5 data file. Must include the `orbits` group used to
  build interpolated orbits for waveform generation.
- **`save_path`**: base output directory (all result files are written here).
- **`catalog_path`**: path to catalogues used by the loader (matching stage).
- **Matching-only**
  - **`match_criteria`**: `"overlap"` (higher is better) or `"scaled_error"`
    (lower is better).
  - **`overlap_threshold`**: threshold for counting a match as successful when
    using `"overlap"`.
  - **`scaled_error_threshold`**: threshold for counting a match as successful
    when using `"scaled_error"`.

## 1) Search: `GB_search.py`

`GB_search.py` runs the GB search for **one batch of frequency windows**.

Usage:

```bash
python GB_search.py <which_run> <batch_index>
```

Arguments:

- **`batch_index`**: zero-based index of the window batch (slice) to process.
- **`which_run`**: one of:
  - **`even1st`**: first pass over even-numbered windows (no subtraction)
  - **`odd`**: pass over odd-numbered windows (subtracts even-window sources)
  - **`even`**: second pass over selected even windows (subtracts odd-window sources)

### What it writes

For each run it writes one HDF5 file per batch to a directory under `save_path`:

- Directory:
  - `save_path/found_signals_<data_set>_SNR_threshold_<snr>_<which_run>_seed<seed>/`
- Files inside:
  - `found_signals_batch_index_<batch_index>_<f0>nHz_to<f1>nHz_<...>.h5`

Each per-batch `.h5` contains:

- `recovered_sources`: pre-flattened array of found source parameters
- `wall_times`: per-window wall-clock search times

These per-batch files are intended to be merged using `merge_GB_signal_files.py`.

## 2) Merge per-batch outputs: `merge_GB_signal_files.py`

After a set of `GB_search.py` jobs completes for a given `which_run`, merge the
per-batch HDF5 files into a single recovered-source catalogue.

Usage:

```bash
python merge_GB_signal_files.py <which_run>
```

Example:

```bash
python merge_GB_signal_files.py Mojito even1st
```

### What it writes

It produces the merged outputs under `save_path`:

- `found_signals_<data_set>_SNR_threshold_<snr>_<which_run>_seed<seed>.h5`

The HDF5 file contains the dataset:

- `recovered_sources`: \(N \times P\) array of recovered sources, sorted by frequency

### Why merging matters

The later search passes use subtraction and expect merged catalogues to exist:

- Before running **`odd`**, merge **`even1st`**.
- Before running **`even`**, merge **`odd`**.

## 3) Match recovered vs injected: `match_GBs.py`

`match_GBs.py` compares recovered sources against injected sources from the
catalogue, using either **overlap** or **scaled error**.

Run:

```bash
python match_GBs.py
```

### Inputs it expects

It reads:

- `globalGB/GB_search_config.json`
- the data file at `data_path` (to build waveforms consistently)
- a merged recovered-source file under `save_path` named:
  - `found_signals_<save_name>.h5`

where `save_name` is constructed inside `match_GBs.py` as:

- `<data_set>_SNR_threshold_<snr>_seed<seed>`

So make sure the merged file you want to match is present under `save_path` with
that naming convention (or adjust the script if you want to match `even1st/odd/even`
specifically).

### What it writes

- Cached matching arrays (so reruns are fast):
  - `save_path/found_sources_<save_name>_<match_criteria>.h5`
- Final results:
  - `save_path/match_results_<save_name>_<match_criteria>.h5`

## End-to-end example

From the repository root, with `globalGB/GB_search_config.json` pointing to the
correct `data_path` and `save_path`:

1) even windows first pass (run many batch indices)
```bash
python GB_search.py even1st 150 # will search four signals in the 150th frequency segment batch
```
To run all batches in parallel using slurm. The highest batch index is determined by int(len(frequencies_even)/batch_size)
```bash
#!/bin/bash
#SBATCH -n 1
#SBATCH --time=4:00:00
#SBATCH --array=0-248
#SBATCH --mem-per-cpu=8000
python GB_search.py even1st $SLURM_ARRAY_TASK_ID # $SLURM_ARRAY_TASK_ID is the integer of the batch index
```

2) merge even1st (needed for odd subtraction)
```bash
python merge_GB_signal_files.py even1st
```
3) odd windows (run many batch indices)
```bash
python GB_search.py odd $SLURM_ARRAY_TASK_ID # $SLURM_ARRAY_TASK_ID is the integer of the batch index
python merge_GB_signal_files.py odd
```
4) even windows second pass
```bash
python GB_search.py even $SLURM_ARRAY_TASK_ID # $SLURM_ARRAY_TASK_ID is the integer of the batch index
python merge_GB_signal_files.py even 
```
5) merge odd and even segments
```bash
python merge_GB_signal_files.py global
```
6) match recovered vs injected
```bash
python match_GBs.py
```
7) TODO: get posterior