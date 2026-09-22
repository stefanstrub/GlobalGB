# Mojito barkeeper

Interactive desktop tool and Python library for working with **Mojito L1** HDF5
data: combine noise, MBHB, GB, EMRI, and SOBHB packages by addition and
subtraction, run ``MojitoProcessor.process_pipeline``, inspect time series and
spectra, and save results in a compact or Mojito L1–compatible format.

## Installation

Clone or copy this directory, then create a conda environment and install the
package in editable mode:

```bash
./install_laptop.sh
```

This creates the conda environment ``mojito_barkeeper`` (Python 3.12) and runs
``pip install -e .``.

Manual install:

```bash
conda create -n mojito_barkeeper python=3.12 -y
conda activate mojito_barkeeper
pip install -r requirements.txt
pip install -e .
```

### Dependencies

Runtime dependencies are listed in ``requirements.txt``:

- ``numpy``, ``matplotlib``, ``h5py``
- ``mojito`` — read Mojito L1 files
- ``mojito-processor`` — ``process_pipeline`` preprocessing

Tkinter is required for the GUI (included with most Python builds on Linux).

## Running the GUI

After installation:

```bash
mojito-barkeeper-gui
```

or:

```bash
./run_gui.sh
```

or:

```bash
python -m mojito_barkeeper.gui
```

Optional arguments:

- ``--data-root PATH`` — initial directory to scan for ``.h5`` packages
- ``--dt SECONDS`` — default target cadence (default 10 s)
- ``--ui-scale FACTOR`` — display scale for HiDPI screens (e.g. 2 on 4K)

GUI preferences (last data directory, save options, display scale) are stored in
``~/.config/mojito_barkeeper/gui.json``.

## Library use

```python
from mojito_barkeeper import (
    DataPackage,
    PipelineSettings,
    TimeWindow,
    discover_packages,
    process,
    save_result,
    SaveOptions,
)

packages = discover_packages("/path/to/LDC/Mojito/data")
packages[0].enabled = True

settings = PipelineSettings.from_search_config(dt=10.0, channels="AET")
result = process(packages, settings, window=TimeWindow(start=0, end=30 * 86400))

save_result(
    "processed.h5",
    result,
    options=SaveOptions(as_l1=True, include_auxiliary=False),
)
```

### Skipping preprocessing

For **saved pipeline outputs** (files with a ``globalgb/`` provenance group), use
``PipelineSettings(apply_preprocessing=False)`` or enable **Skip preprocessing**
in the GUI so the time series is not downsampled or trimmed multiple times.

### Default preprocessing parameters

``mojito_barkeeper.defaults.mojito_preprocessing_pipeline_kwargs(dt)`` returns
the same filter, downsample, trim, and window settings used by the GB search
pipeline for a target cadence ``dt`` in seconds.

## Unit tests

```bash
./run_unit_tests.sh
```

or, with the environment activated:

```bash
pytest
```

## Package layout

```
mojito_barkeeper-develop/
├── README.md
├── pyproject.toml
├── requirements.txt
├── install_laptop.sh
├── run_gui.sh
├── run_unit_tests.sh
├── src/mojito_barkeeper/
│   ├── __init__.py      # public API
│   ├── defaults.py      # default process_pipeline kwargs
│   ├── lab.py           # core logic (no Tk)
│   └── gui.py           # Tkinter / matplotlib GUI
└── test/
    └── test_lab.py
```
