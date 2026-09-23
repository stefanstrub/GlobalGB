"""Default Mojito ``process_pipeline`` keyword arguments."""

from __future__ import annotations

from typing import Any


def mojito_preprocessing_pipeline_kwargs(dt: float) -> dict[str, Any]:
    """
    Return default preprocessing kwargs for a target cadence *dt* in seconds.

    Values match ``DataLoader.data_loader.LISADataLoader._load_mojito`` and the
    GB search pipeline configuration.
    """
    target_fs = 1.0 / dt
    return {
        "downsample_kwargs": {
            "target_fs": target_fs,
            "kaiser_window": 31.0,
        },
        "filter_kwargs": {
            "highpass_cutoff": 5e-6,
            "lowpass_cutoff": 0.5 * target_fs,
            "order": 2,
            "zero_phase": True,
        },
        "trim_kwargs": {
            "fraction": 0.02,
        },
        "window_kwargs": {
            "window": "tukey",
            "alpha": 0.0125,
        },
    }
