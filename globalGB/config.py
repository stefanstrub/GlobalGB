"""Locate and load GB search and L2/L3 submission configuration files."""

from __future__ import annotations

import json
import math
import os
from importlib.resources import files
from typing import Any

from l2l3_datamodel.structure.datamodel.metadata.metadata import PreprocessingMetadata


def get_config_path() -> str:
    """Return the path to ``GB_search_config.json``.

    Resolution order:
    1. ``LDC_CONFIG`` environment variable
    2. ``./globalGB/GB_search_config.json`` in the current working directory
    """
    if env_path := os.environ.get("LDC_CONFIG"):
        if not os.path.isfile(env_path):
            raise FileNotFoundError(f"LDC_CONFIG points to missing file: {env_path}")
        return env_path

    cwd_path = os.path.join(os.getcwd(), "globalGB", "GB_search_config.json")
    if os.path.isfile(cwd_path):
        return cwd_path

    example = files("globalGB").joinpath("GB_search_config.json.example")
    raise FileNotFoundError(
        "GB search config not found. Either:\n"
        "  - set LDC_CONFIG to your config file path, or\n"
        f"  - create globalGB/GB_search_config.json in the working directory\n"
        f"    (see example shipped with the package: {example})"
    )


def load_config() -> dict[str, Any]:
    """Load and return the GB search configuration as a dictionary."""
    with open(get_config_path(), encoding="utf-8") as f:
        return json.load(f)


def get_l2l3_config_path() -> str:
    """Return the path to ``GB_l2l3_config.json``.

    Resolution order:
    1. ``L2L3_CONFIG`` environment variable
    2. ``./globalGB/GB_l2l3_config.json`` in the current working directory
    """
    if env_path := os.environ.get("L2L3_CONFIG"):
        if not os.path.isfile(env_path):
            raise FileNotFoundError(f"L2L3_CONFIG points to missing file: {env_path}")
        return env_path

    cwd_path = os.path.join(os.getcwd(), "globalGB", "GB_l2l3_config.json")
    if os.path.isfile(cwd_path):
        return cwd_path

    example = files("globalGB").joinpath("GB_l2l3_config.json.example")
    raise FileNotFoundError(
        "GB L2/L3 config not found. Either:\n"
        "  - set L2L3_CONFIG to your config file path, or\n"
        f"  - create globalGB/GB_l2l3_config.json in the working directory\n"
        f"    (see example shipped with the package: {example})"
    )


def load_l2l3_config() -> dict[str, Any]:
    """Load and return the GB L2/L3 submission configuration."""
    with open(get_l2l3_config_path(), encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(config: dict[str, Any], path: str) -> str:
    """Resolve a config path relative to ``save_path`` when not absolute."""
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.join(config["save_path"], path)


def mojito_preprocessing_pipeline_kwargs(dt: float) -> dict[str, Any]:
    """
    Return Mojito preprocessing pipeline kwargs used by ``LISADataLoader``.

    Delegates to :func:`mojito_barkeeper.defaults.mojito_preprocessing_pipeline_kwargs`.
    """
    from mojito_barkeeper.defaults import (
        mojito_preprocessing_pipeline_kwargs as _pipeline_kwargs,
    )

    return _pipeline_kwargs(dt)


def build_preprocessing_metadata(
    search_config: dict[str, Any],
    l2l3_config: dict[str, Any] | None = None,
) -> PreprocessingMetadata:
    """Build L2/L3 ``preprocessing_metadata`` from the GB search pipeline settings."""
    pipeline = mojito_preprocessing_pipeline_kwargs(float(search_config["dt"]))
    overrides = (l2l3_config or {}).get("preprocessing", {})

    downsample = {**pipeline["downsample_kwargs"], **overrides.get("downsample_kwargs", {})}
    filters = {**pipeline["filter_kwargs"], **overrides.get("filter_kwargs", {})}
    trim = {**pipeline["trim_kwargs"], **overrides.get("trim_kwargs", {})}
    window = {**pipeline["window_kwargs"], **overrides.get("window_kwargs", {})}

    lowpass_cutoff = filters.get("lowpass_cutoff")
    if lowpass_cutoff is None:
        lowpass_cutoff = math.nan

    metadata = PreprocessingMetadata()
    metadata.highpass_kwargs.cutoff = float(filters["highpass_cutoff"])
    metadata.highpass_kwargs.order = int(filters["order"])
    metadata.highpass_kwargs.zero_phase = bool(filters.get("zero_phase", True))

    metadata.lowpass_kwargs.cutoff = float(lowpass_cutoff)
    metadata.lowpass_kwargs.order = int(filters["order"])
    metadata.lowpass_kwargs.zero_phase = bool(filters.get("zero_phase", True))

    metadata.downsample_kwargs.target_fs = float(downsample["target_fs"])
    metadata.downsample_kwargs.window = [str(window["window"])]

    metadata.trim_kwargs.duration = float(trim["fraction"])
    metadata.trim_kwargs.is_percent = True
    metadata.trim_kwargs.trimming_type = "symmetric"

    if "tobs" in overrides:
        metadata.tobs = float(overrides["tobs"])
    else:
        metadata.tobs = math.nan

    return metadata


def populate_global_metadata_from_search(
    metadata,
    search_config: dict[str, Any],
    *,
    l2l3_config: dict[str, Any] | None = None,
) -> None:
    """Fill global metadata fields derived from the GB search preprocessing pipeline."""
    pipeline = mojito_preprocessing_pipeline_kwargs(float(search_config["dt"]))
    window = pipeline["window_kwargs"]

    metadata.preprocessing_metadata = build_preprocessing_metadata(
        search_config,
        l2l3_config,
    )
    metadata.window_type = str(window["window"])
    metadata.window_alpha = float(window["alpha"])
    metadata.time_step = float(search_config["dt"])

    channels = search_config.get("channel_combination", "")
    metadata.tdi_channels = list(channels) if isinstance(channels, str) else list(channels)

    frequency_range = search_config.get("frequency_range")
    if frequency_range:
        metadata.domain_metadata.kwargs.min_freq = float(frequency_range[0])
        metadata.domain_metadata.kwargs.max_freq = float(frequency_range[1])

    metadata.sensitivity_metadata.kwargs.tdi_generation = int(
        search_config.get("tdi_generation", 0)
    )


def get_l3_submission_settings(
    search_config: dict[str, Any] | None = None,
    l2l3_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return resolved L3 submission names and paths from GB configs."""
    search_config = search_config or load_config()
    l2l3_config = l2l3_config or load_l2l3_config()

    path_base = {
        "save_path": l2l3_config.get("save_path") or search_config["save_path"],
    }

    run_name = l2l3_config.get("run_name", "CD1Lrun2")
    codename = l2l3_config.get("codename", "GlobalGB")
    version = l2l3_config.get("version", "v1")
    prefix = f"{run_name}_{codename}_{version}"

    grouped_default = (
        f"grouped_found_sources_{search_config['data_set']}_"
        f"SNR_threshold_{int(search_config['snr_threshold'])}_"
        f"seed{search_config['seed']}.h5"
    )
    grouped_sources = l2l3_config.get("grouped_sources_path") or grouped_default

    input_h5_default = (
        f"found_signals_{search_config['data_set']}_SNR_threshold_"
        f"{int(search_config['snr_threshold'])}_seed{search_config['seed']}.h5"
    )
    input_h5 = l2l3_config.get("input_h5") or input_h5_default

    store_pdf_files = l2l3_config.get("store_pdf_files")
    if store_pdf_files is not None:
        store_pdf_files = bool(store_pdf_files)

    return {
        "run_name": run_name,
        "codename": codename,
        "version": version,
        "prefix": prefix,
        "which_run": l2l3_config.get("which_run", "global"),
        "contact": l2l3_config.get("contact", ""),
        "code_link": l2l3_config.get("code_link", ""),
        "input_reference": l2l3_config.get("input_reference") or search_config.get(
            "data_set", ""
        ),
        "frequency_atol": float(l2l3_config.get("frequency_atol", 1e-12)),
        "pack_size": int(l2l3_config.get("pack_size", 20)),
        "store_estimated_parameters": bool(
            l2l3_config.get("store_estimated_parameters", False)
        ),
        "store_inline_posteriors": bool(
            l2l3_config.get("store_inline_posteriors", False)
        ),
        "store_pdf_files": store_pdf_files,
        "input_h5": resolve_config_path(path_base, input_h5) if input_h5 else None,
        "output_dir": resolve_config_path(
            path_base, l2l3_config.get("output_dir", "l3_submission")
        ),
        "posterior_dir": resolve_config_path(
            path_base, l2l3_config.get("posterior_dir", f"{prefix}_GB_posteriordir")
        ),
        "grouped_sources_path": resolve_config_path(path_base, grouped_sources),
        "search_config": search_config,
        "l2l3_config": l2l3_config,
    }


def l3_config_arg_defaults() -> dict[str, Any]:
    """Return argparse defaults loaded from ``GB_l2l3_config.json``."""
    settings = get_l3_submission_settings()
    return {
        "which_run": settings["which_run"],
        "output_dir": settings["output_dir"],
        "run_name": settings["run_name"],
        "codename": settings["codename"],
        "version": settings["version"],
        "contact": settings["contact"],
        "code_link": settings["code_link"],
        "input_reference": settings["input_reference"],
        "posterior_dir": settings["posterior_dir"],
        "grouped_sources": settings["grouped_sources_path"],
        "frequency_atol": settings["frequency_atol"],
        "pack_size": settings["pack_size"],
        "store_estimated_parameters": settings["store_estimated_parameters"],
        "store_inline_posteriors": settings["store_inline_posteriors"],
        "store_pdf_files": settings["store_pdf_files"],
        "input_h5": settings["input_h5"],
    }


def merged_sources_path(search_config: dict[str, Any], which_run: str) -> str:
    """Return the merged recovered-sources HDF5 path for a window set."""
    base_found_dir = search_config["save_path"]
    if which_run == "global":
        file_name = (
            f"/found_signals_{search_config['data_set']}_SNR_threshold_"
            f"{int(search_config['snr_threshold'])}_seed{search_config['seed']}.h5"
        )
        output_base = base_found_dir
    else:
        file_name = (
            f"/found_signals_{search_config['data_set']}_SNR_threshold_"
            f"{int(search_config['snr_threshold'])}_{which_run}_"
            f"seed{search_config['seed']}.h5"
        )
        output_base = base_found_dir
        if which_run in ("odd", "even"):
            output_base = (
                base_found_dir
                + f"/found_signals_{search_config['data_set']}_SNR_threshold_"
                f"{int(search_config['snr_threshold'])}_global_seed"
                f"{search_config['seed']}"
            )
    return output_base + file_name


def l3_posterior_chain_path(
    settings: dict[str, Any],
    *,
    n_leaves: int,
    group_index: int,
) -> str:
    """Return the MCMC posterior HDF5 path for one grouped source batch."""
    filename = f"{settings['prefix']}_GB_posteriors{n_leaves}_{group_index}.h5"
    return os.path.join(settings["posterior_dir"], filename)
