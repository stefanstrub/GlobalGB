"""
Build an L2/L3-compliant GB submission from a merged GlobalGB catalogue.

Reads the HDF5 file written by ``merge_GB_signal_files.py`` (dataset
``recovered_sources``) and produces a submission directory compatible with the
``l2l3_datamodel`` checker.

Each recovered source becomes one detection entry.  By default, detections follow
the Mojito L2 convention (source metadata only, no astrophysical point estimates
in the detection table).

Posteriors can be built either from MCMC chain files produced by
``GB_posterior.py`` (recommended) or, when no chains are available, from
single-sample point estimates taken from the merged catalogue.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

from globalGB.config import (
    get_l3_submission_settings,
    l3_config_arg_defaults,
    load_config,
    load_l2l3_config,
    merged_sources_path,
    populate_global_metadata_from_search,
    resolve_config_path,
)
from globalGB.search_utils_GB import PARAM_INDICES, PARAM_NAMES
from l2l3_datamodel.structure.datamodel.metadata.metadata import GlobalMetadata
from l2l3_datamodel.structure.datamodel.posteriors.binary_posterior import (
    BinarySourcePosterior,
)
from l2l3_datamodel.structure.datamodel.posteriors.gb import GalacticBinaryPosterior
from l2l3_datamodel.structure.datamodel.sources.gb import GalacticBinary
from l2l3_datamodel.structure.io.file_factory import save_metadata_to_json
from l2l3_datamodel.structure.mock.mock_sim import (
    create_mock_dataset_metadata,
    create_mock_posterior_metadata,
)


def load_recovered_sources(path: str) -> np.ndarray:
    """Load the merged ``recovered_sources`` array from an HDF5 file."""
    with h5py.File(path, "r") as handle:
        if "recovered_sources" not in handle:
            raise KeyError(f"Dataset 'recovered_sources' not found in {path}")
        sources = handle["recovered_sources"][:]
    if sources.ndim != 2 or sources.shape[1] != len(PARAM_NAMES):
        raise ValueError(
            f"Expected recovered_sources shape (N, {len(PARAM_NAMES)}), got {sources.shape}"
        )
    return sources


def make_source_id(index: int, frequency_hz: float) -> str:
    """Build the GB source identifier ``XXXXX_YYYYYYY`` (frequency in microHz)."""
    return f"{index:05d}_{int(frequency_hz * 1e6)}"


def build_detection(
    row: np.ndarray,
    index: int,
    *,
    store_estimated_parameters: bool,
) -> GalacticBinary:
    """Create one ``GalacticBinary`` detection entry."""
    frequency = float(row[PARAM_INDICES["Frequency"]])
    source = GalacticBinary()
    source.source_id.val = make_source_id(index, frequency)
    source.quality_flag.val = 0
    source.known_injection.val = ""
    source.comment.val = ""

    if store_estimated_parameters:
        source.amplitude.val = float(row[PARAM_INDICES["Amplitude"]])
        source.frequency.val = frequency
        source.frequency_dot.val = float(row[PARAM_INDICES["FrequencyDerivative"]])
        source.right_ascension.val = float(row[PARAM_INDICES["RightAscension"]])
        source.declination.val = float(row[PARAM_INDICES["Declination"]])
        source.inclination.val = float(row[PARAM_INDICES["Inclination"]])
        source.initial_phase.val = float(row[PARAM_INDICES["InitialPhase"]])
        source.polarization.val = float(row[PARAM_INDICES["Polarization"]])

    return source


POSTERIOR_FILE_PATTERN = re.compile(r"_posteriors(\d+)_(\d+)\.h5$")


def row_to_posterior_sample(row: np.ndarray) -> GalacticBinaryPosterior:
    """Map one GlobalGB parameter row to a ``GalacticBinaryPosterior`` sample."""
    return GalacticBinaryPosterior(
        amplitude=float(row[PARAM_INDICES["Amplitude"]]),
        frequency=float(row[PARAM_INDICES["Frequency"]]),
        frequency_dot=float(row[PARAM_INDICES["FrequencyDerivative"]]),
        right_ascension=float(row[PARAM_INDICES["RightAscension"]]),
        declination=float(row[PARAM_INDICES["Declination"]]),
        inclination=float(row[PARAM_INDICES["Inclination"]]),
        initial_phase=float(row[PARAM_INDICES["InitialPhase"]]),
        polarization=float(row[PARAM_INDICES["Polarization"]]),
        loglikelihood=math.nan,
        logprior=math.nan,
    )


def build_posterior_from_row(row: np.ndarray, source_id: str) -> BinarySourcePosterior:
    """Create a single-sample posterior chain from one recovered source."""
    return BinarySourcePosterior(id=source_id, chain=[row_to_posterior_sample(row)])


def build_posterior_from_chain(
    chain: np.ndarray,
    source_id: str,
) -> BinarySourcePosterior:
    """Create a full MCMC posterior chain for one source."""
    if chain.ndim != 2 or chain.shape[1] != len(PARAM_NAMES):
        raise ValueError(
            f"Expected chain shape (N, {len(PARAM_NAMES)}), got {chain.shape}"
        )
    samples = [row_to_posterior_sample(row) for row in chain]
    return BinarySourcePosterior(id=source_id, chain=samples)


def load_grouped_source(path: str, group_index: int) -> dict | None:
    """Load one group from the grouped-sources HDF5 without reading the full file."""
    with h5py.File(path, "r") as handle:
        n_groups = int(handle.attrs["n_groups"])
        if group_index >= n_groups:
            return None
        group = handle[f"group_{group_index}"]
        return {
            "frequency_range": group["frequency_range"][:],
            "sources": group["sources"][:],
        }


def grouped_source_count(path: str) -> int:
    """Return the number of groups in a grouped-sources HDF5 file."""
    with h5py.File(path, "r") as handle:
        return int(handle.attrs["n_groups"])


def match_source_index(
    source: np.ndarray,
    recovered_sources: np.ndarray,
    used_indices: set[int],
    *,
    frequency_atol: float,
) -> int | None:
    """Match one grouped source row to an index in ``recovered_sources``."""
    frequencies = recovered_sources[:, PARAM_INDICES["Frequency"]]
    target_frequency = float(source[PARAM_INDICES["Frequency"]])
    order = np.argsort(np.abs(frequencies - target_frequency))
    for index in order:
        if index in used_indices:
            continue
        if abs(frequencies[index] - target_frequency) <= frequency_atol:
            return int(index)
    return None


def load_posterior_chains(path: str) -> Tuple[List[np.ndarray], np.ndarray | None]:
    """Load MCMC chains from one ``GB_posterior.py`` output file."""
    with h5py.File(path, "r") as handle:
        chain_group = handle["chains"]
        chains = [
            chain_group[key][:]
            for key in sorted(
                chain_group.keys(),
                key=lambda name: int(name.split("_")[1]),
            )
        ]
        initial_parameters = (
            handle["initial_parameters"][:] if "initial_parameters" in handle else None
        )
    return chains, initial_parameters


def discover_posterior_files(posterior_dir: str) -> Dict[int, str]:
    """Return ``group_index -> posterior file path``."""
    files_by_group: Dict[int, str] = {}
    for path in sorted(glob.glob(os.path.join(posterior_dir, "*_posteriors*.h5"))):
        match = POSTERIOR_FILE_PATTERN.search(os.path.basename(path))
        if match is None:
            continue
        n_leaves = int(match.group(1))
        group_index = int(match.group(2))
        if n_leaves <= 0:
            continue
        files_by_group[group_index] = path
    return files_by_group


def process_posterior_group(
    path: str,
    group: dict,
    recovered_sources: np.ndarray,
    source_ids: List[str],
    used_indices: set[int],
    *,
    frequency_atol: float,
) -> Tuple[List[Tuple[int, BinarySourcePosterior]], int, int, bool]:
    """
    Load one posterior file and pair chains with ``group["sources"]`` index-by-index.

    Returns ``(group_posteriors, mcmc_added, skipped_leaves, trimmed_extra_chains)``.
    """
    chains, _initial_parameters = load_posterior_chains(path)
    group_sources = group["sources"]
    n_sources = len(group_sources)
    trimmed = len(chains) > n_sources
    mcmc_added = 0
    skipped_leaves = 0
    group_posteriors: List[Tuple[int, BinarySourcePosterior]] = []

    try:
        for leaf_index in range(n_sources):
            if leaf_index >= len(chains):
                continue

            chain = chains[leaf_index]
            if chain.size == 0:
                continue

            source_index = match_source_index(
                group_sources[leaf_index],
                recovered_sources,
                used_indices,
                frequency_atol=frequency_atol,
            )
            if source_index is None:
                skipped_leaves += 1
                continue

            used_indices.add(source_index)
            group_posteriors.append(
                (source_index, build_posterior_from_chain(chain, source_ids[source_index]))
            )
            mcmc_added += 1
    finally:
        del chains

    return group_posteriors, mcmc_added, skipped_leaves, trimmed


def build_detections(
    recovered_sources: np.ndarray,
    *,
    store_estimated_parameters: bool,
) -> Tuple[List[GalacticBinary], List[str]]:
    """Build detection entries and source IDs for the merged catalogue."""
    detections: List[GalacticBinary] = []
    for index, row in enumerate(recovered_sources):
        detections.append(
            build_detection(
                row,
                index,
                store_estimated_parameters=store_estimated_parameters,
            )
        )
    source_ids = [detection.source_id.val for detection in detections]
    return detections, source_ids


class IncrementalSubmissionWriter:
    """Write L3 submission posteriors one source at a time (one HDF5 file each)."""

    def __init__(
        self,
        *,
        output_dir: str,
        detections: List[GalacticBinary],
        run_name: str,
        codename: str,
        version: str,
        contact: str,
        code_link: str,
        input_reference: str,
        store_estimated_parameters: bool,
        search_config: dict,
        l2l3_config: dict | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.detections = detections
        self.run_name = run_name
        self.codename = codename
        self.version = version
        self.store_estimated_parameters = store_estimated_parameters
        self.source_type = "GB"
        self.timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "")

        self.metadata = GlobalMetadata()
        self.metadata.global_fit_run_name = run_name
        self.metadata.global_fit_codename = codename
        self.metadata.global_fit_version = version
        self.metadata.global_fit_contact = contact
        self.metadata.global_fit_code_link = code_link
        self.metadata.input_reference = input_reference
        self.metadata.searched_source_types_list = ["GB"]
        self.metadata.found_source_types_list = ["GB"]
        populate_global_metadata_from_search(
            self.metadata,
            search_config,
            l2l3_config=l2l3_config,
        )
        self.metadata.submission_timestamp = self.timestamp
        self.metadata.submission_parent_folder = os.path.abspath(output_dir)

        self.dataset_metadata = create_mock_dataset_metadata([GalacticBinary()])
        self.posterior_metadata = create_mock_posterior_metadata([])

        os.makedirs(output_dir, exist_ok=True)

        self.dm_file = os.path.join(
            output_dir,
            f"{run_name}_{codename}_{version}_{self.source_type}_{self.timestamp}.h5",
        )
        self._posterior_file_by_index: Dict[int, str] = {}
        self._saved_mcmc_indices: set[int] = set()
        self._posterior_counter = 0

        save_metadata_to_json(
            os.path.join(output_dir, f"{self.source_type}_metadata.json"),
            self.dataset_metadata,
        )
        save_metadata_to_json(
            os.path.join(output_dir, "global_metadata.json"),
            self.metadata,
        )

    def save_group(
        self,
        group_posteriors: List[Tuple[int, BinarySourcePosterior]],
    ) -> None:
        """Write posteriors from one processed MCMC group and release them."""
        for source_index, posterior in group_posteriors:
            self._append_posterior(source_index, posterior)
            self._saved_mcmc_indices.add(source_index)
        group_posteriors.clear()

    def finalize(
        self,
        recovered_sources: np.ndarray,
        source_ids: List[str],
    ) -> Tuple[int, int]:
        """Write point-estimate fallbacks and the detection table."""
        mcmc_count = len(self._saved_mcmc_indices)
        point_estimate_count = 0

        for index, row in enumerate(recovered_sources):
            if index in self._saved_mcmc_indices:
                continue
            self._append_posterior(
                index,
                build_posterior_from_row(row, source_ids[index]),
            )
            point_estimate_count += 1

        self._write_detection_table()
        return mcmc_count, point_estimate_count

    def _posterior_subdir(self) -> str:
        return (
            f"{self.run_name}_{self.codename}_{self.version}_"
            f"{self.source_type}_posteriordir"
        )

    def _posterior_relative_path(self, file_index: int) -> str:
        return os.path.join(
            self._posterior_subdir(),
            f"{self.run_name}_{self.codename}_{self.version}_"
            f"{self.source_type}_posteriors_{file_index}_{self.timestamp}.h5",
        )

    def _append_posterior(
        self,
        source_index: int,
        posterior: BinarySourcePosterior,
    ) -> None:
        post_file = self._posterior_relative_path(self._posterior_counter)
        post_path = os.path.join(self.output_dir, post_file)
        os.makedirs(os.path.dirname(post_path), exist_ok=True)
        with h5py.File(post_path, "w") as handle:
            handle.create_dataset(name=f"{posterior.id}", data=np.array(posterior))
        self._posterior_file_by_index[source_index] = post_file
        self._posterior_counter += 1

    def _build_detection_array(self) -> np.ndarray:
        params = np.array(self.detections).flatten()
        if self.store_estimated_parameters:
            return params

        columns = [
            "source_id",
            "posterior_file",
            "comment",
            "quality_flag",
            "known_injection",
            "detection_statistic",
        ]
        formats = ["S100", "S100", "S100", "i8", "S100", "f8"]
        df = pd.DataFrame(params)
        detection = np.rec.fromarrays(
            [df[c].to_numpy() for c in columns],
            names=columns,
            formats=formats,
        )
        for index, post_file in self._posterior_file_by_index.items():
            detection["posterior_file"][index] = post_file
        return detection

    def _write_detection_table(self) -> None:
        detection = self._build_detection_array()
        with h5py.File(self.dm_file, "w") as h5file:
            sources_group = h5file.create_group("sources")
            sources_group.create_dataset("detection", data=detection)
            self._write_posterior_metadata_attrs(group=sources_group)

    def _write_posterior_metadata_attrs(
        self,
        group: h5py.Group,
    ) -> None:
        if self.posterior_metadata is None:
            return
        params_desc = getattr(self.posterior_metadata, "parameters_description", None)
        if not params_desc:
            return
        for key, value in params_desc.items():
            msg = ""
            for desc in value.items():
                for part in desc:
                    msg = msg + str(part) + " "
                msg = msg + " ; "
            group.attrs[key] = msg[:-2]


WHICH_RUN_CHOICES = ("even1st", "even", "odd", "global")


def normalize_argv(argv: list[str] | None) -> list[str]:
    """Map legacy positional which_run / batch-index args to argparse flags."""
    if not argv:
        return []
    if argv[0] in WHICH_RUN_CHOICES:
        rest = [arg for arg in argv[1:] if not arg.isdigit()]
        return ["--which-run", argv[0], *rest]
    return list(argv)


def resolve_settings(args: argparse.Namespace) -> dict:
    """Build final settings from config defaults plus CLI values."""
    search_config = load_config()
    l2l3_config = load_l2l3_config()
    path_base = {
        "save_path": l2l3_config.get("save_path") or search_config["save_path"],
    }

    def resolve_path(value: str | None) -> str | None:
        if not value:
            return value
        if os.path.isabs(value):
            return value
        return resolve_config_path(path_base, value)

    settings = get_l3_submission_settings(search_config, l2l3_config)
    settings.update(
        {
            "which_run": args.which_run,
            "run_name": args.run_name,
            "codename": args.codename,
            "version": args.version,
            "contact": args.contact,
            "code_link": args.code_link,
            "input_reference": args.input_reference,
            "frequency_atol": args.frequency_atol,
            "pack_size": args.pack_size,
            "store_estimated_parameters": args.store_estimated_parameters,
            "store_inline_posteriors": args.store_inline_posteriors,
            "store_pdf_files": args.store_pdf_files,
            "input_h5": resolve_path(args.input_h5),
            "output_dir": resolve_path(args.output_dir) or settings["output_dir"],
            "posterior_dir": resolve_path(args.posterior_dir) or settings["posterior_dir"],
            "grouped_sources_path": resolve_path(args.grouped_sources)
            or settings["grouped_sources_path"],
        }
    )
    settings["prefix"] = (
        f"{settings['run_name']}_{settings['codename']}_{settings['version']}"
    )
    return settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = l3_config_arg_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Convert a merged GlobalGB recovered-sources HDF5 file into an "
            "L2/L3 datamodel submission directory. Defaults are loaded from "
            "globalGB/GB_l2l3_config.json."
        )
    )
    parser.set_defaults(**defaults)
    parser.add_argument(
        "input_h5",
        nargs="?",
        default=defaults["input_h5"],
        help=(
            "Path to merged HDF5 file "
            "(default: GB_l2l3_config.input_h5, or inferred from which_run)"
        ),
    )
    parser.add_argument(
        "--which-run",
        choices=["even1st", "even", "odd", "global"],
        default=defaults["which_run"],
        help="Window set for merged catalogue inference",
    )
    parser.add_argument(
        "--output-dir",
        default=defaults["output_dir"],
        help="Submission output directory",
    )
    parser.add_argument(
        "--run-name",
        default=defaults["run_name"],
        help="global_fit_run_name",
    )
    parser.add_argument(
        "--codename",
        default=defaults["codename"],
        help="global_fit_codename",
    )
    parser.add_argument(
        "--version",
        default=defaults["version"],
        help="global_fit_version",
    )
    parser.add_argument(
        "--contact",
        default=defaults["contact"],
        help="Contact e-mail stored in global metadata",
    )
    parser.add_argument(
        "--code-link",
        default=defaults["code_link"],
        help="Analysis code link in global metadata",
    )
    parser.add_argument(
        "--input-reference",
        default=defaults["input_reference"],
        help="Input dataset reference",
    )
    parser.add_argument(
        "--store-estimated-parameters",
        action=argparse.BooleanOptionalAction,
        default=defaults["store_estimated_parameters"],
        help="Store astrophysical point estimates in sources/detection",
    )
    parser.add_argument(
        "--posterior-dir",
        default=defaults["posterior_dir"],
        help="MCMC posterior directory",
    )
    parser.add_argument(
        "--grouped-sources",
        default=defaults["grouped_sources"],
        help="Grouped-sources HDF5",
    )
    parser.add_argument(
        "--frequency-atol",
        type=float,
        default=defaults["frequency_atol"],
        help="Frequency match tolerance",
    )
    parser.add_argument(
        "--store-pdf-files",
        action=argparse.BooleanOptionalAction,
        default=defaults["store_pdf_files"],
        help=(
            "Write posteriors to external HDF5 packs "
            "(default from config, or auto when MCMC posteriors exist)"
        ),
    )
    parser.add_argument(
        "--store-inline-posteriors",
        action=argparse.BooleanOptionalAction,
        default=defaults["store_inline_posteriors"],
        help="Store posterior chains inside the main submission HDF5 file",
    )
    parser.add_argument(
        "--pack-size",
        type=int,
        default=defaults["pack_size"],
        help="Posteriors per external HDF5 pack",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(normalize_argv(argv))
    settings = resolve_settings(args)

    input_path = settings["input_h5"] or merged_sources_path(
        settings["search_config"], settings["which_run"]
    )
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    recovered_sources = load_recovered_sources(input_path)
    print(f"Loaded {len(recovered_sources)} recovered sources from {input_path}")

    posterior_dir = settings["posterior_dir"]
    grouped_sources_path = settings["grouped_sources_path"]
    has_posterior_dir = bool(posterior_dir) and os.path.isdir(posterior_dir)
    posterior_files: Dict[int, str] = {}
    if has_posterior_dir:
        posterior_files = discover_posterior_files(posterior_dir)

    use_mcmc_posteriors = bool(posterior_files)
    if use_mcmc_posteriors:
        if not os.path.isfile(grouped_sources_path):
            raise FileNotFoundError(
                f"Grouped sources file not found: {grouped_sources_path}"
            )
        print(
            f"Grouped sources catalogue: "
            f"{grouped_source_count(grouped_sources_path)} groups "
            f"in {grouped_sources_path}"
        )
        print(f"Loaded MCMC posteriors from: {posterior_dir}")
    elif has_posterior_dir:
        print(
            f"No MCMC chain files in {posterior_dir}; "
            "using point-estimate posteriors"
        )
    else:
        print(
            "MCMC posterior directory not found; using point-estimate posteriors: "
            f"{settings['posterior_dir']}"
        )

    detections, source_ids = build_detections(
        recovered_sources,
        store_estimated_parameters=settings["store_estimated_parameters"],
    )
    mcmc_count = 0
    point_estimate_count = 0
    trimmed_chain_files = 0
    skipped_group_files = 0
    skipped_leaves = 0

    writer = IncrementalSubmissionWriter(
        output_dir=settings["output_dir"],
        detections=detections,
        run_name=settings["run_name"],
        codename=settings["codename"],
        version=settings["version"],
        contact=settings["contact"],
        code_link=settings["code_link"],
        input_reference=settings["input_reference"],
        store_estimated_parameters=settings["store_estimated_parameters"],
        search_config=settings["search_config"],
        l2l3_config=settings["l2l3_config"],
    )

    if use_mcmc_posteriors:
        used_indices: set[int] = set()
        for group_index, path in sorted(posterior_files.items()):
            group = load_grouped_source(grouped_sources_path, group_index)
            if group is None:
                skipped_group_files += 1
                continue

            print(f"Processing posterior group {group_index}: {os.path.basename(path)}")
            group_posteriors, added, skipped, trimmed = process_posterior_group(
                path,
                group,
                recovered_sources,
                source_ids,
                used_indices,
                frequency_atol=settings["frequency_atol"],
            )
            del group

            writer.save_group(group_posteriors)
            del group_posteriors

            skipped_leaves += skipped
            if trimmed:
                trimmed_chain_files += 1
            print(f"  Saved {added} MCMC posterior(s)")

        if trimmed_chain_files:
            print(
                f"Trimmed extra RJMCMC chains in {trimmed_chain_files} "
                f"posterior file(s)"
            )
        if skipped_group_files:
            print(
                f"Skipped {skipped_group_files} posterior file(s) with "
                f"out-of-range group indices"
            )
        if skipped_leaves:
            print(
                f"Skipped {skipped_leaves} chain leaf(s) with no matching detection"
            )

    mcmc_count, point_estimate_count = writer.finalize(recovered_sources, source_ids)

    if use_mcmc_posteriors:
        print(f"  MCMC posteriors: {mcmc_count}")
        if point_estimate_count:
            print(
                f"  Point-estimate fallbacks: {point_estimate_count} "
                "(no matching MCMC chain found)"
            )
    else:
        print(f"  Point-estimate posteriors: {point_estimate_count}")

    print(f"Saved L3 submission to: {os.path.abspath(settings['output_dir'])}")
    print(
        "Validate with: "
        f"l2l3_datamodel-develop/run_checker.sh --dir {settings['output_dir']}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
