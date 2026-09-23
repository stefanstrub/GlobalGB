"""
Core logic for interactive Mojito L1 data processing.

Loads Mojito L1 data packages (noise, MBHB, GB, EMRI, SOBHB), combines them by
adding or subtracting their TDI time series, and runs ``MojitoProcessor``'s
``process_pipeline`` on the result.

The GUI in ``mojito_barkeeper.gui`` is a thin layer on top of this module;
everything here is importable and usable from a script or notebook.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Sequence

import numpy as np

SECONDS_PER_DAY = 86400.0
SECONDS_PER_HOUR = 3600.0

#: Multipliers converting a user-facing time unit into seconds.
TIME_UNITS = {"days": SECONDS_PER_DAY, "hours": SECONDS_PER_HOUR, "seconds": 1.0}

#: Target cadences offered in the GUI; ``None`` keeps the file's native rate.
CADENCE_CHOICES = (None, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 15.0, 30.0)

#: Where the GUI remembers its settings between sessions.
STATE_PATH = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config")),
    "mojito_barkeeper",
    "gui.json",
)

#: Previous GlobalGB GUI config location, read once for migration.
_LEGACY_STATE_PATH = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config")),
    "globalgb",
    "mojito_gui.json",
)

#: Display scale factors offered in the GUI.
UI_SCALE_CHOICES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0)
MIN_UI_SCALE = 0.5
MAX_UI_SCALE = 4.0

XYZ_CHANNELS = ("X", "Y", "Z")
AET_CHANNELS = ("A", "E", "T")

#: Maps the L1 channel name to the ``MojitoL1File.tdis`` attribute holding it.
TDI_ATTRIBUTES = {
    "X": "x2",
    "Y": "y2",
    "Z": "z2",
    "A": "a2",
    "E": "e2",
    "T": "t2",
}

#: Maps GUI / pipeline channel names to Mojito L1 ``tdis`` dataset names.
L1_TDI_DATASETS = {
    "X": "X2",
    "Y": "Y2",
    "Z": "Z2",
    "A": "A2",
    "E": "E2",
    "T": "T2",
}

#: Directory name -> human readable package family, used when scanning a data root.
PACKAGE_FAMILIES = {
    "INSTRUMENT": "Noise",
    "MBHB": "MBHB",
    "GB": "GB",
    "EMRI": "EMRI",
    "SOBHB": "SOBHB",
    "COMBINED": "Combined",
}

#: Matches the ``source0`` / ``source_all`` tag Mojito puts in L1 filenames.
SOURCE_TAG = re.compile(r"source_?([A-Za-z0-9]+)")


class PipelineError(RuntimeError):
    """Raised when packages cannot be combined or the pipeline cannot run."""


@dataclass(frozen=True)
class TimeWindow:
    """
    Slice of an L1 time series, in seconds measured from the start of the file.

    ``end=None`` reads to the end of the file. Trimming here happens while
    reading from disk, which is what keeps a handful of 5.6 GB packages within
    memory.
    """

    start: float = 0.0
    end: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must not be negative, got {self.start}")
        if self.end is not None and self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be later than start ({self.start})")

    @classmethod
    def from_unit(
        cls, start: float, end: float | None, unit: str = "days"
    ) -> "TimeWindow":
        """Build a window from values given in ``days``, ``hours``, or ``seconds``."""
        try:
            scale = TIME_UNITS[unit]
        except KeyError as error:
            raise ValueError(
                f"Unknown time unit '{unit}'; expected one of {sorted(TIME_UNITS)}"
            ) from error
        return cls(start=start * scale, end=None if end is None else end * scale)

    @property
    def duration(self) -> float | None:
        return None if self.end is None else self.end - self.start

    def sample_range(self, fs: float, n_total: int) -> tuple[int, int]:
        """Convert the window into ``[start, stop)`` sample indices."""
        start = max(0, min(int(round(self.start * fs)), n_total))
        stop = n_total if self.end is None else min(n_total, int(round(self.end * fs)))
        return start, stop

    def describe(self) -> str:
        end = "end of file" if self.end is None else f"{self.end / SECONDS_PER_DAY:g} d"
        return f"{self.start / SECONDS_PER_DAY:g} d → {end}"


@dataclass
class GuiState:
    """Preferences the GUI carries over between sessions."""

    data_root: str = ""
    ui_scale: float | None = None
    save_as_l1: bool = True
    save_include_auxiliary: bool = False
    skip_preprocessing: bool = False

    def as_dict(self) -> dict:
        state: dict = {
            "data_root": self.data_root,
            "save_as_l1": self.save_as_l1,
            "save_include_auxiliary": self.save_include_auxiliary,
            "skip_preprocessing": self.skip_preprocessing,
        }
        if self.ui_scale is not None:
            state["ui_scale"] = self.ui_scale
        return state


def clamp_ui_scale(scale: float) -> float:
    """Keep a display scale inside a range that stays usable."""
    return min(MAX_UI_SCALE, max(MIN_UI_SCALE, float(scale)))


def suggest_ui_scale(screen_width: int, screen_height: int) -> float:
    """
    Guess a display scale from the screen size.

    X servers routinely report a placeholder 96 DPI even on HiDPI panels, which
    leaves Tk drawing everything at half size, so go by pixel count instead.
    """
    if screen_height >= 2000 or screen_width >= 3400:
        return 2.0
    if screen_height >= 1400 or screen_width >= 2400:
        return 1.5
    return 1.0


def load_gui_state() -> GuiState:
    """Read stored preferences, ignoring anything stale or unreadable."""
    for path in (STATE_PATH, _LEGACY_STATE_PATH):
        try:
            with open(path, encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(stored, dict):
            continue
        break
    else:
        return GuiState()

    root = stored.get("data_root")
    scale = stored.get("ui_scale")
    try:
        scale = None if scale is None else clamp_ui_scale(scale)
    except (TypeError, ValueError):
        scale = None

    return GuiState(
        data_root=root if isinstance(root, str) and os.path.isdir(root) else "",
        ui_scale=scale,
        save_as_l1=bool(stored.get("save_as_l1", True)),
        save_include_auxiliary=bool(stored.get("save_include_auxiliary", False)),
        skip_preprocessing=bool(stored.get("skip_preprocessing", False)),
    )


def save_gui_state(state: GuiState) -> None:
    """Persist preferences; a read-only home must not stop the GUI from running."""
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as handle:
            json.dump(state.as_dict(), handle)
    except OSError:
        pass


@dataclass
class DataPackage:
    """One Mojito L1 file contributing to the combined data stream."""

    path: str
    label: str = ""
    sign: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, got {self.sign}")
        if not self.label:
            self.label = default_package_label(self.path)

    @property
    def operation(self) -> str:
        return "+" if self.sign > 0 else "−"

    @property
    def short_label(self) -> str:
        """Compact name for plot titles, e.g. ``Noise[0]`` or ``MBHB[all]``."""
        family = package_family(self.path)
        stem = os.path.splitext(os.path.basename(self.path))[0]
        tag = SOURCE_TAG.search(stem)
        if family is None:
            return stem if len(stem) <= 20 else f"{stem[:19]}…"
        return f"{family}[{tag.group(1)}]" if tag else family

    def toggled(self) -> "DataPackage":
        return replace(self, sign=-self.sign)


@dataclass
class PipelineSettings:
    """Parameters forwarded to ``MojitoProcessor.process_pipeline``."""

    channels: tuple[str, ...] = AET_CHANNELS
    #: Target cadence in seconds; ``None`` keeps the file's native sampling.
    target_dt: float | None = 10.0
    kaiser_beta: float = 31.0
    highpass_cutoff: float = 5e-6
    lowpass_cutoff: float | None = None
    filter_order: int = 2
    zero_phase: bool = True
    trim_fraction: float = 0.02
    truncate_days: float | None = None
    window: str = "tukey"
    window_alpha: float = 0.0125
    apply_downsample: bool = True
    apply_filter: bool = True
    apply_window: bool = True
    apply_trim: bool = True
    apply_segments: bool = False
    #: When False, packages are only loaded and combined; no ``process_pipeline``.
    apply_preprocessing: bool = True

    def __post_init__(self) -> None:
        if self.target_dt is not None and self.target_dt <= 0:
            raise ValueError(f"target_dt must be positive, got {self.target_dt}")

    @property
    def target_fs(self) -> float | None:
        """Target sampling rate, or ``None`` when downsampling is disabled."""
        return None if self.target_dt is None else 1.0 / self.target_dt

    @classmethod
    def from_search_config(
        cls, dt: float | None = 10.0, channels: str = "AET"
    ) -> "PipelineSettings":
        """Build settings matching ``mojito_preprocessing_pipeline_kwargs``."""
        from mojito_barkeeper.defaults import mojito_preprocessing_pipeline_kwargs

        # Defaults derive the anti-alias cutoff from the target cadence, so
        # fall back to the standard 10 s grid when downsampling is disabled.
        pipeline = mojito_preprocessing_pipeline_kwargs(10.0 if dt is None else dt)
        return cls(
            channels=tuple(channels),
            target_dt=dt,
            kaiser_beta=float(pipeline["downsample_kwargs"]["kaiser_window"]),
            highpass_cutoff=float(pipeline["filter_kwargs"]["highpass_cutoff"]),
            lowpass_cutoff=float(pipeline["filter_kwargs"]["lowpass_cutoff"]),
            filter_order=int(pipeline["filter_kwargs"]["order"]),
            zero_phase=bool(pipeline["filter_kwargs"]["zero_phase"]),
            trim_fraction=float(pipeline["trim_kwargs"]["fraction"]),
            window=str(pipeline["window_kwargs"]["window"]),
            window_alpha=float(pipeline["window_kwargs"]["alpha"]),
        )

    def to_pipeline_kwargs(self, source_dt: float | None = None) -> dict:
        """
        Translate the settings into ``process_pipeline`` keyword arguments.

        ``source_dt`` is the sampling of the input data; when the requested
        low-pass cutoff sits at (or above) the input Nyquist frequency the
        filter is dropped, mirroring ``LISADataLoader._load_mojito``.
        """
        if (
            self.apply_downsample
            and source_dt is not None
            and self.target_dt is not None
            and self.target_dt < source_dt
        ):
            raise PipelineError(
                f"Target cadence {self.target_dt:g} s is finer than the native "
                f"{source_dt:g} s of the data; resampling cannot create samples."
            )

        kwargs: dict = {"channels": list(self.channels)}

        if self.apply_filter:
            filter_kwargs = {
                "highpass_cutoff": self.highpass_cutoff,
                "lowpass_cutoff": self.lowpass_cutoff,
                "order": self.filter_order,
                "zero_phase": self.zero_phase,
            }
            if source_dt is not None and self.lowpass_cutoff is not None:
                if self.lowpass_cutoff >= 1.0 / (2.0 * source_dt):
                    filter_kwargs["lowpass_cutoff"] = None
            kwargs["filter_kwargs"] = filter_kwargs

        if self.apply_downsample:
            kwargs["downsample_kwargs"] = {
                "target_fs": self.target_fs,
                "kaiser_window": self.kaiser_beta,
            }

        if self.apply_trim:
            kwargs["trim_kwargs"] = {"fraction": self.trim_fraction}
        if self.apply_window:
            kwargs["window_kwargs"] = {"window": self.window, "alpha": self.window_alpha}
        if self.apply_segments and self.truncate_days:
            kwargs["truncate_kwargs"] = {"days": self.truncate_days}
        return kwargs


@contextmanager
def _skip_pipeline_filter():
    """
    Bypass MojitoProcessor's mandatory filter step.

    ``process_pipeline`` always calls ``SignalProcessor.filter``; this patch
    makes that call a no-op for the duration of the context.
    """
    from MojitoProcessor.process.sigprocess import SignalProcessor

    original = SignalProcessor.filter

    def noop(self, **kwargs):
        return {channel: self._data[channel] for channel in self.channels}

    SignalProcessor.filter = noop
    try:
        yield
    finally:
        SignalProcessor.filter = original


def package_family(path: str) -> str | None:
    """Return the package family (``Noise``, ``MBHB``, …) implied by the directory."""
    parts = os.path.normpath(path).split(os.sep)
    return next(
        (PACKAGE_FAMILIES[part] for part in reversed(parts) if part in PACKAGE_FAMILIES),
        None,
    )


def default_package_label(path: str) -> str:
    """Derive a short label like ``Noise / NOISE_731d_2.5s...`` from a file path."""
    stem = os.path.splitext(os.path.basename(path))[0]
    family = package_family(path)
    short = stem if len(stem) <= 46 else f"{stem[:28]}…{stem[-16:]}"
    return f"{family} / {short}" if family else short


def is_saved_pipeline_output(path: str) -> bool:
    """Return whether *path* looks like output from this pipeline."""
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            if "globalgb" in handle:
                return True
            return "tdi" in handle and "channels" in handle.attrs
    except OSError:
        return False


def discover_packages(root: str) -> list[DataPackage]:
    """Find Mojito L1 HDF5 files under *root*, sorted by family then filename."""
    if not root or not os.path.isdir(root):
        return []

    found: list[DataPackage] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in sorted(filenames):
            if not filename.endswith((".h5", ".hdf5")):
                continue
            path = os.path.join(dirpath, filename)
            found.append(DataPackage(path=path, enabled=False))

    return sorted(found, key=lambda package: (package.label.lower(), package.path))


def load_package_tdi(
    path: str,
    channels: Sequence[str],
    *,
    window: TimeWindow | None = None,
) -> dict:
    """
    Load only the requested TDI channels from one Mojito L1 file.

    Reading a slice keeps memory manageable: the full 731-day, 2.5 s files hold
    about 25 million samples per channel.
    """
    from mojito import MojitoL1File

    window = window or TimeWindow()
    unknown = [channel for channel in channels if channel not in TDI_ATTRIBUTES]
    if unknown:
        raise PipelineError(
            f"Unknown TDI channel(s) {unknown}; expected any of {sorted(TDI_ATTRIBUTES)}"
        )

    with MojitoL1File(path) as handle:
        sampling = handle.tdis.time_sampling
        fs = float(sampling.fs)
        dt = float(sampling.dt)
        total = int(sampling.size)

        start, stop = window.sample_range(fs, total)
        if stop <= start:
            raise PipelineError(
                f"Time selection {window.describe()} is empty for {os.path.basename(path)}, "
                f"which spans {total / fs / SECONDS_PER_DAY:.2f} days"
            )

        tdis = {
            channel: np.asarray(
                getattr(handle.tdis, TDI_ATTRIBUTES[channel])[start:stop],
                dtype=np.float64,
            )
            for channel in channels
        }
        times = np.asarray(sampling.t()[start:stop], dtype=np.float64)
        laser_frequency = float(handle.laser_frequency)
        pipeline_names = list(handle.pipeline_names)

    return {
        "tdis": tdis,
        "fs": fs,
        "dt": dt,
        "t_tdi": times,
        "metadata": {
            "laser_frequency": laser_frequency,
            "pipeline_name": pipeline_names,
            "source_path": path,
            "n_samples_available": total,
        },
    }


def combine_packages(
    packages: Iterable[DataPackage],
    channels: Sequence[str],
    *,
    window: TimeWindow | None = None,
    loader: Callable[..., dict] = load_package_tdi,
    progress: Callable[[str], None] | None = None,
    on_package: Callable[[DataPackage, dict], None] | None = None,
) -> dict:
    """
    Add and subtract the TDI streams of several packages into one data dict.

    The first enabled package sets the reference sampling rate; the rest must
    match it. Streams of unequal length are cut to the shortest one so that,
    for example, an EMRI file can be added onto a noise realisation.
    """
    enabled = [package for package in packages if package.enabled]
    if not enabled:
        raise PipelineError("No data packages are enabled")
    enabled_paths = [package.path for package in enabled]

    channels = tuple(channels)
    combined: dict[str, np.ndarray] | None = None
    reference: dict | None = None
    contributions: list[str] = []
    summary: list[str] = []

    for package in enabled:
        if progress is not None:
            progress(f"Loading {package.operation} {package.label}")

        loaded = loader(package.path, channels, window=window)

        missing = [channel for channel in channels if channel not in loaded["tdis"]]
        if missing:
            raise PipelineError(
                f"{package.label} does not provide channel(s) {missing}; "
                f"it has {sorted(loaded['tdis'])}"
            )

        # Multiplying always allocates, so these never alias the file buffers.
        signed = {channel: package.sign * loaded["tdis"][channel] for channel in channels}

        if combined is None:
            combined = signed
            reference = loaded
        else:
            assert reference is not None
            if not np.isclose(loaded["fs"], reference["fs"], rtol=1e-12):
                raise PipelineError(
                    f"Sampling rate mismatch: {package.label} has fs={loaded['fs']} Hz "
                    f"but the first package has fs={reference['fs']} Hz"
                )

            n_common = min(
                len(combined[channels[0]]),
                len(loaded["tdis"][channels[0]]),
            )
            if n_common < len(combined[channels[0]]):
                combined = {channel: combined[channel][:n_common] for channel in channels}
                reference["t_tdi"] = reference["t_tdi"][:n_common]
            for channel in channels:
                combined[channel] += signed[channel][:n_common]

        contributions.append(f"{package.operation} {package.label}")
        summary.append(f"{package.operation} {package.short_label}")

        # Hand this package's own signed stream to the caller while it is still
        # in memory, so a per-package breakdown costs no extra file reads. The
        # callback must consume it now: for the first package these arrays are
        # the accumulator and later additions write through them.
        if on_package is not None:
            on_package(package, {**loaded, "tdis": signed})

        del loaded, signed

    assert combined is not None and reference is not None
    return {
        "tdis": combined,
        "fs": reference["fs"],
        "dt": reference["dt"],
        "t_tdi": reference["t_tdi"],
        "metadata": {
            **reference["metadata"],
            "contributions": contributions,
            "summary": " ".join(summary),
            "source_paths": enabled_paths,
        },
    }


def run_pipeline(data: dict, settings: PipelineSettings, *, segment: str | None = "segment0"):
    """Run ``process_pipeline`` and return one segment (or the whole dict)."""
    if not settings.apply_preprocessing:
        raise PipelineError("run_pipeline requires apply_preprocessing=True")
    from MojitoProcessor import process_pipeline

    kwargs = settings.to_pipeline_kwargs(source_dt=data["dt"])
    filter_ctx = nullcontext() if settings.apply_filter else _skip_pipeline_filter()
    with filter_ctx:
        segments = process_pipeline(data, **kwargs)
    if segment is None:
        return segments
    if segment not in segments:
        raise PipelineError(
            f"Segment '{segment}' not produced; available: {sorted(segments)}"
        )
    return segments[segment]


@dataclass
class Contribution:
    """One package's own processed stream, with its +/− sign already applied."""

    label: str
    operation: str
    series: dict[str, np.ndarray]

    @property
    def signed_label(self) -> str:
        return f"{self.operation} {self.label}"


@dataclass
class ProcessedResult:
    """Processed time series plus the context needed to plot and save it."""

    channels: tuple[str, ...]
    series: dict[str, np.ndarray]
    dt: float
    t0: float
    contributions: list[str] = field(default_factory=list)
    summary: str = ""
    settings: PipelineSettings | None = None
    n_segments: int = 1
    time_window: TimeWindow | None = None
    #: Per-package breakdown; empty when only one package was combined.
    components: list[Contribution] = field(default_factory=list)
    laser_frequency: float | None = None
    source_paths: list[str] = field(default_factory=list)
    pipeline_names: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        first = self.series[self.channels[0]]
        return float(len(first)) * self.dt

    def times(self) -> np.ndarray:
        first = self.series[self.channels[0]]
        return self.t0 + np.arange(len(first)) * self.dt

    def spectrum(self, channel: str) -> tuple[np.ndarray, np.ndarray]:
        return amplitude_spectral_density(self.series[channel], self.dt)


def segment_to_result(
    segment,
    settings: PipelineSettings,
    contributions: Sequence[str] = (),
    n_segments: int = 1,
    summary: str = "",
    time_window: TimeWindow | None = None,
    components: list[Contribution] | None = None,
    laser_frequency: float | None = None,
    source_paths: Sequence[str] = (),
    pipeline_names: Sequence[str] = (),
) -> ProcessedResult:
    """Convert a ``SignalProcessor`` segment into a plain-array result."""
    channels = tuple(segment.channels)
    # Copy so the result owns its data: with every optional stage disabled the
    # pipeline can hand back the very array it was given.
    series = {
        channel: np.array(segment.data[channel], dtype=np.float64, copy=True)
        for channel in channels
    }
    return ProcessedResult(
        channels=channels,
        series=series,
        dt=float(segment.dt),
        t0=float(getattr(segment, "t0", 0.0)),
        contributions=list(contributions),
        summary=summary or " ".join(contributions),
        settings=settings,
        n_segments=n_segments,
        time_window=time_window,
        components=list(components or []),
        laser_frequency=laser_frequency,
        source_paths=list(source_paths),
        pipeline_names=list(pipeline_names),
    )


def amplitude_spectral_density(series: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return positive frequencies and the one-sided ASD of a real time series."""
    series = np.asarray(series, dtype=np.float64)
    n_samples = len(series)
    if n_samples < 2:
        raise PipelineError("Need at least two samples to compute a spectrum")

    spectrum = np.fft.rfft(series) * dt
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    duration = n_samples * dt
    asd = np.sqrt(2.0 / duration) * np.abs(spectrum)

    return freqs[1:], asd[1:]


def _align_components(components: list[Contribution], n_samples: int) -> None:
    """
    Cut every component to the length of the total.

    Packages normally share a length and this is a no-op; it only bites when a
    shorter file forced the combination to be truncated after that package had
    already been processed on its own.
    """
    for component in components:
        for channel, series in component.series.items():
            if len(series) > n_samples:
                component.series[channel] = series[:n_samples]


def _result_from_combined_data(
    data: dict,
    settings: PipelineSettings,
    *,
    contributions: Sequence[str],
    summary: str,
    time_window: TimeWindow | None,
    components: list[Contribution],
    n_segments: int = 1,
) -> ProcessedResult:
    """Build a :class:`ProcessedResult` directly from loaded TDI arrays."""
    channels = tuple(settings.channels)
    metadata = data["metadata"]
    return ProcessedResult(
        channels=channels,
        series={
            channel: np.array(data["tdis"][channel], dtype=np.float64, copy=True)
            for channel in channels
        },
        dt=float(data["dt"]),
        t0=float(data["t_tdi"][0]),
        contributions=list(contributions),
        summary=summary or " ".join(contributions),
        settings=settings,
        n_segments=n_segments,
        time_window=time_window,
        components=components,
        laser_frequency=metadata.get("laser_frequency"),
        source_paths=list(metadata.get("source_paths", [])),
        pipeline_names=list(metadata.get("pipeline_name", [])),
    )


def process(
    packages: Iterable[DataPackage],
    settings: PipelineSettings,
    *,
    window: TimeWindow | None = None,
    loader: Callable[..., dict] = load_package_tdi,
    progress: Callable[[str], None] | None = None,
    breakdown: bool = True,
) -> ProcessedResult:
    """
    Load, combine, and process packages in one call.

    With ``breakdown`` set, each package is additionally run through the
    pipeline on its own so the total can be plotted against the signals that
    make it up.
    """
    components: list[Contribution] = []

    def on_package(package: DataPackage, package_data: dict) -> None:
        if not settings.apply_preprocessing:
            if progress is not None:
                progress(f"Loading {package.operation} {package.short_label}")
            components.append(
                Contribution(
                    label=package.short_label,
                    operation=package.operation,
                    series={
                        channel: np.array(
                            package_data["tdis"][channel], dtype=np.float64, copy=True
                        )
                        for channel in settings.channels
                    },
                )
            )
            return

        if progress is not None:
            progress(f"Processing {package.operation} {package.short_label}")
        segment = run_pipeline(package_data, settings)
        components.append(
            Contribution(
                label=package.short_label,
                operation=package.operation,
                series={
                    channel: np.array(segment.data[channel], dtype=np.float64, copy=True)
                    for channel in segment.channels
                },
            )
        )

    data = combine_packages(
        packages,
        settings.channels,
        window=window,
        loader=loader,
        progress=progress,
        on_package=on_package if breakdown else None,
    )
    contributions = data["metadata"].get("contributions", [])
    summary = data["metadata"].get("summary", "")

    if not settings.apply_preprocessing:
        if progress is not None:
            progress("Skipping preprocessing (combine only)")
        n_samples = len(data["tdis"][settings.channels[0]])
        if len(components) < 2:
            components = []
        else:
            _align_components(components, n_samples)
        return _result_from_combined_data(
            data,
            settings,
            contributions=contributions,
            summary=summary,
            time_window=window,
            components=components,
        )

    if progress is not None:
        progress("Running preprocessing pipeline on the total")

    segments = run_pipeline(data, settings, segment=None)
    if "segment0" not in segments:
        raise PipelineError(f"Pipeline produced no segments (got {sorted(segments)})")

    # A single package is its own total, so a coloured copy of the black curve
    # would add nothing.
    if len(components) < 2:
        components = []
    else:
        _align_components(components, len(segments["segment0"].data[settings.channels[0]]))

    metadata = data["metadata"]
    return segment_to_result(
        segments["segment0"],
        settings,
        components=components,
        contributions=contributions,
        n_segments=len(segments),
        summary=summary,
        time_window=window,
        laser_frequency=metadata.get("laser_frequency"),
        source_paths=list(metadata.get("source_paths", [])),
        pipeline_names=list(metadata.get("pipeline_name", [])),
    )


@dataclass(frozen=True)
class SaveOptions:
    """Controls how :func:`save_result` writes HDF5 output."""

    as_l1: bool = False
    include_auxiliary: bool = False


def _time_window_sample_range(
    t0: float, dt: float, size: int, t_start: float, t_end: float
) -> tuple[int, int]:
    """Map an absolute time interval onto ``[start, stop)`` sample indices."""
    start = int(round((t_start - t0) / dt))
    stop = int(round((t_end - t0) / dt)) + 1
    start = max(0, min(start, size))
    stop = max(start + 1, min(stop, size))
    return start, stop


def _write_processing_metadata(group, result: ProcessedResult) -> None:
    """Store pipeline settings and provenance alongside the saved series."""
    group.attrs["channels"] = list(result.channels)
    group.attrs["dt"] = result.dt
    group.attrs["t0"] = result.t0
    group.attrs["duration"] = result.duration
    group.attrs["contributions"] = result.contributions or [""]
    group.attrs["summary"] = result.summary
    group.attrs["n_segments"] = result.n_segments
    if result.source_paths:
        group.attrs["source_paths"] = list(result.source_paths)
    if result.pipeline_names:
        group.attrs["pipeline_names"] = list(result.pipeline_names)
    if result.settings is not None:
        settings = result.settings
        group.attrs["highpass_cutoff"] = settings.highpass_cutoff
        group.attrs["lowpass_cutoff"] = (
            np.nan if settings.lowpass_cutoff is None else settings.lowpass_cutoff
        )
        group.attrs["filter_order"] = settings.filter_order
        group.attrs["zero_phase"] = settings.zero_phase
        group.attrs["trim_fraction"] = settings.trim_fraction
        group.attrs["window"] = settings.window
        group.attrs["window_alpha"] = settings.window_alpha
        group.attrs["kaiser_beta"] = settings.kaiser_beta
        group.attrs["target_dt"] = np.nan if settings.target_dt is None else settings.target_dt
        group.attrs["apply_downsample"] = settings.apply_downsample
        group.attrs["apply_filter"] = settings.apply_filter
        group.attrs["apply_trim"] = settings.apply_trim
        group.attrs["apply_window"] = settings.apply_window
        group.attrs["apply_segments"] = settings.apply_segments
        group.attrs["apply_preprocessing"] = settings.apply_preprocessing
    if result.time_window is not None:
        group.attrs["selection_start"] = result.time_window.start
        group.attrs["selection_end"] = (
            np.nan if result.time_window.end is None else result.time_window.end
        )


def _copy_time_sliced_group(
    source_group,
    dest_parent,
    name: str,
    start: int,
    stop: int,
) -> None:
    """Copy an HDF5 group, slicing uniform-length datasets along axis 0."""
    import h5py

    src = source_group[name]
    dst = dest_parent.create_group(name)
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            if key == "sampling":
                sub = dst.create_group(key)
                if "fmin" in item.attrs:
                    for attr_name, value in item.attrs.items():
                        sub.attrs[attr_name] = value
                else:
                    t0 = float(item.attrs["t0"])
                    dt = float(item.attrs["dt"])
                    sub.attrs["t0"] = t0 + start * dt
                    sub.attrs["dt"] = dt
                    sub.attrs["size"] = stop - start
            else:
                _copy_time_sliced_group(src, dst, key, start, stop)
        elif item.shape and item.shape[0] >= stop:
            dst.create_dataset(key, data=item[start:stop], compression="gzip")
        else:
            dst.create_dataset(key, data=item[()])


def _copy_auxiliary_groups(
    dest,
    source_path: str,
    *,
    t_start: float,
    t_end: float,
) -> None:
    """Copy LTT, orbit, and noise data from a source Mojito L1 file."""
    import h5py

    with h5py.File(source_path, "r") as source:
        if "noise_estimates" in source:
            source.copy("noise_estimates", dest)

        for group_name in ("ltts", "orbits"):
            if group_name not in source:
                continue
            sampling = source[group_name].get("sampling")
            if sampling is None or "fmin" in sampling.attrs:
                continue
            t0 = float(sampling.attrs["t0"])
            dt = float(sampling.attrs["dt"])
            size = int(sampling.attrs["size"])
            start, stop = _time_window_sample_range(t0, dt, size, t_start, t_end)
            _copy_time_sliced_group(source, dest, group_name, start, stop)

        for attr_name in ("lolipops_version",):
            if attr_name in source.attrs and attr_name not in dest.attrs:
                dest.attrs[attr_name] = source.attrs[attr_name]


def _save_compact(path: str, result: ProcessedResult) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        group = handle.create_group("tdi")
        for channel in result.channels:
            group.create_dataset(channel, data=result.series[channel], compression="gzip")
        _write_processing_metadata(handle, result)


def _save_l1(path: str, result: ProcessedResult, *, include_auxiliary: bool) -> None:
    import h5py

    if result.laser_frequency is None:
        raise PipelineError(
            "Cannot save as Mojito L1: laser frequency was not recorded. "
            "Re-run the pipeline from source packages."
        )

    n_samples = len(result.series[result.channels[0]])
    t_end = result.t0 + max(0, n_samples - 1) * result.dt
    pipeline_name = "mojito-barkeeper-processed"
    if result.summary:
        pipeline_name = f"{pipeline_name} ({result.summary})"

    with h5py.File(path, "w") as handle:
        handle.attrs["laser_frequency"] = float(result.laser_frequency)
        handle.attrs["pipeline_name"] = pipeline_name

        tdis = handle.create_group("tdis")
        sampling = tdis.create_group("sampling")
        sampling.attrs["t0"] = result.t0
        sampling.attrs["dt"] = result.dt
        sampling.attrs["size"] = n_samples
        for channel in result.channels:
            dataset_name = L1_TDI_DATASETS[channel]
            tdis.create_dataset(
                dataset_name,
                data=result.series[channel],
                compression="gzip",
            )

        provenance = handle.create_group("globalgb")
        _write_processing_metadata(provenance, result)

        if include_auxiliary:
            if not result.source_paths:
                raise PipelineError(
                    "Cannot copy auxiliary data: no source package paths were recorded."
                )
            _copy_auxiliary_groups(
                handle,
                result.source_paths[0],
                t_start=result.t0,
                t_end=t_end,
            )


def save_result(
    path: str,
    result: ProcessedResult,
    *,
    options: SaveOptions | None = None,
    as_l1: bool | None = None,
    include_auxiliary: bool | None = None,
) -> str:
    """
    Write a processed result to HDF5 for later reuse.

    With ``as_l1`` set, the file follows the Mojito L1 layout so it can be
    loaded again with :func:`load_package_tdi`. Optional auxiliary groups
    (LTTs, orbits, noise estimates) are copied from the first source package.
    """
    opts = options or SaveOptions()
    if as_l1 is not None:
        opts = SaveOptions(as_l1=as_l1, include_auxiliary=opts.include_auxiliary)
    if include_auxiliary is not None:
        opts = SaveOptions(as_l1=opts.as_l1, include_auxiliary=include_auxiliary)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if opts.as_l1:
        _save_l1(path, result, include_auxiliary=opts.include_auxiliary)
    else:
        _save_compact(path, result)
    return path
