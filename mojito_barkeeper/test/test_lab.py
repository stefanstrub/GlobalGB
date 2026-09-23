"""Tests for the Mojito data-processing core used by the GUI."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mojito_barkeeper import (
    DataPackage,
    GuiState,
    PipelineError,
    PipelineSettings,
    ProcessedResult,
    SaveOptions,
    TimeWindow,
    amplitude_spectral_density,
    clamp_ui_scale,
    combine_packages,
    default_package_label,
    discover_packages,
    is_saved_pipeline_output,
    load_gui_state,
    load_package_tdi,
    process,
    save_gui_state,
    save_result,
    suggest_ui_scale,
)

LASER_FREQUENCY = 281600000000000.0
SOURCE_DT = 2.5
SOURCE_FS = 0.4


def make_loader(levels: dict[str, float], *, lengths: dict[str, int] | None = None, fs=SOURCE_FS):
    """Build a loader returning constant-valued streams, one level per path."""

    def loader(path, channels, window=None):
        n_total = (lengths or {}).get(path, int(4 * 86400 / SOURCE_DT))
        start, stop = (window or TimeWindow()).sample_range(fs, n_total)
        n_samples = stop - start
        sampling = 1.0 / fs
        return {
            "tdis": {
                channel: np.full(n_samples, levels[path], dtype=np.float64)
                for channel in channels
            },
            "fs": fs,
            "dt": sampling,
            "t_tdi": np.arange(n_samples) * sampling,
            "metadata": {"laser_frequency": LASER_FREQUENCY, "pipeline_name": ["test"]},
        }

    return loader


def make_wave_loader(frequencies: dict[str, float], n_samples=int(6 * 86400 / SOURCE_DT)):
    """
    Loader emitting a sinusoid per path.

    Constant streams are useless for checking the pipeline's output because the
    5e-6 Hz high-pass flattens them; these frequencies sit inside the passband.
    """

    def loader(path, channels, window=None):
        start, stop = (window or TimeWindow()).sample_range(SOURCE_FS, n_samples)
        times = np.arange(start, stop) * SOURCE_DT
        wave = np.sin(2 * np.pi * frequencies[path] * times)
        return {
            "tdis": {channel: wave.copy() for channel in channels},
            "fs": SOURCE_FS,
            "dt": SOURCE_DT,
            "t_tdi": times,
            "metadata": {"laser_frequency": LASER_FREQUENCY, "pipeline_name": ["test"]},
        }

    return loader


@pytest.fixture
def packages():
    return [
        DataPackage(path="noise.h5", label="Noise", sign=1),
        DataPackage(path="emri.h5", label="EMRI", sign=1),
        DataPackage(path="mbhb.h5", label="MBHB", sign=-1),
    ]


class TestDataPackage:
    def test_label_defaults_to_family_and_filename(self):
        package = DataPackage(path="/data/EMRI/L1_0p4Hz/EMRI_731d_2.5s_L1_source0.h5")
        assert package.label.startswith("EMRI / ")
        assert "EMRI_731d_2.5s_L1_source0" in package.label

    def test_label_without_known_family_uses_stem(self):
        assert default_package_label("/tmp/custom_run.h5") == "custom_run"

    def test_long_label_is_shortened(self):
        long_name = "A" * 90
        assert len(default_package_label(f"/tmp/{long_name}.h5")) < 60

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("/data/EMRI/L1_0p4Hz/EMRI_731d_2.5s_L1_source0_0_2025.h5", "EMRI[0]"),
            ("/data/MBHB/MBHB_731d_2.5s_L1_source_all_0_20251203.h5", "MBHB[all]"),
            ("/data/INSTRUMENT/L1_0p4Hz/NOISE_731d_2.5s_L1_source0_0_2025.h5", "Noise[0]"),
            ("/tmp/custom_run.h5", "custom_run"),
        ],
    )
    def test_short_label_is_compact(self, path, expected):
        assert DataPackage(path=path).short_label == expected

    def test_rejects_invalid_sign(self):
        with pytest.raises(ValueError, match="sign must be"):
            DataPackage(path="x.h5", sign=0)

    def test_toggled_flips_sign_without_mutating(self):
        package = DataPackage(path="x.h5", sign=1)
        flipped = package.toggled()
        assert package.sign == 1
        assert flipped.sign == -1
        assert flipped.operation == "−"


class TestCombinePackages:
    def test_adds_and_subtracts_streams(self, packages):
        loader = make_loader({"noise.h5": 1.0, "emri.h5": 5.0, "mbhb.h5": 9.0})
        data = combine_packages(packages, ("A", "E"), loader=loader)

        assert set(data["tdis"]) == {"A", "E"}
        assert np.allclose(data["tdis"]["A"], 1.0 + 5.0 - 9.0)
        assert data["metadata"]["contributions"] == ["+ Noise", "+ EMRI", "− MBHB"]

    def test_summary_uses_compact_package_names(self):
        packages = [
            DataPackage(path="/data/INSTRUMENT/NOISE_L1_source0_0.h5"),
            DataPackage(path="/data/MBHB/MBHB_L1_source_all_0.h5", sign=-1),
        ]
        loader = make_loader(
            {"/data/INSTRUMENT/NOISE_L1_source0_0.h5": 1.0, "/data/MBHB/MBHB_L1_source_all_0.h5": 4.0}
        )
        data = combine_packages(packages, ("A",), loader=loader)

        assert data["metadata"]["summary"] == "+ Noise[0] − MBHB[all]"

    def test_disabled_packages_are_skipped(self, packages):
        packages[1].enabled = False
        loader = make_loader({"noise.h5": 1.0, "emri.h5": 5.0, "mbhb.h5": 9.0})
        data = combine_packages(packages, ("A",), loader=loader)

        assert np.allclose(data["tdis"]["A"], 1.0 - 9.0)
        assert "EMRI" not in " ".join(data["metadata"]["contributions"])

    def test_subtracting_leading_package_negates_it(self):
        packages = [DataPackage(path="mbhb.h5", label="MBHB", sign=-1)]
        loader = make_loader({"mbhb.h5": 3.0})
        data = combine_packages(packages, ("A",), loader=loader)

        assert np.allclose(data["tdis"]["A"], -3.0)

    def test_streams_are_cut_to_the_shortest(self, packages):
        loader = make_loader(
            {"noise.h5": 1.0, "emri.h5": 5.0, "mbhb.h5": 9.0},
            lengths={"noise.h5": 1000, "emri.h5": 400, "mbhb.h5": 900},
        )
        data = combine_packages(packages, ("A",), loader=loader)

        assert len(data["tdis"]["A"]) == 400
        assert len(data["t_tdi"]) == 400
        assert np.allclose(data["tdis"]["A"], 1.0 + 5.0 - 9.0)

    def test_time_selection_is_passed_to_the_loader(self, packages):
        loader = make_loader(
            {"noise.h5": 1.0, "emri.h5": 5.0, "mbhb.h5": 9.0},
            lengths={path: int(10 * 86400 / SOURCE_DT) for path in ("noise.h5", "emri.h5", "mbhb.h5")},
        )
        window = TimeWindow.from_unit(2, 5, "days")
        data = combine_packages(packages, ("A",), window=window, loader=loader)

        assert len(data["tdis"]["A"]) == int(3 * 86400 * SOURCE_FS)

    def test_sampling_rate_mismatch_is_rejected(self):
        def loader(path, channels, window=None):
            fs = 0.4 if path == "a.h5" else 0.1
            return {
                "tdis": {channel: np.zeros(100) for channel in channels},
                "fs": fs,
                "dt": 1.0 / fs,
                "t_tdi": np.arange(100),
                "metadata": {"laser_frequency": LASER_FREQUENCY},
            }

        packages = [DataPackage(path="a.h5"), DataPackage(path="b.h5")]
        with pytest.raises(PipelineError, match="Sampling rate mismatch"):
            combine_packages(packages, ("A",), loader=loader)

    def test_no_enabled_packages_is_rejected(self, packages):
        for package in packages:
            package.enabled = False
        with pytest.raises(PipelineError, match="No data packages are enabled"):
            combine_packages(packages, ("A",), loader=make_loader({}))

    def test_progress_reports_each_package(self, packages):
        messages: list[str] = []
        loader = make_loader({"noise.h5": 1.0, "emri.h5": 5.0, "mbhb.h5": 9.0})
        combine_packages(packages, ("A",), loader=loader, progress=messages.append)

        assert messages == [
            "Loading + Noise",
            "Loading + EMRI",
            "Loading − MBHB",
        ]


class TestTimeWindow:
    def test_defaults_span_the_whole_file(self):
        window = TimeWindow()

        assert window.sample_range(SOURCE_FS, 1000) == (0, 1000)
        assert window.duration is None

    def test_start_and_end_map_to_sample_indices(self):
        window = TimeWindow(start=100.0, end=300.0)

        assert window.sample_range(SOURCE_FS, 10000) == (40, 120)
        assert window.duration == pytest.approx(200.0)

    def test_end_is_clamped_to_the_file_length(self):
        window = TimeWindow(start=0.0, end=1e9)

        assert window.sample_range(SOURCE_FS, 500) == (0, 500)

    @pytest.mark.parametrize(
        "unit, expected_start, expected_end",
        [("days", 86400.0, 172800.0), ("hours", 3600.0, 7200.0), ("seconds", 1.0, 2.0)],
    )
    def test_from_unit_converts_to_seconds(self, unit, expected_start, expected_end):
        window = TimeWindow.from_unit(1, 2, unit)

        assert window.start == pytest.approx(expected_start)
        assert window.end == pytest.approx(expected_end)

    def test_open_ended_window_from_unit(self):
        assert TimeWindow.from_unit(3, None, "days").end is None

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown time unit"):
            TimeWindow.from_unit(0, 1, "fortnights")

    def test_end_before_start_is_rejected(self):
        with pytest.raises(ValueError, match="must be later than start"):
            TimeWindow(start=100.0, end=50.0)

    def test_negative_start_is_rejected(self):
        with pytest.raises(ValueError, match="must not be negative"):
            TimeWindow(start=-1.0)


class TestGuiState:
    @pytest.fixture
    def state_file(self, tmp_path, monkeypatch):
        path = tmp_path / "state" / "mojito_gui.json"
        monkeypatch.setattr("mojito_barkeeper.lab.STATE_PATH", str(path))
        monkeypatch.setattr(
            "mojito_barkeeper.lab._LEGACY_STATE_PATH",
            str(tmp_path / "state" / "legacy_missing.json"),
        )
        return path

    def test_roundtrip_remembers_directory_and_scale(self, state_file, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        save_gui_state(GuiState(data_root=str(data_dir), ui_scale=2.0))
        restored = load_gui_state()

        assert restored.data_root == str(data_dir)
        assert restored.ui_scale == pytest.approx(2.0)

    def test_missing_state_falls_back_to_defaults(self, state_file):
        restored = load_gui_state()

        assert restored.data_root == ""
        assert restored.ui_scale is None

    def test_directory_that_disappeared_is_ignored(self, state_file, tmp_path):
        save_gui_state(GuiState(data_root=str(tmp_path / "gone")))

        assert load_gui_state().data_root == ""

    def test_corrupt_state_file_is_ignored(self, state_file):
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("{not json", encoding="utf-8")

        assert load_gui_state() == GuiState()

    def test_absurd_stored_scale_is_clamped(self, state_file):
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text('{"data_root": "", "ui_scale": 99}', encoding="utf-8")

        assert load_gui_state().ui_scale == pytest.approx(4.0)

    def test_unset_scale_is_not_written(self, state_file):
        save_gui_state(GuiState(data_root=""))

        assert "ui_scale" not in json.loads(state_file.read_text(encoding="utf-8"))


class TestUiScale:
    @pytest.mark.parametrize(
        "width, height, expected",
        [
            (3840, 2160, 2.0),   # 4K panel reporting a placeholder 96 DPI
            (2560, 1440, 1.5),
            (1920, 1080, 1.0),
            (1366, 768, 1.0),
            (3440, 1440, 2.0),   # ultrawide, caught by the width rule
        ],
    )
    def test_scale_is_suggested_from_screen_size(self, width, height, expected):
        assert suggest_ui_scale(width, height) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "value, expected", [(0.1, 0.5), (1.0, 1.0), (2.5, 2.5), (10.0, 4.0)]
    )
    def test_scale_is_clamped_to_a_usable_range(self, value, expected):
        assert clamp_ui_scale(value) == pytest.approx(expected)


class TestPipelineSettings:
    def test_defaults_follow_search_config(self):
        settings = PipelineSettings.from_search_config(dt=10.0, channels="AET")

        assert settings.channels == ("A", "E", "T")
        assert settings.target_fs == pytest.approx(0.1)
        assert settings.highpass_cutoff == pytest.approx(5e-6)
        assert settings.lowpass_cutoff == pytest.approx(0.05)
        assert settings.window == "tukey"

    def test_lowpass_at_source_nyquist_is_dropped(self):
        settings = PipelineSettings(target_dt=SOURCE_DT, lowpass_cutoff=0.2)
        kwargs = settings.to_pipeline_kwargs(source_dt=SOURCE_DT)

        assert kwargs["filter_kwargs"]["lowpass_cutoff"] is None

    def test_lowpass_below_source_nyquist_is_kept(self):
        settings = PipelineSettings(target_dt=10.0, lowpass_cutoff=0.05)
        kwargs = settings.to_pipeline_kwargs(source_dt=SOURCE_DT)

        assert kwargs["filter_kwargs"]["lowpass_cutoff"] == pytest.approx(0.05)

    def test_disabled_stages_are_omitted(self):
        settings = PipelineSettings(apply_trim=False, apply_window=False)
        kwargs = settings.to_pipeline_kwargs()

        assert "trim_kwargs" not in kwargs
        assert "window_kwargs" not in kwargs
        assert "truncate_kwargs" not in kwargs

    def test_disabled_downsample_and_filter_are_omitted(self):
        settings = PipelineSettings(
            target_dt=10.0,
            apply_downsample=False,
            apply_filter=False,
        )
        kwargs = settings.to_pipeline_kwargs(source_dt=SOURCE_DT)

        assert "downsample_kwargs" not in kwargs
        assert "filter_kwargs" not in kwargs

    def test_segment_days_becomes_truncate_kwargs(self):
        kwargs = PipelineSettings(truncate_days=4.0, apply_segments=True).to_pipeline_kwargs()

        assert kwargs["truncate_kwargs"] == {"days": 4.0}

    def test_segments_are_ignored_when_disabled(self):
        kwargs = PipelineSettings(truncate_days=4.0, apply_segments=False).to_pipeline_kwargs()

        assert "truncate_kwargs" not in kwargs

    def test_downsample_validation_is_skipped_when_stage_is_off(self):
        settings = PipelineSettings(target_dt=0.25, apply_downsample=False)

        kwargs = settings.to_pipeline_kwargs(source_dt=SOURCE_DT)

        assert "downsample_kwargs" not in kwargs

    def test_native_cadence_disables_downsampling(self):
        settings = PipelineSettings(target_dt=None)

        assert settings.target_fs is None
        assert settings.to_pipeline_kwargs()["downsample_kwargs"]["target_fs"] is None

    def test_native_cadence_still_gets_config_defaults(self):
        settings = PipelineSettings.from_search_config(dt=None)

        assert settings.target_dt is None
        assert settings.highpass_cutoff == pytest.approx(5e-6)

    def test_cadence_finer_than_the_data_is_rejected(self):
        settings = PipelineSettings(target_dt=0.25)

        with pytest.raises(PipelineError, match="finer than the native"):
            settings.to_pipeline_kwargs(source_dt=SOURCE_DT)

    def test_non_positive_cadence_is_rejected(self):
        with pytest.raises(ValueError, match="target_dt must be positive"):
            PipelineSettings(target_dt=0.0)

    def test_preprocessing_can_be_disabled_entirely(self):
        settings = PipelineSettings(apply_preprocessing=False)

        assert settings.apply_preprocessing is False


class TestPassthrough:
    def test_skipping_preprocessing_preserves_sample_count(self, packages):
        n_samples = int(6 * 86400 / SOURCE_DT)
        loader = make_loader(
            {"noise.h5": 1.0, "emri.h5": 5.0, "mbhb.h5": 9.0},
            lengths={path: n_samples for path in ("noise.h5", "emri.h5", "mbhb.h5")},
        )
        settings = PipelineSettings(
            channels=("A", "E", "T"),
            apply_preprocessing=False,
        )

        result = process(packages, settings, loader=loader, breakdown=False)

        assert len(result.series["A"]) == n_samples
        assert result.dt == pytest.approx(SOURCE_DT)

    def test_preprocessed_output_is_shorter_than_passthrough(self, packages):
        n_samples = int(6 * 86400 / SOURCE_DT)
        loader = make_loader(
            {"noise.h5": 1.0, "emri.h5": 5.0, "mbhb.h5": 9.0},
            lengths={path: n_samples for path in ("noise.h5", "emri.h5", "mbhb.h5")},
        )
        passthrough = process(
            packages,
            PipelineSettings(channels=("A", "E", "T"), apply_preprocessing=False),
            loader=loader,
            breakdown=False,
        )
        processed = process(
            packages,
            PipelineSettings.from_search_config(dt=10.0, channels="AET"),
            loader=loader,
            breakdown=False,
        )

        assert len(processed.series["A"]) < len(passthrough.series["A"])

    def test_run_pipeline_rejects_passthrough_settings(self):
        data = {
            "tdis": {"A": np.zeros(8)},
            "fs": SOURCE_FS,
            "dt": SOURCE_DT,
            "t_tdi": np.arange(8) * SOURCE_DT,
            "metadata": {"laser_frequency": LASER_FREQUENCY},
        }
        settings = PipelineSettings(channels=("A",), apply_preprocessing=False)

        with pytest.raises(PipelineError, match="apply_preprocessing"):
            from mojito_barkeeper import run_pipeline

            run_pipeline(data, settings)


class TestSavedPipelineDetection:
    def test_detects_globalgb_provenance(self, tmp_path):
        import h5py

        path = tmp_path / "processed.h5"
        with h5py.File(path, "w") as handle:
            handle.create_group("globalgb")

        assert is_saved_pipeline_output(str(path)) is True

    def test_detects_compact_format(self, tmp_path):
        import h5py

        path = tmp_path / "processed.h5"
        with h5py.File(path, "w") as handle:
            handle.create_group("tdi")
            handle.attrs["channels"] = ["A"]

        assert is_saved_pipeline_output(str(path)) is True

    def test_raw_mojito_file_is_not_flagged(self, tmp_path):
        import h5py

        path = tmp_path / "noise.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["laser_frequency"] = LASER_FREQUENCY
            tdis = handle.create_group("tdis")
            sampling = tdis.create_group("sampling")
            sampling.attrs.update({"t0": 0.0, "dt": 2.5, "size": 10})
            tdis.create_dataset("A2", data=np.zeros(10))

        assert is_saved_pipeline_output(str(path)) is False


class TestProcess:
    def test_end_to_end_downsamples_and_trims(self, packages):
        rng = np.random.default_rng(0)
        n_samples = int(6 * 86400 / SOURCE_DT)

        def loader(path, channels, window=None):
            level = {"noise.h5": 1.0, "emri.h5": 5.0, "mbhb.h5": 9.0}[path]
            return {
                "tdis": {
                    channel: level + rng.normal(scale=1e-3, size=n_samples)
                    for channel in channels
                },
                "fs": SOURCE_FS,
                "dt": SOURCE_DT,
                "t_tdi": np.arange(n_samples) * SOURCE_DT,
                "metadata": {"laser_frequency": LASER_FREQUENCY},
            }

        settings = PipelineSettings.from_search_config(dt=10.0, channels="AET")
        result = process(packages, settings, loader=loader)

        assert result.channels == ("A", "E", "T")
        assert result.dt == pytest.approx(10.0)
        # 6 days downsampled to 10 s, minus the 2 % trimmed off the edges.
        assert result.duration / 86400 == pytest.approx(6 * 0.98, rel=0.01)
        assert np.isfinite(result.series["A"]).all()

    def test_breakdown_returns_one_component_per_package(self, packages):
        loader = make_loader({"noise.h5": 1.0, "emri.h5": 5.0, "mbhb.h5": 9.0})
        settings = PipelineSettings.from_search_config(dt=10.0, channels="AET")

        result = process(packages, settings, loader=loader)

        # Components carry the compact path-derived name used in plot legends.
        assert [c.signed_label for c in result.components] == [
            "+ noise",
            "+ emri",
            "− mbhb",
        ]
        assert all(len(c.series["A"]) == len(result.series["A"]) for c in result.components)

    def test_components_sum_to_the_total(self, packages):
        """The pipeline is linear, so the coloured curves must add to the black one."""
        loader = make_wave_loader(
            {"noise.h5": 1e-3, "emri.h5": 3e-4, "mbhb.h5": 7e-4}
        )
        settings = PipelineSettings.from_search_config(dt=10.0, channels="AET")

        result = process(packages, settings, loader=loader)
        total = result.series["A"]
        stacked = sum(c.series["A"] for c in result.components)

        # Compare against the signal's own amplitude: a pointwise relative test
        # would blow up at the zero crossings.
        peak = np.abs(total).max()
        assert peak > 0
        assert np.abs(stacked - total).max() < 1e-8 * peak

    def test_subtracted_package_is_the_negative_of_an_added_one(self, packages):
        loader = make_wave_loader(
            {"noise.h5": 1e-3, "emri.h5": 3e-4, "mbhb.h5": 7e-4}
        )
        settings = PipelineSettings.from_search_config(dt=10.0, channels="AET")

        subtracted = process(packages, settings, loader=loader)
        packages[2].sign = 1
        added = process(packages, settings, loader=loader)

        minus_mbhb = next(c for c in subtracted.components if c.label == "mbhb")
        plus_mbhb = next(c for c in added.components if c.label == "mbhb")

        assert minus_mbhb.operation == "−"
        assert plus_mbhb.operation == "+"
        assert np.abs(plus_mbhb.series["A"]).max() > 0
        assert np.allclose(minus_mbhb.series["A"], -plus_mbhb.series["A"])

    def test_single_package_has_no_redundant_breakdown(self):
        loader = make_loader({"noise.h5": 1.0})
        settings = PipelineSettings.from_search_config(dt=10.0, channels="AET")

        result = process([DataPackage(path="noise.h5")], settings, loader=loader)

        assert result.components == []

    def test_breakdown_can_be_switched_off(self, packages):
        loader = make_loader({"noise.h5": 1.0, "emri.h5": 5.0, "mbhb.h5": 9.0})
        settings = PipelineSettings.from_search_config(dt=10.0, channels="AET")

        result = process(packages, settings, loader=loader, breakdown=False)

        assert result.components == []
        assert np.isfinite(result.series["A"]).all()

    def test_components_are_cut_to_the_total_length(self, packages):
        loader = make_loader(
            {"noise.h5": 1.0, "emri.h5": 5.0, "mbhb.h5": 9.0},
            lengths={
                "noise.h5": int(6 * 86400 / SOURCE_DT),
                "emri.h5": int(6 * 86400 / SOURCE_DT),
                "mbhb.h5": int(4 * 86400 / SOURCE_DT),
            },
        )
        settings = PipelineSettings.from_search_config(dt=10.0, channels="AET")

        result = process(packages, settings, loader=loader)

        assert all(
            len(c.series["A"]) == len(result.series["A"]) for c in result.components
        )

    def test_missing_channel_is_reported(self):
        def loader(path, channels, window=None):
            return {
                "tdis": {"E": np.zeros(1000)},
                "fs": SOURCE_FS,
                "dt": SOURCE_DT,
                "t_tdi": np.arange(1000) * SOURCE_DT,
                "metadata": {"laser_frequency": LASER_FREQUENCY},
            }

        with pytest.raises(PipelineError, match=r"does not provide channel\(s\) \['A'\]"):
            process(
                [DataPackage(path="a.h5", label="Noise")],
                PipelineSettings(channels=("A",)),
                loader=loader,
            )


class TestSpectrum:
    def test_asd_recovers_a_monochromatic_line(self):
        dt, n_samples, f0 = 10.0, 8192, 1e-3
        times = np.arange(n_samples) * dt
        series = np.sin(2 * np.pi * f0 * times)

        freqs, asd = amplitude_spectral_density(series, dt)

        assert freqs[0] > 0
        assert freqs[np.argmax(asd)] == pytest.approx(f0, rel=0.02)

    def test_short_series_is_rejected(self):
        with pytest.raises(PipelineError, match="at least two samples"):
            amplitude_spectral_density(np.array([1.0]), 10.0)


class TestProcessedResult:
    @pytest.fixture
    def result(self):
        return ProcessedResult(
            channels=("A", "E"),
            series={"A": np.arange(100.0), "E": np.arange(100.0) * 2},
            dt=10.0,
            t0=500.0,
            contributions=["+ Noise", "− MBHB"],
            settings=PipelineSettings(),
        )

    def test_times_start_at_t0(self, result):
        times = result.times()

        assert times[0] == pytest.approx(500.0)
        assert times[1] - times[0] == pytest.approx(10.0)
        assert result.duration == pytest.approx(1000.0)

    def test_roundtrip_through_hdf5(self, result, tmp_path):
        import h5py

        path = save_result(str(tmp_path / "processed.h5"), result)
        with h5py.File(path, "r") as handle:
            assert list(handle.attrs["channels"]) == ["A", "E"]
            assert handle.attrs["dt"] == pytest.approx(10.0)
            assert list(handle.attrs["contributions"]) == ["+ Noise", "− MBHB"]
            assert np.allclose(handle["tdi/A"][:], result.series["A"])

    def test_l1_roundtrip_is_reloadable(self, tmp_path):
        result = ProcessedResult(
            channels=("A", "E"),
            series={"A": np.arange(100.0), "E": np.arange(100.0) * 2},
            dt=10.0,
            t0=500.0,
            laser_frequency=LASER_FREQUENCY,
            source_paths=[str(tmp_path / "unused.h5")],
        )
        path = save_result(
            str(tmp_path / "processed_l1.h5"),
            result,
            options=SaveOptions(as_l1=True),
        )

        loaded = load_package_tdi(path, ("A", "E"))

        assert loaded["dt"] == pytest.approx(10.0)
        assert loaded["t_tdi"][0] == pytest.approx(500.0)
        assert np.allclose(loaded["tdis"]["A"], result.series["A"])
        assert loaded["metadata"]["laser_frequency"] == pytest.approx(LASER_FREQUENCY)

    def test_l1_save_copies_auxiliary_groups(self, tmp_path):
        source = tmp_path / "source.h5"
        n_samples = 120
        t0 = 500.0
        dt = 10.0
        import h5py

        with h5py.File(source, "w") as handle:
            handle.attrs["laser_frequency"] = LASER_FREQUENCY
            handle.attrs["pipeline_name"] = "test-source"
            handle.attrs["lolipops_version"] = "0.0-test"
            tdis = handle.create_group("tdis")
            sampling = tdis.create_group("sampling")
            sampling.attrs["t0"] = t0
            sampling.attrs["dt"] = dt
            sampling.attrs["size"] = n_samples
            tdis.create_dataset("A2", data=np.zeros(n_samples))
            ltts = handle.create_group("ltts")
            ltt_sampling = ltts.create_group("sampling")
            ltt_sampling.attrs["t0"] = t0
            ltt_sampling.attrs["dt"] = dt
            ltt_sampling.attrs["size"] = n_samples
            ltts.create_dataset("ltt_12", data=np.arange(n_samples, dtype=float))

        result = ProcessedResult(
            channels=("A",),
            series={"A": np.linspace(0.0, 1.0, 80)},
            dt=dt,
            t0=t0,
            laser_frequency=LASER_FREQUENCY,
            source_paths=[str(source)],
        )
        path = save_result(
            str(tmp_path / "with_aux.h5"),
            result,
            options=SaveOptions(as_l1=True, include_auxiliary=True),
        )

        with h5py.File(path, "r") as handle:
            assert "ltts" in handle
            assert handle["ltts/sampling"].attrs["size"] == 80
            assert handle["ltts/ltt_12"].shape == (80,)
            assert handle.attrs["lolipops_version"] == "0.0-test"


class TestDiscovery:
    def test_finds_hdf5_files_recursively(self, tmp_path):
        (tmp_path / "EMRI" / "L1_0p4Hz").mkdir(parents=True)
        (tmp_path / "INSTRUMENT").mkdir()
        (tmp_path / "EMRI" / "L1_0p4Hz" / "EMRI_source0.h5").touch()
        (tmp_path / "INSTRUMENT" / "NOISE_source0.h5").touch()
        (tmp_path / "notes.txt").touch()

        found = discover_packages(str(tmp_path))

        assert [package.label for package in found] == [
            "EMRI / EMRI_source0",
            "Noise / NOISE_source0",
        ]
        assert all(package.enabled is False for package in found)

    def test_missing_root_returns_empty_list(self, tmp_path):
        assert discover_packages(str(tmp_path / "nope")) == []

    def test_unset_root_returns_empty_list(self):
        assert discover_packages("") == []
