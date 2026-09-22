"""Smoke tests that the mojito_barkeeper package is installed."""

from mojito_barkeeper import PipelineSettings


def test_package_imports():
    assert PipelineSettings.from_search_config(dt=10.0).target_dt == 10.0
