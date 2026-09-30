"""Tests for the benchmark suite's plumbing (not its timings)."""

import json

import numpy as np
import pytest

from tmswarp.bench import data, report
from tmswarp.bench.__main__ import SUITES, build_jobs, discretization, run_job
from tmswarp.bench.worker import sanitize


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TMSWARP_CACHE", str(tmp_path))
    return tmp_path


def test_sanitize_makes_strict_json():
    raw = {"a": np.float64("nan"), "b": [np.int32(3), np.bool_(True)],
           "c": float("inf"), "d": np.float32(1.5)}
    clean = sanitize(raw)
    assert clean == {"a": None, "b": [3, True], "c": None, "d": 1.5}
    json.dumps(clean, allow_nan=False)


def test_refined_dataset_is_cached_with_parents(cache):
    mesh, tag1, parent = data.load("sphere3-r1", log=lambda *_: None)
    base, _, _ = data.load("sphere3", log=lambda *_: None)
    assert len(mesh.elements) == 8 * len(base.elements)
    assert np.array_equal(parent, np.arange(len(mesh.elements)) // 8)
    assert (cache / "sphere3-r1.npz").exists()


def test_every_suite_uses_known_datasets():
    for suite in SUITES.values():
        for key in ("datasets", "slow_cpu", "optimization"):
            assert set(suite[key]) <= set(data.DATASETS)


def test_jobs_run_and_report_renders(cache):
    data.ensure("sphere3", log=lambda *_: None)
    data.ensure("sphere3-r1", log=lambda *_: None)
    jobs = build_jobs(SUITES["quick"], devices=[], has_cuda=False,
                      simnibs_python=None, tmsservice=None,
                      datasets=["sphere3", "sphere3-r1"])
    assert {j["kind"] for j in jobs} == {"reference", "scipy-cg", "numpy-direct"}

    results = [run_job(job, timeout=600) for job in jobs]
    assert [r["status"] for r in results] == ["ok"] * len(results)
    reference = results[0]
    assert reference["rdm_vs_analytical"] < 0.2

    run = {
        "machine": {"label": "test", "gpus": [], "cpu": "x", "cpu_threads": 1,
                    "memory_gb": 1.0, "os": "x", "suite": "quick"},
        "datasets": {n: {"n_nodes": 1, "n_elements": 1}
                     for n in ("sphere3", "sphere3-r1")},
        "jobs": results,
        "discretization": discretization(["sphere3", "sphere3-r1"]),
    }
    assert len(run["discretization"]) == 1
    json.dumps(sanitize(run), allow_nan=False)
    text = report.render([sanitize(run)])
    assert "sphere3-r1" in text
    # Every row of a table must have as many columns as its header
    width = None
    for line in text.splitlines():
        if line.startswith("|"):
            width = width or line.count("|")
            assert line.count("|") == width, line
        else:
            width = None


def test_failed_job_is_reported_not_raised(cache):
    result = run_job({"kind": "reference", "dataset": "no-such-mesh",
                      "dipole": "far"}, timeout=120)
    assert result["status"] == "error"
    assert "no-such-mesh" in result["error"]
