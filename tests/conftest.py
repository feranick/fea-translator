"""Tests run in local mode: a machine's config (e.g. FEAT_EXECUTOR=slurm from llmt tune) must not send the
unit tests' studies to the real SLURM queue. tests/test_executor.py opts in, with fake SLURM commands."""
import pytest


@pytest.fixture(autouse=True)
def _local_executor(monkeypatch):
    monkeypatch.setenv("FEAT_EXECUTOR", "local")
    for k in ("FEAT_SLURM_ARGS", "FEAT_SLURM_MIN_MINUTES", "FEAT_PROCS", "FEAT_SLURM_MEM"):
        monkeypatch.delenv(k, raising=False)
