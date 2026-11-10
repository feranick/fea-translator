"""Tests run in local mode: a machine's config (e.g. FEAT_EXECUTOR=slurm from llmt tune) must not send the
unit tests' studies to the real SLURM queue. tests/test_executor.py opts in, with llmt-runtime's fake SLURM commands.
Version: 2026.11.09.8
"""
import pytest


@pytest.fixture(autouse=True)
def _local_executor(monkeypatch):
    monkeypatch.setenv("FEAT_EXECUTOR", "local")
    for k in ("FEAT_SLURM_ARGS", "FEAT_SLURM_MIN_MINUTES", "FEAT_PROCS", "FEAT_SLURM_MEM",
              "FEAT_MAX_TIMEOUT", "FEAT_MAX_MEM", "FEAT_MAX_PROCS", "FEAT_TIMEOUT", "FEAT_SLURM_NODE",
              "FEAT_SLURM_TIME", "FEAT_EXIT_WHEN_STALE"):
        monkeypatch.delenv(k, raising=False)
    from llmt_runtime import jobs as J
    from llmt_runtime import slurm as S

    from fea_translator import executor
    executor._NODES.update(t=0.0, nodes=None)   # the cluster's nodes are remembered for a minute: not across tests
    # (executor._NODES is llmt-runtime's cache); forget() also drops a failed scontrol (not asked again within 5 s)
    # and the partition's limit
    S.forget()
    J._PENDING.update(t=0.0, v=None)            # the pending jobs (kept 3 s) and a slow sbatch (30 s): not across tests
    J._SBATCH_DOWN.update(until=0.0, why="")
