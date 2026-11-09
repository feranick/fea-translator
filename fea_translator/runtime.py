"""runtime.py — feat's llmt-runtime: the executor code the translators share, with feat's settings and hooks.
Version: 2026.11.09.6

    RT          llmt_runtime.Runtime for feat (FEAT_ settings, the `feat` CLI, threads: --cpus-per-task)
    RUNTIME_VERSION   llmt-runtime's version (tool_status().executor.runtime)

Phase 1 (2026.11.09.6): workers, memory and out of memory, ended_state, version recording and ran_with, request_name
come from llmt-runtime. The SLURM layer (settings, caps, time limits, job.sh, states, jobs, cancel, extend) and the MCP
server's stale handling stay in executor.py and mcp_server.py until phase 2 and 3.
"""
from __future__ import annotations

from pathlib import Path

from llmt_runtime import Hooks, Runtime
from llmt_runtime import __version__ as RUNTIME_VERSION  # noqa: F401

from . import __version__


def _oom_advice(folder: Path) -> str:
    from .executor import oom_advice
    return oom_advice(folder)


RT = Runtime(prefix="FEAT", cli="feat", package="fea_translator", version=__version__, kind="threads",
             out_default="~/feat_runs", min_minutes=1.0, timeout_default=900.0, slurm_time_default="max",
             hooks=Hooks(oom_advice=_oom_advice, timeout_hint="or coarsen the mesh / drop converge"))
