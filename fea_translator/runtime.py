"""runtime.py — feat's llmt-runtime: the executor code the translators share, with feat's settings and hooks.
Version: 2026.11.09.8

    RT          llmt_runtime.Runtime for feat (FEAT_ settings, the `feat` CLI, threads: --cpus-per-task)
    RUNTIME_VERSION   llmt-runtime's version (tool_status().executor.runtime)

Phase 1 (2026.11.09.6): workers, memory and out of memory, ended_state, version recording and ran_with, request_name
come from llmt-runtime. Phase 2 (2026.11.09.8): the SLURM layer (settings, caps, time limits, job.sh, submission, states,
the queue, jobs, cancel, extend) is llmt-runtime's too, with feat's sizing as hooks (procs_for, mem_auto,
time_limit_min, solves_of, job_extra, oom_advice, the progress of running jobs, the running solve's deadline).
The MCP server's stale handling stays in mcp_server.py until phase 3.

The hooks are small wrappers that import executor and api when called: executor imports RT from here.

FEAT_SLURM_TIME: feat's former 'estimate' (until 2026.11.09.7: the runs' timeouts, the meshing and 10 min) is the
runtime's 'timeout'; time_aliases keeps existing FEAT_SLURM_TIME=estimate settings (config/env) meaning that.
"""
from __future__ import annotations

from pathlib import Path

from llmt_runtime import Hooks, Runtime
from llmt_runtime import __version__ as RUNTIME_VERSION  # noqa: F401

from . import __version__


def _oom_advice(folder: Path) -> str:
    from .executor import oom_advice
    return oom_advice(folder)


def _procs(study: dict) -> int:
    from .executor import procs_for
    return procs_for(study)


def _mem_auto(study: dict) -> str:
    from .executor import mem_auto
    return mem_auto(study)


def _time_need_min(study: dict) -> int:
    from .executor import time_limit_min
    return time_limit_min(study)


def _solves(study: dict) -> int:
    from .executor import solves_of
    return solves_of(study)


def _job_extra(study: dict) -> dict:
    from .executor import job_extra
    return job_extra(study)


def _running_progress(folder: Path, study: dict) -> dict:
    from .api import job_progress
    return job_progress(folder, study)


def _solver_left_s(folder: Path, study: dict) -> float:
    from .api import _feat_left_s
    return _feat_left_s(folder, study)


RT = Runtime(prefix="FEAT", cli="feat", package="fea_translator", version=__version__, kind="threads",
             out_default="~/feat_runs", min_minutes=1.0, timeout_default=900.0, slurm_time_default="max",
             time_aliases={"estimate": "timeout"},
             hooks=Hooks(procs=_procs, mem_auto=_mem_auto, time_need_min=_time_need_min, solves=_solves,
                         job_extra=_job_extra, oom_advice=_oom_advice,
                         timeout_hint="or coarsen the mesh / drop converge",
                         running_progress=_running_progress, solver_left_s=_solver_left_s))
