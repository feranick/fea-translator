"""runtime.py — feat's llmt-runtime: the executor code the translators share, with feat's settings and hooks.
Version: 2026.11.10.1

    RT          llmt_runtime.Runtime for feat (FEAT_ settings, the `feat` CLI, threads: --cpus-per-task)
    RUNTIME_VERSION   llmt-runtime's version (tool_status().executor.runtime)

Phase 1 (2026.11.09.6): workers, memory and out of memory, ended_state, version recording and ran_with, request_name
come from llmt-runtime. Phase 2 (2026.11.09.8): the SLURM layer (settings, caps, time limits, job.sh, submission, states,
the queue, jobs, cancel, extend) is llmt-runtime's too, with feat's sizing as hooks (procs_for, mem_auto,
time_limit_min, solves_of, job_extra, oom_advice, the progress of running jobs, the running solve's deadline).
Phase 3 (2026.11.09.10): the MCP server's stale handling, exit, log, signal handlers and tool wrapper are
llmt_runtime.server.Server (mcp_server.GUARD, built on RT: FEAT_EXIT_WHEN_STALE, FEAT_STALE_CHECK_S, FEAT_OUT/.server).

The hooks are small wrappers that import executor and api when called: executor imports RT from here.

FEAT_SLURM_TIME takes the runtime's words, max (feat's default) | timeout | estimate | minutes, with the meanings they
have in every translator. feat's former 'estimate' (until 2026.11.09.7: the runs' timeouts, the meshing and 10 min) is
'timeout'; the alias that kept it (time_aliases) was dropped in 2026.11.10.1, when no machine's config/env set
FEAT_SLURM_TIME (carbonio, mochi, spark1, spark2). 'estimate' now sizes the job from the study's own run-time estimate
(api.start passes it), and each CalculiX run's timeout fits that limit (hooks.timeout_for_limit: executor.
timeout_for_limit; `feat run-study` reads it from job.json).
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


def _timeout_for_limit(study: dict, minutes: int) -> float:
    from .executor import timeout_for_limit
    return timeout_for_limit(study, minutes)


def _running_progress(folder: Path, study: dict) -> dict:
    from .api import job_progress
    return job_progress(folder, study)


def _solver_left_s(folder: Path, study: dict) -> float:
    from .api import _feat_left_s
    return _feat_left_s(folder, study)


RT = Runtime(prefix="FEAT", cli="feat", package="fea_translator", version=__version__, kind="threads",
             out_default="~/feat_runs", min_minutes=1.0, timeout_default=900.0, slurm_time_default="max",
             hooks=Hooks(procs=_procs, mem_auto=_mem_auto, time_need_min=_time_need_min, solves=_solves,
                         timeout_for_limit=_timeout_for_limit,
                         job_extra=_job_extra, oom_advice=_oom_advice,
                         timeout_hint="or coarsen the mesh / drop converge",
                         running_progress=_running_progress, solver_left_s=_solver_left_s))
