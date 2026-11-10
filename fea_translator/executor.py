"""executor.py — where studies run: the local worker pool, or SLURM jobs (llm-translators docs/EXECUTION.md).
Version: 2026.11.10.1

From 2026.11.09.8 the SLURM layer is llmt-runtime's (llmt_runtime.slurm and llmt_runtime.jobs, configured by
runtime.RT): the settings, caps and the node's memory, the machine (cluster nodes, this node, pinning, cores), time
limits, a call's time budget, job.sh and submission, squeue/sacct and the states of jobs, the queue, scancel and
SLURM's side of extend. The names below stay as thin wrappers (api, runner, study, bench, mcp_server and the tests call
them). What stays feat's: the sizing and its models (solves_of, modal runs, the meshing margin, time_limit_min, the node
estimate before meshing and its calibration, the part-volume pool, the equation solver, threads by size and memory,
the memory model, memory_plan and its note, mem_auto, measured_seconds, oom_advice), given to the runtime as hooks
(runtime.py). (2026.11.09.6: ended_state and the runtime's version in status_key were the first pieces.)

Settings (environment): FEAT_EXECUTOR local | slurm (local); FEAT_SLURM_MIN_MINUTES (1): studies whose
time estimate is below run locally even with slurm; FEAT_SLURM_ARGS: extra sbatch arguments, split as a shell
does (a --mem there wins over FEAT_SLURM_MEM, a --nodelist/-w over FEAT_SLURM_NODE); FEAT_SLURM_NODE local
(default: --nodelist=<this node> when the cluster has several nodes) | any | a node list;
FEAT_PROCS (1): CalculiX threads per study (a study's "procs" overrides it); FEAT_TIMEOUT (900 s): per
CalculiX solve. A study's "queue": true sends it to SLURM, "queue": false keeps it local (not with
FEAT_EXECUTOR=local, which is always local). FEAT_SLURM_MEM: auto (default) | a size (8G) | none:
the job's memory. Where SLURM gives a job without --mem the whole node (CR_Core_Memory with
DefMemPerNode=UNLIMITED, as on mochi: 60000M each), only one job runs at a time; auto asks for what
CalculiX needs, from the study's node count when it is known before meshing, else 8G (never above the node's memory).
FEAT_SLURM_TIME: max (default: the partition's own limit) | timeout (the runs' timeouts, the meshing and 10 min) |
estimate (from the study's run-time estimate, each run's timeout cut to fit: timeout_for_limit) | minutes.

Per study (docs/EXECUTION.md "Per-study run settings"): "run": {"timeout_s", "mem", "procs", "queue"}
overrides FEAT_TIMEOUT, the job's --mem (any --mem/--mem-per-cpu in FEAT_SLURM_ARGS is dropped, the
other arguments kept), the threads (still size-capped) and queue, within FEAT_MAX_TIMEOUT (s),
FEAT_MAX_MEM and FEAT_MAX_PROCS from config/env (unset: no cap).

One SLURM job per study: <request>/<name>/job.sh (a complete script: `sbatch job.sh` reruns it) runs
`python -m fea_translator.cli run-study REQUEST NAME` with --ntasks=1 --cpus-per-task=N (1 for modal and
buckling: their eigenvalues are only repeatable single-threaded); job.json records the id. The state of a study is
read from the files and the queue (squeue once per call; sacct, else job.log, for jobs that ended
without a result), never from this process, so it works after restarts and from other conversations.
SLURM that does not answer (scontrol, squeue) is asked again next time, never remembered as 'no SLURM' or
'job gone'. In an MCP call, every SLURM command, submission and part estimate keeps to the call's deadline.
"""
from __future__ import annotations

import json
import math
import os
import socket  # noqa: F401 (llmt_runtime.slurm asks socket.gethostname(); tests patch it as executor.socket)
from pathlib import Path

from llmt_runtime import jobs as J
from llmt_runtime import slurm as S

from .runtime import RT

PREFIX = "feat:"           # job names: feat:<request folder name>:<study name>
FINAL = J.FINAL            # ("timed_out", "out_of_memory", "cancelled", "failed")


# ----------------------------------------------------------------------------- settings (llmt-runtime's)

def kind() -> str:
    return S.kind(RT)


def min_minutes() -> float:
    return S.min_minutes(RT)


def default_procs() -> int:
    return S.default_procs(RT)


run_of = S.run_of


def timeout_s(study: dict | None = None) -> float:
    """Seconds per CalculiX solve: the study's run.timeout_s, else FEAT_TIMEOUT (900)."""
    return S.timeout_s(RT, study)


mem_mb = S.mem_mb          # a SLURM memory size (500M, 8G, 1T, 8GB, a plain number in MB) in MB; None if malformed


def caps() -> dict:
    """Ceilings on what a study may ask: FEAT_MAX_TIMEOUT (s), FEAT_MAX_MEM, FEAT_MAX_PROCS (config/env), and with
    SLURM the memory of the node jobs go to, when it is lower or no FEAT_MAX_MEM is set; mem_mb (MB) and mem_from say
    how much and why (the router leaves out machines below a study's run.mem). llmt-runtime's: the node's memory is
    exact from 2026.11.09.8 ('64000M', mem_mb 64000; it read '62G', 63488)."""
    return S.caps(RT)


def job_mem_mb() -> float | None:
    """The most memory one of this translator's jobs can ask for (MB): the RealMemory of the nodes its jobs may go to
    (the pinned ones, else the cluster's), the largest; None without SLURM or when SLURM does not know it."""
    return S.node_mem_mb(RT)


def procs_of(study: dict) -> int:
    """Threads asked for: run.procs, then the study's procs, then FEAT_PROCS (before the size cap)."""
    return S.procs_of(RT, study)


queue_of = S.queue_of      # run.queue, then the study's queue; None: decided by the time estimate


def slurm_args() -> list[str]:
    """FEAT_SLURM_ARGS split as a shell does (several options in one value: "--mem=8G --nodelist=spark-5da5")."""
    return S.slurm_args(RT)


opt_value = S.opt_value    # an sbatch option's value among arguments ('' without a value, None when absent)


def args_mem() -> str | None:
    """A --mem / --mem-per-cpu in FEAT_SLURM_ARGS (it wins over FEAT_SLURM_MEM)."""
    return S.args_mem(RT)


def args_nodelist() -> str | None:
    """A --nodelist / -w in FEAT_SLURM_ARGS (it wins over FEAT_SLURM_NODE)."""
    return S.args_nodelist(RT)


def sbatch_args(study: dict) -> list[str]:
    """FEAT_SLURM_ARGS for this study's sbatch: with run.mem, its --mem/--mem-per-cpu are dropped."""
    return S.sbatch_args(RT, study)


# ----------------------------------------------------------------------------- feat's time model

def solves_of(study: dict) -> int:
    """CalculiX runs of a study with the full timeout_s each: joule 2, +1 with stress, x2 with converge (an
    assembly's modal runs: modal_runs_of)."""
    n = {"joule": 2}.get(study.get("analysis"), 1) + (1 if study.get("stress") else 0)
    return n * (2 if study.get("converge") else 1)


MODAL_FRACTION = 0.25      # an assembly's modal step: a linear frequency run, a quarter of timeout_s (at least 10 min)


def modal_timeout_s(timeout: float) -> float:
    return max(600.0, MODAL_FRACTION * float(timeout))


def modal_runs_of(study: dict) -> int:
    """An assembly's (or joint's) modal steps: each a separate CalculiX run (joint.modal_run)."""
    if study.get("analysis") not in ("assembly", "joint"):
        return 0
    try:
        from .joint import deck_steps
        return sum(1 for s in deck_steps(study) if s["kind"] == "modal")
    except Exception:  # noqa: BLE001 (an invalid study: no modal runs counted)
        return 0


MESH_MARGIN_MIN = 10       # meshing before the first solve: at least 10 min in the job's time limit ...
MESH_NODES_PER_S = 3000    # ... or the expected nodes at 3000 per second (an S50 casing: 1.7 M nodes in 171 s)


def mesh_margin_min(study: dict) -> int:
    n = estimate_nodes(study) or 0
    return int(math.ceil(max(MESH_MARGIN_MIN * 60, n / MESH_NODES_PER_S) / 60)) * (2 if study.get("converge") else 1)


def time_limit_min(study: dict | None = None) -> int:
    """The job's time limit: every CalculiX run's timeout (the solves, an assembly's modal runs), the meshing,
    and 10 minutes, so the translator's own timeout (which writes a clean result) fires before SLURM's.
    (A job's meshing came out of a 10-minute margin before 2026.11.05.2, and modal runs were not counted.)
    The runtime's hooks.time_need_min: FEAT_SLURM_TIME=timeout."""
    st = study or {}
    t = solves_of(st) * timeout_s(study) + modal_runs_of(st) * modal_timeout_s(timeout_s(study))
    return int(math.ceil(t / 60)) + mesh_margin_min(st) + 10


# ----------------------------------------------------------------------------- time limits (llmt-runtime's)

slurm_minutes = S.slurm_minutes   # a SLURM time ('infinite', '7-00:00:00', '2:00:00', '30', '5:30') in minutes
_PART = S._PART                   # the partition's limit, kept 5 minutes (the runtime's: tests reset it here)


def partition_max_min() -> float | None:
    """The time limit (minutes; inf: none) of the partition this translator's jobs go to (FEAT_SLURM_ARGS
    --partition, else the default one), from sinfo; None when SLURM does not say. Kept for 5 minutes."""
    return S.partition_max_min(RT)


def job_time(study: dict | None = None, estimate_min: float | None = None) -> tuple[str, dict]:
    """(the job's --time, details). FEAT_SLURM_TIME: 'max' (default: the partition's own limit, UNLIMITED where it
    has none, so that extend(request) needs only feat's own CalculiX timeout: SLURM lets only an admin raise a
    running job's limit, mochi job 3177, 2026.11.09), 'timeout' (feat's time_limit_min: the runs' timeouts, the
    meshing and 10 min), 'estimate' (from the study's run-time estimate, estimate_min: max(FEAT_SLURM_TIME_MIN 20,
    3 x estimate + 10) min, at most 'timeout'; each run's timeout then fits it: timeout_for_limit), or minutes (at
    least 'timeout'). feat's own timeout still ends CalculiX and writes a clean result; the SLURM limit is the
    backstop. (Until 2026.11.10.1 'estimate' was read as 'timeout', feat's word for it before 2026.11.09.8.)"""
    return S.job_time(RT, study, estimate_min)


def timeout_for_limit(study: dict, minutes: int) -> float:
    """Seconds each CalculiX run of a study may take when its job has `minutes` (FEAT_SLURM_TIME=estimate): the
    inverse of time_limit_min. The limit less 10 min and the meshing margin is shared by the solves (each the
    timeout) and an assembly's modal runs (each max(600 s, a quarter of it)); at least 60 s."""
    st = study or {}
    avail = (int(minutes) - 10 - mesh_margin_min(st)) * 60.0
    n, m = solves_of(st), modal_runs_of(st)
    t = (avail - 600.0 * m) / n                      # the modal runs at their 10-minute floor
    if m and MODAL_FRACTION * t > 600.0:             # ... or a quarter of the timeout each
        t = avail / (n + MODAL_FRACTION * m)
    return max(60.0, t)


def job_time_left_min(job_id: str) -> float | None:
    """Minutes a running job has left before SLURM's limit (squeue %L); inf without a limit; None when unknown."""
    return S.job_time_left_min(job_id)


def status_key() -> dict:
    """The 'executor' key of tool_status (llmt status reads it): llmt-runtime's (feat's keys, plus 'time' and
    'runtime')."""
    return S.status_key(RT)


# ----------------------------------------------------------------------------- a call's time budget
# Every MCP tool answers within its wait (at most 50 s): the server sets a deadline for the call's thread, and
# SLURM commands, submissions and estimates that need CAD (a worker process) use only the time that is left.
# The runtime's: one budget for feat's code and the runtime's SLURM commands.

deadline = S.deadline
remaining = S.remaining
_CALL = S._CALL


def _run(cmd: list[str], timeout: float = 15):
    """A SLURM (or system) command's result; None when it is missing, fails to run or does not answer in time
    (within the call's remaining time, when in an MCP call)."""
    return S.run(cmd, timeout)


# ----------------------------------------------------------------------------- clusters of several nodes
# llm-translators docs/EXECUTION.md "Clusters of several nodes": the DGX Sparks spark-5da5 (controller) and
# spark-0808 are one SLURM cluster without shared home folders, so a job must run on the node that submitted it
# (its run folder is only there). FEAT_SLURM_NODE: local (default: --nodelist=<this node> when the cluster has
# several nodes) | any (SLURM chooses) | a node list; a --nodelist / -w in FEAT_SLURM_ARGS wins. llmt-runtime's.

NODES_TTL = S.NODES_TTL
_NODES = S._NODES          # the runtime's cache (t, nodes, failed): tests reset it here


def cluster_nodes() -> list[dict] | None:
    """The cluster's nodes from `scontrol -o show node`, one {key: value} per node; None when SLURM does not
    answer. An answer is reused for 60 s; a failure is asked again, but not within 5 s."""
    return S.cluster_nodes()


def _addresses() -> set[str]:
    """This machine's addresses (`hostname -I`, the host name's own)."""
    return S.addresses()


def this_node(nodes: list[dict] | None) -> dict | None:
    """The cluster node that is this machine (by name, else by address); None if none matches."""
    return S.this_node(nodes)


def node_setting() -> str:
    return S.node_setting(RT)


def node_pin() -> tuple[str | None, str | None, dict]:
    """(--nodelist this translator adds to a job or None, the job's effective nodelist or None when SLURM may
    choose, details: cluster_nodes, this_node, warning)."""
    return S.node_pin(RT)


def node_status() -> dict:
    """tool_status().executor.node: {setting, nodelist (as jobs ask: null when SLURM may choose; the router's
    sbatch --test-only uses the same), cluster_nodes, this_node; from_args when FEAT_SLURM_ARGS sets the nodelist}."""
    return S.node_status(RT)


def node_cores() -> int:
    """Cores a job can use (jobs ask for one thread per core): the smallest node of a pinned nodelist, else this node
    (or the only one), else the cluster's smallest; without SLURM, this machine's CPUs. Thread caps use these, never
    the cluster's total. llmt-runtime's (cores from Sockets x CoresPerSocket when SLURM gives them)."""
    return S.node_cores(RT)


# ----------------------------------------------------------------------------- feat's size estimates

def _joint_parts(study: dict) -> list[dict] | None:
    """A joint study's parts as single-part studies ({part, mesh}: the part's mesh over the study's), else None."""
    if study.get("analysis") not in ("joint", "assembly") or not isinstance(study.get("parts"), dict):
        return None
    base = study.get("mesh") if isinstance(study.get("mesh"), dict) else {}
    return [{"part": p.get("part", {}), "mesh": {**base, **(p.get("mesh") if isinstance(p.get("mesh"), dict) else {})}}
            for p in study["parts"].values() if isinstance(p, dict)]


def _sum(values) -> int | None:
    values = list(values)
    return None if not values or any(v is None for v in values) else int(sum(values))


def nodes_hint(study: dict) -> int | None:
    """Nodes of a study's (first) mesh when they are known before meshing: box parts (their divisions,
    or the element size); None for STEP files and goals. A joint: the sum over its parts."""
    jp = _joint_parts(study)
    if jp is not None:
        return _sum(nodes_hint(p) for p in jp)
    part = study.get("part", {})
    if not isinstance(part, dict) or "box" not in part:
        return None
    try:
        dims = [float(v) for v in part["box"]]
        div = part.get("divisions")
        if not div:
            h = float(study.get("mesh", {}).get("size") or (sum(d * d for d in dims) ** 0.5) / 20)
            div = [max(1, round(d / h)) for d in dims]
        k = 2 if int(study.get("mesh", {}).get("order", 2)) == 2 else 1
        n = 1
        for d in div:
            n *= k * int(d) + 1
        return n
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# ---- sizing from this machine's measurements (tools/measure_threads.py, mochi 2026-10-20: a 175k-node box,
# direct solver 409 / 230 / 167 / 125 s at 1 / 6 / 10 / 19 threads with 6.6 / 12.2 / 12.5 / 14.0 GB; iterative
# 167 s at 1 and at 10 threads, 0.64 GB; a 45k-node box 30 / 21 / 18 s, 1.1 / 1.9 / 2.0 GB)

FINE = 1 / 0.6 ** 3          # converge: true meshes again at 0.6 x the size: ~4.6x the nodes
NODES_PER_H3 = {2: 12.0, 1: 2.0}   # Gmsh tetrahedra, nodes per (element size)^3 of volume: an upper estimate
NODES_PER_THREAD = 10_000    # direct solver: a thread per 10k nodes at most (45k nodes gain 1.7x at 10 threads)
MEM_DEFAULT_GB = 8           # when the size cannot be estimated
_VOLUMES: dict = {}


VOLUME_WAIT_S = 20.0   # an MCP call waits at most this long for a part's volume (then: size unknown)
_PENDING: dict = {}
_VPOOL: dict = {"ex": None}


def _part_volume(part: dict) -> tuple[float, float, float] | None:
    """(volume, bounding-box diagonal, smallest bounding-box side) of a goal (built with build123d-translator,
    milliseconds to seconds) or a STEP file (read with build123d, about a second); None if it cannot be read.
    In an MCP call (a deadline is set) the CAD work runs in a worker process and the call waits only for the time
    it has (at most VOLUME_WAIT_S): building a goal never blocks the server; a late answer is kept for next time."""
    key = json.dumps(part, sort_keys=True, default=str)
    if key in _VOLUMES:
        return _VOLUMES[key]
    left = remaining()
    if left is None:
        _VOLUMES[key] = _volume_of(part)
        return _VOLUMES[key]
    fut = _PENDING.get(key)
    if fut is None:
        try:
            fut = _volume_pool().submit(_volume_of, part)
        except Exception:  # noqa: BLE001 (a broken pool: no estimate this time; a new pool next time)
            _VPOOL["ex"] = None
            return None
        _PENDING[key] = fut

        def keep(f, key=key):
            _PENDING.pop(key, None)
            try:
                _VOLUMES[key] = f.result()
            except Exception:  # noqa: BLE001
                _VOLUMES[key] = None
        fut.add_done_callback(keep)
    try:
        return fut.result(timeout=max(0.05, min(VOLUME_WAIT_S, left - 5)))
    except Exception:  # noqa: BLE001 (not ready in time, or failed: the size is unknown for now)
        return _VOLUMES.get(key)


def _volume_pool():
    if _VPOOL["ex"] is None:
        import atexit
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        from .workers import watch_parent
        _VPOOL["ex"] = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                                           initializer=watch_parent, initargs=(os.getpid(),))
        atexit.register(_close_volume_pool)
    return _VPOOL["ex"]


def close_volume_pool(terminate: bool = False) -> None:
    """Stops the volume pool; terminate: also ends its worker processes (shutdown(wait=False) leaves a busy or idle
    spawn worker running)."""
    ex, _VPOOL["ex"] = _VPOOL["ex"], None
    _PENDING.clear()
    if ex is None:
        return
    procs = list((getattr(ex, "_processes", None) or {}).values())
    ex.shutdown(wait=False, cancel_futures=True)
    if terminate:
        for p in procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass
        for p in procs:
            try:
                p.join(2)
            except Exception:  # noqa: BLE001
                pass


def _close_volume_pool() -> None:
    close_volume_pool(terminate=True)


def _volume_of(part: dict) -> tuple[float, float, float] | None:
    out = None
    try:
        if "goal" in part:
            from build123d_translator import translate as T

            solid, _ = T.build(part["goal"])
        elif "step" in part:
            from build123d import import_step

            solid = import_step(str(Path(part["step"]).expanduser()))
        else:
            solid = None
        if solid is not None:
            bb = solid.bounding_box()
            out = (float(solid.volume), float(bb.diagonal), float(min(bb.size.X, bb.size.Y, bb.size.Z)))
    except Exception:  # noqa: BLE001
        out = None
    return out


_FACTORS: dict = {"key": None, "f": {}}


def node_factor(analysis: str | None) -> float:
    """How many more nodes this machine's meshes of an analysis had than estimated (the median of meshed /
    raw estimate in timing.jsonl, its last 500 records); 1 without records, never below 1 (the estimate stays an
    upper one where it was). carbonio job 5807: an S50 sleeve assembly estimated at 107,171 nodes had 424,303 (curved
    faces, small features, fine parts), so its memory came out 6x low."""
    from statistics import median
    if not analysis:
        return 1.0
    try:
        from .runner import stats_path
        f = stats_path()
        key = (str(f), f.stat().st_mtime)
    except (OSError, ImportError):
        return 1.0
    if _FACTORS["key"] != key:
        ratios: dict = {}
        try:
            for line in f.read_text().splitlines()[-500:]:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("nodes_estimate_raw") and r.get("nodes"):
                    ratios.setdefault(r.get("analysis"), []).append(r["nodes"] / r["nodes_estimate_raw"])
        except OSError:
            pass
        _FACTORS.update(key=key, f={a: min(max(median(v), 1.0), 20.0) for a, v in ratios.items()})
    return _FACTORS["f"].get(analysis, 1.0)


def estimate_nodes(study: dict) -> int | None:
    """Nodes of a study's (first) mesh before meshing: estimate_nodes_raw, times what this machine's meshes of the
    same analysis had more (node_factor); exact for box parts (no factor)."""
    n = estimate_nodes_raw(study)
    if n is None or nodes_hint(study) is not None:
        return n
    return int(n * node_factor(study.get("analysis")))


def estimate_nodes_raw(study: dict) -> int | None:
    """Nodes of a study's (first) mesh before meshing: exact for box parts, else an estimate from the
    part's volume and the element size (plus the refinement spheres); None if it cannot be told."""
    jp = _joint_parts(study)
    if jp is not None:
        return _sum(estimate_nodes_raw(p) for p in jp)
    n = nodes_hint(study)
    if n is not None:
        return n
    part = study.get("part", {})
    if not isinstance(part, dict):
        return None
    vd = _part_volume(part)
    if vd is None or vd[0] <= 0:
        return None
    vol, diag, thin = vd
    mesh = study.get("mesh", {}) if isinstance(study.get("mesh"), dict) else {}
    k = NODES_PER_H3[2 if int(mesh.get("order", 2)) == 2 else 1]
    h = float(mesh.get("size") or diag / 20)
    n = k * vol / h ** 3
    for r in mesh.get("refine", []) or []:
        try:
            rr = float(r["r"])   # the part of the sphere inside the part: at most a slab as thick as its thinnest side
            region = min(4 / 3 * math.pi * rr ** 3, (2 * rr) ** 2 * min(2 * rr, thin), vol)
            n += k * region / float(r["size"]) ** 3
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
    return int(n)


def _meshes(study: dict, nodes: int) -> list[int]:
    return [nodes] + ([int(nodes * FINE)] if study.get("converge") else [])


def method_of(study: dict, nodes: int) -> str:
    from .deck import equation_solver

    return equation_solver(study, nodes)


def _threads_by_size(study: dict) -> int:
    """Threads before the memory check: 1 for modal and buckling, 1 when every solve uses the iterative solver, else
    the study's procs capped at a thread per NODES_PER_THREAD nodes of its largest direct-solver mesh."""
    if study.get("analysis") in ("modal", "buckling"):
        return 1
    procs = min(procs_of(study), node_cores())
    nodes = estimate_nodes(study)
    if nodes is None:
        return procs
    direct = [n for n in _meshes(study, nodes) if method_of(study, n) == "direct"]
    if not direct:
        return 1
    return max(1, min(procs, int(math.ceil(max(direct) / NODES_PER_THREAD))))


def given_mem_gb(study: dict) -> tuple[float | None, str]:
    """(GB, where from) of the memory a study's CalculiX run gets when that is fixed: run.mem, a size in FEAT_SLURM_MEM
    or FEAT_SLURM_ARGS's --mem; with FEAT_SLURM_MEM=auto, what the machine allows (FEAT_MAX_MEM, the node's
    memory); (None, '') when nothing limits it. (llmt-runtime's given_mem_mb, in GB.)"""
    mb, src = S.given_mem_mb(RT, study)
    return (mb / 1024 if mb else None), src


def mem_need_gb(study: dict, threads: int) -> float | None:
    """GB CalculiX needs for a study with `threads` threads (mem_gb on its largest mesh); None: size unknown."""
    nodes = estimate_nodes(study)
    if nodes is None:
        return None
    return max(mem_gb(n, method_of(study, n), threads) for n in _meshes(study, nodes))


def procs_for(study: dict) -> int:
    """CalculiX threads (CPUs) for a study: _threads_by_size, then 1 when the memory it gets (given_mem_gb) is below
    what those threads need but one thread fits better: threads multiply the direct solver's memory, and a run that
    runs out of memory is restarted on one thread anyway after the time it lost (carbonio, 2026.11.05: 6 threads in
    32 GB, killed after 53 min, while 5 CPUs were held for nothing). The SLURM job then asks for one CPU.
    The runtime's hooks.procs."""
    t = _threads_by_size(study)
    if t <= 1:
        return t
    have, _ = given_mem_gb(study)
    need = mem_need_gb(study, t)
    if have is not None and need is not None and need > have:
        return 1
    return t


def memory_plan(study: dict) -> dict:
    """What a study gets and what feat estimates: {given_gb, given_from, threads_asked, threads, estimate_gb: {threads:
    GB}, note}; the note when the memory is below the estimate for the threads asked (and the way out)."""
    asked = _threads_by_size(study)
    t = procs_for(study)
    have, src = given_mem_gb(study)
    est = {n: mem_need_gb(study, n) for n in sorted({asked, 1})}
    n, raw = estimate_nodes(study), estimate_nodes_raw(study)
    if nodes_hint(study) is not None:
        basis = "an exact box mesh"
    elif raw:
        basis = (f"nodes estimated before meshing: {raw:,} from the volumes and element sizes"
                 + (f", x{n / raw:.2f} from this machine's meshes of {study.get('analysis')}" if n and n > raw else
                    " (meshes with curved faces and small features can have several times more)")
                 + "; CalculiX's run checks the threads again on the meshed count")
    else:
        basis = None
    out = {"given_gb": None if have is None else round(have, 1), "given_from": src or None, "threads_asked": asked,
           "threads": t, "estimate_gb": {str(k): (None if v is None else round(v, 1)) for k, v in est.items()},
           "nodes_estimate": n, "estimate_basis": basis}
    note = memory_note(have, src, asked, t, est)
    if note:
        out["note"] = note
    return out


def ask_gb(need: float) -> int:
    """The memory to ask for (GB) when CalculiX needs `need`: 1 GB more for feat itself, which shares the job."""
    return int(math.ceil(need + 1))


def memory_note(have, src, asked, t, est) -> str | None:
    """'run.mem 32G is below the 62 GB estimate for 6 threads: give 62G, or 1 thread fits in 32G (it runs on 1)'."""
    if have is None:
        return None
    one, many = est.get(1), est.get(asked)
    if asked > 1 and t == 1 and many is not None:
        fits = (f"1 thread fits in {have:.0f} GB" if one is not None and one <= have else
                f"even 1 thread needs about {one:.0f} GB")
        return (f"{src} is below the {many:.0f} GB estimate for {asked} threads: give {ask_gb(many)}G for the "
                f"threads, or {fits} (feat runs it on 1 thread and asks SLURM for 1 CPU)")
    need = est.get(t)
    if need is not None and need > have:
        return (f"{src} is below the {need:.0f} GB estimate for {t} thread(s): CalculiX may run out of memory; "
                f"give {ask_gb(need)}G or use a coarser mesh")
    return None


def mem_gb(nodes: int, method: str, threads: int) -> float:
    """Memory to ask for (GB), fitted to the measurements with a margin: the iterative solver needs ~3.7 kB
    per node; the direct one grows as nodes^1.43 (2.0 GB at 45k, 14 GB at 175k), half that single-threaded."""
    if method == "iterative":
        return 1 + 1.5 * 3.7e-6 * nodes
    gb = 2.3 * (nodes / 5e4) ** 1.43 * (0.5 if threads <= 1 else 1.0)
    return 1 + 1.25 * gb


def mem_for(study: dict) -> str | None:
    """The --mem value of a study's job: its run.mem, else FEAT_SLURM_MEM (auto | size | none); None: do
    not ask (also when FEAT_SLURM_ARGS has its own --mem). llmt-runtime's mem_plan: sizes as sbatch takes them
    ('8GB' becomes '8192M'), auto (mem_auto) never above the node's memory."""
    return S.mem_for(RT, study)


def mem_auto(study: dict) -> str:
    """The memory CalculiX needs for a study (FEAT_SLURM_MEM=auto): from its estimated size, else MEM_DEFAULT_GB.
    The runtime's hooks.mem_auto (it caps the answer at the node's memory)."""
    nodes = estimate_nodes(study)
    if nodes is None:
        return f"{MEM_DEFAULT_GB}G"
    t = procs_for(study)
    gb = max(mem_gb(n, method_of(study, n), t) for n in _meshes(study, nodes))
    return f"{max(2, int(math.ceil(gb)))}G"


# single-threaded CalculiX on mochi (tools/measure_threads.py, 2026-10-20): a 175,329-node box took 409 s with
# the direct solver (~nodes^1.43) and 167 s with the iterative one (~nodes); threads speed the direct solver
# up ~threads^0.4 (1.8 / 2.5 / 3.3 x at 6 / 10 / 19). For estimates before this machine has timing records.
MEASURED_NODES, MEASURED_DIRECT_S, MEASURED_ITERATIVE_S = 175_329, 409.0, 167.0


def measured_seconds(study: dict) -> float | None:
    """Seconds of a study's CalculiX solves from the mochi measurements (meshing ~0.1 ms per node, plus
    30 s of start-up per solve); None if its size cannot be estimated."""
    nodes = estimate_nodes(study)
    if nodes is None:
        return None
    t = procs_for(study)
    per_mesh = solves_of(study) // (2 if study.get("converge") else 1)
    s = 0.0
    for n in _meshes(study, nodes):
        if method_of(study, n) == "iterative":
            solve = MEASURED_ITERATIVE_S * n / MEASURED_NODES
        else:
            solve = MEASURED_DIRECT_S * (n / MEASURED_NODES) ** 1.43 / max(1, t) ** 0.4
        s += 1e-4 * n + per_mesh * (solve + 30.0)
    return s


def use_slurm(study: dict, estimate_min: float | None) -> bool:
    """Whether a study becomes a SLURM job: only with FEAT_EXECUTOR=slurm; then "queue" decides if
    given, else the estimate (no estimate: SLURM)."""
    return S.use_slurm(RT, study, estimate_min)


def job_extra(study: dict) -> dict:
    """feat's keys in job.json (the runtime's hooks.job_extra): nodes_estimate, nodes_estimate_raw, equation_solver,
    memory (memory_plan; left out when it fails: advice only)."""
    n = estimate_nodes(study)
    out = {"nodes_estimate": n, "nodes_estimate_raw": estimate_nodes_raw(study),
           "equation_solver": method_of(study, n) if n else None}
    try:
        out["memory"] = memory_plan(study)
    except Exception:  # noqa: BLE001 (advice only)
        pass
    return out


# ----------------------------------------------------------------------------- submission (llmt-runtime's)

def job_script(request: Path, name: str, study: dict) -> str:
    """job.sh of a study: its name feat:<request>:<name>, folder, log, --ntasks=1 --cpus-per-task (procs_for),
    --time (job_time), --mem (mem_for), --nodelist (node_pin), then `python -m fea_translator.cli run-study`."""
    return J.job_script(RT, Path(request), name, study, procs_for(study), job_time(study)[0], mem_for(study),
                        node_pin()[0])


def submit(request: Path, name: str, study: dict, estimate_min: float | None = None) -> dict:
    """Writes job.sh and submits it; returns job.json's content, or {'ok': False, 'why': ...} (never a
    silent fallback to local). A job that sbatch took but answered too late for is found by its name (job.json
    'adopted'). job.json: the runtime's keys and feat's (job_extra). estimate_min: the study's run-time estimate,
    for FEAT_SLURM_TIME=estimate."""
    return J.submit(RT, Path(request), name, study, estimate_min)


# ----------------------------------------------------------------------------- queue (llmt-runtime's)

_minutes = S.elapsed_minutes   # squeue's elapsed time ([D-]HH:MM:SS, MM:SS) in minutes
_tail = J.tail


def squeue(ids: list[str]) -> dict | None:
    """{id: {state, reason, start, elapsed_min, node}} of the given jobs still in the queue (one call); None when
    SLURM does not answer (then nothing can be said about the jobs: they are not 'gone'). Ended jobs that squeue
    still lists (MinJobAge) are among them, with their final state."""
    return J.squeue(ids)


def pending_ids() -> list[int] | None:
    """Ids of every pending job in the queue (one squeue call); None when SLURM does not answer."""
    p = J.pending_jobs()
    return None if p is None else [i for _, i in p]


def ahead_of(job_id: str, pending: list[tuple[float, int]] | None = None) -> int | None:
    """Pending jobs SLURM considers before this one: higher priority, or equal and submitted earlier (all of the
    queue's, not only the translators'). pending: llmt_runtime.jobs.pending_jobs()'s (priority, id) pairs."""
    return J.ahead_of(job_id, pending)


def sacct(ids: list[str]) -> dict:
    """{id: STATE} from SLURM's accounting; {} when there is none (common on single-node installs)."""
    return J.sacct(ids)


def oom_advice(folder: Path) -> str:
    """Why and what to ask for when SLURM killed a study's job for memory (the whole batch step, feat with it: no
    result): the job's memory and threads (job.json), feat's estimate for the meshed nodes (progress.json)."""
    try:
        job = json.loads((folder / "job.json").read_text())
    except (OSError, ValueError):
        job = {}
    try:
        nodes = int(json.loads((folder / "progress.json").read_text()).get("nodes") or 0)
    except (OSError, ValueError, TypeError):
        nodes = 0
    t, mem = int(job.get("procs") or 1), job.get("mem")
    head = "SLURM killed the job for memory" + (f" ({mem} with {t} thread{'s' if t > 1 else ''})" if mem else "")
    if not nodes:
        return head + ", before meshing finished: give the study more memory (run.mem) or coarsen the mesh"
    many, one = mem_gb(nodes, "direct", t), mem_gb(nodes, "direct", 1)
    ask = (f"run.mem {ask_gb(many)}G for {t} threads, or run.procs 1 with run.mem {ask_gb(one)}G" if t > 1 else
           f"run.mem {ask_gb(one)}G")
    return (f"{head}; for the meshed {nodes:,} nodes feat estimates about {many:.0f} GB with {t} thread(s)"
            + (f" and {one:.0f} GB with 1" if t > 1 else "") + f": {ask} (or a coarser mesh)")


def ended_state(acct: str | None, log: Path) -> tuple[str, str]:
    """(state, why) of a job that left the queue without writing a result: llmt-runtime's ended_state (sacct's state,
    else job.log: 'DUE TO TIME LIMIT', 'CANCELLED', SLURM's out-of-memory wording 'Detected N oom_kill events'), with
    feat's oom_advice and timeout hint (runtime.RT's hooks)."""
    return J.ended_state(RT, acct, Path(log))


def job_states(request: Path, names: list[str]) -> dict:
    """{name: {state, job_id, ...}} for the request's SLURM studies, from the files and the queue (llmt-runtime's):
    done, queued (reason, expected_start, ahead), running (job_elapsed_min, node), unknown (SLURM did not answer), a
    final state (FINAL, with why); a study with no job.json yet is queued ('submitting') for 10 minutes after its
    request was written, then failed."""
    return J.job_states(RT, Path(request), names)


def queue_summary() -> dict:
    """This translator's jobs in the queue ({queue: running, queued, cpus_in_use}), and the CPUs of the nodes its jobs
    go to ({machine: cpus, cpus_allocated, cpus_idle}, each node once)."""
    return J.queue_summary(RT)


def extend_job(job_id: str, minutes: int) -> str | None:
    """Raises a SLURM job's time limit by `minutes` (scontrol update TimeLimit=+N); None, or why not (raising a
    limit usually needs an operator or admin)."""
    return S.extend_job(job_id, minutes)


def scancel(ids: list[str]) -> str | None:
    """Cancels the jobs; returns an error message or None."""
    return J.scancel(ids)
