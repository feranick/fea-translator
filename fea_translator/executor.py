"""executor.py — where studies run: the local worker pool, or SLURM jobs (llm-translators docs/EXECUTION.md).
Version: 2026.11.09.5

Settings (environment): FEAT_EXECUTOR local | slurm (local); FEAT_SLURM_MIN_MINUTES (1): studies whose
time estimate is below run locally even with slurm; FEAT_SLURM_ARGS: extra sbatch arguments, split as a shell
does (a --mem there wins over FEAT_SLURM_MEM, a --nodelist/-w over FEAT_SLURM_NODE); FEAT_SLURM_NODE local
(default: --nodelist=<this node> when the cluster has several nodes) | any | a node list;
FEAT_PROCS (1): CalculiX threads per study (a study's "procs" overrides it); FEAT_TIMEOUT (900 s): per
CalculiX solve. A study's "queue": true sends it to SLURM, "queue": false keeps it local (not with
FEAT_EXECUTOR=local, which is always local). FEAT_SLURM_MEM: auto (default) | a size (8G) | none:
the job's memory. Where SLURM gives a job without --mem the whole node (CR_Core_Memory with
DefMemPerNode=UNLIMITED, as on mochi: 60000M each), only one job runs at a time; auto asks for what
CalculiX needs, from the study's node count when it is known before meshing, else 8G.

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
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

PREFIX = "feat:"           # job names: feat:<request folder name>:<study name>
FINAL = ("timed_out", "out_of_memory", "cancelled", "failed")


# ----------------------------------------------------------------------------- settings

def kind() -> str:
    k = os.environ.get("FEAT_EXECUTOR", "local").strip().lower()
    return k if k in ("local", "slurm") else "local"


def min_minutes() -> float:
    try:
        return float(os.environ.get("FEAT_SLURM_MIN_MINUTES", "1"))
    except ValueError:
        return 1.0


def default_procs() -> int:
    try:
        return max(1, int(os.environ.get("FEAT_PROCS", "1")))
    except ValueError:
        return 1


def run_of(study: dict | None) -> dict:
    r = (study or {}).get("run")
    return r if isinstance(r, dict) else {}


def timeout_s(study: dict | None = None) -> float:
    """Seconds per CalculiX solve: the study's run.timeout_s, else FEAT_TIMEOUT (900)."""
    t = run_of(study).get("timeout_s")
    return float(t) if t else float(os.environ.get("FEAT_TIMEOUT", "900"))


def mem_mb(v) -> float | None:
    """A SLURM memory size (500M, 8G, 1T, a plain number in MB) in MB; None if malformed."""
    import re

    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*", str(v), re.I)
    if not m:
        return None
    return float(m.group(1)) * {"K": 1 / 1024, "": 1, "M": 1, "G": 1024, "T": 1024 ** 2}[m.group(2).upper()]


def caps() -> dict:
    """Ceilings on what a study may ask: FEAT_MAX_TIMEOUT (s), FEAT_MAX_MEM, FEAT_MAX_PROCS (config/env), and with
    SLURM the memory of the node jobs go to (job_mem_mb), when it is lower or no FEAT_MAX_MEM is set; mem_mb (MB) and
    mem_from say how much and why (the router leaves out machines below a study's run.mem). Before 2026.11.07.2 the
    node's memory was used only to size feat's own requests: a run.mem above it failed at submission."""
    out = {}
    for key, var in (("timeout_s", "FEAT_MAX_TIMEOUT"), ("mem", "FEAT_MAX_MEM"), ("procs", "FEAT_MAX_PROCS")):
        v = os.environ.get(var, "").strip()
        if v:
            out[key] = v if key == "mem" else (float(v) if key == "timeout_s" else int(float(v)))
    if out.get("mem"):
        out["mem_from"] = "FEAT_MAX_MEM"
    nm = job_mem_mb()
    if nm and (not out.get("mem") or (mem_mb(out["mem"]) or 0) > nm):
        out["mem"] = f"{int(nm // 1024)}G" if nm >= 1024 else f"{int(nm)}M"
        out["mem_from"] = "a node's memory (SLURM RealMemory)"
    if out.get("mem") and mem_mb(out["mem"]):
        out["mem_mb"] = int(mem_mb(out["mem"]))
    return out


def job_mem_mb() -> float | None:
    """The most memory one of this translator's jobs can ask for (MB): the RealMemory of the node its jobs are pinned
    to (FEAT_SLURM_NODE, FEAT_SLURM_ARGS --nodelist), else the largest among the cluster's nodes; None without SLURM
    or when SLURM does not know it (unset nodes report 1 MB)."""
    if kind() != "slurm":
        return None
    nodes = cluster_nodes() or []
    try:
        pin = node_pin()[1]
    except Exception:  # noqa: BLE001
        pin = None
    mem = {}
    for n in nodes:
        try:
            v = float(n.get("RealMemory") or 0)
        except ValueError:
            continue
        if v > 1:
            mem[n.get("NodeName")] = v
    if pin and pin in mem:
        return mem[pin]
    return max(mem.values()) if mem else None


def procs_of(study: dict) -> int:
    """Threads asked for: run.procs, then the study's procs, then FEAT_PROCS (before the size cap)."""
    try:
        return max(1, int(run_of(study).get("procs") or study.get("procs") or default_procs()))
    except (TypeError, ValueError):
        return default_procs()


def queue_of(study: dict):
    """run.queue, then the study's queue; None: decided by the time estimate."""
    q = run_of(study).get("queue")
    return q if q is not None else study.get("queue")


def slurm_args() -> list[str]:
    """FEAT_SLURM_ARGS split as a shell does (several options in one value: "--mem=8G --nodelist=spark-5da5")."""
    try:
        return shlex.split(os.environ.get("FEAT_SLURM_ARGS", ""))
    except ValueError:  # unbalanced quotes: what a shell would refuse; keep the plain words
        return os.environ.get("FEAT_SLURM_ARGS", "").split()


def opt_value(args: list[str], longs: tuple, shorts: tuple = ()) -> str | None:
    """The value of an sbatch option among the arguments ('--nodelist=a', '--nodelist a', '-w a', '-wa');
    '' when given without a value; None when absent."""
    for i, a in enumerate(args):
        for lo in longs:
            if a == lo:
                return args[i + 1] if i + 1 < len(args) else ""
            if a.startswith(lo + "="):
                return a[len(lo) + 1:]
        for sh in shorts:
            if a == sh:
                return args[i + 1] if i + 1 < len(args) else ""
            if a.startswith(sh) and not a.startswith("--"):
                return a[len(sh):].lstrip("=")
    return None


def args_mem() -> str | None:
    """A --mem / --mem-per-cpu in FEAT_SLURM_ARGS (it wins over FEAT_SLURM_MEM)."""
    return opt_value(slurm_args(), ("--mem", "--mem-per-cpu"))


def args_nodelist() -> str | None:
    """A --nodelist / -w in FEAT_SLURM_ARGS (it wins over FEAT_SLURM_NODE)."""
    return opt_value(slurm_args(), ("--nodelist",), ("-w",))


def sbatch_args(study: dict) -> list[str]:
    """FEAT_SLURM_ARGS for this study's sbatch: with run.mem, its --mem/--mem-per-cpu are dropped."""
    args = slurm_args()
    if not run_of(study).get("mem"):
        return args
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("--mem", "--mem-per-cpu"):
            skip = True
            continue
        if a.startswith(("--mem=", "--mem-per-cpu=")):
            continue
        out.append(a)
    return out


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
    (A job's meshing came out of a 10-minute margin before 2026.11.05.2, and modal runs were not counted.)"""
    st = study or {}
    t = solves_of(st) * timeout_s(study) + modal_runs_of(st) * modal_timeout_s(timeout_s(study))
    return int(math.ceil(t / 60)) + mesh_margin_min(st) + 10


def slurm_minutes(t: str | None) -> float | None:
    """A SLURM time ('infinite', 'UNLIMITED', '7-00:00:00', '2:00:00', '30', '1-12') in minutes; inf for no limit;
    None when it cannot be read."""
    if not t:
        return None
    t = t.strip().lower()
    if t in ("infinite", "unlimited"):
        return math.inf
    try:
        days = 0
        if "-" in t:
            d, t = t.split("-", 1)
            days = int(d)
            f = [int(x) for x in t.split(":")] + [0, 0]
            h, m, sec = f[0], f[1], f[2]
        else:
            f = [int(x) for x in t.split(":")]
            h, m, sec = {1: (0, f[0], 0), 2: (0, f[0], f[-1]), 3: tuple(f)}[len(f)] if len(f) <= 3 else (0, 0, 0)
        return days * 1440 + h * 60 + m + sec / 60
    except (ValueError, KeyError):
        return None


_PART: dict = {"t": 0.0, "v": None}


def partition_max_min() -> float | None:
    """The time limit (MaxTime, minutes; inf: none) of the partition this translator's jobs go to (FEAT_SLURM_ARGS
    --partition, else the default one), from sinfo; None when SLURM does not say. Kept for 5 minutes."""
    if time.time() - _PART["t"] < 300 and _PART["v"] is not None:
        return _PART["v"]
    p = _run(["sinfo", "-h", "-o", "%P|%l"], timeout=10)
    if p is None or p.returncode != 0:
        return None
    want = opt_value(slurm_args(), ("--partition",), ("-p",))
    rows = [ln.split("|", 1) for ln in p.stdout.splitlines() if "|" in ln]
    pick = None
    for name, lim in rows:
        if (want and name.rstrip("*") == want) or (not want and name.endswith("*")):
            pick = lim
            break
    if pick is None and rows and not want:
        pick = rows[0][1]
    v = slurm_minutes(pick)
    if v is not None:
        _PART.update(t=time.time(), v=v)
    return v


def job_time(study: dict | None = None) -> tuple[str, dict]:
    """(the job's --time, details). FEAT_SLURM_TIME: 'max' (default: the partition's own limit, UNLIMITED where it
    has none, so that extend(request) needs only feat's own CalculiX timeout: SLURM lets only an admin raise a
    running job's limit, mochi job 3177, 2026.11.09), 'estimate' (feat's time_limit_min: the runs' timeouts, the
    meshing and 10 min), or minutes (at least the estimate). feat's own timeout still ends CalculiX and writes a
    clean result; the SLURM limit is the backstop."""
    need = time_limit_min(study)
    setting = os.environ.get("FEAT_SLURM_TIME", "max").strip().lower() or "max"
    info = {"needed_min": need, "setting": setting}
    if setting == "estimate":
        return str(need), info
    if setting != "max":
        try:
            return str(max(need, int(float(setting)))), info
        except ValueError:
            info["warning"] = f"FEAT_SLURM_TIME={setting!r} is not max, estimate or minutes: the estimate"
            return str(need), info
    mx = partition_max_min()
    if mx is None:
        info["note"] = "the partition's limit is unknown (sinfo): the estimate"
        return str(need), info
    if math.isinf(mx):
        info["partition_max"] = "infinite"
        return "UNLIMITED", info
    info["partition_max_min"] = int(mx)
    if mx < need:
        info["warning"] = (f"the partition's limit ({int(mx)} min) is below what the runs may take ({need} min): "
                           "SLURM may stop the job before feat's own timeout")
    return str(int(mx)), info


def job_time_left_min(job_id: str) -> float | None:
    """Minutes a running job has left before SLURM's limit (squeue %L); inf without a limit; None when unknown."""
    p = _run(["squeue", "-h", "-j", str(job_id), "-o", "%L"], timeout=10)
    if p is None or p.returncode != 0 or not p.stdout.strip():
        return None
    return slurm_minutes(p.stdout.strip().splitlines()[0])


def status_key() -> dict:
    """The 'executor' key of tool_status (llmt status reads it)."""
    return {"kind": kind(), "sbatch": shutil.which("sbatch"), "min_minutes": min_minutes(),
            "procs": default_procs(), "timeout_s": timeout_s(), "time_limit_min": time_limit_min(),
            "slurm_time": os.environ.get("FEAT_SLURM_TIME", "max"),
            "mem": os.environ.get("FEAT_SLURM_MEM", "auto"), "caps": caps(), "node": node_status(),
            "node_cores": node_cores()}


# ----------------------------------------------------------------------------- a call's time budget
# Every MCP tool answers within its wait (at most 50 s): the server sets a deadline for the call's thread, and
# SLURM commands, submissions and estimates that need CAD (a worker process) use only the time that is left.

_CALL = threading.local()


@contextmanager
def deadline(seconds: float):
    """Bounds the SLURM commands, submissions and part estimates of this thread to `seconds` from now."""
    prev = getattr(_CALL, "until", None)
    until = time.time() + max(0.0, float(seconds))
    _CALL.until = until if prev is None else min(prev, until)
    try:
        yield
    finally:
        _CALL.until = prev


def remaining() -> float | None:
    """Seconds left in this thread's call; None outside a call (CLI, SLURM jobs, workers: no limit)."""
    until = getattr(_CALL, "until", None)
    return None if until is None else until - time.time()


# ----------------------------------------------------------------------------- clusters of several nodes
# llm-translators docs/EXECUTION.md "Clusters of several nodes": the DGX Sparks spark-5da5 (controller) and
# spark-0808 are one SLURM cluster without shared home folders, so a job must run on the node that submitted it
# (its run folder is only there). FEAT_SLURM_NODE: local (default: --nodelist=<this node> when the cluster has
# several nodes) | any (SLURM chooses) | a node list; a --nodelist / -w in FEAT_SLURM_ARGS wins.

NODES_TTL = 60.0
_NODES: dict = {"t": 0.0, "nodes": None}


def cluster_nodes() -> list[dict] | None:
    """The cluster's nodes from `scontrol -o show node`, one {key: value} per node; None when SLURM does not
    answer. An answer is reused for NODES_TTL s; a failure is never remembered (a controller that was down
    when the server started must not count as no SLURM for the server's lifetime)."""
    now = time.time()
    if _NODES["nodes"] is not None and now - _NODES["t"] < NODES_TTL:
        return _NODES["nodes"]
    p = _run(["scontrol", "-o", "show", "node"], timeout=10)
    if p is None or p.returncode != 0:
        return None
    nodes = [dict(re.findall(r"(\w+)=(\S*)", ln)) for ln in p.stdout.splitlines() if "NodeName=" in ln]
    if not nodes:
        return None
    _NODES.update(t=now, nodes=nodes)
    return nodes


def _addresses() -> set[str]:
    """This machine's addresses: `hostname -I` (Linux; the Sparks' Tailscale addresses among them), and the
    host name's own."""
    out = set()
    p = _run(["hostname", "-I"], timeout=5)
    if p is not None and p.returncode == 0:
        out.update(p.stdout.split())
    try:
        out.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return out


def this_node(nodes: list[dict] | None) -> dict | None:
    """The cluster node that is this machine: NodeName or NodeHostName equal to the host name (or its short
    form), else NodeAddr among this machine's addresses; None if none matches."""
    if not nodes:
        return None
    host = socket.gethostname()
    names = {host, host.split(".")[0]}
    for n in nodes:
        if n.get("NodeName") in names or n.get("NodeHostName") in names \
                or (n.get("NodeHostName") or "").split(".")[0] in names:
            return n
    addrs = _addresses()
    for n in nodes:
        if n.get("NodeAddr") in addrs:
            return n
    return None


def node_setting() -> str:
    return os.environ.get("FEAT_SLURM_NODE", "").strip() or "local"


def node_pin() -> tuple[str | None, str | None, dict]:
    """(--nodelist this translator adds to a job or None, the job's effective nodelist or None when SLURM may
    choose, details: cluster_nodes, this_node, warning)."""
    nodes = cluster_nodes()
    me = this_node(nodes)
    info = {"cluster_nodes": len(nodes) if nodes else None, "this_node": me.get("NodeName") if me else None}
    given = args_nodelist()
    if given:
        return None, given, info
    setting = node_setting()
    if setting.lower() == "any":
        return None, None, info
    if setting.lower() != "local":
        return setting, setting, info
    if not nodes or len(nodes) < 2:
        return None, None, info
    if me is None:
        info["warning"] = ("FEAT_SLURM_NODE=local, but this machine is none of the cluster's nodes (by name or "
                           "address): jobs may run on another node; set FEAT_SLURM_NODE to this node's name")
        return None, None, info
    return me["NodeName"], me["NodeName"], info


def node_status() -> dict:
    """tool_status().executor.node: {setting, nodelist (as jobs ask: null when SLURM may choose; the router's
    sbatch --test-only uses the same), cluster_nodes, this_node; from_args when FEAT_SLURM_ARGS sets the nodelist}."""
    add, nodelist, info = node_pin()
    out = {"setting": node_setting(), "nodelist": nodelist, **info}
    if args_nodelist():
        out["from_args"] = True
    return out


def node_cores() -> int:
    """Cores of this node (CPUs / threads per core, from SLURM: jobs ask for one thread per core), else this
    machine's CPUs. Thread caps use these, never the cluster's total."""
    nodes = cluster_nodes()
    me = this_node(nodes) or (nodes[0] if nodes and len(nodes) == 1 else None)
    if me:
        try:
            return max(1, int(me.get("CPUTot", 0)) // max(1, int(me.get("ThreadsPerCore", 1) or 1)))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


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
    memory); (None, ...) when nothing limits it."""
    r = run_of(study).get("mem")
    if r:
        mb = mem_mb(str(r).strip())
        return (mb / 1024 if mb else None), f"run.mem {str(r).strip()}"
    a = opt_value(slurm_args(), ("--mem",))   # (a --mem-per-cpu grows with the CPUs: not a fixed amount)
    if a:
        mb = mem_mb(a)
        if mb:
            return mb / 1024, f"FEAT_SLURM_ARGS --mem={a}"
    v = os.environ.get("FEAT_SLURM_MEM", "auto").strip()
    if v and v.lower() not in ("auto", "none"):
        mb = mem_mb(v)
        if mb:
            return mb / 1024, f"FEAT_SLURM_MEM={v}"
    c = caps()   # FEAT_MAX_MEM, or the node's memory when lower
    if mem_mb(c.get("mem", "")):
        return mem_mb(c["mem"]) / 1024, f"{c.get('mem_from', 'FEAT_MAX_MEM')} {c['mem']}"
    return None, ""


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
    32 GB, killed after 53 min, while 5 CPUs were held for nothing). The SLURM job then asks for one CPU."""
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
    not ask (also when FEAT_SLURM_ARGS has its own --mem)."""
    if run_of(study).get("mem"):
        return str(run_of(study)["mem"]).strip()
    v = os.environ.get("FEAT_SLURM_MEM", "auto").strip()
    if v.lower() == "none" or args_mem() is not None:
        return None
    if v and v.lower() != "auto":
        return v
    return mem_auto(study)


def mem_auto(study: dict) -> str:
    """The memory CalculiX needs for a study (FEAT_SLURM_MEM=auto): from its estimated size, else MEM_DEFAULT_GB."""
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
    if kind() != "slurm":
        return False
    q = queue_of(study)
    if q is not None:
        return bool(q)
    return estimate_min is None or estimate_min >= min_minutes()


# ----------------------------------------------------------------------------- submission

def job_script(request: Path, name: str, study: dict) -> str:
    d = request / name
    p = procs_for(study)
    pin = node_pin()[0]
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name={PREFIX}{request.name}:{name}",
        f"#SBATCH --chdir={d}",
        f"#SBATCH --output={d / 'job.log'}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={p}",
        f"#SBATCH --time={job_time(study)[0]}",
        *([f"#SBATCH --mem={mem_for(study)}"] if mem_for(study) else []),
        *([f"#SBATCH --nodelist={pin}"] if pin else []),
        "#SBATCH --export=ALL",
        f"exec {shlex.quote(sys.executable)} -m fea_translator.cli run-study {shlex.quote(str(request))} {shlex.quote(name)}",
        ""])


def submit(request: Path, name: str, study: dict) -> dict:
    """Writes job.sh and submits it; returns job.json's content, or {'ok': False, 'why': ...} (never a
    silent fallback to local)."""
    d = request / name
    d.mkdir(parents=True, exist_ok=True)
    script = d / "job.sh"
    script.write_text(job_script(request, name, study))
    script.chmod(0o755)
    cmd = ["sbatch", "--parsable", *sbatch_args(study), str(script)]
    exe = shutil.which("sbatch")
    if not exe:
        return {"ok": False, "why": "could not submit: sbatch not found (FEAT_EXECUTOR=slurm needs SLURM)"}
    left = remaining()
    if left is not None and left < 2:
        return {"ok": False, "why": "could not submit: SLURM was too slow to answer within this call; run the study again"}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60 if left is None else min(60, left - 1))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "why": f"could not submit: {exc}"}
    if p.returncode != 0 or not p.stdout.strip():
        return {"ok": False, "why": f"could not submit: {(p.stderr or p.stdout).strip()[:500]}"}
    job_id = p.stdout.strip().split(";")[0].split()[0]
    n = estimate_nodes(study)
    info = {"executor": "slurm", "job_id": job_id, "submitted": round(time.time(), 1), "procs": procs_for(study),
            "time_min": time_limit_min(study), "slurm_time": job_time(study)[0], "slurm_time_info": job_time(study)[1],
            "mem": mem_for(study), "nodelist": node_pin()[1], "nodes_estimate": n,
            "nodes_estimate_raw": estimate_nodes_raw(study),
            "equation_solver": method_of(study, n) if n else None, "run": run_of(study), "command": cmd}
    try:
        info["memory"] = memory_plan(study)
    except Exception:  # noqa: BLE001 (advice only)
        pass
    (d / "job.json").write_text(json.dumps(info))
    return info


# ----------------------------------------------------------------------------- queue

def _run(cmd: list[str], timeout: float = 15) -> subprocess.CompletedProcess | None:
    """A SLURM (or system) command's result; None when it is missing, fails to run or does not answer in time
    (within the call's remaining time, when in an MCP call)."""
    if not shutil.which(cmd[0]):
        return None
    left = remaining()
    if left is not None:
        if left < 1:
            return None
        timeout = min(timeout, left - 0.5)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _minutes(t: str) -> float | None:
    """squeue's elapsed time ([D-]HH:MM:SS, MM:SS) in minutes."""
    try:
        days = 0
        if "-" in t:
            d, t = t.split("-", 1)
            days = int(d)
        parts = [int(x) for x in t.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        h, m, s = parts
        return round(days * 1440 + h * 60 + m + s / 60, 1)
    except ValueError:
        return None


def squeue(ids: list[str]) -> dict | None:
    """{id: {state, reason, start, elapsed_min}} of the given jobs still in the queue (one call); None when
    SLURM does not answer (then nothing can be said about the jobs: they are not 'gone')."""
    if not ids:
        return {}
    p = _run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T|%r|%S|%M"])
    out = {}
    if p is None:
        return None
    if p.returncode != 0:
        # squeue errors on ids that have all left the queue ("Invalid job id"): none listed; anything else
        # (the controller is down) is no answer
        return out if "invalid job id" in (p.stderr + p.stdout).lower() else None
    for line in p.stdout.splitlines():
        f = line.strip().split("|")
        if len(f) >= 5:
            out[f[0]] = {"state": f[1], "reason": f[2], "start": f[3], "elapsed_min": _minutes(f[4])}
    return out


def pending_ids() -> list[int] | None:
    """Ids of every pending job in the queue (one squeue call); None when SLURM does not answer."""
    p = _run(["squeue", "-h", "-t", "PENDING", "-o", "%i"])
    if p is None or p.returncode != 0:
        return None
    return [int(x.split("_")[0]) for x in p.stdout.split() if x.split("_")[0].isdigit()]


def ahead_of(job_id: str, pending: list[int] | None = None) -> int | None:
    """Pending jobs submitted before this one (all of the queue's, not only the translators')."""
    if pending is None:
        pending = pending_ids()
    if pending is None:
        return None
    try:
        me = int(job_id.split("_")[0])
    except ValueError:
        return None
    return sum(1 for x in pending if x < me)


def sacct(ids: list[str]) -> dict:
    """{id: STATE} from SLURM's accounting; {} when there is none (common on single-node installs)."""
    if not ids:
        return {}
    p = _run(["sacct", "-n", "-P", "-X", "-j", ",".join(ids), "-o", "JobID,State"])
    if p is None or p.returncode != 0:
        return {}
    out = {}
    for line in p.stdout.splitlines():
        f = line.strip().split("|")
        if len(f) >= 2 and f[0]:
            out[f[0].split(".")[0]] = f[1].strip()   # e.g. "CANCELLED by 1000": keep who
    return out


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


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
    """(state, why) of a job that left the queue without writing a result."""
    a = (acct or "").upper()
    tail = _tail(log)
    if a.startswith("TIMEOUT") or (not a and "DUE TO TIME LIMIT" in tail):
        return "timed_out", ("SLURM's time limit stopped it before feat's own timeout: raise FEAT_TIMEOUT "
                             "(it also sets the job's limit) or coarsen the mesh / drop converge")
    if a.startswith("OUT_OF_MEMORY") or re.search(r"oom[-_ ]kill|out of memory", tail, re.I):
        return "out_of_memory", oom_advice(Path(log).parent)
    if a.startswith("CANCELLED") or (not a and "CANCELLED" in tail):
        who = acct.split("by", 1)[1].strip() if acct and "by" in acct else None
        return "cancelled", "cancelled" + (f" by {who}" if who else "") + " (scancel or cancel())"
    return "failed", "the job ended without a result" + (f"; job.log:\n{tail}" if tail else " (no job.log)")


def job_states(request: Path, names: list[str]) -> dict:
    """{name: {state, job_id, ...}} for the request's SLURM studies, from the files and the queue."""
    jobs = {}
    for n in names:
        try:
            jobs[n] = json.loads((request / n / "job.json").read_text())
        except (OSError, json.JSONDecodeError):
            pass
    ids = [j["job_id"] for j in jobs.values()]
    q = squeue([i for n, i in ((n, j["job_id"]) for n, j in jobs.items()) if not (request / n / f"{n}.json").is_file()])
    if q is None:   # SLURM did not answer: the jobs are as they were, not gone
        return {n: ({"state": "done", "job_id": j["job_id"]} if (request / n / f"{n}.json").is_file() else
                    {"state": "unknown", "job_id": j["job_id"], "why": "SLURM did not answer (squeue); ask again later"})
                for n, j in jobs.items()}
    gone = [j["job_id"] for n, j in jobs.items() if j["job_id"] not in q and not (request / n / f"{n}.json").is_file()]
    acct = sacct(gone) if gone else {}
    pend = pending_ids() if any(v["state"] in ("PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED") for v in q.values()) else []
    out = {}
    for n, j in jobs.items():
        jid = j["job_id"]
        if (request / n / f"{n}.json").is_file():
            out[n] = {"state": "done", "job_id": jid}
        elif jid in q:
            st = q[jid]
            if st["state"] in ("PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED"):
                rec = {"state": "queued", "job_id": jid, "reason": st["reason"]}
                if st["start"] and st["start"] not in ("N/A", "Unknown", "None"):
                    rec["expected_start"] = st["start"]
                rec["ahead"] = ahead_of(jid, pend)
            else:
                rec = {"state": "running", "job_id": jid, "job_elapsed_min": st["elapsed_min"]}
            out[n] = rec
        elif (request / n / f"{n}.json").is_file():   # finished between the two looks
            out[n] = {"state": "done", "job_id": jid}
        else:
            state, why = ended_state(acct.get(jid), request / n / "job.log")
            if state == "failed" and (request / n / "cancel").exists():
                state, why = "cancelled", "cancelled with cancel(request) (it left the queue before running)"
            out[n] = {"state": state, "job_id": jid, "why": why}
    return out


def queue_summary() -> dict:
    """This translator's jobs in the queue, and the machine's CPUs."""
    out = {"queue": {"running": 0, "queued": 0, "cpus_in_use": 0}, "machine": {}}
    p = _run(["squeue", "-h", "-o", "%i|%j|%T|%C"])
    if p is not None and p.returncode == 0:
        for line in p.stdout.splitlines():
            f = line.strip().split("|")
            if len(f) >= 4 and f[1].startswith(PREFIX):
                if f[2] in ("RUNNING", "COMPLETING"):
                    out["queue"]["running"] += 1
                    out["queue"]["cpus_in_use"] += int(f[3]) if f[3].isdigit() else 0
                else:
                    out["queue"]["queued"] += 1
    p = _run(["sinfo", "-h", "-o", "%C"])
    if p is not None and p.returncode == 0 and p.stdout.strip():
        try:
            alloc, idle, other, total = (int(x) for x in p.stdout.split()[0].split("/"))
            out["machine"] = {"cpus": total, "cpus_allocated": alloc}
        except ValueError:
            pass
    return out


def extend_job(job_id: str, minutes: int) -> str | None:
    """Raises a SLURM job's time limit by `minutes` (scontrol update TimeLimit=+N); None, or why not (raising a
    limit usually needs an operator or admin)."""
    p = _run(["scontrol", "update", f"JobId={job_id}", f"TimeLimit=+{int(minutes)}"])
    if p is None:
        return "scontrol not found"
    return None if p.returncode == 0 else ((p.stderr or p.stdout).strip()[:300] or f"exit {p.returncode}")


def scancel(ids: list[str]) -> str | None:
    """Cancels the jobs; returns an error message or None."""
    if not ids:
        return None
    p = _run(["scancel", *ids])
    if p is None:
        return "scancel not found"
    return None if p.returncode == 0 else (p.stderr or p.stdout).strip()[:300]
