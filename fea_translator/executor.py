"""executor.py — where studies run: the local worker pool, or SLURM jobs (llm-translators docs/EXECUTION.md).
Version: 2026.10.25.1

Settings (environment): FEAT_EXECUTOR local | slurm (local); FEAT_SLURM_MIN_MINUTES (5): studies whose
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
        return float(os.environ.get("FEAT_SLURM_MIN_MINUTES", "5"))
    except ValueError:
        return 5.0


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
    """Ceilings on what a study may ask (config/env): FEAT_MAX_TIMEOUT (s), FEAT_MAX_MEM, FEAT_MAX_PROCS."""
    out = {}
    for key, var in (("timeout_s", "FEAT_MAX_TIMEOUT"), ("mem", "FEAT_MAX_MEM"), ("procs", "FEAT_MAX_PROCS")):
        v = os.environ.get(var, "").strip()
        if v:
            out[key] = v if key == "mem" else (float(v) if key == "timeout_s" else int(float(v)))
    return out


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
    """CalculiX runs of a study (each has its timeout_s): joule 2, +1 with stress, x2 with converge."""
    n = {"joule": 2}.get(study.get("analysis"), 1) + (1 if study.get("stress") else 0)
    return n * (2 if study.get("converge") else 1)


def time_limit_min(study: dict | None = None) -> int:
    """The job's time limit: every solve's timeout plus 10 minutes, so the translator's own timeout
    (which writes a clean result) fires before SLURM's."""
    return int(math.ceil(solves_of(study or {}) * timeout_s(study) / 60)) + 10


def status_key() -> dict:
    """The 'executor' key of tool_status (llmt status reads it)."""
    return {"kind": kind(), "sbatch": shutil.which("sbatch"), "min_minutes": min_minutes(),
            "procs": default_procs(), "timeout_s": timeout_s(), "time_limit_min": time_limit_min(),
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


def nodes_hint(study: dict) -> int | None:
    """Nodes of a study's (first) mesh when they are known before meshing: box parts (their divisions,
    or the element size); None for STEP files and goals."""
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

        _VPOOL["ex"] = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        atexit.register(_close_volume_pool)
    return _VPOOL["ex"]


def _close_volume_pool() -> None:
    ex, _VPOOL["ex"] = _VPOOL["ex"], None
    if ex is not None:
        ex.shutdown(wait=False, cancel_futures=True)


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


def estimate_nodes(study: dict) -> int | None:
    """Nodes of a study's (first) mesh before meshing: exact for box parts, else an upper estimate from the
    part's volume and the element size (plus the refinement spheres); None if it cannot be told."""
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


def procs_for(study: dict) -> int:
    """CalculiX threads (CPUs) for a study: 1 for modal and buckling (repeatable eigenvalues); 1 when every solve
    uses the iterative solver (single-threaded in CalculiX); else the study's procs or FEAT_PROCS (1),
    capped at a thread per NODES_PER_THREAD nodes of its largest direct-solver mesh."""
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
        f"#SBATCH --time={time_limit_min(study)}",
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
            "time_min": time_limit_min(study), "mem": mem_for(study), "nodelist": node_pin()[1], "nodes_estimate": n,
            "equation_solver": method_of(study, n) if n else None, "run": run_of(study), "command": cmd}
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


def ended_state(acct: str | None, log: Path) -> tuple[str, str]:
    """(state, why) of a job that left the queue without writing a result."""
    a = (acct or "").upper()
    tail = _tail(log)
    if a.startswith("TIMEOUT") or (not a and "DUE TO TIME LIMIT" in tail):
        return "timed_out", ("SLURM's time limit stopped it before feat's own timeout: raise FEAT_TIMEOUT "
                             "(it also sets the job's limit) or coarsen the mesh / drop converge")
    if a.startswith("OUT_OF_MEMORY") or (not a and ("oom-kill" in tail.lower() or "out of memory" in tail.lower())):
        return "out_of_memory", "add --mem=... to FEAT_SLURM_ARGS, or coarsen the mesh"
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


def scancel(ids: list[str]) -> str | None:
    """Cancels the jobs; returns an error message or None."""
    if not ids:
        return None
    p = _run(["scancel", *ids])
    if p is None:
        return "scancel not found"
    return None if p.returncode == 0 else (p.stderr or p.stdout).strip()[:300]
