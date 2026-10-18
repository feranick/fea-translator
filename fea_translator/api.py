"""api.py — what other programs and the MCP server call. JSON in, JSON out.
Version: 2026.10.18.5

    study_format()                   the study format, for the caller to read
    validate(studies)                static checks, no meshing
    run(studies, solver, wait)       meshes, solves and checks each study; files in a request folder;
                                     with wait, returns after that many seconds with the rest pending
    start(studies, solver)           the same in the background: returns the request folder
    status(request, wait)            finished studies with results, the others pending with their
                                     progress and time left (expected and at most), and when to look again
    deck(study, name)                the CalculiX input deck of one study, as text (meshes the part)
    bench(names, solver)             the benchmark studies with closed-form answers
    read_file(path)                  a written file as base64 (only under the output folder)
    jobs(limit, active)              recent requests with SLURM jobs and their states; the queue
    cancel(request, names)           stops a request's studies (scancel)
    tool_status()                    versions, solver and mesher found, the executor

Studies run in worker processes with a time limit each, or (FEAT_EXECUTOR=slurm, executor.py) the
long ones as SLURM jobs. Environment: FEAT_OUT (default ~/feat_runs), FEAT_WORKERS (2), FEAT_TIMEOUT
(seconds per CalculiX solve, 900), FEAT_CCX (path of ccx), FEAT_SOLVER (auto | ccx | ref; auto =
CalculiX when found, else the reference solver), FEAT_PROCS (CalculiX threads, 1), FEAT_EXECUTOR,
FEAT_SLURM_MIN_MINUTES (5), FEAT_SLURM_ARGS, FEAT_SLURM_MEM (auto).
"""
from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path

from . import __version__, executor
from .study import study_format as _format
from .study import validate as _validate
from .study import validate_study


def out_root() -> Path:
    return Path(os.environ.get("FEAT_OUT", "~/feat_runs")).expanduser()


def _solver(solver: str | None) -> str:
    s = (solver or os.environ.get("FEAT_SOLVER", "auto")).lower()
    if s == "auto":
        from .runner import find_ccx
        return "ccx" if find_ccx() else "ref"
    if s not in ("ccx", "ref"):
        raise ValueError("solver: auto, ccx or ref")
    return s


def tool_status() -> dict:
    from .runner import find_ccx

    out = {"translator": __version__, "ccx": find_ccx(), "default_solver": _solver(None)}
    try:
        import gmsh
        out["gmsh"] = getattr(gmsh, "__version__", "yes")
    except ImportError:
        out["gmsh"] = None
    try:
        import build123d_translator
        out["build123d_translator"] = build123d_translator.__version__
    except ImportError:
        out["build123d_translator"] = None
    out["executor"] = executor.status_key()
    return out


def study_format() -> dict:
    return {**_format(), "translator": __version__, "status": tool_status()}


def validate(studies: dict) -> dict:
    return _validate(studies)


def _run_one(job: dict) -> dict:
    from .runner import run_study

    if job.get("out"):  # the worker may outlive a change of FEAT_OUT: timing records go with the request
        os.environ["FEAT_OUT"] = job["out"]
    # CalculiX threads: the study's procs or FEAT_PROCS when given, else up to 4 (set every time: the
    # worker process runs one study after another)
    given = job["study"].get("procs") or os.environ.get("FEAT_PROCS")
    os.environ["OMP_NUM_THREADS"] = str(int(given) if given else min(4, os.cpu_count() or 1))
    os.environ["CCX_NPROC_EQUATION_SOLVER"] = os.environ["OMP_NUM_THREADS"]
    t0 = time.perf_counter()
    r = run_study(job["name"], job["study"], Path(job["folder"]) / job["name"], job["solver"], job.get("timeout", 900))
    r.setdefault("seconds", {})
    if isinstance(r["seconds"], dict):
        r["seconds"]["wall"] = round(time.perf_counter() - t0, 3)
    return r


class _Pool:
    """Worker processes that live across calls; jobs run in the background and write
    <request>/<name>/<name>.json when done, so their status can be read at any time."""

    def __init__(self):
        self.pool = None
        self.jobs: dict[str, dict] = {}   # request folder -> {name: AsyncResult}

    def submit(self, folder: Path, jobs: list[dict]) -> None:
        workers = int(os.environ.get("FEAT_WORKERS", "2"))
        timeout = float(os.environ.get("FEAT_TIMEOUT", "900"))
        if self.pool is None:
            self.pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=20)
        self.jobs[str(folder)] = {j["name"]: self.pool.apply_async(_run_one, (dict(j, timeout=timeout),)) for j in jobs}


_POOL = _Pool()


def _request_dir(prefix: str) -> Path:
    d = out_root() / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _collect(folder: Path) -> dict:
    """Status of every study of a request: finished ones with their results."""
    meta = json.loads((folder / "request.json").read_text())
    handles = _POOL.jobs.get(str(folder), {})
    out, pending = dict(meta.get("invalid", {})), []
    slurm = set(meta.get("slurm", []))
    todo = [n for n in slurm if not (folder / n / f"{n}.json").is_file()]
    states = executor.job_states(folder, todo) if todo else {}
    for name in meta["names"]:
        f = folder / name / f"{name}.json"
        h = handles.get(name)
        if f.is_file() and (h is None or h.ready()):
            try:
                out[name] = json.loads(f.read_text())
                continue
            except json.JSONDecodeError:
                pass
        if name in slurm:
            st = states.get(name, {"state": "failed", "why": "no job.json"})
            if st["state"] in executor.FINAL:   # a finished state is a result
                out[name] = {"ok": False, **st}
            elif st["state"] == "done":         # written between the two looks
                try:
                    out[name] = json.loads(f.read_text())
                except (OSError, json.JSONDecodeError):
                    pending.append(name)
            else:
                pending.append(name)
            continue
        if h is not None and h.ready():
            try:
                out[name] = h.get()
            except Exception as exc:  # noqa: BLE001
                out[name] = {"ok": False, "why": "worker failed", "error": str(exc)[:500]}
        elif h is None:
            out[name] = {"ok": False, "why": "not finished and not running here (the server restarted?); run it again"}
        else:
            pending.append(name)
    started = meta.get("started", time.time())
    res = {"request": str(folder), "translator": __version__, "solver": meta["solver"], "done": not pending,
           "pending": pending, "elapsed_s": round(time.time() - started, 1), "studies": out}
    if pending:
        studies = json.loads((folder / "studies.json").read_text()) if (folder / "studies.json").is_file() else {}
        timeout = float(meta.get("timeout", os.environ.get("FEAT_TIMEOUT", "900")))
        res["progress"] = {}
        for n in pending:
            pr = _progress(folder / n, studies.get(n, {}), meta["solver"], timeout)
            if n in slurm:
                st = states.get(n, {})
                pr.update({k: v for k, v in st.items() if k != "state"})
                pr["state"] = st.get("state", "queued")
                if pr["state"] == "queued":   # not started: the run's own estimate; the queue wait comes first
                    pr["stage"] = "queued"
                elif pr["stage"] == "queued":  # started by SLURM, no progress written yet
                    pr["stage"] = "starting"
            else:
                pr["state"] = "queued" if pr["stage"] == "queued" else "running"
            res["progress"][n] = pr
        sl = {n: res["progress"][n].get("job_id") for n in pending if n in slurm}
        if sl:
            res["time_left"], res["next"] = _timing_slurm(folder, res["progress"], sl)
        else:
            res["time_left"], res["next"] = _timing(folder, res["progress"])
    else:
        (folder / "results.json").write_text(json.dumps(res, indent=1))
    return res


# ----------------------------------------------------------------------------- progress and time left

STAGES = ("mesh", "solve", "mesh_fine", "solve_fine")
FINE_NODES = 1 / 0.6 ** 3   # the finer mesh of converge: true (size x 0.6) has about 4.6x the nodes
SOLVE_EXP = 1.5              # solve time ~ nodes^1.5 (sparse direct solver on 3D meshes)
POLL_LIMIT_MIN = 5.0         # longer than this: tell the user when to come back instead of polling


def _rates(analysis: str, solver: str) -> dict | None:
    """From this machine's finished solves (timing.jsonl): solve s per nodes^1.5, mesh s per node,
    and typical (median) mesh and solve times; None without records."""
    from statistics import median
    from .runner import stats_path

    p = stats_path()
    if not p.is_file():
        return None
    recs = []
    for line in p.read_text().splitlines()[-500:]:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("analysis") == analysis and r.get("solver") == solver and r.get("nodes", 0) > 0:
            recs.append(r)
    if not recs:
        return None
    return {"solve_k": median(r["solve_s"] / r["nodes"] ** SOLVE_EXP for r in recs),
            "mesh_k": median(r["mesh_s"] / r["nodes"] for r in recs),
            "mesh_s": median(r["mesh_s"] for r in recs), "solve_s": median(r["solve_s"] for r in recs),
            "runs": len(recs)}


def _stage_s(stage: str, nodes: int | None, rt: dict | None) -> float | None:
    """Expected seconds of one stage; `nodes` is the study's (first) mesh, the finer one is scaled."""
    if rt is None:
        return None
    fine = stage.endswith("_fine")
    if nodes:
        n = nodes * (FINE_NODES if fine else 1)
        return rt["mesh_k"] * n if stage.startswith("mesh") else rt["solve_k"] * n ** SOLVE_EXP
    if stage.startswith("mesh"):
        return rt["mesh_s"] * (FINE_NODES if fine else 1)
    return rt["solve_s"] * (FINE_NODES ** SOLVE_EXP if fine else 1)


def _nodes_hint(study: dict) -> int | None:
    return executor.nodes_hint(study)


def _progress(d: Path, study: dict, solver: str, timeout: float) -> dict:
    """Where a running study is: step k of n (mesh, solve; with converge: true also the finer mesh),
    seconds per step, and minutes left: expected (from this machine's past solves) and at most (the
    time limit of each remaining solve). The expected time never exceeds the limit."""
    steps = 4 if study.get("converge") else 2
    per_stage = max(1, executor.solves_of(study) // (2 if study.get("converge") else 1))  # joule: 2-3 solves
    timeout = timeout * per_stage
    rt = _rates(study.get("analysis", ""), solver)
    pf = d / "progress.json"
    try:
        p = json.loads(pf.read_text())
    except (OSError, json.JSONDecodeError):
        p = None
    now = time.time()
    if p is None:  # not started: estimate from the mesh size when it is known before meshing
        nodes = _nodes_hint(study)
        worst = steps / 2 * timeout + (sum(_stage_s(s, nodes, rt) or 0 for s in STAGES[:steps] if s.startswith("mesh"))
                                      if rt and nodes else 0)
        exp = min(sum(_stage_s(s, nodes, rt) or 0 for s in STAGES[:steps]), worst) if rt and nodes else None
        out = {"stage": "queued", "iteration": 0, "max_iterations": steps,
               "eta_min": None if exp is None else round(exp / 60, 1), "eta_min_max": round(worst / 60, 1)}
        if exp is None:
            out["note"] = ("expected run time unknown until the study is meshed" if rt else
                           "no finished solves on this machine yet: expected run time unknown")
        return out
    step, stage = int(p["step"]), p["stage"]
    el_stage = now - p.get("stage_started", now)
    done = p.get("done_steps_s", [])
    nodes = p.get("nodes")
    exp = None
    if rt:
        cur = _stage_s(stage, nodes, rt)
        exp = max((cur or 0) - el_stage, 0.1 * el_stage) + sum(_stage_s(s, nodes, rt) or 0 for s in STAGES[step:steps])
    solves_left = [s for s in STAGES[step - 1:steps] if s.startswith("solve")]
    worst = sum(timeout for _ in solves_left) - (el_stage if stage.startswith("solve") else 0)
    # meshing has no time limit of its own: count its expected time (the rest of the current one too)
    for s in STAGES[step - 1:steps]:
        if s.startswith("mesh") and rt:
            worst += max((_stage_s(s, nodes, rt) or 0) - (el_stage if s == stage else 0), 0)
    worst = max(worst, 0)
    if exp is not None:
        exp = min(exp, worst)   # the time limits bound it: a slower run stops at its timeout
    out = {"stage": stage, "iteration": step, "max_iterations": steps,
           "s_per_iteration": round(sum(done) / len(done), 1) if done else None,
           "elapsed_min": round((now - p.get("started", now)) / 60, 1), "nodes": nodes,
           "eta_min": None if exp is None else round(exp / 60, 1), "eta_min_max": round(worst / 60, 1)}
    if rt is None:
        out["note"] = "no finished solves on this machine yet: eta_min is unknown until one finishes"
    return out


def _clock(t: float) -> str:
    return time.strftime("%H:%M", time.localtime(t)) + (" (tomorrow)" if time.localtime(t).tm_yday != time.localtime().tm_yday else "")


def _slurm_time(s: str | None) -> float | None:
    """SLURM's expected start (local ISO time, e.g. 2026-10-18T22:22:27) as epoch seconds."""
    if not s:
        return None
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def _timing_slurm(folder: Path, progress: dict, ids: dict) -> tuple[dict, str]:
    """Request-level time left when studies are SLURM jobs: a running study ends after its own time left;
    a queued one starts at SLURM's expected start (planned from the time limits of the jobs ahead, so it
    is a latest start) and then needs its run time."""
    now = time.time()
    exp_end, max_end = [], []
    for p in progress.values():
        start = now if p.get("state") != "queued" else _slurm_time(p.get("expected_start"))
        exp_end.append(None if start is None or p.get("eta_min") is None else start + 60 * p["eta_min"])
        max_end.append(None if start is None else start + 60 * p["eta_min_max"])
    t_exp = max(exp_end) if exp_end and all(e is not None for e in exp_end) else None
    t_max = max(max_end) if max_end and all(e is not None for e in max_end) else None
    t = {"eta_min": None if t_exp is None else round((t_exp - now) / 60, 1),
         "eta_min_max": None if t_max is None else round((t_max - now) / 60, 1),
         "expected_finish": None if t_exp is None else _clock(t_exp),
         "latest_finish": None if t_max is None else _clock(t_max), "includes_queue_wait": True}
    if t_exp is not None and t_max is not None:
        when = f"around {t['expected_finish']} (at the latest {t['latest_finish']})"
    elif t_max is not None:
        when = f"by {t['latest_finish']} at the latest (no estimate of the run time yet)"
    else:
        when = "when the jobs ahead of them in the queue are done (SLURM gives no start time yet)"
    nxt = (f"The studies are SLURM jobs on this machine (ids {', '.join(str(v) for v in ids.values())}). They keep "
           f"running without this conversation. Tell the user they should finish {when} and stop; collect them "
           f"later with study_status(request='{folder}') or find them with jobs().")
    return t, nxt


def _timing(folder: Path, progress: dict) -> tuple[dict, str]:
    """Request-level time left and the advice for the caller."""
    workers = max(1, int(os.environ.get("FEAT_WORKERS", "2")))
    queued = [p for p in progress.values() if p["stage"] == "queued"]
    running = [p for p in progress.values() if p["stage"] != "queued"]
    exp = [p["eta_min"] for p in running if p.get("eta_min") is not None]
    exp_q = [p["eta_min"] for p in queued if p.get("eta_min") is not None]
    rounds = -(-len(queued) // workers)  # queued studies start as workers free up
    known = len(exp) == len(running) and len(exp_q) == len(queued)
    eta = max(exp, default=0) + rounds * max(exp_q, default=0) if known else None
    eta_max = max((p["eta_min_max"] for p in running), default=0) + rounds * max((p["eta_min_max"] for p in queued), default=0)
    now = time.time()
    t = {"eta_min": None if eta is None else round(eta, 1), "eta_min_max": round(eta_max, 1),
         "expected_finish": None if eta is None else _clock(now + 60 * eta), "latest_finish": _clock(now + 60 * eta_max)}
    basis = eta if eta is not None else eta_max
    if basis > POLL_LIMIT_MIN:
        when = t["expected_finish"] or f"at the latest {t['latest_finish']}"
        nxt = (f"this run needs about {basis:.0f} more min: tell the user it should finish around {when} "
               f"(at the latest {t['latest_finish']}) and stop; do not poll. Collect the results later with "
               f"study_status(request='{folder}') when the user asks, or from a task scheduled after that time.")
    else:
        nxt = f"call study_status(request='{folder}', wait_s={min(50, int(60 * basis) + 10)}) to collect the rest"
    return t, nxt


def start(studies: dict, solver: str | None = None) -> Path:
    """Validates and starts the studies in the background; returns the request folder."""
    folder = _request_dir("run")
    (folder / "studies.json").write_text(json.dumps(studies, indent=1))
    s = _solver(solver)
    invalid, jobs = {}, []
    for n, st in studies.items():
        v = validate_study(st)
        if not v["ok"]:
            invalid[n] = {"ok": False, "why": "invalid study", "errors": v["errors"]}
        else:
            jobs.append({"name": n, "study": st, "folder": str(folder), "solver": s, "out": str(out_root())})
    slurm = [j["name"] for j in jobs if executor.use_slurm(j["study"], estimate_min(j["study"], s))]
    (folder / "request.json").write_text(json.dumps({"names": [j["name"] for j in jobs], "invalid": invalid,
                                                     "solver": s, "started": time.time(), "slurm": slurm,
                                                     "executor": executor.kind(),
                                                     "timeout": float(os.environ.get("FEAT_TIMEOUT", "900"))}))
    for n in slurm:   # one SLURM job per study; a failed submission is that study's result (no local fallback)
        info = executor.submit(folder, n, studies[n])
        if info.get("ok") is False:
            (folder / n).mkdir(parents=True, exist_ok=True)
            (folder / n / f"{n}.json").write_text(json.dumps({**info, "state": "failed"}))
    local = [j for j in jobs if j["name"] not in slurm]
    if local:
        _POOL.submit(folder, local)
    return folder


def estimate_min(study: dict, solver: str) -> float | None:
    """Expected minutes of a study from this machine's finished solves of its analysis (timing.jsonl);
    None without records (such a study goes to SLURM when FEAT_EXECUTOR=slurm)."""
    rt = _rates(study.get("analysis", ""), solver)
    if rt is None:
        return None
    s = rt["mesh_s"] + rt["solve_s"]
    if study.get("converge"):
        s += rt["mesh_s"] * FINE_NODES + rt["solve_s"] * FINE_NODES ** SOLVE_EXP
    return s / 60


def status(request: str, wait: float = 0.0) -> dict:
    """The studies of a request: finished ones with results, the others pending. Waits up to
    `wait` seconds for the pending ones to finish."""
    folder = Path(request).expanduser()
    if not (folder / "request.json").is_file():
        return {"error": f"no request {folder}"}
    t_end = time.time() + max(0.0, float(wait))
    while True:
        res = _collect(folder)
        if res["done"] or time.time() >= t_end:
            return res
        slurm = any(p.get("job_id") for p in res.get("progress", {}).values())
        time.sleep(min(5.0 if slurm else 1.0, max(0.1, t_end - time.time())))   # one squeue per look


def run(studies: dict, solver: str | None = None, wait: float | None = None) -> dict:
    """Runs the studies into a new request folder under FEAT_OUT: per study <name>/<name>.inp,
    .dat, .frd (CalculiX), mesh.msh, part.step, <name>.json; and studies.json, results.json.
    Waits for all of them (wait None) or at most `wait` seconds; then the unfinished ones are
    listed as pending and status(request) collects them later."""
    folder = start(studies, solver)
    return status(str(folder), 1e9 if wait is None else wait)


def jobs(limit: int = 20, active: bool = False) -> dict:
    """The newest requests under FEAT_OUT with SLURM jobs, their studies' states, job ids and times;
    the translator's queue (running, queued, CPUs) and the machine's CPUs. active: only requests with
    queued or running studies."""
    root = out_root()
    reqs = sorted((d for d in root.glob("run_*") if (d / "request.json").is_file()),
                  key=lambda d: d.stat().st_mtime, reverse=True) if root.is_dir() else []
    out = []
    for d in reqs:
        if len(out) >= max(1, int(limit)):
            break
        try:
            meta = json.loads((d / "request.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        names = meta.get("slurm", [])
        if not names or not any((d / n / "job.json").is_file() for n in names):
            continue
        st = executor.job_states(d, names)
        if active and not any(v["state"] in ("queued", "running") for v in st.values()):
            continue
        studies = {}
        for n, v in st.items():
            rec = dict(v)
            if v["state"] == "done":
                try:
                    r = json.loads((d / n / f"{n}.json").read_text())
                    rec["ok"] = r.get("ok")
                    if r.get("ok") is False:
                        rec["why"] = r.get("why")
                except (OSError, json.JSONDecodeError):
                    pass
            elif v["state"] == "running":
                pr = _progress(d / n, {}, meta.get("solver", "ccx"), float(meta.get("timeout", 900)))
                rec.update({k: pr[k] for k in ("stage", "eta_min", "eta_min_max") if k in pr})
            studies[n] = rec
        out.append({"request": str(d), "started": _clock(meta.get("started", d.stat().st_mtime)), "studies": studies})
    return {"executor": executor.kind(), "requests": out, **executor.queue_summary()}


def cancel(request: str, names: list[str] | None = None) -> dict:
    """Stops a request's studies (or the named ones): scancel for SLURM jobs; a local study only if
    this server process runs it. Returns their states."""
    folder = Path(request).expanduser()
    try:
        meta = json.loads((folder / "request.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {"error": f"no request {folder}"}
    want = list(names) if names else list(meta.get("names", []))
    slurm = [n for n in want if n in meta.get("slurm", [])]
    ids = []
    for n in slurm:
        try:
            if not (folder / n / f"{n}.json").is_file():
                ids.append(json.loads((folder / n / "job.json").read_text())["job_id"])
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    err = executor.scancel(ids)
    out = {"cancelled_jobs": ids}
    if err:
        out["error"] = err
    local = [n for n in want if n not in slurm]
    if local:
        handles = _POOL.jobs.get(str(folder), {})
        running = [n for n in local if n in handles and not handles[n].ready()]
        others = [h for f, hs in _POOL.jobs.items() for n, h in hs.items()
                  if not h.ready() and not (f == str(folder) and n in running)]
        if running and not others and _POOL.pool is not None:   # the pool runs only these: stop it
            _POOL.pool.terminate()
            _POOL.pool = None
            for n in running:
                (folder / n).mkdir(parents=True, exist_ok=True)
                (folder / n / f"{n}.json").write_text(json.dumps({"ok": False, "state": "cancelled",
                                                                  "why": "cancelled (local worker pool stopped)"}))
            out["cancelled_local"] = running
        elif running:
            out["note"] = ("local studies run in this server's worker pool together with other studies; they cannot "
                           "be stopped alone and end at their time limit: " + ", ".join(running))
        elif any(n not in handles for n in local if not (folder / n / f"{n}.json").is_file()):
            out["note"] = "local studies of this request are not run by this server process; they cannot be cancelled here"
    t_end = time.time() + 10   # give SLURM a moment to end the jobs
    while ids and time.time() < t_end and any(executor.squeue(ids)):
        time.sleep(1)
    st = _collect(folder)
    out["states"] = {n: (st["progress"][n]["state"] if n in st.get("pending", []) else
                         st["studies"].get(n, {}).get("state", "done" if st["studies"].get(n, {}).get("ok") else "failed"))
                     for n in want}
    return out


def deck(study: dict, name: str = "study") -> dict:
    from . import deck as D
    from .mesher import mesh_part

    v = validate_study(study)
    if not v["ok"]:
        return {"error": "invalid study", "errors": v["errors"]}
    folder = _request_dir("deck")
    mesh, info = mesh_part(study, folder)
    text, plan = D.write(name, study, mesh)
    p = folder / f"{name}.inp"
    p.write_text(text)
    head = text if len(text) < 20000 else text[:20000] + "\n** ... (truncated; the whole deck is in the file)\n"
    return {"path": str(p), "mesh": info, "bytes": len(text), "deck": head}


def bench(names=None, solver: str | None = None) -> dict:
    from . import bench as B

    s = _solver(solver)
    folder = _request_dir("bench")
    r = B.run(s, names, folder)
    passed = [k for k, v in r.items() if v.get("passed")]
    failed = [k for k, v in r.items() if v.get("passed") is False or v.get("ok") is False]
    out = {"request": str(folder), "solver": s, "passed": passed, "failed": failed,
           "skipped": [k for k, v in r.items() if "skipped" in v], "cases": r}
    (folder / "bench.json").write_text(json.dumps(out, indent=1))
    return out


def read_file(path: str, max_mb: float = 20) -> dict:
    p = Path(path).expanduser().resolve()
    root = out_root().resolve()
    if root not in p.parents:
        return {"error": f"only files under {root} can be read"}
    if not p.is_file():
        return {"error": f"no file {p}"}
    if p.stat().st_size > max_mb * 1e6:
        return {"error": f"{p.stat().st_size / 1e6:.1f} MB is more than max_mb"}
    return {"name": p.name, "bytes": p.stat().st_size, "base64": base64.b64encode(p.read_bytes()).decode()}
