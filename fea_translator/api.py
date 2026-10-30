"""api.py — what other programs and the MCP server call. JSON in, JSON out.
Version: 2026.10.30.1

    study_format()                   the study format, for the caller to read
    validate(studies)                static checks, no meshing
    run(studies, solver, wait)       meshes, solves and checks each study; files in a request folder;
                                     with wait, returns after that many seconds with the rest pending
    start(studies, solver)           the same in the background: returns the request folder
    status(request, wait)            finished studies with results, the others pending with their
                                     progress and time left (expected and at most), and when to look again
    deck(study, name)                the CalculiX input deck of one study, as text (meshes the part)
    bench(names, solver, wait)       the benchmark studies with closed-form answers, as one request
    bench_status(bench, wait)        passed, failed, pending benchmark cases
    read_file(path)                  a written file as base64 (only under the output folder)
    jobs(limit, active)              recent requests with SLURM jobs and their states; the queue
    cancel(request, names)           stops a request's studies (scancel)
    tool_status()                    versions, solver and mesher found, the executor, the renderers
    render_images(study_folder, images)  new pictures of a finished study from its saved state (no solving)
    in_worker(what, fn_name, kwargs, wait)  slow work of a tool call in a worker process, within wait
    image_path(path)                 a PNG written by a run or render_images (only under the output folder)

Studies run in worker processes with a time limit each, or (FEAT_EXECUTOR=slurm, executor.py) the
long ones as SLURM jobs. Environment: FEAT_OUT (default ~/feat_runs), FEAT_WORKERS (2), FEAT_TIMEOUT
(seconds per CalculiX solve, 900), FEAT_CCX (path of ccx), FEAT_SOLVER (auto | ccx | ref; auto =
CalculiX when found, else the reference solver), FEAT_PROCS (CalculiX threads, 1), FEAT_EXECUTOR,
FEAT_SLURM_MIN_MINUTES (1), FEAT_SLURM_ARGS, FEAT_SLURM_MEM (auto), FEAT_SLURM_NODE (local), FEAT_RENDER
(matplotlib: no VTK), FEAT_RENDER_TIMEOUT (900 s), FEAT_AUX_WORKERS (2: rendering workers).
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
    from .render_vtk import status as render_status
    out["render"] = render_status()
    return out


def study_format() -> dict:
    return {**_format(), "translator": __version__, "status": tool_status()}


def validate(studies: dict) -> dict:
    """Static checks; each valid study also gets "estimate": {minutes, procs, mem_mb} (llm-translators
    docs/EXECUTION.md, "Estimates in validation"), the sizing its SLURM job would get."""
    out = _validate(studies)
    for name, v in out.items():
        if v.get("ok"):
            try:
                v["estimate"] = estimate(studies[name])
            except Exception:  # noqa: BLE001 (an estimate is advice; validation stands without it)
                pass
    return out


def estimate(study: dict, solver: str | None = None) -> dict:
    """{minutes, procs, mem_mb} of a study, from the functions that size its SLURM job: procs after the size
    cap (executor.procs_for), memory as its --mem (or what CalculiX needs when the job asks for none),
    minutes from this machine's timing records scaled to the study's estimated mesh (when they include
    meshes within 10x of its size: small benchmark meshes scaled as nodes^1.5 overshoot), else from the
    mochi measurements, else the time limit (every solve's timeout); never above the time limit."""
    s = _solver(solver)
    limit = executor.solves_of(study) * executor.timeout_s(study) / 60
    rt = _rates(study.get("analysis", ""), s)
    nodes = executor.estimate_nodes(study)
    stages = ["mesh", "solve"] + (["mesh_fine", "solve_fine"] if study.get("converge") else [])
    if rt is not None and (nodes is None or rt["max_nodes"] * 10 >= nodes or s != "ccx"):
        sec = sum(_stage_s(st, nodes, rt) or 0 for st in stages)
    elif s == "ccx":
        sec = executor.measured_seconds(study)
    else:
        sec = None
    minutes = limit if sec is None else min(limit, sec / 60)
    mem = executor.mem_for(study) or executor.mem_auto(study)
    return {"minutes": round(max(minutes, 0.1), 1), "procs": executor.procs_for(study),
            "mem_mb": int(round(executor.mem_mb(mem) or executor.MEM_DEFAULT_GB * 1024))}


def _run_one(job: dict) -> dict:
    from .runner import run_study

    if job.get("out"):  # the worker may outlive a change of FEAT_OUT: timing records go with the request
        os.environ["FEAT_OUT"] = job["out"]
    # CalculiX threads, sized as for a SLURM job (executor.procs_for: 1 by default, 1 for the iterative
    # solver; set every time: the worker process runs one study after another)
    os.environ["OMP_NUM_THREADS"] = str(executor.procs_for(job["study"]))
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
        if self.pool is None:
            self.pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=20)
        self.jobs[str(folder)] = {j["name"]: self.pool.apply_async(_run_one, (dict(j, timeout=executor.timeout_s(j["study"])),)) for j in jobs}


_POOL = _Pool()


def _close_pool() -> None:
    """At exit: stop the worker pools quietly (no Pool.__del__ noise after the interpreter has torn down)."""
    if _POOL.pool is not None:
        _POOL.pool.terminate()
        _POOL.pool = None
    if _AUX["pool"] is not None:
        _AUX["pool"].terminate()
        _AUX["pool"] = None


# ---- other slow work of a tool call (rendering) in worker processes, as cfd-translator's in_worker: a call
# returns within its wait (at most 50 s) and a repeated call with the same arguments picks the result up
_AUX: dict = {"pool": None, "pending": {}}


def _aux_pool():
    if _AUX["pool"] is None:
        _AUX["pool"] = mp.get_context("spawn").Pool(int(os.environ.get("FEAT_AUX_WORKERS", "2") or 2), maxtasksperchild=20)
    return _AUX["pool"]


def _aux_run(fn_name: str, kwargs: dict, out: str | None = None):
    if out:   # the worker may outlive a change of FEAT_OUT
        os.environ["FEAT_OUT"] = out
    return globals()[fn_name](**kwargs)


def aux_busy() -> bool:
    return any(not h.ready() for h, _ in _AUX["pending"].values())


def in_worker(what: str, fn_name: str, kwargs: dict, wait: float) -> dict:
    """fn_name(**kwargs) in a worker; waits at most `wait` s (<= 50, and within the call's deadline). Unfinished:
    {state: running, ...}; the same call again waits for the same work."""
    key = f"{fn_name}:{json.dumps(kwargs, sort_keys=True, default=str)}"
    if key not in _AUX["pending"]:
        _AUX["pending"][key] = (_aux_pool().apply_async(_aux_run, (fn_name, kwargs, str(out_root()))), time.time())
    h, t0 = _AUX["pending"][key]
    h.wait(max(0.0, _until(min(float(wait), 50.0)) - time.time()))
    if not h.ready():
        return {"state": "running", "what": what, "elapsed_s": round(time.time() - t0, 1),
                "next": f"still {what}: call {fn_name} again with the same arguments (the work goes on meanwhile)"}
    _AUX["pending"].pop(key, None)
    try:
        return h.get()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def render_images(study_folder: str, images=True) -> dict:
    """New pictures of a finished study (its folder: <request>/<name>) from the saved state, without solving."""
    from .render import image_errors, load_state, render

    d = Path(study_folder).expanduser().resolve()
    root = out_root().resolve()
    if root not in d.parents:
        return {"error": f"only study folders under {root}"}
    if not (d / "state.npz").is_file():
        return {"error": f"no saved state in {d} (a finished study's folder: <request>/<name>, holding <name>.json)"}
    study = load_state(d)["meta"].get("study", {})
    errs = image_errors(images, study)
    if errs:
        return {"error": "invalid images", "errors": errs}
    out, warns = render(d, images, d.name)
    return {"images": out, "warnings": warns}


def image_path(path: str) -> Path:
    """A PNG written by a run or by render_images (only under the output folder)."""
    p = Path(path).expanduser().resolve()
    root = out_root().resolve()
    if root not in p.parents or p.suffix.lower() != ".png" or not p.is_file():
        raise ValueError(f"no image {p} under {root}")
    return p


import atexit  # noqa: E402

atexit.register(_close_pool)


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
        res["progress"] = {}
        for n in pending:   # each study's own time limit (run.timeout_s), else the request's default
            st_ = studies.get(n, {})
            t_ = executor.timeout_s(st_) if executor.run_of(st_).get("timeout_s") else float(meta.get("timeout", executor.timeout_s()))
            pr = _progress(folder / n, st_, meta["solver"], t_)
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
            "runs": len(recs), "max_nodes": max(r["nodes"] for r in recs)}


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
    return executor.estimate_nodes(study)


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
    # SLURM plans a queued job's start from the time limits of the jobs ahead: a latest start. The expected
    # start is when the first running study of the request is expected to free its CPUs, if that is sooner.
    run_ends = sorted(now + 60 * p["eta_min"] for p in progress.values()
                      if p.get("state") != "queued" and p.get("eta_min") is not None)
    for p in progress.values():
        if p.get("state") != "queued":
            s_exp = s_max = now
        else:
            s_max = _slurm_time(p.get("expected_start"))
            s_exp = min(s_max, run_ends[0]) if s_max is not None and run_ends else s_max
        exp_end.append(None if s_exp is None or p.get("eta_min") is None else s_exp + 60 * p["eta_min"])
        max_end.append(None if s_max is None else s_max + 60 * p["eta_min_max"])
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


def start(studies: dict, solver: str | None = None, prefix: str = "run") -> Path:
    """Validates and starts the studies in the background; returns the request folder."""
    folder = _request_dir(prefix)
    (folder / "studies.json").write_text(json.dumps(studies, indent=1))
    s = _solver(solver)
    invalid, jobs = {}, []
    for n, st in studies.items():
        v = validate_study(st)
        if not v["ok"]:
            invalid[n] = {"ok": False, "why": "invalid study", "errors": v["errors"]}
        else:
            jobs.append({"name": n, "study": st, "folder": str(folder), "solver": s, "out": str(out_root())})
    # SLURM unless the study is short (its estimate, as validate_studies reports it, below FEAT_SLURM_MIN_MINUTES):
    # a SLURM job can be cancelled and runs on, whatever happens to this server
    slurm = [j["name"] for j in jobs if executor.use_slurm(j["study"], _route_minutes(j["study"], s))]
    (folder / "request.json").write_text(json.dumps({"names": [j["name"] for j in jobs], "invalid": invalid,
                                                     "solver": s, "started": time.time(), "slurm": slurm,
                                                     "executor": executor.kind(),
                                                     "timeout": executor.timeout_s()}))
    for n in slurm:   # one SLURM job per study; a failed submission is that study's result (no local fallback)
        info = executor.submit(folder, n, studies[n])
        if info.get("ok") is False:
            (folder / n).mkdir(parents=True, exist_ok=True)
            (folder / n / f"{n}.json").write_text(json.dumps({**info, "state": "failed"}))
    local = [j for j in jobs if j["name"] not in slurm]
    if local:
        _POOL.submit(folder, local)
    return folder


def _route_minutes(study: dict, solver: str) -> float | None:
    """The minutes routing goes by: estimate()'s (size-scaled; the time limit when the size is unknown)."""
    try:
        return float(estimate(study, solver)["minutes"])
    except Exception:  # noqa: BLE001
        return None


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
    t_end = _until(wait)
    while True:
        res = _collect(folder)
        if res["done"] or time.time() >= t_end:
            return res
        slurm = any(p.get("job_id") for p in res.get("progress", {}).values())
        time.sleep(min(5.0 if slurm else 1.0, max(0.1, t_end - time.time())))   # one squeue per look


def _until(wait: float) -> float:
    """When a wait of `wait` s ends: within the MCP call's own deadline (executor.deadline), 3 s to spare for the
    answer."""
    t = time.time() + max(0.0, float(wait))
    left = executor.remaining()
    return t if left is None else min(t, time.time() + left - 3.0)


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
    reqs = sorted((d for g in ("run_*", "bench_*") for d in root.glob(g) if (d / "request.json").is_file()),
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
        left = executor.remaining()
        if left is not None and left < 5:   # within the call's time: say so rather than answer late
            out.append({"request": str(d), "started": _clock(meta.get("started", d.stat().st_mtime)),
                        "note": "not checked: SLURM was slow and the call ran out of time; ask again"})
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
    """Stops a request's studies (or the named ones), wherever they run: a cancel flag in each unfinished study's
    folder, which the runner checks while CalculiX runs (it ends CalculiX's process group within a second) and
    before meshing and each solve, in any process (the local pool of this or another server, a SLURM job); and
    scancel for SLURM jobs, queued or running. A study not started yet ends as cancelled when it would start.
    Returns their states (after waiting up to 10 s for them to stop)."""
    folder = Path(request).expanduser()
    try:
        meta = json.loads((folder / "request.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {"error": f"no request {folder}"}
    want = list(names) if names else list(meta.get("names", []))
    unknown = [n for n in want if n not in meta.get("names", [])]
    if unknown:
        return {"error": f"no studies {unknown} in this request ({meta.get('names', [])})"}
    todo = [n for n in want if not (folder / n / f"{n}.json").is_file()]
    for n in todo:
        (folder / n).mkdir(parents=True, exist_ok=True)
        (folder / n / "cancel").write_text(str(time.time()))
    slurm = [n for n in todo if n in meta.get("slurm", [])]
    ids = []
    for n in slurm:
        try:
            ids.append(json.loads((folder / n / "job.json").read_text())["job_id"])
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    err = executor.scancel(ids)
    out = {"request": str(folder), "cancelled_jobs": ids, "already_finished": [n for n in want if n not in todo]}
    if err:
        out["error"] = err
    local = [n for n in todo if n not in slurm]
    here = _POOL.jobs.get(str(folder), {})
    t_end = _until(10)                       # give CalculiX and SLURM a moment to stop
    while time.time() < t_end:
        q = executor.squeue(ids) if ids else {}
        if (q is not None and not q) and all((folder / n / f"{n}.json").is_file() or (n in here and here[n].ready())
                                             or n not in here for n in local):
            break
        time.sleep(0.5)
    st = _collect(folder)
    out["states"] = {n: (st["progress"][n]["state"] if n in st.get("pending", []) else
                         st["studies"].get(n, {}).get("state", "done" if st["studies"].get(n, {}).get("ok") else "failed"))
                     for n in want}
    elsewhere = [n for n in local if n not in here and n in st.get("pending", [])]
    if elsewhere:
        out["note"] = ("local studies not run by this server process: their cancel flag stops them within a second "
                       "in the process that runs them (if one does): " + ", ".join(elsewhere))
    return out


def deck(study: dict, name: str = "study", wait: float | None = None) -> dict:
    """The CalculiX deck of a study, meshed but not solved. With `wait` (the MCP tool) the meshing runs in a
    worker process and the call returns after at most `wait` s: if it is not done by then, the answer says
    where the deck will be (read it with get_file). Without (CLI): meshed here."""
    v = validate_study(study)
    if not v["ok"]:
        return {"error": "invalid study", "errors": v["errors"]}
    folder = _request_dir("deck")
    if wait is None:
        return _deck_into(study, name, folder)
    proc = mp.get_context("spawn").Process(target=_deck_job, args=(study, name, str(folder), str(out_root())),
                                           daemon=True, name="feat-deck")
    proc.start()
    t_end = _until(wait)
    res = folder / "deck.json"
    while time.time() < t_end and proc.is_alive() and not res.is_file():
        time.sleep(0.2)
    try:
        return json.loads(res.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    if not proc.is_alive():
        return {"error": "meshing failed in the worker process", "folder": str(folder)}
    return {"pending": True, "path": str(folder / f"{name}.inp"), "folder": str(folder),
            "next": f"the part is still being meshed; the deck will be at {folder / (name + '.inp')} "
                    "(read it with get_file when it exists), or ask get_deck again later"}


def _deck_job(study: dict, name: str, folder: str, out: str) -> None:
    """Worker process of deck(): writes deck.json (the answer) next to the deck, atomically."""
    os.environ["FEAT_OUT"] = out
    try:
        r = _deck_into(study, name, Path(folder))
    except Exception as exc:  # noqa: BLE001
        r = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
    tmp = Path(folder) / ".deck.json.tmp"
    tmp.write_text(json.dumps(r))
    tmp.replace(Path(folder) / "deck.json")


def _deck_into(study: dict, name: str, folder: Path) -> dict:
    from . import deck as D
    from .mesher import mesh_part

    mesh, info = mesh_part(study, folder)
    text, plan = D.write(name, study, mesh)
    p = folder / f"{name}.inp"
    p.write_text(text)
    head = text if len(text) < 20000 else text[:20000] + "\n** ... (truncated; the whole deck is in the file)\n"
    return {"path": str(p), "mesh": info, "bytes": len(text), "deck": head}


def bench(names=None, solver: str | None = None, wait: float | None = None) -> dict:
    """Runs the benchmark studies as one request (SLURM jobs or the local pool, as any study) and
    returns after at most `wait` seconds (None: when all are done); bench_status(bench) collects the rest."""
    from . import bench as B

    s = _solver(solver)
    try:
        import gmsh  # noqa: F401
        have_gmsh = True
    except ImportError:
        have_gmsh = False
    cases, skipped = {}, {}
    for n, st in B.cases(s).items():
        if names and n not in names:
            continue
        need = B.NEEDS.get(n, set())
        if "ccx" in need and s != "ccx":
            skipped[n] = "needs CalculiX"
        elif "gmsh" in need and not have_gmsh:
            skipped[n] = "needs Gmsh"
        else:
            cases[n] = st
    folder = start(cases, s, prefix="bench")
    (folder / "bench_meta.json").write_text(json.dumps({"skipped": skipped, "names": list(cases)}))
    return bench_status(str(folder), 1e9 if wait is None else wait)


def bench_status(bench: str, wait: float = 0.0) -> dict:
    """A benchmark request: passed, failed, skipped and pending cases (waits up to `wait` s for the pending)."""
    folder = Path(bench).expanduser()
    try:
        meta = json.loads((folder / "bench_meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {"error": f"no benchmark request {folder}"}
    st = status(str(folder), wait)
    cases = {k: {x: v.get(x) for x in ("ok", "passed", "checks", "error", "why", "state", "seconds", "mesh",
                                     "warnings", "solver_reruns") if x in v}
             for k, v in st.get("studies", {}).items()}
    out = {"bench": str(folder), "solver": st.get("solver"), "done": st.get("done"),
           "passed": [k for k, v in cases.items() if v.get("passed")],
           "failed": [k for k, v in cases.items() if v.get("passed") is False or v.get("ok") is False],
           "pending": st.get("pending", []), "skipped": meta.get("skipped", {}), "cases": cases}
    for k in ("progress", "time_left"):
        if k in st:
            out[k] = st[k]
    if st.get("next"):
        out["next"] = st["next"].replace(f"study_status(request='{folder}')", f"bench_status(bench='{folder}')")
    if out["done"]:
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
