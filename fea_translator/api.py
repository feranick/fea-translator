"""api.py — what other programs and the MCP server call. JSON in, JSON out.
Version: 2026.10.16.10

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
    tool_status()                    versions, solver and mesher found

Studies run in worker processes with a time limit each. Environment: FEAT_OUT (default
~/feat_runs), FEAT_WORKERS (2), FEAT_TIMEOUT (seconds per study, 900), FEAT_CCX (path of ccx),
FEAT_SOLVER (auto | ccx | ref; auto = CalculiX when found, else the reference solver).
"""
from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path

from . import __version__
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
    return out


def study_format() -> dict:
    return {**_format(), "translator": __version__, "status": tool_status()}


def validate(studies: dict) -> dict:
    return _validate(studies)


def _run_one(job: dict) -> dict:
    from .runner import run_study

    if job.get("out"):  # the worker may outlive a change of FEAT_OUT: timing records go with the request
        os.environ["FEAT_OUT"] = job["out"]
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
    for name in meta["names"]:
        f = folder / name / f"{name}.json"
        h = handles.get(name)
        if f.is_file() and (h is None or h.ready()):
            try:
                out[name] = json.loads(f.read_text())
                continue
            except json.JSONDecodeError:
                pass
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
        res["progress"] = {n: _progress(folder / n, studies.get(n, {}), meta["solver"], timeout) for n in pending}
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


def _progress(d: Path, study: dict, solver: str, timeout: float) -> dict:
    """Where a running study is: step k of n (mesh, solve; with converge: true also the finer mesh),
    seconds per step, and minutes left: expected (from this machine's past solves) and at most (the
    time limit of each remaining solve)."""
    steps = 4 if study.get("converge") else 2
    rt = _rates(study.get("analysis", ""), solver)
    pf = d / "progress.json"
    try:
        p = json.loads(pf.read_text())
    except (OSError, json.JSONDecodeError):
        p = None
    now = time.time()
    if p is None:  # waiting for a worker
        exp = sum(_stage_s(s, None, rt) or 0 for s in STAGES[:steps]) if rt else None
        return {"stage": "queued", "iteration": 0, "max_iterations": steps,
                "eta_min": None if exp is None else round(exp / 60, 1),
                "eta_min_max": round(steps / 2 * timeout / 60, 1)}
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
    out = {"stage": stage, "iteration": step, "max_iterations": steps,
           "s_per_iteration": round(sum(done) / len(done), 1) if done else None,
           "elapsed_min": round((now - p.get("started", now)) / 60, 1), "nodes": nodes,
           "eta_min": None if exp is None else round(exp / 60, 1), "eta_min_max": round(max(worst, 0) / 60, 1)}
    if rt is None:
        out["note"] = "no finished solves on this machine yet: eta_min is unknown until one finishes"
    return out


def _clock(t: float) -> str:
    return time.strftime("%H:%M", time.localtime(t)) + (" (tomorrow)" if time.localtime(t).tm_yday != time.localtime().tm_yday else "")


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
    (folder / "request.json").write_text(json.dumps({"names": [j["name"] for j in jobs], "invalid": invalid,
                                                     "solver": s, "started": time.time(),
                                                     "timeout": float(os.environ.get("FEAT_TIMEOUT", "900"))}))
    if jobs:
        _POOL.submit(folder, jobs)
    return folder


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
        time.sleep(1.0)


def run(studies: dict, solver: str | None = None, wait: float | None = None) -> dict:
    """Runs the studies into a new request folder under FEAT_OUT: per study <name>/<name>.inp,
    .dat, .frd (CalculiX), mesh.msh, part.step, <name>.json; and studies.json, results.json.
    Waits for all of them (wait None) or at most `wait` seconds; then the unfinished ones are
    listed as pending and status(request) collects them later."""
    folder = start(studies, solver)
    return status(str(folder), 1e9 if wait is None else wait)


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
