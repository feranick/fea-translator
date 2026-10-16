"""api.py — what other programs and the MCP server call. JSON in, JSON out.
Version: 2026.10.16.4

    study_format()                   the study format, for the caller to read
    validate(studies)                static checks, no meshing
    run(studies, solver, wait)       meshes, solves and checks each study; files in a request folder;
                                     with wait, returns after that many seconds with the rest pending
    start(studies, solver)           the same in the background: returns the request folder
    status(request, wait)            finished studies with results, the others pending
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
        res["next"] = f"call study_status(request='{folder}') to collect the rest"
    else:
        (folder / "results.json").write_text(json.dumps(res, indent=1))
    return res


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
            jobs.append({"name": n, "study": st, "folder": str(folder), "solver": s})
    (folder / "request.json").write_text(json.dumps({"names": [j["name"] for j in jobs], "invalid": invalid,
                                                     "solver": s, "started": time.time()}))
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
