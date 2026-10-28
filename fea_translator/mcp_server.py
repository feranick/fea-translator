"""mcp_server.py — MCP server: lets LLM-based tools run FEA studies with Gmsh and CalculiX.
Version: 2026.10.28.1

Tools (all JSON):
  study_format      the study format: analyses, materials, boundary conditions, face selectors. Read it first.
  validate_studies  static checks of {name: study} (fast; fix errors before running)
  run_studies       meshes, solves and checks {name: study}: results, checks, files (returns within ~45 s;
                    long studies stay pending)
  study_status      collects the pending studies of a run_studies request (progress, time left, advice)
  get_deck          the CalculiX input deck of one study, as text
  run_benchmarks    the benchmark studies with closed-form answers (is the toolchain right?); returns
                    within 30 s
  bench_status      collects a run_benchmarks request
  get_file          a written file as base64 (for callers on another machine)
  jobs              recent requests with SLURM jobs (also from other conversations) and the queue
  cancel            stops a request's studies
  render_images     new pictures of a finished study from its saved state (no solving; in a worker)
  get_image         a PNG written by run_studies or render_images, as an image

Run:  feat-mcp                       (stdio)
      feat-mcp --http 8767 [--host 0.0.0.0]

Time: every tool answers within its wait (at most 50 s): tools run in a worker thread under a deadline that
      bounds SLURM commands and submissions; CAD (part sizes) and meshing (get_deck) run in worker processes,
      solves in the worker pool or in SLURM jobs.
Stale: a server keeps the code it started with. When a newer release is installed (shared install: <prefix>/current
      points elsewhere; checkout: __init__.py has another version), a call that finds the server idle is refused
      with STALE_SERVER (llm-translators' router repeats it on a new server) and the server exits; an idle stale
      server exits within ~20 s by itself. Busy servers answer (old code, with a server_stale note) and exit once
      idle. Only a server started by main() refuses or exits (tests and run-study in a release that is not yet
      current run normally). FEAT_EXIT_WHEN_STALE=1 always, 0 never; default: shared installs only.
"""
from __future__ import annotations

import argparse
import functools
import os
import re
import sys
import threading
import time
from pathlib import Path

import anyio.to_thread

from . import api, executor

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import Image
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
    _V2 = True
except ImportError:  # 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp import Image
    from mcp.server.fastmcp.exceptions import ToolError
    _V2 = False

BUDGET_S = 48.0   # a call's deadline: the router (and the Claude app) count on an answer within 50 s

INSTRUCTIONS = """fea-translator runs finite-element analyses of one solid part: linear static (stresses,
displacements, reactions, safety factor), modal (natural frequencies), linear buckling (load factors)
steady heat transfer (temperatures, heat flows; with stress: true, thermal stresses), DC electric
conduction (terminal currents, resistance, power, current density), electrostatics (charges,
capacitance, peak field, margin to breakdown) and Joule heating (electric -> heat -> temperatures,
optionally -> stresses). A study names the part (a STEP file or a
build123d-translator goal), the material, the analysis, the supports and loads on faces chosen by
normal, point or cylinder radius, and optional expected values to check. Gmsh meshes the part,
CalculiX solves it.
Workflow: call study_format once and follow its conventions; write one study per analysis;
validate_studies and fix every error; run_studies (long studies come back as 'pending' with
'progress' per study and 'time_left': eta_min expected, eta_min_max at most, expected_finish and
latest_finish as clock times). Polling costs tokens: when a run needs more than about 5 minutes, tell
the user when it should finish and stop; collect it later with study_status (when the user asks, or
from a task scheduled after that time). Follow 'next' in the result.
Check equilibrium_error, warnings and 'selection' (what each
face selector took), and rerun with converge: true before trusting peak stresses; peaks at supports
are singular: judge designs on peak_von_mises_off_bc and safety_factor_off_bc.
Long studies run as SLURM jobs on the machine (when FEAT_EXECUTOR=slurm): run_studies returns them pending
with job ids. They do not depend on this conversation. Report the expected finish time and stop; do not poll.
Later, study_status(request) collects them; jobs() lists recent and active runs (also those started in other
conversations); cancel(request) stops them. A study's "run": {timeout_s, mem, procs, queue} overrides the
machine's defaults for that study, within the caps in study_format's status.
Pictures: "images": true (or views: a field on the deformed part, mode shapes, the setup view showing the faces each
bc took, section cuts) writes PNGs (results images.<name>.path); get_image shows each; render_images draws other
views of a finished study without solving again. Check 'selection' by eye with the setup view.
Units: mm, N, MPa, kg/m3, W, degrees C; V, A, ohm m, kV/mm."""

server = _Server("fea-translator", instructions=INSTRUCTIONS)


# ---- a stale server: a newer version installed since this process started (llm-translators docs/SHARED.md) ----
STALE_MARK = "STALE_SERVER"
VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.M)
_STATE = {"inflight": 0, "exiting": False, "checked": 0.0, "why": None, "shared": False, "serving": False}
_LOCK = threading.Lock()


def stale_reason(pkg_dir: Path, running: str) -> tuple[str | None, bool]:
    """(why this process runs old code or None, shared install?). Shared install (a release under
    <prefix>/releases with RELEASE.json): <prefix>/current now points to another release. Otherwise: the
    package's __init__.py on disk carries another version (a checkout or editable install updated in place)."""
    rel = pkg_dir.parent.parent
    if (rel / "RELEASE.json").is_file() and rel.parent.name == "releases":
        cur = rel.parent.parent / "current"
        try:
            if cur.exists() and cur.resolve() != rel.resolve():
                return f"release {cur.resolve().name} is installed; this server runs {rel.name} ({running})", True
        except OSError:
            pass
        return None, True
    try:
        m = VERSION_RE.search((pkg_dir / "__init__.py").read_text())
    except OSError:
        return None, False
    if m and m.group(1) != running:
        return f"version {m.group(1)} is installed; this server runs {running}", False
    return None, False


def _stale() -> str | None:
    """stale_reason for this process, checked at most every 2 s."""
    now = time.time()
    if now - _STATE["checked"] > 2.0:
        _STATE["why"], _STATE["shared"] = stale_reason(Path(api.__file__).resolve().parent, api.__version__)
        _STATE["checked"] = now
    return _STATE["why"]


def _may_exit() -> bool:
    """Only a server started by main() (never tools called in-process: tests, scripts, llmt deploy --test before
    current is switched). Then FEAT_EXIT_WHEN_STALE: 1 always, 0 never; default: in a shared install."""
    if not _STATE["serving"]:
        return False
    v = os.environ.get("FEAT_EXIT_WHEN_STALE", "").strip()
    return v == "1" or (v != "0" and _STATE["shared"])


def _pool_busy() -> bool:
    """Studies still running in this process's worker pool (they would die with it)."""
    return any(not r.ready() for jobs in api._POOL.jobs.values() for r in jobs.values()) or api.aux_busy()


def _exit_soon(why: str, delay: float) -> None:
    """Exits in `delay` s (an answer in flight reaches the caller first). Caller holds _LOCK."""
    if _STATE["exiting"]:
        return
    _STATE["exiting"] = True
    sys.stderr.write(f"feat: {STALE_MARK}: exiting to pick up the new code: {why}\n")
    sys.stderr.flush()

    def go():
        time.sleep(delay)
        if api._POOL.pool is not None:
            api._POOL.pool.terminate()
        os._exit(0)
    threading.Thread(target=go, daemon=True, name="feat-exit").start()


def _enter_call() -> str | None:
    """Counts a call in. Returns a STALE_MARK message (the call is refused) when this server is stale, may exit
    and has nothing else running; otherwise None."""
    with _LOCK:
        why = _stale()
        if why and _may_exit() and (_STATE["exiting"] or (not _STATE["inflight"] and not _pool_busy())):
            _exit_soon(why, 1.0)
            return (f"{STALE_MARK}: {why}. This server is exiting so that the next one runs the installed version: "
                    "call again.")
        _STATE["inflight"] += 1
        return None


def _leave_call() -> None:
    with _LOCK:
        _STATE["inflight"] -= 1


def _watch_install(period: float = 10.0) -> None:
    """Exits this server when it is stale, may exit and is idle (no calls, no pool studies): within ~20 s."""
    def loop():
        while True:
            time.sleep(period)
            with _LOCK:
                if _stale() and _may_exit() and not _STATE["inflight"] and not _pool_busy():
                    _exit_soon(_STATE["why"], 0.0)
    threading.Thread(target=loop, daemon=True, name="feat-watch-install").start()


def _tool(fn):
    """Every tool: refused with STALE_SERVER when the server is stale and idle; otherwise run in a worker thread
    (a blocking call must not stop the server's other requests) under a deadline of BUDGET_S, so it answers in
    time; errors come back with their text (SDK 2.x hides the text of other exceptions)."""
    def bounded(*args, **kwargs):
        with executor.deadline(BUDGET_S):
            return fn(*args, **kwargs)

    @functools.wraps(fn)
    async def wrap(*args, **kwargs):
        refused = _enter_call()
        if refused:
            raise ToolError(refused)
        try:
            r = await anyio.to_thread.run_sync(functools.partial(bounded, *args, **kwargs))
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            _leave_call()
        if _STATE["why"] and isinstance(r, dict):
            r = {**r, "server_stale": f"{_STATE['why']}: this answer comes from the old code"
                                      + ("; the server restarts when idle" if _may_exit() else "; restart the MCP server")}
        return r
    return wrap


@server.tool()
@_tool
def study_format() -> dict:
    """The study format: analyses, materials, boundary-condition kinds with their parameters,
    face selectors, conventions, an example; and which solver and mesher are installed."""
    return api.study_format()


@server.tool()
@_tool
def validate_studies(studies: dict) -> dict:
    """Static checks of studies given as {name: study}. Returns per study ok, errors, warnings, and for a valid
    study "estimate": {minutes, procs, mem_mb} (expected run time, at most its time limit; threads; memory)."""
    return api.validate(studies)


@server.tool()
@_tool
def run_studies(studies: dict, solver: str = "auto", wait_s: float = 45) -> dict:
    """Meshes, solves and checks each study. Per study: results (max displacement, von Mises peaks
    with and without the support/load regions, safety factors, reactions, frequencies, buckling
    factors, temperatures, heat flows, probes), what each face selector matched, checks against
    'expect', equilibrium, warnings, and files (.inp deck, .frd and .dat results, mesh, STEP).
    Returns after at most wait_s seconds: studies still running are listed in 'pending' with
    'progress' (step of steps, s per step, eta_min, eta_min_max) and 'time_left' (expected_finish,
    latest_finish); 'next' says whether to collect them with study_status soon or to tell the user
    when to come back (runs over about 5 min). Fine meshes and converge: true can take minutes.
    With FEAT_EXECUTOR=slurm, long studies (and those with "queue": true) are SLURM jobs: their
    progress has 'state' (queued, running) and 'job_id'; they keep running without this conversation."""
    return api.run(studies, solver, wait=min(max(float(wait_s), 0.0), 50.0))


@server.tool()
@_tool
def study_status(request: str, wait_s: float = 45) -> dict:
    """The studies of a run_studies request (its 'request' path): finished ones with their results,
    the others in 'pending' with progress and time left. Waits up to wait_s seconds (at most 50) for
    them to finish. Do not poll long runs: follow 'next'."""
    return api.status(request, min(max(float(wait_s), 0.0), 50.0))


@server.tool()
@_tool
def get_deck(study: dict, name: str = "study") -> dict:
    """Meshes the part and returns the CalculiX input deck (.inp) of one study, without solving. Returns within
    45 s: a part that takes longer to mesh comes back 'pending' with the path where the deck will be."""
    return api.deck(study, name, wait=45.0)


@server.tool()
@_tool
def run_benchmarks(names: list[str] | None = None, solver: str = "auto", wait_s: float = 30) -> dict:
    """Runs the benchmark studies with closed-form answers (axial bar, cantilever, natural frequency,
    Euler column, conduction, Lame ring, plate with a hole, electrical cases) as one request, and
    returns within wait_s seconds (at most 50): passed, failed, skipped and still pending cases. Collect
    the pending ones with bench_status(bench); do not poll long runs: follow 'next'."""
    return api.bench(names, solver, wait=min(max(float(wait_s), 0.0), 50.0))


@server.tool()
@_tool
def bench_status(bench: str, wait_s: float = 30) -> dict:
    """A run_benchmarks request (its 'bench' path): passed, failed, skipped and pending cases, with
    progress for the pending ones. Waits up to wait_s seconds (at most 50)."""
    return api.bench_status(bench, min(max(float(wait_s), 0.0), 50.0))


@server.tool()
@_tool
def get_file(path: str, max_mb: float = 20) -> dict:
    """A file written by run_studies (a path from its result) as base64."""
    return api.read_file(path, max_mb)


@server.tool()
@_tool
def jobs(limit: int = 20, active: bool = False) -> dict:
    """Recent requests that have SLURM jobs, with each study's state (queued, running, done, timed_out,
    out_of_memory, cancelled, failed), job id and expected time; the translator's queue and the machine's
    CPUs. active=True: only requests with queued or running studies. Finds runs started in other
    conversations too."""
    return api.jobs(limit, active)


@server.tool()
@_tool
def get_image(path: str) -> Image:
    """A PNG written by run_studies (a result's images.<name>.path) or render_images, as an image you can see and
    show: fields on the (deformed) part, mode shapes, the setup view, section cuts."""
    return Image(path=str(api.image_path(path)))


@server.tool()
@_tool
def render_images(study_folder: str, images: list | bool = True, wait_s: float = 45) -> dict:
    """New pictures of a finished study without solving again: study_folder is <request>/<study name> (the folder
    holding <name>.json and state.npz). images: true (the defaults) or [{name, field | mode: n | setup: true,
    surface: part | <bc name>, deformed: true | scale, view: [elev, azim], clim: [lo, hi] | full | robust, quality:
    high (VTK: lit, smooth) | basic, slice: {axis, at}}] as in study_format. Returns within wait_s (at most 50): if
    still drawing, state 'running' (call again with the same arguments). Then get_image each path."""
    return api.in_worker("rendering", "render_images", {"study_folder": study_folder, "images": images},
                         min(max(float(wait_s), 0.0), 50.0))


@server.tool()
@_tool
def cancel(request: str, names: list[str] | None = None) -> dict:
    """Stops a request's studies (or the named ones): their SLURM jobs are cancelled. Returns their states."""
    return api.cancel(request, names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", type=int, help="serve MCP over streamable HTTP on this port (default: stdio)")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    _STATE["serving"] = True
    _watch_install()
    if a.http is None:
        server.run()
    elif _V2:
        server.run(transport="streamable-http", host=a.host, port=a.http)
    else:
        server.settings.host, server.settings.port = a.host, a.http
        server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
