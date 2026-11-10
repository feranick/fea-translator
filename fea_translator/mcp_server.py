"""mcp_server.py — MCP server: lets LLM-based tools run FEA studies with Gmsh and CalculiX.
Version: 2026.11.09.10

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
  extend            more time for a request's running studies (CalculiX's limit and the SLURM job's)
  cancel            stops a request's studies
  render_images     new pictures of a finished study from its saved state (no solving; in a worker)
  get_image         a PNG written by run_studies or render_images, as an image

Run:  feat-mcp                       (stdio)
      feat-mcp --http 8767 [--host 0.0.0.0]

Time: every tool answers within its wait (at most 50 s): tools run in a worker thread under a deadline that
      bounds SLURM commands and submissions; CAD (part sizes) and meshing (get_deck) run in worker processes,
      solves in the worker pool or in SLURM jobs.
Stale: a server keeps the code it started with. When a newer release is installed (shared install: <prefix>/current
      points elsewhere; checkout: __init__.py, feat's or llmt-runtime's, has another version), a call that finds the
      server idle is refused with STALE_SERVER (llm-translators' router repeats it on a new server) and the server
      exits; an idle stale server exits by itself within about two periods (FEAT_STALE_CHECK_S, default 10 s). Busy
      servers answer (old code, with a server_stale note) and exit once idle. Only a server started by main()
      refuses or exits (tests and run-study in a release that is not yet current run normally).
      FEAT_EXIT_WHEN_STALE=1 always, 0 never; default: shared installs only.

Since 2026.11.09.10 the stale handling and the tool wrapper are llmt-runtime's (llmt_runtime.server.Server: GUARD
below), as are the server's log (<FEAT_OUT>/.server/<time>_<host>_<pid>.log: calls, failures with their traceback,
the exit) and the signal handlers (SIGTERM/SIGHUP/SIGINT stop the workers first). The names below (stale_reason,
_STATE, _stale, _may_exit, _exit_soon, _enter_call, _leave_call, _watch_install, STALE_MARK, BUDGET_S) are thin
wrappers over GUARD for tests and scripts.
"""
# No `from __future__ import annotations` here (since 2026.11.09.10): the tools' wrapper is defined in
# llmt_runtime.server, and MCP SDK 1.x evaluates string annotations in the wrapper's globals (that module's), where
# `Image` is unknown; real annotation objects need no evaluation (all valid at run time on Python >= 3.10).
import argparse

from llmt_runtime import server as _rt_server
from llmt_runtime.server import Server

from . import api
from .runtime import RT

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

# a call's deadline: the router (and the Claude app) count on an answer within 50 s. llmt-runtime's (48 s), bound when
# GUARD.tool decorates each tool: changing this name later changes nothing (tests patch llmt_runtime.server.deadline)
BUDGET_S = _rt_server.BUDGET_S

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
Bolted joints: analysis "joint" takes an assembly (parts), gaskets (contact with a closure-stress law and a
sealing stress), pretensioned bolts, and a pressure or force raised in steps; results give the leak and
lift-off value per gasket and the bolt forces per step.
Assemblies: analysis "assembly" takes parts with contacts between them (hard, penalty or bonded, friction,
clearance or interference), gaskets, bolts and nuts (a nut clamping parts on a meshed shaft), and steps: preload,
static steps with their loads, a sweep, and modal steps on the preloaded state (frequencies with contact).
Long studies run as SLURM jobs on the machine (when FEAT_EXECUTOR=slurm): run_studies returns them pending
with job ids. They do not depend on this conversation. Report the expected finish time and stop; do not poll.
Later, study_status(request) collects them; jobs() lists recent and active runs (also those started in other
conversations); cancel(request) stops them; extend(request, hours) gives running ones more time (when
study_status warns that CalculiX will not finish within its limit: its 'calculix' progress says where it is). A study's "run": {timeout_s, mem, procs, queue} overrides the
machine's defaults for that study, within the caps in study_format's status.
Pictures: "images": true (or views: a field on the deformed part, mode shapes, the setup view showing the faces each
bc took, section cuts) writes PNGs (results images.<name>.path); get_image shows each; render_images draws other
views of a finished study without solving again. Check 'selection' by eye with the setup view.
Units: mm, N, MPa, kg/m3, W, degrees C; V, A, ohm m, kV/mm."""

server = _Server("fea-translator", instructions=INSTRUCTIONS)


# ---- a stale server: a newer version installed since this process started (llm-translators docs/SHARED.md) ----
# llmt-runtime's since 2026.11.09.10: GUARD holds the state, the checks, the exit and the tool wrapper.


def _pool_busy() -> bool:
    """Studies still running in this process's worker pool (they would die with it)."""
    return any(not r.ready() for jobs in api._POOL.jobs.values() for r in jobs.values()) or api.aux_busy()


GUARD = Server(RT, api.__file__, busy=_pool_busy, shutdown=api.shutdown_workers)   # period: FEAT_STALE_CHECK_S (10 s)

STALE_MARK = _rt_server.STALE_MARK
stale_reason = _rt_server.stale_reason  # (pkg_dir, running[, runtime_dir, runtime_running]) -> (why or None, shared?)
_STATE = GUARD.state                    # inflight, background, exiting, checked, why, shared, serving
_LOCK = GUARD.lock


def _stale() -> str | None:
    """stale_reason for this process (feat's and llmt-runtime's versions), checked at most every 2 s."""
    return GUARD.stale()


def _may_exit() -> bool:
    """Only a server started by main() (never tools called in-process: tests, scripts, llmt deploy --test before
    current is switched). Then FEAT_EXIT_WHEN_STALE: 1 always, 0 never; default: in a shared install."""
    return GUARD.may_exit()


def _exit_soon(why: str, delay: float) -> None:
    """Exits in `delay` s after stopping every worker (api.shutdown_workers, at most SHUTDOWN_S: os._exit skips
    atexit). GUARD's own calls hold its lock."""
    GUARD.exit_soon(why, delay)


def _enter_call() -> str | None:
    """Counts a call in. Returns a STALE_MARK message (the call is refused) when this server is stale, may exit
    and has nothing else running; otherwise None."""
    return GUARD.enter_call()


def _leave_call() -> None:
    GUARD.leave_call()


def _watch_install(period: float | None = None) -> None:
    """Exits this server when it is stale, may exit and is idle (no calls, no pool studies), checking every
    `period` s (default GUARD.period: FEAT_STALE_CHECK_S, 10 s). GUARD.serve() starts it."""
    if period is not None:
        GUARD.period = float(period)
    GUARD.watch()


@server.tool()
@GUARD.tool(ToolError)
def study_format() -> dict:
    """The study format: analyses, materials, boundary-condition kinds with their parameters,
    face selectors, conventions, an example; and which solver and mesher are installed."""
    return api.study_format()


@server.tool()
@GUARD.tool(ToolError)
def validate_studies(studies: dict) -> dict:
    """Static checks of studies given as {name: study}. Returns per study ok, errors, warnings, and for a valid
    study "estimate": {minutes, procs, mem_mb} (expected run time, at most its time limit; threads; memory)."""
    return api.validate(studies)


@server.tool()
@GUARD.tool(ToolError)
def run_studies(studies: dict, solver: str = "auto", wait_s: float = 45, request_name: str | None = None) -> dict:
    """Meshes, solves and checks each study. Per study: results (max displacement, von Mises peaks
    with and without the support/load regions, safety factors, reactions, frequencies, buckling
    factors, temperatures, heat flows, probes), what each face selector matched, checks against
    'expect', equilibrium, warnings, and files (.inp deck, .frd and .dat results, mesh, STEP).
    Returns after at most wait_s seconds: studies still running are listed in 'pending' with
    'progress' (step of steps, s per step, eta_min, eta_min_max) and 'time_left' (expected_finish,
    latest_finish); 'next' says whether to collect them with study_status soon or to tell the user
    when to come back (runs over about 5 min). Fine meshes and converge: true can take minutes.
    With FEAT_EXECUTOR=slurm, long studies (and those with "queue": true) are SLURM jobs: their
    progress has 'state' (queued, running) and 'job_id'; they keep running without this conversation.
    request_name ('run_<id>', optional): the request folder's name, so a caller knows the request path before the
    answer; repeating it returns that request's status (no second submission)."""
    return api.run(studies, solver, wait=min(max(float(wait_s), 0.0), 50.0), request_name=request_name)


@server.tool()
@GUARD.tool(ToolError)
def study_status(request: str, wait_s: float = 45) -> dict:
    """The studies of a run_studies request (its 'request' path): finished ones with their results,
    the others in 'pending' with progress and time left. Waits up to wait_s seconds (at most 50) for
    them to finish. Do not poll long runs: follow 'next'."""
    return api.status(request, min(max(float(wait_s), 0.0), 50.0))


@server.tool()
@GUARD.tool(ToolError)
def get_deck(study: dict, name: str = "study") -> dict:
    """Meshes the part (a joint: every part) and returns the CalculiX input deck (.inp) of one study and its mesh
    size (nodes, elements; a joint: per part, with the selection, bolts, gaskets' gap and law), without solving:
    check a mesh's size before running it. Returns within 45 s: a part that takes longer to mesh comes back
    'pending' with the paths where the deck and this answer will be."""
    return api.deck(study, name, wait=45.0)


@server.tool()
@GUARD.tool(ToolError)
def run_benchmarks(names: list[str] | None = None, solver: str = "auto", wait_s: float = 30) -> dict:
    """Runs the benchmark studies with closed-form answers (axial bar, cantilever, natural frequency,
    Euler column, conduction, Lame ring, plate with a hole, electrical cases) as one request, and
    returns within wait_s seconds (at most 50): passed, failed, skipped and still pending cases. Collect
    the pending ones with bench_status(bench); do not poll long runs: follow 'next'."""
    return api.bench(names, solver, wait=min(max(float(wait_s), 0.0), 50.0))


@server.tool()
@GUARD.tool(ToolError)
def bench_status(bench: str, wait_s: float = 30) -> dict:
    """A run_benchmarks request (its 'bench' path): passed, failed, skipped and pending cases, with
    progress for the pending ones. Waits up to wait_s seconds (at most 50)."""
    return api.bench_status(bench, min(max(float(wait_s), 0.0), 50.0))


@server.tool()
@GUARD.tool(ToolError)
def get_file(path: str, max_mb: float = 20) -> dict:
    """A file written by run_studies (a path from its result) as base64."""
    return api.read_file(path, max_mb)


@server.tool()
@GUARD.tool(ToolError)
def jobs(limit: int = 20, active: bool = False) -> dict:
    """Recent requests that have SLURM jobs, with each study's state (queued, running, done, timed_out,
    out_of_memory, cancelled, failed), job id and expected time; the translator's queue and the machine's
    CPUs. active=True: only requests with queued or running studies. Finds runs started in other
    conversations too."""
    return api.jobs(limit, active)


@server.tool()
@GUARD.tool(ToolError)
def get_image(path: str) -> Image:
    """A PNG written by run_studies (a result's images.<name>.path) or render_images, as an image you can see and
    show: fields on the (deformed) part, mode shapes, the setup view, section cuts."""
    return Image(path=str(api.image_path(path)))


@server.tool()
@GUARD.tool(ToolError)
def render_images(study_folder: str, images: list | bool = True, wait_s: float = 45) -> dict:
    """New pictures of a finished study without solving again: study_folder is <request>/<study name> (the folder
    holding <name>.json and state.npz). images: true (the defaults) or [{name, field | mode: n | setup: true,
    surface: part | <bc name>, deformed: true | scale, view: [elev, azim], clim: [lo, hi] | full | robust, quality:
    high (VTK: lit, smooth) | basic, slice: {axis, at}}] as in study_format. Returns within wait_s (at most 50): if
    still drawing, state 'running' (call again with the same arguments). Then get_image each path."""
    return api.in_worker("rendering", "render_images", {"study_folder": study_folder, "images": images},
                         min(max(float(wait_s), 0.0), 50.0))


@server.tool()
@GUARD.tool(ToolError)
def extend(request: str, hours: float, names: list[str] | None = None) -> dict:
    """More time for a request's running studies (or the named ones): each CalculiX run's time limit moves back by
    `hours` within seconds, and its SLURM job's limit is raised by as much (SLURM may refuse: the answer says so,
    and an admin can raise it). Use it when study_status warns that CalculiX will not finish in time."""
    return api.extend(request, hours, names)


@server.tool()
@GUARD.tool(ToolError)
def cancel(request: str, names: list[str] | None = None) -> dict:
    """Stops a request's studies (or the named ones), wherever they run: SLURM jobs are cancelled, and studies in a
    local worker pool (of any server) stop within a second (CalculiX and its threads end). Returns their states."""
    return api.cancel(request, names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", type=int, help="serve MCP over streamable HTTP on this port (default: stdio)")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    GUARD.serve()     # may refuse stale calls and exit; the install watch, the log in $FEAT_OUT/.server, signals
    if a.http is None:
        server.run()
    elif _V2:
        server.run(transport="streamable-http", host=a.host, port=a.http)
    else:
        server.settings.host, server.settings.port = a.host, a.http
        server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
