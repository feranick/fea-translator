"""mcp_server.py — MCP server: lets LLM-based tools run FEA studies with Gmsh and CalculiX.
Version: 2026.10.18.2

Tools (all JSON):
  study_format      the study format: analyses, materials, boundary conditions, face selectors. Read it first.
  validate_studies  static checks of {name: study} (fast; fix errors before running)
  run_studies       meshes, solves and checks {name: study}: results, checks, files (returns within ~45 s;
                    long studies stay pending)
  study_status      collects the pending studies of a run_studies request (progress, time left, advice)
  get_deck          the CalculiX input deck of one study, as text
  run_benchmarks    the benchmark studies with closed-form answers (is the toolchain right?)
  get_file          a written file as base64 (for callers on another machine)
  jobs              recent requests with SLURM jobs (also from other conversations) and the queue
  cancel            stops a request's studies

Run:  feat-mcp                       (stdio)
      feat-mcp --http 8767 [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse

from . import api

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
    _V2 = True
except ImportError:  # 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    _V2 = False

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
conversations); cancel(request) stops them.
Units: mm, N, MPa, kg/m3, W, degrees C; V, A, ohm m, kV/mm."""

server = _Server("fea-translator", instructions=INSTRUCTIONS)


@server.tool()
def study_format() -> dict:
    """The study format: analyses, materials, boundary-condition kinds with their parameters,
    face selectors, conventions, an example; and which solver and mesher are installed."""
    return api.study_format()


@server.tool()
def validate_studies(studies: dict) -> dict:
    """Static checks of studies given as {name: study}. Returns per study ok, errors, warnings."""
    return api.validate(studies)


@server.tool()
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
def study_status(request: str, wait_s: float = 45) -> dict:
    """The studies of a run_studies request (its 'request' path): finished ones with their results,
    the others in 'pending' with progress and time left. Waits up to wait_s seconds (at most 50) for
    them to finish. Do not poll long runs: follow 'next'."""
    return api.status(request, min(max(float(wait_s), 0.0), 50.0))


@server.tool()
def get_deck(study: dict, name: str = "study") -> dict:
    """Meshes the part and returns the CalculiX input deck (.inp) of one study, without solving."""
    return api.deck(study, name)


@server.tool()
def run_benchmarks(names: list[str] | None = None, solver: str = "auto") -> dict:
    """Runs the benchmark studies with closed-form answers (axial bar, cantilever, natural
    frequency, Euler column, conduction, Lame ring, plate with a hole) and reports pass or fail."""
    return api.bench(names, solver)


@server.tool()
def get_file(path: str, max_mb: float = 20) -> dict:
    """A file written by run_studies (a path from its result) as base64."""
    return api.read_file(path, max_mb)


@server.tool()
def jobs(limit: int = 20, active: bool = False) -> dict:
    """Recent requests that have SLURM jobs, with each study's state (queued, running, done, timed_out,
    out_of_memory, cancelled, failed), job id and expected time; the translator's queue and the machine's
    CPUs. active=True: only requests with queued or running studies. Finds runs started in other
    conversations too."""
    return api.jobs(limit, active)


@server.tool()
def cancel(request: str, names: list[str] | None = None) -> dict:
    """Stops a request's studies (or the named ones): their SLURM jobs are cancelled. Returns their states."""
    return api.cancel(request, names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", type=int, help="serve MCP over streamable HTTP on this port (default: stdio)")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    if a.http is None:
        server.run()
    elif _V2:
        server.run(transport="streamable-http", host=a.host, port=a.http)
    else:
        server.settings.host, server.settings.port = a.host, a.http
        server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
