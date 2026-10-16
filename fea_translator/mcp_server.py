"""mcp_server.py — MCP server: lets LLM-based tools run FEA studies with Gmsh and CalculiX.
Version: 2026.10.16.1

Tools (all JSON):
  study_format      the study format: analyses, materials, boundary conditions, face selectors. Read it first.
  validate_studies  static checks of {name: study} (fast; fix errors before running)
  run_studies       meshes, solves and checks {name: study}: results, checks, files
  get_deck          the CalculiX input deck of one study, as text
  run_benchmarks    the benchmark studies with closed-form answers (is the toolchain right?)
  get_file          a written file as base64 (for callers on another machine)

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
and steady heat transfer (temperatures, heat flows). A study names the part (a STEP file or a
build123d-translator goal), the material, the analysis, the supports and loads on faces chosen by
normal, point or cylinder radius, and optional expected values to check. Gmsh meshes the part,
CalculiX solves it.
Workflow: call study_format once and follow its conventions; write one study per analysis;
validate_studies and fix every error; run_studies. Check equilibrium_error and warnings, and rerun
with converge: true (or a smaller mesh size) before trusting peak stresses.
Units: mm, N, MPa, kg/m3, W, degrees C."""

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
def run_studies(studies: dict, solver: str = "auto") -> dict:
    """Meshes, solves and checks each study. Per study: results (max displacement and von Mises
    stress, reactions, frequencies, buckling factors, temperatures, heat flows, probes), checks
    against 'expect', equilibrium, warnings, and files (.inp deck, .frd and .dat results, mesh, STEP)."""
    return api.run(studies, solver)


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
