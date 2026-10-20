"""cli.py — the `feat` command line. Every command prints JSON.
Version: 2026.10.20.1

    feat status                          versions; CalculiX and Gmsh found or not
    feat format > study_format.json      the study format
    feat validate studies.json           static checks
    feat run studies.json [--solver S]   mesh, solve, check (S: auto | ccx | ref); waits for all
    feat status REQUEST                  results of a request folder (finished and pending studies)
    feat deck studies.json NAME          write the CalculiX deck of one study (no solve)
    feat bench [NAMES...] [--solver S]   the benchmark studies with closed-form answers (waits for all)
    feat bench-status BENCH              a benchmark request's passed, failed and pending cases
    feat run-study REQUEST NAME          run one study of a request folder (what a SLURM job runs)
    feat jobs [--limit N] [--active]     recent requests with SLURM jobs, their states, the queue
    feat cancel REQUEST [NAMES...]       stop a request's studies

'-' reads the studies from stdin.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import api


def _load(path: str) -> dict:
    return json.load(sys.stdin) if path == "-" else json.loads(open(path).read())


def run_study(request: str, name: str) -> int:
    """One study of a request folder, as a pool worker runs it: what a SLURM job's job.sh executes.
    Threads from SLURM_CPUS_PER_TASK (else FEAT_PROCS). Exit 0 when it wrote <name>.json, even for a
    failed study (the result says why); non-zero only on a crash (job.log has the traceback)."""
    import os
    from pathlib import Path

    from . import executor
    from .runner import run_study as _run

    folder = Path(request).expanduser().resolve()
    meta = json.loads((folder / "request.json").read_text())
    study = json.loads((folder / "studies.json").read_text())[name]
    os.environ.setdefault("FEAT_OUT", str(folder.parent))   # timing records next to the request
    n = os.environ.get("SLURM_CPUS_PER_TASK") or str(executor.procs_for(study))
    os.environ["OMP_NUM_THREADS"] = os.environ["CCX_NPROC_EQUATION_SOLVER"] = str(n)
    _run(name, study, folder / name, meta.get("solver", "ccx"), float(meta.get("timeout", executor.timeout_s())))
    return 0 if (folder / name / f"{name}.json").is_file() else 1


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="feat", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status")
    p.add_argument("request", nargs="?")
    sub.add_parser("format")
    p = sub.add_parser("validate")
    p.add_argument("studies")
    p = sub.add_parser("run")
    p.add_argument("studies")
    p.add_argument("--solver", default=None)
    p = sub.add_parser("deck")
    p.add_argument("studies")
    p.add_argument("name")
    p = sub.add_parser("bench")
    p.add_argument("names", nargs="*")
    p.add_argument("--solver", default=None)
    p = sub.add_parser("bench-status")
    p.add_argument("bench")
    p = sub.add_parser("run-study")
    p.add_argument("request")
    p.add_argument("name")
    p = sub.add_parser("jobs")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--active", action="store_true")
    p = sub.add_parser("cancel")
    p.add_argument("request")
    p.add_argument("names", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "run-study":
        sys.exit(run_study(a.request, a.name))
    if a.cmd == "status":
        out = api.status(a.request) if a.request else api.tool_status()
    elif a.cmd == "format":
        out = api.study_format()
    elif a.cmd == "validate":
        out = api.validate(_load(a.studies))
    elif a.cmd == "run":
        out = api.run(_load(a.studies), a.solver)
    elif a.cmd == "deck":
        st = _load(a.studies)
        out = api.deck(st[a.name], a.name)
    elif a.cmd == "bench-status":
        out = api.bench_status(a.bench)
    elif a.cmd == "jobs":
        out = api.jobs(a.limit, a.active)
    elif a.cmd == "cancel":
        out = api.cancel(a.request, a.names or None)
    else:
        out = api.bench(a.names or None, a.solver)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
