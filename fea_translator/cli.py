"""cli.py — the `feat` command line. Every command prints JSON.
Version: 2026.10.16.1

    feat status                          versions; CalculiX and Gmsh found or not
    feat format > study_format.json      the study format
    feat validate studies.json           static checks
    feat run studies.json [--solver S]   mesh, solve, check (S: auto | ccx | ref)
    feat deck studies.json NAME          write the CalculiX deck of one study (no solve)
    feat bench [NAMES...] [--solver S]   the benchmark studies with closed-form answers

'-' reads the studies from stdin.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import api


def _load(path: str) -> dict:
    return json.load(sys.stdin) if path == "-" else json.loads(open(path).read())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="feat", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
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
    a = ap.parse_args(argv)
    if a.cmd == "status":
        out = api.status()
    elif a.cmd == "format":
        out = api.study_format()
    elif a.cmd == "validate":
        out = api.validate(_load(a.studies))
    elif a.cmd == "run":
        out = api.run(_load(a.studies), a.solver)
    elif a.cmd == "deck":
        st = _load(a.studies)
        out = api.deck(st[a.name], a.name)
    else:
        out = api.bench(a.names or None, a.solver)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
