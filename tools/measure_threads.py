#!/usr/bin/env python3
"""measure_threads.py — how CalculiX scales with threads on this machine, as SLURM jobs.
Version: 2026.10.19.1

Runs, one after another on an otherwise idle machine (as cfdt measured on mochi, docs/EXECUTION.md
"Sizing jobs and timing"):
  - a large box cantilever (80 x 16 x 16 cells, ~175k nodes, ~525k DOFs) with the direct solver (SPOOLES)
    at 1, 6, 10 and 19 threads, and with the iterative solver at 1 and 10 threads;
  - a small one (50 x 10 x 10 cells, ~45k nodes) with the direct solver at 1, 6 and 10 threads;
  - two large direct-solver studies side by side at 10 threads each.
For each: solve seconds, CalculiX's peak memory, speed-up over 1 thread; for the pair, the throughput
against running them one after the other at 19 threads. The numbers set the thread cap and the memory
formula for SLURM jobs (executor.py) and FEAT_PROCS for llmt tune.

Run on the SLURM machine, with nothing else running:
    bin/run python3 fea-translator/tools/measure_threads.py            (from ~/Software/llm-translators)
    ... --quick       skips the slowest run (large, direct, 1 thread)
    ... --threads 1,6,10,19 --pair 10 --mem 24G --timeout 1800
Writes measure_<time>.json and .md under FEAT_OUT and prints the table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def study(div, threads, method):
    return {"part": {"box": [100, 10, 10], "divisions": div, "origin": [0, -5, -5]}, "material": "steel",
            "analysis": "static", "procs": threads, "queue": True, "equation_solver": method,
            "bcs": [{"kind": "fixed", "name": "wall", "faces": {"nx": -1}},
                    {"kind": "force", "name": "tip", "faces": {"nx": 1}, "fz": -100}]}


def busy() -> list[str]:
    p = subprocess.run(["squeue", "-h", "-o", "%i %j %T %C"], capture_output=True, text=True)
    return [ln for ln in p.stdout.splitlines() if ln.strip()]


def run(api, studies: dict, solver: str = "ccx") -> dict:
    t0 = time.time()
    r = api.run(studies, solver, wait=None)       # waits for the SLURM jobs (this is a script, not an MCP call)
    out = {}
    for n, res in r["studies"].items():
        s = res.get("seconds", {})
        out[n] = {"ok": res.get("ok"), "state": res.get("state"), "why": res.get("why") or res.get("error"),
                  "nodes": res.get("mesh", {}).get("nodes"), "equation_solver": res.get("mesh", {}).get("equation_solver"),
                  "threads": s.get("ccx_threads"), "mesh_s": s.get("mesh"), "solve_s": s.get("solve"),
                  "rss_gb": round(s["ccx_peak_rss_mb"] / 1024, 2) if s.get("ccx_peak_rss_mb") else None,
                  "tip_uz": res.get("faces", {}).get("tip", {}).get("mean_u", {}).get("uz"),
                  "equilibrium_error": res.get("equilibrium_error"), "reruns": res.get("solver_reruns")}
    print(f"  {', '.join(studies)}: {time.time() - t0:.0f} s wall", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", default="1,6,10,19", help="threads for the large direct runs")
    ap.add_argument("--small", default="1,6,10", help="threads for the small direct runs")
    ap.add_argument("--iterative", default="1,10", help="threads for the large iterative runs")
    ap.add_argument("--pair", type=int, default=10, help="threads of each of the two side-by-side runs (0: skip)")
    ap.add_argument("--mem", default="24G", help="memory per job (FEAT_SLURM_MEM)")
    ap.add_argument("--timeout", default="1800", help="seconds per CalculiX solve (FEAT_TIMEOUT)")
    ap.add_argument("--quick", action="store_true", help="skip the large direct run at 1 thread")
    ap.add_argument("--force", action="store_true", help="run even if other jobs are in the queue")
    ap.add_argument("--solver", default="ccx", help=argparse.SUPPRESS)        # tests: ref
    ap.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)    # tests: small meshes
    a = ap.parse_args()
    os.environ.update(FEAT_EXECUTOR="slurm", FEAT_SLURM_MEM=a.mem, FEAT_TIMEOUT=a.timeout)
    os.environ.pop("FEAT_SLURM_ARGS", None)
    from fea_translator import __version__, api

    others = busy()
    if others and not a.force:
        sys.exit("other jobs are in the SLURM queue (timings would be skewed); wait, or use --force:\n  " + "\n  ".join(others))
    large, small = ([10, 2, 2], [5, 2, 2]) if a.tiny else ([80, 16, 16], [50, 10, 10])
    ths = [int(x) for x in a.threads.split(",") if x]
    if a.quick:
        ths = [t for t in ths if t > 1]
    plan = ([("large", "direct", t) for t in ths] + [("large", "iterative", int(t)) for t in a.iterative.split(",") if t]
            + [("small", "direct", int(t)) for t in a.small.split(",") if t])
    rows = []
    print(f"fea-translator {__version__}: {len(plan)} single runs" + (f" + a pair at {a.pair} threads" if a.pair else ""), flush=True)
    for size, method, t in plan:
        name = f"{size}_{method}_{t}"
        r = run(api, {name: study(large if size == "large" else small, t, method)}, a.solver)[name]
        rows.append({"case": name, "size": size, "method": method, "requested": t, **r})
    pair = None
    if a.pair:
        r = run(api, {f"pair{i}_{a.pair}": study(large, a.pair, "direct") for i in (1, 2)}, a.solver)
        pair = {"threads": a.pair, "runs": r}
    # speed-ups against 1 thread, and the pair against two runs one after the other at the most threads
    for row in rows:
        base = next((b for b in rows if b["size"] == row["size"] and b["method"] == row["method"] and b["requested"] == 1
                     and b.get("solve_s")), None)
        row["speedup"] = round(base["solve_s"] / row["solve_s"], 2) if base and row.get("solve_s") else None
    if pair:
        best = max((b for b in rows if b["size"] == "large" and b["method"] == "direct" and b.get("solve_s")),
                   key=lambda b: b["requested"], default=None)
        times = [v["solve_s"] for v in pair["runs"].values() if v.get("solve_s")]
        if best and len(times) == 2:
            pair["serial_two_s"] = round(2 * best["solve_s"], 1)
            pair["pair_s"] = round(max(times), 1)
            pair["throughput_gain"] = round(2 * best["solve_s"] / max(times) - 1, 3)
            pair["against_threads"] = best["requested"]
    out = Path(os.environ.get("FEAT_OUT", "~/feat_runs")).expanduser()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    res = {"translator": __version__, "host": os.uname().nodename, "when": stamp, "runs": rows, "pair": pair,
           "settings": {"mem": a.mem, "timeout": a.timeout}}
    (out / f"measure_{stamp}.json").write_text(json.dumps(res, indent=1))
    lines = ["| case | nodes | solver | threads | solve s | speed-up | peak GB | tip uz | equilibrium |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['case']} | {r.get('nodes')} | {r.get('equation_solver')} | {r.get('threads')} | {r.get('solve_s')} | "
                     f"{r.get('speedup')} | {r.get('rss_gb')} | {r.get('tip_uz')} | {r.get('equilibrium_error')} |"
                     + (f" {r['state'] or ''} {r['why'] or ''}" if not r.get("ok") else ""))
    if pair:
        for n, v in pair["runs"].items():
            lines.append(f"| {n} (together) | {v.get('nodes')} | {v.get('equation_solver')} | {v.get('threads')} | "
                         f"{v.get('solve_s')} | | {v.get('rss_gb')} | {v.get('tip_uz')} | {v.get('equilibrium_error')} |")
        if "throughput_gain" in pair:
            lines.append(f"\nPair at {pair['threads']} threads: {pair['pair_s']} s for both, against {pair['serial_two_s']} s for "
                         f"two at {pair['against_threads']} threads one after the other: throughput {100 * pair['throughput_gain']:+.0f} %.")
    md = "\n".join(lines)
    (out / f"measure_{stamp}.md").write_text(md + "\n")
    print("\n" + md + f"\n\n{out / f'measure_{stamp}.json'}")


if __name__ == "__main__":
    main()
