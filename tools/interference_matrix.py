"""interference_matrix.py — why do interference fits not converge in CalculiX? A small press fit (the
assembly_press_fit benchmark: a ring ri 20 / ro 40 on a solid shaft r 20, 10 thick, 0.02 mm radial interference,
Lame's plane-stress contact pressure 78.75 MPa) run in variants that each change one thing: the penalty, friction,
supports, slave and master, how the interference is made (clearance or geometry), contact type, the size of the
interference, the increments, the mesh. Each run is cut off by the convergence watch or a time limit.
Version: 2026.11.04.10

    python tools/interference_matrix.py [--out DIR] [--timeout S] [--only base,k1,...] [--size 6] [--reevaluate]

Run from a feat checkout with CalculiX and Gmsh (e.g. in tmux on mochi). Writes DIR/<variant>/ (deck, .sta, .cvg,
.dat, .frd), DIR/results.jsonl (one line per variant) and prints a table at the end.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fea_translator import joint  # noqa: E402
from fea_translator.runner import SolverError, Stopped, run_ccx  # noqa: E402

E, B, C, D, H = 210000.0, 20.0, 40.0, 0.02, 10.0


def lame(d: float = D) -> float:
    """Plane-stress contact pressure of a ring (b, c) on a solid shaft of the same material, radial interference d."""
    return E * d * (C ** 2 - B ** 2) / (2 * B * C ** 2)


def cyl(r, h):
    return {"goal": {"features": [{"kind": "base_cyl", "params": {"r": r, "h": h}}]}}


def ring(ri, ro, h):
    return {"goal": {"features": [{"kind": "base_ring", "params": {"ri": ri, "ro": ro, "h": h}}]}}


def study(size: float, **o) -> dict:
    """The press fit; o: shaft_r, clearance (None: none), law, friction, fixed (shaft held fully), swap."""
    shaft_r = o.get("shaft_r", B)
    contact = {"name": "fit", "faces": {"part": "ring", "cyl_r": B, "side": "inner"},
               "with": {"part": "shaft", "cyl_r": shaft_r}, "law": o.get("law", "hard"), "friction": o.get("friction", 0.2)}
    if o.get("clearance", -D) is not None:
        contact["clearance"] = o.get("clearance", -D)
    if o.get("swap"):
        contact["faces"], contact["with"] = contact["with"], contact["faces"]
    shaft_bc = {"kind": "fixed" if o.get("fixed") else "slider", "name": "shaft_base", "faces": {"part": "shaft", "nz": -1}}
    return {"analysis": "assembly", "mesh": {"size": size},
            "parts": {"shaft": {"part": cyl(shaft_r, H)}, "ring": {"part": ring(B, C, H)}},
            "bcs": [shaft_bc, {"kind": "slider", "name": "ring_base", "faces": {"part": "ring", "nz": -1}}],
            "contacts": [contact], "steps": [{"kind": "static"}]}


def variants(size: float) -> dict:
    """name: (question, study, deck edits [(old, new)], expected pressure)."""
    h = size
    return {
        "base": ("the benchmark, small: does it reproduce the failure?", study(size), [], lame()),
        "k1": ("is the penalty too stiff? E/h", study(size, law={"stiffness": E / h}), [], lame()),
        "k01": ("... 0.1 E/h", study(size, law={"stiffness": 0.1 * E / h}), [], lame()),
        "mu0": ("is it the friction? frictionless", study(size, friction=0.0), [], lame()),
        "fixed": ("are rigid-body modes the problem? the shaft held fully", study(size, fixed=True), [], lame()),
        "swap": ("does the slave side matter? shaft as slave", study(size, swap=True), [], lame()),
        "overlap": ("is it *CLEARANCE? the interference in the geometry (shaft r 20.02)",
                    study(size, shaft_r=B + D, clearance=None), [], lame()),
        "n2s": ("is it face-to-face contact? node-to-surface, interference in the geometry",
                study(size, shaft_r=B + D, clearance=None),
                [("TYPE=SURFACE TO SURFACE", "TYPE=NODE TO SURFACE")], lame()),
        "half": ("does it depend on the interference? 0.01 mm", study(size, clearance=-D / 2), [], lame(D / 2)),
        "fine_inc": ("is the ramp too coarse? increments of at most 2 %", study(size),
                     [("0.05, 1., 1e-05, 0.1", "0.01, 1., 1e-06, 0.02")], lame()),
        "size4": ("the benchmark's mesh (size 4), for reference", study(4.0), [], lame()),
    }


def sta_summary(text: str) -> dict:
    rows = joint.read_sta(text)
    ok = [r for r in rows if r[3]]
    times = []
    for line in text.splitlines():
        tk = line.split()
        if len(tk) >= 7 and tk[0].isdigit() and tk[1].isdigit() and not tk[2].endswith("U"):
            try:
                times.append(float(tk[5]))
            except ValueError:
                pass
    return {"increments_converged": len(ok), "attempts_failed": len(rows) - len(ok),
            "step_time_reached": max(times, default=0.0)}


def cvg_summary(text: str) -> dict:
    rows = joint.read_cvg(text)
    if not rows:
        return {}
    cont = [r[4] for r in rows]
    last = [r for r in rows if r[:3] == rows[-1][:3]]
    return {"iterations": len(rows), "contact_elements_first": cont[0], "contact_elements_last": cont[-1],
            "contact_elements_min": min(cont), "contact_elements_max": max(cont),
            "last_attempt": {"increment": rows[-1][1], "attempt": rows[-1][2], "iterations": len(last),
                             "contact_elements": [r[4] for r in last][-12:],
                             "residual_pct": [r[5] for r in last][-6:]}}


def run(name: str, spec, out: Path, timeout: float) -> dict:
    question, st, edits, expected = spec
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    rec = {"variant": name, "question": question, "expected_MPa": round(expected, 4)}
    t0 = time.time()
    try:
        parts = joint.mesh_parts(st, d)
        text, plan = joint.write(name, st, parts)
        for a, b in edits:
            if a not in text:
                raise ValueError(f"deck edit not found: {a!r}")
            text = text.replace(a, b)
        inp = d / f"{name}.inp"
        inp.write_text(text)
        rec.update(nodes=sum(len(p["mesh"].nodes) for p in parts), law=plan["contacts"][0]["law"])
        try:
            run_ccx(inp, timeout, monitor=joint.Watch(inp, plan))
            rec["status"] = "converged"
        except Stopped as exc:
            rec["status"], rec["why"] = "stopped", str(exc)[:300]
        except subprocess.TimeoutExpired:
            rec["status"] = "timeout"
        except SolverError as exc:
            rec["status"], rec["why"] = "error", str(exc)[:300]
        for sfx, fn in ((".sta", sta_summary), (".cvg", cvg_summary)):
            f = inp.with_suffix(sfx)
            if f.is_file():
                rec.update(fn(f.read_text(errors="replace")))
        if rec["status"] == "converged":
            res = joint.evaluate(st, parts, plan, inp.with_suffix(".dat").read_text(errors="replace"),
                                 inp.with_suffix(".frd").read_text(errors="replace"))
            c = (res["steps"][-1].get("contacts") or {}).get("fit") or {}
            rec["mean_pressure_MPa"] = c.get("mean_pressure_MPa")
            rec["closed_fraction"] = c.get("closed_fraction")
            if rec["mean_pressure_MPa"]:
                rec["pressure_error"] = round(rec["mean_pressure_MPa"] / expected - 1, 4)
            stb = res.get("stabilisation") or {}
            rec["stabilisation_fraction"] = stb.get("fraction_of_load")
    except Exception as exc:  # noqa: BLE001 (one variant must not stop the others)
        rec["status"], rec["why"] = "failed", f"{type(exc).__name__}: {exc}"[:300]
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def reevaluate(name: str, spec, out: Path) -> dict:
    """A finished variant's results again from its .dat and .frd, with this checkout's evaluation (no CalculiX):
    the mesh is made again (Gmsh gives the same one) into a scratch folder."""
    import tempfile
    question, st, edits, expected = spec
    d = out / name
    rec = {"variant": name, "expected_MPa": round(expected, 4)}
    if not (d / f"{name}.dat").is_file() or not (d / f"{name}.frd").is_file():
        return dict(rec, status="no results")
    parts = joint.mesh_parts(st, Path(tempfile.mkdtemp()))
    text, plan = joint.write(name, st, parts)
    for a, b in edits:
        text = text.replace(a, b)
    if text.splitlines()[1:] != (d / f"{name}.inp").read_text().splitlines()[1:]:
        return dict(rec, status="the mesh came out different: not comparable")
    res = joint.evaluate(st, parts, plan, (d / f"{name}.dat").read_text(errors="replace"),
                         (d / f"{name}.frd").read_text(errors="replace"))
    if not res["steps"]:
        return dict(rec, status="no step finished")
    c = (res["steps"][-1].get("contacts") or {}).get("fit") or {}
    rec.update(status="re-evaluated", step=res["steps"][-1]["step"], **{k: c.get(k) for k in (
        "mean_pressure_MPa", "mean_pressure_cpress_MPa", "force_N", "shear_N", "closed_fraction", "max_pressure_MPa",
        "force_from")})
    if c.get("mean_pressure_MPa"):
        rec["pressure_error"] = round(c["mean_pressure_MPa"] / expected - 1, 4)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path.home() / "interference_matrix"))
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per variant (900)")
    ap.add_argument("--size", type=float, default=6.0, help="mesh size of the variants (6; size4 is always 4)")
    ap.add_argument("--only", default="", help="comma-separated variant names")
    ap.add_argument("--reevaluate", action="store_true", help="no runs: the finished variants' pressures again")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    vs = variants(a.size)
    names = [n for n in (a.only.split(",") if a.only else vs) if n]
    if a.reevaluate:
        for n in names:
            print(json.dumps(reevaluate(n, vs[n], out)), flush=True)
        return 0
    log = out / "results.jsonl"
    recs = []
    for n in names:
        print(f"[{time.strftime('%H:%M:%S')}] {n}: {vs[n][0]}", flush=True)
        rec = run(n, vs[n], out, a.timeout)
        recs.append(rec)
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"    {rec['status']} in {rec['seconds']} s; reached step time {rec.get('step_time_reached')}; "
              f"pressure {rec.get('mean_pressure_MPa')} MPa (Lame {rec['expected_MPa']})" + (f"; {rec['why']}" if rec.get("why") else ""),
              flush=True)
    print("\nvariant     status      s      step_t  incs  fails  iters  contacts min-max   pressure  error")
    for r in recs:
        print(f"{r['variant']:<11} {r['status']:<10} {r['seconds']:>6} {r.get('step_time_reached', 0):>7.3f} "
              f"{r.get('increments_converged', 0):>5} {r.get('attempts_failed', 0):>6} {r.get('iterations', 0):>6} "
              f"{r.get('contact_elements_min', 0):>8}-{r.get('contact_elements_max', 0):<8} "
              f"{r.get('mean_pressure_MPa') if r.get('mean_pressure_MPa') is not None else '-':>8}  "
              f"{r.get('pressure_error') if r.get('pressure_error') is not None else '-'}")
    print(f"\n{log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
