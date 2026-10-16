"""runner.py — one study from start to end: part, mesh, deck, solver, results, checks.
Version: 2026.10.16.1

    run_study(name, study, folder, solver="ccx") -> results dict (also written as <name>.json)

Solver "ccx" runs CalculiX (FEAT_CCX, else ccx or ccx_2.* on PATH); "ref" the package's reference
solver (static, modal, thermal; small meshes). Checks on every run: static equilibrium (reactions
balance the loads), heat balance (thermal), the study's `expect` list, and with converge: true the
change of the main results on a finer mesh.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import __version__, deck, results
from .meshdata import Mesh, MeshError
from .mesher import mesh_part
from .study import bc_name, material_of


class SolverError(Exception):
    pass


def find_ccx() -> str | None:
    env = os.environ.get("FEAT_CCX")
    if env:
        return env
    for name in ("ccx", "ccx_2.23", "ccx_2.22", "ccx_2.21", "ccx_2.20", "ccx_2.19"):
        p = shutil.which(name)
        if p:
            return p
    for pat in ("/opt/homebrew/bin/ccx*", "/usr/local/bin/ccx*", "/usr/bin/ccx*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def run_ccx(inp: Path, timeout: float) -> str:
    exe = find_ccx()
    if not exe:
        raise SolverError("CalculiX (ccx) not found: install it or set FEAT_CCX to its path (or use solver 'ref')")
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", str(min(4, os.cpu_count() or 1)))
    env.setdefault("CCX_NPROC_EQUATION_SOLVER", env["OMP_NUM_THREADS"])
    p = subprocess.run([exe, "-i", inp.stem], cwd=inp.parent, capture_output=True, text=True, timeout=timeout, env=env)
    log = p.stdout + p.stderr
    (inp.parent / f"{inp.stem}.log").write_text(log)
    errs = [ln.strip() for ln in log.splitlines() if "*ERROR" in ln.upper()]
    if errs or p.returncode != 0 or not inp.with_suffix(".dat").is_file():
        raise SolverError("CalculiX failed: " + ("; ".join(errs[:5]) if errs else log[-800:]))
    return log


def _node_values(dat: dict, what: str, n: int, width: int) -> np.ndarray:
    b = results.block(dat, what, "NALL")
    if b is None:
        raise SolverError(f"no '{what}' block in the .dat file")
    a = np.full((n, width), np.nan)
    for r in b["rows"]:
        a[int(r[0]) - 1] = r[1:1 + width]
    return a


def _face_mean(mesh: Mesh, tags, values: np.ndarray) -> np.ndarray:
    """Area-weighted mean over the faces of a nodal field (exact for the element's interpolation)."""
    acc, area = 0.0, 0.0
    for t in tags:
        g = mesh.groups[t]
        if g.mids is not None:  # quadratic triangle: the integral of N is 0 at corners, A/3 at mid-edge nodes
            v = values[g.mids].mean(axis=1)
        else:
            v = values[g.corners].mean(axis=1)
        acc = acc + (g.area[:, None] * v.reshape(len(g.area), -1)).sum(axis=0)
        area += g.total_area
    return np.atleast_1d(acc / area)


def _get(res: dict, path: str):
    cur = res
    for key in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(key)]
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            raise KeyError(path)
    return float(cur)


def evaluate(study: dict, mesh: Mesh, plan: dict, dat: dict, frd: dict | None) -> dict:
    an = study["analysis"]
    n = len(mesh.nodes)
    mat = material_of(study)
    out: dict = {"analysis": an}
    sets = plan["sets"]
    if an == "static":
        U = _node_values(dat, "displacements", n, 3)
        mag = np.linalg.norm(U, axis=1)
        i = int(np.nanargmax(mag))
        out["max_displacement"] = round(float(mag[i]), 9)
        out["max_displacement_at"] = np.round(mesh.nodes[i], 4).tolist()
        out["max_displacement_components"] = {c: round(float(np.nanmax(np.abs(U[:, k]))), 9) for k, c in enumerate("xyz")}
        sb = results.block(dat, "stresses", "EALL")
        if sb:
            S = np.array([r[2:8] for r in sb["rows"]])
            vm = results.von_mises(S)
            j = int(np.argmax(vm))
            e = int(sb["rows"][j][0]) - 1
            out["max_von_mises"] = round(float(vm[j]), 6)
            out["max_von_mises_at"] = np.round(mesh.nodes[mesh.tets[e, :4]].mean(axis=0), 4).tolist()
        if frd and "STRESS" in frd:
            vals = np.array([v[:6] for v in frd["STRESS"].values()])
            out["max_von_mises_nodal"] = round(float(results.von_mises(vals).max()), 6)
        peak = out.get("max_von_mises_nodal", out.get("max_von_mises"))
        if peak and mat.get("yield"):
            out["safety_factor"] = round(float(mat["yield"]) / peak, 4)
        out["reactions"] = {}
        total = np.zeros(3)
        taken = [set(), set(), set()]  # nodes whose reaction in x, y, z is already counted
        for s in plan["supports"]:
            b = results.block(dat, "forces", sets[s]["nset"])
            if not b or not b["rows"]:
                raise SolverError(f"no reaction forces for {s} in the .dat file")
            # CalculiX's RF is the external force at a node: the reaction plus the loads applied there.
            # Each constrained direction of a node counts once, for the first support that holds it.
            r = np.zeros(3)
            for row in b["rows"]:
                nd = int(row[0])
                f = np.array(row[1:4]) - plan["nodal_load"][nd - 1]
                for k in plan["dofs"][s]:
                    if nd not in taken[k]:
                        taken[k].add(nd)
                        r[k] += f[k]
            total += r
            out["reactions"][s] = {"fx": round(float(r[0]), 6), "fy": round(float(r[1]), 6), "fz": round(float(r[2]), 6)}
        applied = np.array(plan["applied_force"])
        scale = max(np.linalg.norm(applied), max((np.linalg.norm(list(v.values())) for v in out["reactions"].values()), default=0), 1e-9)
        out["equilibrium_error"] = round(float(np.linalg.norm(total + applied) / scale), 8)
        out["applied_force"] = np.round(applied, 6).tolist()
        out["faces"] = {}
        for bname, tags in plan["tags"].items():
            mu = _face_mean(mesh, tags, U)
            nodes = mesh.nodes_of(tags)
            key = bname.replace("probe:", "")
            tgt = out.setdefault("probes", {}) if bname.startswith("probe:") else out["faces"]
            tgt[key] = {"mean_u": {"ux": round(float(mu[0]), 9), "uy": round(float(mu[1]), 9), "uz": round(float(mu[2]), 9)},
                        "max_u": round(float(np.nanmax(mag[nodes])), 9), "area": round(sets[bname]["area"], 6)}
        out["mass_kg"] = round(float(mat.get("density", 0)) * mesh.volume() * 1e-9, 6)
    elif an == "modal":
        out["frequencies"] = [round(e["hz"], 6) for e in dat["eigen"]]
        if not out["frequencies"]:
            raise SolverError("no frequencies in the .dat file")
        out["mass_kg"] = round(float(mat.get("density", 0)) * mesh.volume() * 1e-9, 6)
    elif an == "buckling":
        f = [round(e["factor"], 6) for e in dat["buckling"]]
        if not f:
            raise SolverError("no buckling factors in the .dat file")
        out["buckling_factors"] = f
        pos = [x for x in f if x > 0]
        out["critical_factor"] = min(pos) if pos else None
    elif an == "thermal":
        T = _node_values(dat, "temperatures", n, 1)[:, 0]
        out["T_max"], out["T_min"] = round(float(np.nanmax(T)), 6), round(float(np.nanmin(T)), 6)
        out["T_max_at"] = np.round(mesh.nodes[int(np.nanargmax(T))], 4).tolist()
        out["heat_flow"] = {}
        into = plan["heat_in_W"]
        film_heat = plan["nodal_heat"] + sum((mesh.film_nodes(fl["tags"], fl["h"], fl["T_inf"], T) for fl in plan["films"]),
                                             np.zeros(n))
        taken = set()
        for s in plan["supports"]:
            b = [x for x in dat["blocks"] if x["set"] == sets[s]["nset"] and x["rows"]]
            if not b:
                raise SolverError(f"no heat flow for {s} in the .dat file")
            # RFL: the heat at a node, applied fluxes and films there included; keep the boundary's own
            q = 0.0
            for row in b[-1]["rows"]:
                nd = int(row[0])
                if nd not in taken:
                    taken.add(nd)
                    q += row[1] - film_heat[nd - 1]
            q /= 1e3  # mW -> W, heat supplied through the boundary
            out["heat_flow"][s] = round(q, 9)
            into += q
        film_out = 0.0
        for fl in plan["films"]:
            tf = float(_face_mean(mesh, fl["tags"], T[:, None])[0])
            q = fl["h_W"] * 1e-6 * mesh.area_of(fl["tags"]) * (tf - fl["T_inf"])  # W/(m2 K) * m2 * K
            out["heat_flow"][fl["name"]] = round(-q, 9)
            film_out += q
        scale = max(sum(abs(v) for v in out["heat_flow"].values()) + abs(plan["heat_in_W"]), 1e-12)
        out["heat_balance_error"] = round(abs(into - film_out) / scale, 8)
        out["faces"] = {}
        for bname, tags in plan["tags"].items():
            key = bname.replace("probe:", "")
            tgt = out.setdefault("probes", {}) if bname.startswith("probe:") else out["faces"]
            nodes = mesh.nodes_of(tags)
            tgt[key] = {"mean_T": round(float(_face_mean(mesh, tags, T[:, None])[0]), 6),
                        "max_T": round(float(np.nanmax(T[nodes])), 6), "min_T": round(float(np.nanmin(T[nodes])), 6)}
    return out


def check_expectations(study: dict, res: dict) -> list[dict]:
    out = []
    for ex in study.get("expect", []):
        tol = float(ex.get("tol", 0.02))
        try:
            got = _get(res, ex["quantity"])
            err = abs(got - ex["value"]) / max(abs(ex["value"]), 1e-12)
            out.append({"quantity": ex["quantity"], "value": ex["value"], "got": got, "rel_error": round(err, 6),
                        "tol": tol, "pass": err <= tol})
        except (KeyError, IndexError, ValueError, TypeError):
            out.append({"quantity": ex["quantity"], "value": ex["value"], "got": None, "pass": False,
                        "why": "no such quantity in the results"})
    return out


MAIN = {"static": ("max_displacement", "max_von_mises"), "modal": ("frequencies.0",),
        "buckling": ("critical_factor",), "thermal": ("T_max", "T_min")}


def solve_once(name: str, study: dict, folder: Path, solver: str, timeout: float, size_factor: float = 1.0) -> dict:
    t0 = time.perf_counter()
    mesh, minfo = mesh_part(study, folder, size_factor)
    t1 = time.perf_counter()
    text, plan = deck.write(name, study, mesh)
    inp = folder / f"{name}.inp"
    inp.write_text(text)
    if solver == "ref":
        from . import refsolver
        refsolver.run(inp)
    else:
        run_ccx(inp, timeout)
    t2 = time.perf_counter()
    dat = results.parse_dat(inp.with_suffix(".dat").read_text())
    frd_path = inp.with_suffix(".frd")
    frd = results.parse_frd(frd_path.read_text()) if frd_path.is_file() else None
    res = evaluate(study, mesh, plan, dat, frd)
    res["mesh"] = minfo
    res["seconds"] = {"mesh": round(t1 - t0, 3), "solve": round(t2 - t1, 3), "total": round(time.perf_counter() - t0, 3)}
    res["files"] = {k: str(p) for k, p in (("inp", inp), ("dat", inp.with_suffix(".dat")), ("frd", frd_path),
                                            ("msh", folder / "mesh.msh"), ("step", Path(minfo.get("step", "")))) if p.is_file()}
    return res


def run_study(name: str, study: dict, folder: Path, solver: str = "ccx", timeout: float = 600) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        res = solve_once(name, study, folder, solver, timeout)
        res["ok"] = True
        if study.get("converge"):
            fine_dir = folder / f"{name}_fine"
            fine_dir.mkdir(exist_ok=True)
            fine = solve_once(name, study, fine_dir, solver, timeout, size_factor=0.6)
            conv = {}
            for q in MAIN[study["analysis"]]:
                try:
                    a, b = _get(res, q), _get(fine, q)
                    conv[q] = {"mesh": a, "finer_mesh": b, "rel_change": round(abs(b - a) / max(abs(b), 1e-12), 6)}
                except KeyError:
                    pass
            res["convergence"] = conv
        res["checks"] = check_expectations(study, res)
        warn = []
        if res.get("equilibrium_error", 0) > 1e-3:
            warn.append(f"reactions do not balance the loads (error {res['equilibrium_error']:.2g})")
        if res.get("heat_balance_error", 0) > 1e-2:
            warn.append(f"heat balance error {res['heat_balance_error']:.2g}")
        for q, c in res.get("convergence", {}).items():
            if c["rel_change"] > 0.05:
                warn.append(f"{q} changes {100 * c['rel_change']:.1f} % on a finer mesh: refine the mesh")
        if warn:
            res["warnings"] = warn
        res["passed"] = all(c["pass"] for c in res["checks"]) if res["checks"] else None
    except (MeshError, SolverError, subprocess.TimeoutExpired) as exc:
        res = {"ok": False, "why": type(exc).__name__, "error": str(exc)[:2000]}
    except Exception as exc:  # noqa: BLE001
        res = {"ok": False, "why": type(exc).__name__, "error": str(exc)[:2000]}
    res["translator"] = __version__
    res["solver"] = solver
    (folder / f"{name}.json").write_text(json.dumps(res, indent=1))
    return res


def bc_names(study: dict) -> list[str]:
    return [bc_name(i, b) for i, b in enumerate(study.get("bcs", []))]
