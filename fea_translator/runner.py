"""runner.py — one study from start to end: part, mesh, deck, solver, results, checks.
Version: 2026.10.18.2

    run_study(name, study, folder, solver="ccx") -> results dict (also written as <name>.json)

Solver "ccx" runs CalculiX (FEAT_CCX, else ccx or ccx_2.* on PATH); "ref" the package's reference
solver (static, modal, thermal; small meshes). Checks on every run: static equilibrium (reactions
balance the loads), heat balance (thermal), the study's `expect` list, and with converge: true the
change of the main results on a finer mesh.

Progress: while it runs, a study writes <folder>/progress.json (step k of n: mesh, solve, and with
converge: true mesh and solve again on the finer mesh; nodes; times), and every finished solve
appends its size and times to timing.jsonl in the output root, from which api.status estimates
the time left.
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
from .study import EPS0, bc_name, material_of, to_deck_units


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


def run_ccx(inp: Path, timeout: float, serial: bool = False) -> str:
    """Runs ccx on the deck. serial: one thread (buckling: the eigenvalues then do not depend on the
    machine's load; with several threads, 1 in 4 concurrent runs of euler_column returned a wrong factor)."""
    exe = find_ccx()
    if not exe:
        raise SolverError("CalculiX (ccx) not found: install it or set FEAT_CCX to its path (or use solver 'ref')")
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", str(min(4, os.cpu_count() or 1)))
    env.setdefault("CCX_NPROC_EQUATION_SOLVER", env["OMP_NUM_THREADS"])
    if serial:
        env["OMP_NUM_THREADS"] = env["CCX_NPROC_EQUATION_SOLVER"] = "1"
    # CalculiX is OpenMP-parallel and its eigenvalue solver (ARPACK) calls BLAS inside it; a threaded
    # OpenBLAS called that way can return wrong results (OpenBLAS's own warning), most often when several
    # jobs run at once: the pump test got a buckling factor of 10.07 for 674.7. One BLAS thread per job.
    for k in ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[k] = "1"
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
        v = values[g.fnodes]                                  # (T, n, k)
        acc = acc + np.einsum("ta,tak->k", g.wN, v.reshape(v.shape[0], v.shape[1], -1))
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


def evaluate(study: dict, mesh: Mesh, plan: dict, dat: dict, frd: dict | None, band: float | None = None) -> dict:
    """Results of one solved study. band (mm): peaks 'off_bc' leave out the material within this
    distance of support and load nodes (default here: the median element edge; runs pass two
    nominal element sizes)."""
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
        # elements touching a support or a load: fixed boundaries and point-like loads make stresses
        # singular there, so peaks are also reported away from them
        bc_names = [s for s in plan["supports"] + plan["loads"] if s in sets]
        bc_node = np.zeros(n, bool)
        owner = {}
        for s in bc_names:
            idx = np.array(sets[s]["nodes"]) - 1
            bc_node[idx] = True
            for k in idx[:]:
                owner.setdefault(int(k), s)
        touch = bc_node[mesh.tets].any(axis=1)                     # (E,) element touches a bc
        # a fixed physical band around the bcs (the same on every mesh, so it converges)
        b = float(band) if band else float(np.median(np.linalg.norm(
            mesh.nodes[mesh.tets[:, 1]] - mesh.nodes[mesh.tets[:, 0]], axis=1)))
        out["off_bc_band_mm"] = round(b, 4)
        if bc_node.any():
            from scipy.spatial import cKDTree

            tree = cKDTree(mesh.nodes[bc_node])
            near = tree.query(mesh.nodes, distance_upper_bound=b)[0] <= b          # nodes within the band
            cen = mesh.nodes[mesh.tets[:, :4]].mean(axis=1)
            band_el = (tree.query(cen, distance_upper_bound=b)[0] <= b) | touch
            near |= np.isin(np.arange(n), mesh.tets[touch].ravel())
        else:
            near, band_el = np.zeros(n, bool), touch
        sb = results.block(dat, "stresses", "EALL")
        peaks = {}
        if sb:
            rows = np.array(sb["rows"])
            el = rows[:, 0].astype(int) - 1
            vm = results.von_mises(rows[:, 2:8])
            j = int(np.argmax(vm))
            out["max_von_mises"] = round(float(vm[j]), 6)
            out["max_von_mises_at"] = np.round(mesh.nodes[mesh.tets[el[j], :4]].mean(axis=0), 4).tolist()
            peaks["ip"] = (float(vm[j]), bool(touch[el[j]]), mesh.tets[el[j]])
            off = ~band_el[el]
            if off.any():
                k = int(np.argmax(np.where(off, vm, -1)))
                out["max_von_mises_off_bc"] = round(float(vm[k]), 6)
                out["max_von_mises_off_bc_at"] = np.round(mesh.nodes[mesh.tets[el[k], :4]].mean(axis=0), 4).tolist()
        if frd and "STRESS" in frd:
            ids = np.array(list(frd["STRESS"].keys())) - 1
            vmn = results.von_mises(np.array([v[:6] for v in frd["STRESS"].values()]))
            j = int(np.argmax(vmn))
            out["max_von_mises_nodal"] = round(float(vmn[j]), 6)
            out["max_von_mises_nodal_at"] = np.round(mesh.nodes[ids[j]], 4).tolist()
            peaks["nodal"] = (float(vmn[j]), bool(near[ids[j]]), np.array([ids[j]]))
            off = ~near[ids]
            if off.any():
                k = int(np.argmax(np.where(off, vmn, -1)))
                out["max_von_mises_nodal_off_bc"] = round(float(vmn[k]), 6)
        if peaks:
            # the larger of the two peaks: nodal values are extrapolated and averaged, and can be lower
            kind = max(peaks, key=lambda q: peaks[q][0])
            val, at_bc, nodes_ = peaks[kind]
            out["peak_von_mises"] = round(val, 6)
            out["peak_from"] = {"ip": "integration points", "nodal": "nodes"}[kind]
            offp_ = max(out.get("max_von_mises_off_bc", 0), out.get("max_von_mises_nodal_off_bc", 0))
            uniform = sb is not None and val <= 1.05 * float(np.median(vm))
            if at_bc and not uniform and not (offp_ and val <= 1.05 * offp_):  # a uniform stress is not a peak
                names = sorted({owner[int(q)] for q in nodes_ if int(q) in owner} or
                               {owner[int(q)] for e in np.nonzero(touch & np.isin(mesh.tets, nodes_).any(axis=1))[0]
                                for q in mesh.tets[e] if int(q) in owner})
                out["peak_at_bc"] = names or True
            offp = max(out.get("max_von_mises_off_bc", 0), out.get("max_von_mises_nodal_off_bc", 0))
            if offp:
                out["peak_von_mises_off_bc"] = round(offp, 6)
            if mat.get("yield"):
                out["safety_factor"] = round(float(mat["yield"]) / val, 4)
                if offp:
                    out["safety_factor_off_bc"] = round(float(mat["yield"]) / offp, 4)
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
        f = [round(e["factor"], 6) for e in dat["buckling"]][:int(study.get("modes", 3))]
        if not f:
            raise SolverError("no buckling factors in the .dat file")
        out["buckling_factors"] = f
        pos = [x for x in f if x > 0]
        out["critical_factor"] = min(pos) if pos else None
    elif an in ("electric", "electrostatic"):
        out.update(_field_results(study, mesh, plan, dat, band))
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
        # floor: a heat flow typical of the part (k L, 1 K), so a uniform temperature (no flow) reads as balanced
        typical = float(mat.get("k", 1.0)) * 1e-3 * mesh.size
        scale = max(sum(abs(v) for v in out["heat_flow"].values()) + abs(plan["heat_in_W"]), typical)
        out["heat_balance_error"] = round(abs(into - film_out) / scale, 8)
        out["faces"] = {}
        for bname, tags in plan["tags"].items():
            key = bname.replace("probe:", "")
            tgt = out.setdefault("probes", {}) if bname.startswith("probe:") else out["faces"]
            nodes = mesh.nodes_of(tags)
            tgt[key] = {"mean_T": round(float(_face_mean(mesh, tags, T[:, None])[0]), 6),
                        "max_T": round(float(np.nanmax(T[nodes])), 6), "min_T": round(float(np.nanmin(T[nodes])), 6)}
    return out


def _field_results(study: dict, mesh: Mesh, plan: dict, dat: dict, band: float | None) -> dict:
    """Electric (DC current) or electrostatic results of a field solved as heat transfer: the potential is
    the temperature; the heat flux is the current density (electric, A/mm2) or eps_r grad(phi)
    (electrostatic); the heat a temperature boundary supplies is the current (A) or charge / eps0."""
    an = study["analysis"]
    n = len(mesh.nodes)
    mat = material_of(study)
    sets = plan["sets"]
    phi = _node_values(dat, "temperatures", n, 1)[:, 0]
    out = {"V_max": round(float(np.nanmax(phi)), 9), "V_min": round(float(np.nanmin(phi)), 9)}
    scale = 1.0 if an == "electric" else EPS0          # A, or C per (eps_r V mm)
    terms, taken = {}, set()
    for s in plan["supports"]:                         # voltage terminals
        b = [x for x in dat["blocks"] if x["set"] == sets[s]["nset"] and x["rows"]]
        if not b:
            raise SolverError(f"no flow through terminal {s} in the .dat file")
        q = 0.0
        for row in b[-1]["rows"]:
            nd = int(row[0])
            if nd not in taken:                        # applied current at a shared node is not this terminal's
                taken.add(nd)
                q += row[1] - plan["nodal_heat"][nd - 1]
        terms[s] = {"V": plan["terminals"][s]["V"], ("I_A" if an == "electric" else "Q_C"): q * scale}
    for s, t in plan["terminals"].items():             # current terminals: the potential they reach (face mean)
        if t["kind"] == "current":
            terms[s] = {"V": float(_face_mean(mesh, t["tags"], phi[:, None])[0]), "I_A": t["I"]}
    key = "I_A" if an == "electric" else "Q_C"
    for t in terms.values():
        t["V"] = round(t["V"], 9)
        t[key] = float(f"{t[key]:.9g}")
    out["terminals"] = terms
    total = sum(t[key] for t in terms.values())
    big = max((abs(t[key]) for t in terms.values()), default=0.0)
    out["current_balance_error" if an == "electric" else "charge_balance_error"] = round(abs(total) / max(big, 1e-300), 8)
    levels = sorted({t["V"] for t in terms.values()})
    if an == "electric":
        out["power_W"] = float(f"{sum(t['V'] * t['I_A'] for t in terms.values()):.9g}")
        if len(terms) == 2:
            (na, a), (nb, b) = sorted(terms.items(), key=lambda kv: -kv[1]["V"])
            if abs(a["I_A"]) > 0:
                out["resistance_ohm"] = float(f"{(a['V'] - b['V']) / a['I_A']:.9g}")
    else:
        out["energy_J"] = float(f"{0.5 * sum(t['V'] * t['Q_C'] for t in terms.values()):.9g}")
        if len(levels) == 2:
            hi = sum(t["Q_C"] for t in terms.values() if t["V"] == levels[1])
            out["capacitance_pF"] = float(f"{abs(hi) / (levels[1] - levels[0]) * 1e12:.9g}")
    hb = results.block(dat, "heat flux", "EALL")
    if hb and hb["rows"]:
        rows = np.array(hb["rows"])
        el = rows[:, 0].astype(int) - 1
        mag = np.linalg.norm(rows[:, 2:5], axis=1)
        if an == "electrostatic":
            mag = mag / float(mat["permittivity"]) / 1e3   # eps_r |grad phi| -> |E| in kV/mm
        j = int(np.argmax(mag))
        where = np.round(mesh.nodes[mesh.tets[el[j], :4]].mean(axis=0), 4).tolist()
        # away from the terminals / electrode edges, where fields concentrate (sharp edges are singular)
        bc_node = np.zeros(n, bool)
        for s in sets:
            if not s.startswith("probe:"):
                bc_node[np.array(sets[s]["nodes"]) - 1] = True
        off = ~bc_node[mesh.tets[el]].any(axis=1)
        if band and bc_node.any():
            from scipy.spatial import cKDTree

            cen = mesh.nodes[mesh.tets[el, :4]].mean(axis=1)
            off &= cKDTree(mesh.nodes[bc_node]).query(cen, distance_upper_bound=band)[0] > band
        offv = float(mag[off].max()) if off.any() else None
        if an == "electric":
            out["max_current_density_A_mm2"] = round(float(mag[j]), 9)
            out["max_current_density_at"] = where
            if offv is not None:
                out["max_current_density_off_terminals_A_mm2"] = round(offv, 9)
        else:
            out["max_field_kV_mm"] = round(float(mag[j]), 9)
            out["max_field_at"] = where
            if offv is not None:
                out["max_field_off_electrodes_kV_mm"] = round(offv, 9)
            if mat.get("dielectric_strength"):
                out["breakdown_safety_factor"] = round(float(mat["dielectric_strength"]) / float(mag[j]), 4)
                if offv:
                    out["breakdown_safety_factor_off_electrodes"] = round(float(mat["dielectric_strength"]) / offv, 4)
    out["faces"] = {}
    for bname, tags in plan["tags"].items():
        k = bname.replace("probe:", "")
        tgt = out.setdefault("probes", {}) if bname.startswith("probe:") else out["faces"]
        tgt[k] = {"mean_V": round(float(_face_mean(mesh, tags, phi[:, None])[0]), 9)}
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


MAIN = {"static": ("max_displacement", "peak_von_mises", "peak_von_mises_off_bc"), "modal": ("frequencies.0",),
        "electric": ("resistance_ohm", "power_W", "max_current_density_off_terminals_A_mm2"),
        "electrostatic": ("capacitance_pF", "energy_J", "max_field_off_electrodes_kV_mm"),
        "joule": ("T_max", "joule_heat_W"),
        "buckling": ("critical_factor",), "thermal": ("T_max", "T_min")}


def stats_path() -> Path:
    """Timing records of finished solves on this machine (the output root's timing.jsonl)."""
    return Path(os.environ.get("FEAT_OUT", "~/feat_runs")).expanduser() / "timing.jsonl"


def _record_timing(study: dict, solver: str, minfo: dict, mesh_s: float, solve_s: float) -> None:
    try:
        p = stats_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"analysis": study["analysis"], "solver": solver, "nodes": int(minfo.get("nodes", 0)),
               "order": int(minfo.get("order", 2)), "mesh_s": round(mesh_s, 3), "solve_s": round(solve_s, 3),
               "t": round(time.time(), 1)}
        with p.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass


class Progress:
    """Writes <folder>/progress.json as a study goes through its steps (read by api.status)."""

    def __init__(self, folder: Path, study: dict, solver: str, timeout: float):
        self.path = Path(folder) / "progress.json"
        self.steps = 4 if study.get("converge") else 2
        self.info = {"analysis": study["analysis"], "solver": solver, "steps": self.steps, "step": 0,
                     "stage": "starting", "timeout_s": timeout, "started": time.time(), "done_steps_s": []}

    def stage(self, stage: str, **extra) -> None:
        now = time.time()
        if self.info["step"] > 0:
            self.info["done_steps_s"].append(round(now - self.info["stage_started"], 3))
        self.info.update(stage=stage, step=self.info["step"] + 1, stage_started=now, **extra)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.info))
        tmp.replace(self.path)


ELECTRIC_BCS = ("voltage", "current")
THERMAL_BCS = ("temperature", "heat_flux", "heat", "convection")
MECH_BCS = ("fixed", "slider", "displacement", "force", "pressure", "gravity")


def _sub(study: dict, analysis: str, kinds) -> dict:
    """The part of a study one solve needs: its analysis and its kinds of bcs (named as in the study)."""
    bcs = [dict(b, name=bc_name(i, b)) for i, b in enumerate(study.get("bcs", [])) if b["kind"] in kinds]
    return dict(study, analysis=analysis, bcs=bcs)


def _solve(name: str, study: dict, mesh: Mesh, folder: Path, solver: str, timeout: float, extra=None):
    """Writes and solves one deck; returns (plan, dat, frd, inp)."""
    text, plan = deck.write(name, study, mesh, extra)
    inp = folder / f"{name}.inp"
    inp.write_text(text)
    if solver == "ref":
        from . import refsolver
        refsolver.run(inp)
    else:
        run_ccx(inp, timeout, serial=study["analysis"] == "buckling")
    dat = results.parse_dat(inp.with_suffix(".dat").read_text())
    frd_path = inp.with_suffix(".frd")
    frd = results.parse_frd(frd_path.read_text()) if frd_path.is_file() else None
    return plan, dat, frd, inp


def joule_heat(study: dict, mesh: Mesh, dat: dict) -> np.ndarray:
    """Joule heat per element (W/mm3) from the current density at the integration points of an electric
    solve: |J|^2 / sigma, averaged over each element's points (J in A/mm2, sigma in S/mm)."""
    b = results.block(dat, "heat flux", "EALL")
    if b is None or not b["rows"]:
        raise SolverError("no current densities in the electric solve's .dat file")
    rows = np.array(b["rows"])
    el = rows[:, 0].astype(int) - 1
    j2 = (rows[:, 2:5] ** 2).sum(axis=1)
    acc, cnt = np.zeros(len(mesh.tets)), np.zeros(len(mesh.tets))
    np.add.at(acc, el, j2)
    np.add.at(cnt, el, 1)
    sigma = to_deck_units(material_of(study))["sigma"]
    return acc / np.maximum(cnt, 1) / sigma


def solve_once(name: str, study: dict, folder: Path, solver: str, timeout: float, size_factor: float = 1.0,
               progress: Progress | None = None) -> dict:
    fine = size_factor != 1.0
    if progress:
        progress.stage("mesh_fine" if fine else "mesh")
    t0 = time.perf_counter()
    mesh, minfo = mesh_part(study, folder, size_factor)
    if size_factor != 1.0 and minfo.get("size"):
        minfo["size_nominal"] = minfo["size"] / size_factor   # the same off-bc band as the nominal mesh
    t1 = time.perf_counter()
    if progress:
        progress.stage("solve_fine" if fine else "solve", **{("nodes_fine" if fine else "nodes"): int(minfo.get("nodes", 0))})
    # default band: two elements of the nominal mesh (one element is still inside the singular zone;
    # the bracket test: a 1-element band moved 14 % on refinement, a 2-element band 0.3 %)
    h = minfo.get("size_nominal") or minfo.get("size")
    band = study.get("stress_band") or (2 * h if h else None)
    an = study["analysis"]
    files, warns, sel = {}, [], {}

    def run(sname, st, extra=None):
        plan, dat, frd, inp = _solve(sname, st, mesh, folder, solver, timeout, extra)
        for k, p in (("inp", inp), ("dat", inp.with_suffix(".dat")), ("frd", inp.with_suffix(".frd"))):
            if p.is_file():
                files[k if sname == name else f"{k}_{sname[len(name) + 1:]}"] = str(p)
        warns.extend(plan["selection_warnings"])
        sel.update(plan["selection"])
        return plan, dat, frd

    if an == "joule":
        st_e = _sub(study, "electric", ELECTRIC_BCS)
        plan_e, dat_e, frd_e = run(f"{name}_electric", st_e)
        res_e = evaluate(st_e, mesh, plan_e, dat_e, frd_e, band)
        q = joule_heat(study, mesh, dat_e)
        st_t = _sub(study, "thermal", THERMAL_BCS)
        plan_t, dat_t, frd_t = run(f"{name}_thermal", st_t, {"body_heat": q})
        res = evaluate(st_t, mesh, plan_t, dat_t, frd_t, band)
        res["analysis"] = "joule"
        res["electric"] = {k: v for k, v in res_e.items() if k not in ("analysis",)}
        jw = float((q * mesh.element_volumes()).sum())
        res["joule_heat_W"] = round(jw, 9)
        pw = res_e.get("power_W")
        if pw:
            res["joule_check_error"] = round(abs(jw - pw) / max(abs(pw), 1e-30), 6)
        T = _node_values(dat_t, "temperatures", len(mesh.nodes), 1)[:, 0]
    else:
        plan, dat, frd = run(name, study if an != "thermal" else _sub(study, "thermal", THERMAL_BCS))
        res = evaluate(study if an != "thermal" else _sub(study, "thermal", THERMAL_BCS), mesh, plan, dat, frd, band)
        T = _node_values(dat, "temperatures", len(mesh.nodes), 1)[:, 0] if an == "thermal" else None
    if study.get("stress") and an in ("thermal", "joule"):
        st_s = _sub(study, "static", MECH_BCS)
        t_ref = float(study.get("T_ref", 20.0))
        plan_s, dat_s, frd_s = run(f"{name}_stress", st_s, {"temperatures": T, "T_ref": t_ref})
        rs = evaluate(st_s, mesh, plan_s, dat_s, frd_s, band)
        rs.pop("analysis", None)
        rs["T_ref"] = t_ref
        res["stress"] = rs
    t2 = time.perf_counter()
    _record_timing(study, solver, minfo, t1 - t0, t2 - t1)
    res["selection"] = sel
    res["_selection_warnings"] = warns
    res["mesh"] = dict(minfo, equation_solver=deck.equation_solver(study, len(mesh.nodes)))
    res["seconds"] = {"mesh": round(t1 - t0, 3), "solve": round(t2 - t1, 3), "total": round(time.perf_counter() - t0, 3)}
    for k, p in (("msh", folder / "mesh.msh"), ("step", Path(minfo.get("step", "")))):
        if p.is_file():
            files[k] = str(p)
    res["files"] = files
    return res


def run_study(name: str, study: dict, folder: Path, solver: str = "ccx", timeout: float = 600) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    prog = Progress(folder, study, solver, timeout)
    try:
        res = solve_once(name, study, folder, solver, timeout, progress=prog)
        res["ok"] = True
        if study.get("converge"):
            fine_dir = folder / f"{name}_fine"
            fine_dir.mkdir(exist_ok=True)
            fine = solve_once(name, study, fine_dir, solver, timeout, size_factor=0.6, progress=prog)
            fine.pop("_selection_warnings", None)
            conv = {}
            keys = MAIN[study["analysis"]] + (("stress.max_displacement", "stress.peak_von_mises",
                                                 "stress.peak_von_mises_off_bc") if study.get("stress") else ())
            for q in keys:
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
        el = res.get("electric", res)
        for key, what in (("current_balance_error", "the terminal currents do not add up to zero"),
                          ("charge_balance_error", "the electrode charges do not add up to zero")):
            if el.get(key, 0) > 1e-3:
                warn.append(f"{what} (error {el[key]:.2g})")
        if res.get("joule_check_error", 0) > 0.03:
            warn.append(f"the Joule heat put into the thermal solve differs from the electric power by "
                        f"{100 * res['joule_check_error']:.1f} % (it is taken constant in each element): refine the mesh")
        st = res.get("stress", {})
        if st.get("equilibrium_error", 0) > 1e-3:
            warn.append(f"stress: reactions do not balance (error {st['equilibrium_error']:.2g})")
        if st.get("peak_at_bc"):
            warn.append("stress: the peak von Mises stress is at a support, where stresses are singular; judge the "
                        "design on stress.peak_von_mises_off_bc / stress.safety_factor_off_bc")
        warn += res.pop("_selection_warnings", [])
        ns = res.get("mesh", {}).get("straightened_elements", 0)
        if ns:
            warn.append(f"{ns} curved second-order elements were inverted after meshing and were straightened "
                        "(their mid-edge nodes moved onto straight edges): there are features thin for the "
                        "element size; stresses right there are approximate; refine near them (mesh.refine)")
        if res.get("peak_at_bc"):
            at = res["peak_at_bc"]
            where = ", ".join(at) if isinstance(at, list) else "a support or load"
            warn.append(f"the peak von Mises stress is at {where}: stresses at fixed or loaded edges are singular "
                        "(they grow with refinement); judge the design on peak_von_mises_off_bc / safety_factor_off_bc, "
                        "or model the support more realistically (contact faces, a fillet at the corner)")
        conv = res.get("convergence", {})
        for q, c in conv.items():
            if c["rel_change"] > 0.05:
                hint = ("widen stress_band (now " + str(res.get("off_bc_band_mm")) + " mm) or refine the mesh"
                        if q.endswith("off_bc") else "refine the mesh")
                warn.append(f"{q} changes {100 * c['rel_change']:.1f} % on a finer mesh: {hint}")
        d, p = conv.get("max_displacement"), conv.get("peak_von_mises")
        if d and p and d["rel_change"] < 0.02 and p["rel_change"] > 0.1:
            warn.append("the displacement has converged but the peak stress keeps growing on a finer mesh: a stress "
                        "singularity (sharp corner, fixed edge, point load); the peak there is not a design value")
        if warn:
            res["warnings"] = warn
        res["passed"] = all(c["pass"] for c in res["checks"]) if res["checks"] else None
    except (MeshError, SolverError, subprocess.TimeoutExpired) as exc:
        res = {"ok": False, "why": type(exc).__name__, "error": str(exc)[:2000]}
    except Exception as exc:  # noqa: BLE001
        res = {"ok": False, "why": type(exc).__name__, "error": str(exc)[:2000]}
    res["translator"] = __version__
    res["solver"] = solver
    tmp = folder / f".{name}.json.tmp"   # atomic: study_status never reads half a file
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(folder / f"{name}.json")
    return res


def bc_names(study: dict) -> list[str]:
    return [bc_name(i, b) for i, b in enumerate(study.get("bcs", []))]
