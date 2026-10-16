"""deck.py — a study and its mesh as a CalculiX input deck (.inp).
Version: 2026.10.16.1

    text, plan = write(name, study, mesh)

`plan` records what the deck applies (node sets, loads, resultants, faces with fluxes and films),
so the results can be checked against it (equilibrium, heat balance) without reading the deck back.
Units in the deck: mm, N, s, t, mW (study.to_deck_units converts the material).
"""
from __future__ import annotations

import re

import numpy as np

from . import __version__
from .meshdata import Mesh, MeshError
from .study import G_SCALE, H_SCALE, P_SCALE, Q_SCALE, bc_name, material_of, to_deck_units

SUPPORT_KINDS = ("fixed", "slider", "displacement", "temperature")


def _g(v: float) -> str:
    return f"{float(v):.10g}"


def set_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name).upper()
    return ("N_" + s)[:70]


def _lines(ids, per: int = 10) -> list[str]:
    ids = list(ids)
    return [", ".join(str(i) for i in ids[k:k + per]) for k in range(0, len(ids), per)]


def resolve(study: dict, mesh: Mesh) -> dict:
    """Surface tags of every bc and probe: {name: [tags]}; MeshError if a selector matches nothing."""
    out = {}
    for i, bc in enumerate(study.get("bcs", [])):
        if "faces" in bc:
            try:
                out[bc_name(i, bc)] = mesh.select(bc["faces"])
            except MeshError as exc:
                raise MeshError(f"bc {bc_name(i, bc)}: {exc}") from None
    for pr in study.get("probes", []):
        try:
            out["probe:" + pr["name"]] = mesh.select(pr["faces"])
        except MeshError as exc:
            raise MeshError(f"probe {pr['name']}: {exc}") from None
    return out


def write(name: str, study: dict, mesh: Mesh) -> tuple[str, dict]:
    an = study["analysis"]
    mat = to_deck_units(material_of(study))
    tags = resolve(study, mesh)
    etype = "C3D10" if mesh.order == 2 else "C3D4"
    L = [f"** {name}: written by fea-translator {__version__} ({an})",
         "** units: mm, N, s, t (tonne), mW; temperatures in degrees C",
         "*NODE, NSET=NALL"]
    L += [f"{i + 1}, {_g(x)}, {_g(y)}, {_g(z)}" for i, (x, y, z) in enumerate(mesh.nodes)]
    L.append(f"*ELEMENT, TYPE={etype}, ELSET=EALL")
    L += [f"{e + 1}, " + ", ".join(str(n + 1) for n in row) for e, row in enumerate(mesh.tets)]
    plan = {"analysis": an, "sets": {}, "applied_force": [0.0, 0.0, 0.0], "supports": [], "loads": [],
            "heat_in_W": 0.0, "films": [], "fluxes": [], "tags": tags, "dofs": {},
            # equivalent nodal loads (N) and heat (mW): CalculiX's RF / RFL at a node include them
            "nodal_load": np.zeros((len(mesh.nodes), 3)), "nodal_heat": np.zeros(len(mesh.nodes))}
    for bname, tg in tags.items():
        nodes = mesh.nodes_of(tg)
        sname = set_name(bname.replace("probe:", "P_"))
        plan["sets"][bname] = {"nset": sname, "nodes": (nodes + 1).tolist(), "area": mesh.area_of(tg)}
        L.append(f"*NSET, NSET={sname}")
        L += _lines((nodes + 1).tolist())
    L += ["*MATERIAL, NAME=MAT", "*ELASTIC", f"{_g(mat['E'])}, {_g(mat['nu'])}"]
    if "density" in mat:
        L += ["*DENSITY", _g(mat["density"])]
    if "k" in mat:
        L += ["*CONDUCTIVITY", _g(mat["k"])]
    if "alpha" in mat:
        L += ["*EXPANSION", _g(mat["alpha"])]
    L.append("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT")
    if an == "thermal":
        temps = [float(b["T"]) for b in study["bcs"] if b["kind"] == "temperature"] + \
                [float(b["T_inf"]) for b in study["bcs"] if b["kind"] == "convection"]
        L += ["*INITIAL CONDITIONS, TYPE=TEMPERATURE", f"NALL, {_g(sum(temps) / len(temps))}"]
    L.append("*STEP")
    modes = int(study.get("modes", 6 if an == "modal" else 3))
    L += {"static": ["*STATIC"], "modal": ["*FREQUENCY", str(modes)], "buckling": ["*BUCKLE", str(modes)],
          "thermal": ["*HEAT TRANSFER, STEADY STATE", "1., 1."]}[an]
    bnd, cload, dload, dflux, film = [], {}, [], [], []
    for i, bc in enumerate(study.get("bcs", [])):
        k, bname = bc["kind"], bc_name(i, bc)
        sname = plan["sets"].get(bname, {}).get("nset")
        tg = tags.get(bname)
        if k == "fixed":
            bnd.append(f"{sname}, 1, 3")
            plan["supports"].append(bname)
            plan["dofs"][bname] = [0, 1, 2]
        elif k == "slider":
            a = mesh.axis_normal(tg) + 1
            bnd.append(f"{sname}, {a}, {a}")
            plan["supports"].append(bname)
            plan["dofs"][bname] = [a - 1]
        elif k == "displacement":
            for d, key in enumerate(("ux", "uy", "uz")):
                if key in bc:
                    bnd.append(f"{sname}, {d + 1}, {d + 1}, {_g(bc[key])}")
            plan["supports"].append(bname)
            plan["dofs"][bname] = [d for d, key in enumerate(("ux", "uy", "uz")) if key in bc]
        elif k == "force":
            f = np.array([float(bc.get(c, 0.0)) for c in ("fx", "fy", "fz")])
            for n, v in mesh.force_nodes(tg, f).items():
                cload[n] = cload.get(n, np.zeros(3)) + v
                plan["nodal_load"][n] += v
            plan["applied_force"] = (np.array(plan["applied_force"]) + f).tolist()
            plan["loads"].append(bname)
        elif k == "pressure":
            p = float(bc["p"])
            dload += [f"{e + 1}, P{s}, {_g(p)}" for e, s in mesh.faces_of(tg)]
            plan["nodal_load"] += mesh.pressure_nodes(tg, p)
            plan["applied_force"] = (np.array(plan["applied_force"]) + mesh.pressure_resultant(tg, p)).tolist()
            plan["loads"].append(bname)
        elif k == "gravity":
            g = np.array([float(bc.get("gx", 0.0)), float(bc.get("gy", 0.0)), float(bc.get("gz", -9.81))])
            gm = float(np.linalg.norm(g))
            dload.append(f"EALL, GRAV, {_g(gm * G_SCALE)}, " + ", ".join(_g(c / gm) for c in g))
            weight = mat["density"] * mesh.volume() * g * G_SCALE  # t/mm3 * mm3 * mm/s2 = N
            plan["nodal_load"] += mesh.body_nodes(mat["density"] * g * G_SCALE)
            plan["applied_force"] = (np.array(plan["applied_force"]) + weight).tolist()
            plan["loads"].append(bname)
        elif k == "temperature":
            bnd.append(f"{sname}, 11, 11, {_g(bc['T'])}")
            plan["supports"].append(bname)
            plan["dofs"][bname] = [0]
        elif k in ("heat_flux", "heat"):
            q = float(bc["q"]) * Q_SCALE if k == "heat_flux" else float(bc["power"]) * P_SCALE / mesh.area_of(tg)
            dflux += [f"{e + 1}, S{s}, {_g(q)}" for e, s in mesh.faces_of(tg)]
            plan["heat_in_W"] += q * mesh.area_of(tg) / P_SCALE
            plan["nodal_heat"] += mesh.flux_nodes(tg, q)
            plan["fluxes"].append(bname)
        elif k == "convection":
            h = float(bc["h"]) * H_SCALE
            film += [f"{e + 1}, F{s}, {_g(bc['T_inf'])}, {_g(h)}" for e, s in mesh.faces_of(tg)]
            plan["films"].append({"name": bname, "tags": tg, "h_W": float(bc["h"]), "h": h, "T_inf": float(bc["T_inf"])})
    if bnd:
        L += ["*BOUNDARY"] + bnd
    if cload:
        L.append("*CLOAD")
        for n in sorted(cload):
            for d in range(3):
                if cload[n][d] != 0:
                    L.append(f"{n + 1}, {d + 1}, {_g(cload[n][d])}")
    if dload:
        L += ["*DLOAD"] + dload
    if dflux:
        L += ["*DFLUX"] + dflux
    if film:
        L += ["*FILM"] + film
    if an == "static":
        L += ["*NODE PRINT, NSET=NALL", "U", "*EL PRINT, ELSET=EALL", "S"]
        for s in plan["supports"]:
            L += [f"*NODE PRINT, NSET={plan['sets'][s]['nset']}", "RF"]  # per node: shared nodes counted once
        L += ["*NODE FILE", "U", "*EL FILE", "S"]
    elif an == "thermal":
        L += ["*NODE PRINT, NSET=NALL", "NT"]
        for s in plan["supports"]:
            L += [f"*NODE PRINT, NSET={plan['sets'][s]['nset']}", "RFL"]
        L += ["*NODE FILE", "NT", "*EL FILE", "HFL"]
    else:
        L += ["*NODE FILE", "U"]
    L.append("*END STEP")
    plan["n_nodes"], plan["n_elements"], plan["element"] = len(mesh.nodes), len(mesh.tets), etype
    return "\n".join(L) + "\n", plan
