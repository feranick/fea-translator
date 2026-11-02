"""joint.py — bolted, gasketed assemblies: several parts in contact, bolt pretension, a gasket law, and the internal
pressure stepped up to find where the joint leaks and where it lifts off (analysis "joint", CalculiX only).
Version: 2026.11.02.4

    parts = mesh_parts(study, folder)              each part meshed on its own (nodes and elements offset when merged)
    text, plan = write(name, study, parts)          the deck: contact pairs, bolt connectors, preload + pressure steps
    res = evaluate(study, parts, plan, dat_text, frd_text)   bolt forces, gasket stresses, leak and lift-off pressures
    errors, warnings = validate(study)

Bolts are connectors (no bolt geometry): each bolt is a spring (SPRINGA) of the thread's tensile stress area over
its grip, k = E A_s / grip, from the head's bearing face to the nut's bearing face or to the engaged thread of a
tapped hole. Each end follows the mean displacement of its bearing nodes (an annulus between the hole and the washer
around the axis, or the bore over the thread depth), one *EQUATION per direction. The preload acts across a cut at
the head: u_C = mean(u of the head's bearing nodes) - u_PT * axis, where the reference node PT takes the preload as a *CLOAD in the first step and
is then held (*BOUNDARY, FIXED), so the bolt force follows the joint. (CalculiX's *PRE-TENSION SECTION on a beam
element deletes the beam and leaves a rigid link: the bolt would not stretch.) The bolt force is the spring's
stiffness times its stretch; at the preload it must equal the preload (bolt_preload_error). Bolts carry no shear
or bending: friction at the gasket holds the parts sideways.

The gasket is the contact between the two flange faces (face-to-face penalty contact, TYPE=SURFACE TO SURFACE, as
CalculiX recommends with 10-node tetrahedra) with a linear pressure-overclosure law and friction: its stiffness
(MPa/mm), or its compression curve's secant at the mean gasket stress under the preload (CalculiX ignores a tabular
law in face-to-face contact: the parts pass through each other). The contact pressure at the gasket's slave
nodes is the gasket stress.

Leak and lift-off: after the preload step, the swept load (the internal-pressure bc, at its full value in the last
step) rises in equal steps. At each step the gasket stress at every slave node is read from the .frd file, and the
seal's ring stress is found: the largest stress such that a closed ring of nodes at or above it still separates the
gasket's inner edge (the one nearest the pressure) from its outer edges (minimax path from inner to outer edge).
The joint leaks when the ring stress falls below the sealing stress (the given stress, or m x p, whichever is
larger), and lifts off when it falls to zero (an open path across the gasket: the relief pressure of a cover).
Both are interpolated between steps. Not modelled: the pressure reaching into an opening gasket (it makes leaks
slightly earlier), bolt bending stresses, thread stripping, creep relaxation of the gasket.
"""
from __future__ import annotations

import heapq
import math
import re
from pathlib import Path

import numpy as np

from . import __version__, results
from .meshdata import EDGE_MID, EDGES, FACES, Mesh, MeshError
from .study import G_SCALE, MATERIALS, _num, _point, _selector, bc_name, material_of, to_deck_units

# ISO 261 coarse threads: nominal diameter d, pitch P (mm); ISO 273 medium clearance; ISO 7089 washer outer diameter
BOLTS = {"M3": (3, 0.5, 3.4, 7), "M4": (4, 0.7, 4.5, 9), "M5": (5, 0.8, 5.5, 10), "M6": (6, 1.0, 6.6, 12),
         "M8": (8, 1.25, 9.0, 16), "M10": (10, 1.5, 11.0, 20), "M12": (12, 1.75, 13.5, 24), "M14": (14, 2.0, 15.5, 28),
         "M16": (16, 2.0, 17.5, 30), "M18": (18, 2.5, 20.0, 34), "M20": (20, 2.5, 22.0, 37), "M22": (22, 2.5, 24.0, 39),
         "M24": (24, 3.0, 26.0, 44), "M27": (27, 3.0, 30.0, 50), "M30": (30, 3.5, 33.0, 56), "M33": (33, 3.5, 36.0, 60),
         "M36": (36, 4.0, 39.0, 66)}
JOINT_KEYS = {"parts", "analysis", "bcs", "bolts", "gaskets", "sweep", "mesh", "expect", "notes", "procs", "queue", "run",
              "equation_solver", "images"}
SUPPORTS = ("fixed", "slider", "displacement")
LOADS = ("pressure", "force", "gravity")
FREQ = 1000000          # output only the end of each step (CalculiX always writes the final increment)


def stress_area(d: float, p: float) -> float:
    """Tensile stress area of a metric thread (ISO 898-1), mm2."""
    return math.pi / 4 * (d - 0.9382 * p) ** 2


def bolt_size(b: dict) -> dict:
    d, p, clear, washer = BOLTS[b["size"]]
    return {"d": d, "pitch": p, "As": stress_area(d, p), "hole_d": float(b.get("hole_d", clear)),
            "washer_d": float(b.get("washer_d", washer)), "tap_d": d - p}


def preload_N(b: dict) -> float:
    """The preload: given, or from the tightening torque T = K F d (K 0.2 unless given)."""
    if "preload" in b:
        return float(b["preload"])
    return float(b["torque"]) * 1e3 / (float(b.get("k", 0.2)) * bolt_size(b)["d"])


# ---- parts ------------------------------------------------------------------------------------
def mesh_parts(study: dict, folder: Path, size_factor: float = 1.0) -> list[dict]:
    """Each part meshed on its own (in <folder>/<part>/), moved by its 'move' [dx, dy, dz]; with the node and
    element offsets it gets in the merged deck."""
    from .mesher import mesh_part
    out, n0, e0 = [], 0, 0
    for pname, spec in study["parts"].items():
        sub = {"part": spec["part"], "mesh": spec.get("mesh", study.get("mesh", {}))}
        d = Path(folder) / f"part_{pname}"
        d.mkdir(parents=True, exist_ok=True)
        mesh, info = mesh_part(sub, d, size_factor)
        mv = spec.get("move")
        if mv:
            mesh = Mesh(mesh.nodes + np.asarray(mv, float), mesh.tets, mesh.surfaces, mesh.surface_info).build()
        mat = material_of({"material": spec.get("material", "steel")})
        out.append({"name": pname, "mesh": mesh, "info": info, "n0": n0, "e0": e0, "material": mat})
        n0 += len(mesh.nodes)
        e0 += len(mesh.tets)
    orders = {p["mesh"].order for p in out}
    if len(orders) > 1:
        raise MeshError("all parts need the same element order (mesh.order)")
    return out


def _part(parts: list[dict], name: str) -> dict:
    for p in parts:
        if p["name"] == name:
            return p
    raise MeshError(f"no part named {name!r}")


def select(parts: list[dict], sel) -> list[tuple[dict, list[int]]]:
    """Faces of a part-qualified selector ({part, nx, ...}, or a list of them): [(part, surface tags)]."""
    out = []
    for s in (sel if isinstance(sel, list) else [sel]):
        p = _part(parts, s["part"])
        tags = p["mesh"].select({k: v for k, v in s.items() if k != "part"})
        hit = next((o for o in out if o[0] is p), None)
        if hit:
            hit[1].extend(t for t in tags if t not in hit[1])
        else:
            out.append((p, list(tags)))
    return out


def _gnodes(sel_parts) -> np.ndarray:
    """Global node indices (0-based, merged numbering) of selected faces."""
    return np.unique(np.concatenate([p["mesh"].nodes_of(t) + p["n0"] for p, t in sel_parts])) if sel_parts else np.zeros(0, int)


def _gfaces(sel_parts) -> list[tuple[int, int]]:
    """(global element index, face number) of selected faces."""
    return [(e + p["e0"], s) for p, t in sel_parts for e, s in p["mesh"].faces_of(t)]


TAG_STRIDE = 100000         # face tags in the merged state: (part index + 1) * TAG_STRIDE + the part's own tag


def gtags(parts: list[dict], sel_parts) -> list[int]:
    """Globally unique face tags of a selection (for the pictures' state)."""
    return [(parts.index(p) + 1) * TAG_STRIDE + int(t) for p, tg in sel_parts for t in tg]


def all_nodes(parts: list[dict]) -> np.ndarray:
    return np.concatenate([p["mesh"].nodes for p in parts])


# ---- bolts ------------------------------------------------------------------------------------
def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _perp(a: np.ndarray) -> np.ndarray:
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    return _unit(t - (t @ a) * a)


def bolt_positions(b: dict, axis: np.ndarray) -> list[np.ndarray]:
    """Points on each bolt's axis: 'at' (a list of [x, y, z]), or a 'pattern' {center, pcd, n, start_deg} around
    the axis direction."""
    if "at" in b:
        return [np.asarray(q, float) for q in b["at"]]
    pt = b["pattern"]
    c = np.asarray(pt["center"], float)
    u = _perp(axis)
    v = np.cross(axis, u)
    r = float(pt["pcd"]) / 2
    a0 = math.radians(float(pt.get("start_deg", 0.0)))
    n = int(pt["n"])
    return [c + r * (math.cos(a0 + 2 * math.pi * k / n) * u + math.sin(a0 + 2 * math.pi * k / n) * v) for k in range(n)]


def _axial_radial(X: np.ndarray, c: np.ndarray, a: np.ndarray):
    rel = X - c
    s = rel @ a
    return s, np.linalg.norm(rel - s[:, None] * a, axis=1)


def bolt_geometry(parts: list[dict], b: dict) -> list[dict]:
    """Per bolt: the head and nut (or thread) bearing nodes (global), the beam ends B1 (head) and B2 (nut), the
    axis from head to nut, the grip length."""
    sz = bolt_size(b)
    head = select(parts, b["head"])
    hp, htags = head[0]
    if "axis" in b:
        axis = _unit(b["axis"])
    else:   # into the joint: against the head bearing face's outward normal
        g = hp["mesh"].groups[htags[0]]
        if not g.planar:
            raise MeshError(f"bolts {b['name']}: the head face is not flat; give 'axis' (from head to nut)")
        axis = -_unit(g.normal[0])
    X = all_nodes(parts)
    hn = _gnodes(head)
    if "nut" in b:
        nut = select(parts, b["nut"])
        nn = _gnodes(nut)
    else:
        th = b["thread"]
        tp = _part(parts, th["part"])
        # the bore's faces: not the flat faces across the axis (the hole's entry face, the flange faces)
        bore = [g.nodes() for g in tp["mesh"].groups.values() if not (g.planar and abs(g.normal[0] @ axis) > 0.9)]
        nn = (np.unique(np.concatenate(bore)) if bore else np.zeros(0, int)) + tp["n0"]
    out = []
    for k, c in enumerate(bolt_positions(b, axis)):
        s, r = _axial_radial(X[hn], c, axis)
        m = (r >= 0.5 * sz["hole_d"] * 0.999) & (r <= 0.5 * sz["washer_d"] * 1.001)
        h = hn[m]
        if not len(h):
            raise MeshError(f"bolts {b['name']} #{k + 1} at {np.round(c, 3).tolist()}: no head-face nodes between the hole "
                            f"(d {sz['hole_d']:g}) and the washer (d {sz['washer_d']:g}) around the axis")
        s1 = float(s[m].mean())
        s_n, r_n = _axial_radial(X[nn], c, axis)
        if "nut" in b:
            mn = (r_n >= 0.5 * sz["hole_d"] * 0.999) & (r_n <= 0.5 * sz["washer_d"] * 1.001)
            what = "nut-face nodes between the hole and the washer"
        else:   # the engaged thread: the bore between the tap drill and the nominal diameter, over its depth
            band = (r_n >= 0.4 * sz["tap_d"]) & (r_n <= 0.6 * sz["d"] + 0.05)
            band &= s_n > s1 + 1e-6
            mn = band.copy()
            if band.any():
                s0 = float(s_n[band].min())
                mn &= s_n <= s0 + float(b["thread"]["depth"])
            what = f"bore nodes (radius {0.5 * sz['tap_d']:g} to {0.5 * sz['d']:g}) below the head"
        n_ = nn[mn]
        if not len(n_):
            raise MeshError(f"bolts {b['name']} #{k + 1} at {np.round(c, 3).tolist()}: no {what}")
        s2 = float(s_n[mn].mean())
        if s2 <= s1:
            raise MeshError(f"bolts {b['name']} #{k + 1}: the nut side is not beyond the head along the axis "
                            f"{np.round(axis, 4).tolist()}; check 'axis' or the faces")
        out.append({"k": k + 1, "axis": axis, "B1": c + s1 * axis, "B2": c + s2 * axis, "head": h, "nut": n_,
                    "grip": s2 - s1})
    return out


# ---- the deck ---------------------------------------------------------------------------------
def _g(v: float) -> str:
    return f"{float(v):.10g}"


def _lines(ids, per: int = 10) -> list[str]:
    ids = list(ids)
    return [", ".join(str(i) for i in ids[k:k + per]) for k in range(0, len(ids), per)]


def _name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s).upper()[:60]


def _equation(dep: int, others: list[int], ref: int, d: int, avoid: set) -> list[str]:
    """n u_ref - sum(u_j) = 0 in direction d (1-based nodes): the reference node follows the mean of the nodes.
    The reference node (a bolt end, in no other equation as dependent, never held) is the dependent term: a bearing
    node can be held by a support (a nut on a fixed face), and CalculiX refuses a held DOF on the dependent side."""
    nodes = [dep] + [o for o in others if o != dep]
    terms = [(ref, d, float(len(nodes)))] + [(q, d, -1.0) for q in nodes]
    out = ["*EQUATION", str(len(terms))]
    for k in range(0, len(terms), 4):
        out.append(", ".join(f"{q}, {dd}, {_g(c)}" for q, dd, c in terms[k:k + 4]))
    return out


MIN_SLOPE = 0.02          # no gasket-curve segment softer than 2 % of the curve's secant slope (stress / closure at its end)
PRELOAD_STICK = 0.01      # the friction's stick slope in the preload step, relative to the later steps'
STAB_FRACTION = 1e-6      # stabilisation springs: each 1e-6 of the softest bolt's stiffness
STAB_NODES = 6            # per part that no support holds: 6 well-spread nodes, 3 springs each
STAB_WARN = 0.01          # warn when the springs carry more than 1 % of the preload
GAP_TOL = 0.01            # mm: gasket faces further apart than this (median) do not touch at the start


def face_gap(X: np.ndarray, slave_nodes: np.ndarray, master_nodes: np.ndarray, normal) -> dict:
    """How far the gasket's faces are apart before the preload: for each slave node, the distance along the gasket's
    normal to the nearest master node (median and largest, mm)."""
    from scipy.spatial import cKDTree
    if not len(slave_nodes) or not len(master_nodes):
        return {"median_mm": None, "max_mm": None}
    _, j = cKDTree(X[master_nodes]).query(X[slave_nodes])
    d = np.abs((X[master_nodes][j] - X[slave_nodes]) @ np.asarray(normal, float))
    return {"median_mm": round(float(np.median(d)), 6), "max_mm": round(float(d.max()), 6)}


def gasket_curve(g: dict) -> tuple[list[tuple[float, float]], bool]:
    """The compression curve from (0, 0), every segment at least MIN_SLOPE of the secant slope (a segment of zero
    stiffness, e.g. a data sheet's seating range, leaves the first contact without stiffness: the preload then does
    not converge); and whether it was raised."""
    pts = sorted((float(c), float(s)) for c, s in g["curve"])
    if pts[0] != (0.0, 0.0):
        pts = [(0.0, 0.0)] + pts
    smin = MIN_SLOPE * pts[-1][1] / max(pts[-1][0], 1e-12)
    out, raised = [pts[0]], False
    for c, s in pts[1:]:
        s2 = max(s, out[-1][1] + smin * (c - out[-1][0]))
        raised |= s2 > s + 1e-12 * max(1.0, abs(s))
        out.append((c, s2))
    return out, raised


def gap_warning(g: dict) -> str | None:
    gp = g.get("gap") or {}
    if gp.get("median_mm") is None or gp["median_mm"] <= GAP_TOL:
        return None
    return (f"gasket {g['name']}: its faces are {gp['median_mm']:.3g} mm apart (median; up to {gp['max_mm']:.3g} mm) "
            "before the preload: they should touch (parts.move); the gasket's thickness is not meshed")


def closure_at(g: dict, stress: float) -> float:
    """The closure (mm) of a gasket curve at a stress (MPa): linear between its points, the last segment beyond."""
    pts, _ = gasket_curve(g)
    for (c1, s1), (c2, s2) in zip(pts, pts[1:]):
        if stress <= s2:
            return c1 + (c2 - c1) * (stress - s1) / max(s2 - s1, 1e-300)
    (c1, s1), (c2, s2) = pts[-2], pts[-1]
    return c2 + (c2 - c1) * (stress - s2) / max(s2 - s1, 1e-300)


def gasket_linear(g: dict, preload_MPa: float) -> dict:
    """The gasket's linear law for CalculiX: its stiffness, or its curve's secant at the mean preload stress (the
    curve's own secant at its last point when there is no preload)."""
    if "curve" not in g:
        return {"kind": "stiffness", "stiffness_MPa_mm": float(g["stiffness"])}
    pts, raised = gasket_curve(g)
    at = preload_MPa if preload_MPa > 0 else pts[-1][1]
    c = closure_at(g, at)
    return {"kind": "secant", "stiffness_MPa_mm": at / c, "at_MPa": round(at, 6), "closure_mm": round(c, 6),
            "beyond_curve": bool(at > pts[-1][1]), "curve_raised": raised}


def friction_lines(k: float, g: dict, stick: float = 1.0) -> list[str]:
    """*FRICTION: mu and the stick slope (100 times below the normal stiffness k, x stick)."""
    return ["*FRICTION", f"{_g(float(g.get('friction', 0.2)))}, {_g(k / 100.0 * stick)}"]


def _nearest_gasket(X: np.ndarray, b: dict, axis, gasket_nodes: dict) -> str | None:
    """The gasket a bolt clamps: the one with a node nearest to the bolt's axis."""
    best, name = math.inf, None
    a = np.asarray(axis, float)
    for gname, nodes in gasket_nodes.items():
        if not len(nodes):
            continue
        rel = X[nodes] - np.asarray(b["B1"], float)
        d = np.linalg.norm(rel - np.outer(rel @ a, a), axis=1).min()
        if d < best:
            best, name = d, gname
    return name


def spread_nodes(X: np.ndarray, cand: np.ndarray, n: int) -> list[int]:
    """n well-spread nodes of cand (indices into X): the farthest from the centre, then each the farthest from
    those chosen."""
    if not len(cand):
        return []
    P = X[cand]
    pick = [int(np.argmax(np.linalg.norm(P - P.mean(axis=0), axis=1)))]
    d = np.linalg.norm(P - P[pick[0]], axis=1)
    while len(pick) < min(n, len(cand)):
        k = int(np.argmax(d))
        if d[k] <= 0:
            break
        pick.append(k)
        d = np.minimum(d, np.linalg.norm(P - P[k], axis=1))
    return [int(cand[k]) for k in pick]


def write(name: str, study: dict, parts: list[dict]) -> tuple[str, dict]:
    """The deck of a joint study, and its plan (node sets, bolts, gaskets, steps) for reading the results."""
    order = parts[0]["mesh"].order
    etype = "C3D10" if order == 2 else "C3D4"
    L = [f"** {name}: written by fea-translator {__version__} (joint: contact, bolt pretension, gasket)",
         "** units: mm, N, s, t (tonne); stresses MPa", "*NODE, NSET=NALL"]
    for p in parts:
        L += [f"{p['n0'] + i + 1}, {_g(x)}, {_g(y)}, {_g(z)}" for i, (x, y, z) in enumerate(p["mesh"].nodes)]
    n_next = sum(len(p["mesh"].nodes) for p in parts) + 1
    e_next = sum(len(p["mesh"].tets) for p in parts) + 1
    for p in parts:
        L.append(f"*ELEMENT, TYPE={etype}, ELSET=E_{_name(p['name'])}")
        L += [f"{p['e0'] + e + 1}, " + ", ".join(str(p["n0"] + n + 1) for n in row) for e, row in enumerate(p["mesh"].tets)]
    plan = {"analysis": "joint", "sets": {}, "supports": [], "loads": [], "kinds": {}, "bolts": [], "gaskets": [],
            "steps": [], "sweep": None, "selection": {}, "n_nodes": n_next - 1, "n_elements": e_next - 1, "element": etype}
    sweep = study.get("sweep", {})
    swept = sweep.get("bc")
    nsteps = int(sweep.get("steps", 10))
    # boundary conditions
    bnd, held, cload_const, dload_const, swept_cload, swept_dload = [], set(), {}, [], {}, []
    X = all_nodes(parts)
    for i, bc in enumerate(study.get("bcs", [])):
        k, bname = bc["kind"], bc_name(i, bc)
        plan["kinds"][bname] = k
        sel = select(parts, bc["faces"]) if "faces" in bc else []
        if sel:
            plan["selection"][bname] = [dict(p["mesh"].groups[t].describe(), part=p["name"]) for p, tg in sel for t in tg]
        nodes = _gnodes(sel)
        sname = "N_" + _name(bname)
        plan.setdefault("bc_tags", {})[bname] = gtags(parts, sel)
        if k in SUPPORTS:
            plan["sets"][bname] = {"nset": sname, "nodes": (nodes + 1).tolist()}
            L.append(f"*NSET, NSET={sname}")
            L += _lines((nodes + 1).tolist())
            held.update((nodes + 1).tolist())
            plan.setdefault("held_parts", set()).update(p["name"] for p, _ in sel)
            if k == "fixed":
                bnd.append(f"{sname}, 1, 3")
            elif k == "slider":
                p, tg = sel[0]
                a = p["mesh"].axis_normal(tg) + 1
                bnd.append(f"{sname}, {a}, {a}")
            else:
                for d, key in enumerate(("ux", "uy", "uz")):
                    if key in bc:
                        bnd.append(f"{sname}, {d + 1}, {d + 1}, {_g(bc[key])}")
            plan["supports"].append(bname)
        elif k == "pressure":
            faces = _gfaces(sel)
            rows = [(e + 1, s) for e, s in faces]
            (swept_dload if bname == swept else dload_const).append((rows, float(bc["p"])))
            plan["loads"].append(bname)
            if bname == swept:
                c = np.concatenate([(p["mesh"].groups[t].centroid * p["mesh"].groups[t].area[:, None]).sum(axis=0, keepdims=True)
                                    / p["mesh"].groups[t].total_area for p, tg in sel for t in tg]).mean(axis=0)
                plan["sweep"] = {"bc": bname, "kind": "pressure", "max": float(bc["p"]), "steps": nsteps, "center": c.tolist()}
        elif k == "force":
            f = np.array([float(bc.get(c, 0.0)) for c in ("fx", "fy", "fz")])
            area = sum(p["mesh"].area_of(tg) for p, tg in sel)
            tgt = swept_cload if bname == swept else cload_const
            for p, tg in sel:
                fp = f * p["mesh"].area_of(tg) / area
                for n, v in p["mesh"].force_nodes(tg, fp).items():
                    tgt[n + p["n0"]] = tgt.get(n + p["n0"], np.zeros(3)) + v
            plan["loads"].append(bname)
            if bname == swept:
                plan["sweep"] = {"bc": bname, "kind": "force", "max": float(np.linalg.norm(f)), "steps": nsteps,
                                 "center": X[_gnodes(sel)].mean(axis=0).tolist()}
        elif k == "gravity":
            gv = np.array([float(bc.get("gx", 0.0)), float(bc.get("gy", 0.0)), float(bc.get("gz", -9.81))])
            gm = float(np.linalg.norm(gv))
            for p in parts:
                dload_const.append(([(f"E_{_name(p['name'])}", "GRAV")], (gm * G_SCALE, gv / gm)))
            plan["loads"].append(bname)
    # materials and sections
    for p in parts:
        m = to_deck_units(p["material"])
        L += [f"*MATERIAL, NAME=M_{_name(p['name'])}", "*ELASTIC", f"{_g(m['E'])}, {_g(m['nu'])}"]
        if "density" in m:
            L += ["*DENSITY", _g(m["density"])]
        L.append(f"*SOLID SECTION, ELSET=E_{_name(p['name'])}, MATERIAL=M_{_name(p['name'])}")
    # bolts: a spring (SPRINGA, k = E A_s / grip) from a cut node C at the head to the nut, with the pretension as
    # an equation across the cut: u_C = mean(u_head) - u_PT * axis (CalculiX's own *PRE-TENSION SECTION on a beam deletes the
    # beam and leaves a rigid link: the bolt would have no stretch). B1 and B2 follow their bearing nodes' mean.
    springs, eqs, pret = [], [], []
    gasket_nodes = {g["name"]: _gnodes(select(parts, g["faces"])) for g in study.get("gaskets", [])}
    for gi, b in enumerate(study.get("bolts", [])):
        bn = _name(b["name"])
        sz = bolt_size(b)
        P = preload_N(b)
        mat = to_deck_units(material_of({"material": b.get("material", "steel")}))
        geo = bolt_geometry(parts, b)
        rec = {"name": b["name"], "size": b["size"], "preload_N": P, "As": sz["As"], "bolts": []}
        for bg in geo:
            n2, nc, npt = n_next, n_next + 1, n_next + 2
            n_next += 3
            L += ["*NODE"] + [f"{n}, {_g(q[0])}, {_g(q[1])}, {_g(q[2])}" for n, q in
                              ((n2, bg["B2"]), (nc, bg["B1"]), (npt, bg["B1"]))]
            eid = e_next
            e_next += 1
            k = mat["E"] * sz["As"] / bg["grip"]
            springs.append((eid, nc, n2, bn, k))
            in_eq = plan.setdefault("eq_nodes", set())
            in_eq.update(int(q) for q in bg["head"])
            in_eq.update(int(q) for q in bg["nut"])
            head = [int(q) + 1 for q in bg["head"]]
            for d in (1, 2, 3):
                # the nut end B2 follows the mean of its bearing nodes
                eqs += _equation(int(bg["nut"][0]) + 1, [int(q) + 1 for q in bg["nut"]], n2, d, held)
                # the cut C at the head: the mean of the head's bearing nodes, less the pretension along the axis,
                # n u_C - sum(u_head) + n axis_d u_PT = 0. Each dependent DOF (B2, C) is in no other equation: a chain
                # (C on B1, B1 on the head nodes) gave CalculiX's wrong bolt forces (joint_two_bolt, 2026.11.02.2)
                n = float(len(head))
                terms = [(nc, d, n)] + [(q, d, -1.0) for q in head] + (
                    [(npt, 1, n * float(bg["axis"][d - 1]))] if abs(bg["axis"][d - 1]) > 1e-12 else [])
                eqs += ["*EQUATION", str(len(terms))] + [", ".join(f"{q}, {dd}, {_g(c)}" for q, dd, c in terms[k:k + 4])
                                                         for k in range(0, len(terms), 4)]
            pret.append((eid, npt, bg["axis"], P))
            rec["bolts"].append({"k": bg["k"], "element": eid, "pret_node": npt, "ends": [nc, n2], "k_N_mm": k,
                                 "B1": np.round(bg["B1"], 4).tolist(), "B2": np.round(bg["B2"], 4).tolist(),
                                 "grip_mm": round(bg["grip"], 4), "head_nodes": len(bg["head"]),
                                 "nut_nodes": len(bg["nut"])})
        rec["elset"], rec["axis"] = f"EB_{bn}", geo[0]["axis"].tolist()
        rec["k_bolt_N_mm"] = float(np.mean([q["k_N_mm"] for q in rec["bolts"]]))
        plan["bolts"].append(rec)
    for eid, nc, n2, bn, k in springs:   # one element set per spring: each has its own stiffness (its grip)
        L += [f"*ELEMENT, TYPE=SPRINGA, ELSET=ES{eid}", f"{eid}, {nc}, {n2}", f"*SPRING, ELSET=ES{eid}", _g(k)]
    L += eqs
    # gaskets: face-to-face contact. CalculiX ignores a TABULAR pressure-overclosure law in face-to-face contact (the
    # parts pass through each other: joint_two_bolt and an S50 cover, 2026.11.02): a curve becomes the LINEAR law of
    # its secant at the gasket's mean stress under the preload (the bolts nearest to it, over its area)
    for g in study.get("gaskets", []):
        gn = _name(g["name"])
        slave, master = select(parts, g["faces"]), select(parts, g["with"])
        sf, mf = _gfaces(slave), _gfaces(master)
        sn = _gnodes(slave)
        area = sum(p["mesh"].area_of(tg) for p, tg in slave)
        clamp = 0.0
        for rec in plan["bolts"]:
            for b_ in rec["bolts"]:
                if _nearest_gasket(X, b_, rec["axis"], gasket_nodes) == g["name"]:
                    clamp += rec["preload_N"]
        law = gasket_linear(g, clamp / area if area else 0.0)
        L += [f"*SURFACE, NAME=GS_{gn}"] + [f"{e + 1}, S{s}" for e, s in sf]
        L += [f"*SURFACE, NAME=GM_{gn}"] + [f"{e + 1}, S{s}" for e, s in mf]
        L += [f"*SURFACE INTERACTION, NAME=GI_{gn}", "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
              _g(law["stiffness_MPa_mm"])] + friction_lines(law["stiffness_MPa_mm"], g, PRELOAD_STICK)
        L += [f"*CONTACT PAIR, INTERACTION=GI_{gn}, TYPE=SURFACE TO SURFACE", f"GS_{gn}, GM_{gn}"]
        nrm = sum((p["mesh"].groups[tg_].normal * p["mesh"].groups[tg_].area[:, None]).sum(axis=0)
                  for p, tgs in slave for tg_ in tgs)
        gap = face_gap(X, sn, _gnodes(master), _unit(nrm))
        plan["gaskets"].append({"name": g["name"], "slave_faces": sf, "seal": g.get("seal", {}),
                                "inner_at": g.get("inner_at"), "tags": gtags(parts, slave),
                                "with_tags": gtags(parts, master), "normal": _unit(nrm).tolist(),
                                "law": law, "friction": friction_lines(law["stiffness_MPa_mm"], g),
                                "interaction": f"GI_{gn}", "gap": gap, "area": area})
        plan["selection"][f"gasket {g['name']}"] = [dict(p["mesh"].groups[t].describe(), part=p["name"]) for p, tg in slave for t in tg]
        plan["selection"][f"gasket {g['name']} (with)"] = [dict(p["mesh"].groups[t].describe(), part=p["name"]) for p, tg in master for t in tg]
    # stabilisation: weak springs to ground on every part no support holds (before the gasket is pressed it has no
    # stiffness and friction none: the part could slide and rock about the bolt line, and the preload chatters)
    k_stab = STAB_FRACTION * min((q["k_N_mm"] for rec in plan["bolts"] for q in rec["bolts"]), default=1.0)
    stab, ground, stab_el = {}, [], []
    held_parts = plan.pop("held_parts", set())
    eq_nodes = plan.pop("eq_nodes", set())
    for p in parts:
        if p["name"] in held_parts:
            continue
        surf = np.unique(np.concatenate([np.asarray(tri).ravel() for tri in p["mesh"].surfaces.values()])) + p["n0"]
        cand = np.array([q for q in surf if int(q) not in eq_nodes and int(q) + 1 not in held], int)
        stab[p["name"]] = [q + 1 for q in spread_nodes(X, cand, STAB_NODES)]
        for q in stab[p["name"]]:
            for d in range(3):   # a SPRINGA of unit length along x, y, z to a fixed ground node
                gpos = X[q - 1] + np.eye(3)[d]
                L += ["*NODE", f"{n_next}, {_g(gpos[0])}, {_g(gpos[1])}, {_g(gpos[2])}"]
                ground.append(n_next)
                stab_el.append((e_next, q, n_next))
                n_next += 1
                e_next += 1
    if stab_el:
        L += ["*ELEMENT, TYPE=SPRINGA, ELSET=E_STAB"] + [f"{e}, {a}, {b}" for e, a, b in stab_el]
        L += ["*SPRING, ELSET=E_STAB", _g(k_stab)]
        L += ["*NSET, NSET=N_STAB"] + _lines(sorted({q for v in stab.values() for q in v}))
        L += ["*NSET, NSET=N_STABGROUND"] + _lines(ground)
        bnd.append("N_STABGROUND, 1, 3")
    plan["stabilisation"] = {"k_N_mm": k_stab, "parts": stab, "springs": len(stab_el)}
    pnodes = [npt for _, npt, _, _ in pret]
    ends = [n for _, nc, n2, _, _ in springs for n in (nc, n2)]
    L += ["*NSET, NSET=N_PRETENSION"] + _lines(pnodes) + ["*NSET, NSET=N_BOLTENDS"] + _lines(ends)
    plan["pret_nodes"] = pnodes

    def outputs():   # bolt forces from the beam ends' displacements (CalculiX reports no reaction at a pretension
        # node held with *BOUNDARY, FIXED); gasket stresses from the contact output; no element stresses (large)
        stab_out = [f"*NODE PRINT, NSET=N_STAB, FREQUENCY={FREQ}", "U"] if stab_el else []
        return ([f"*NODE PRINT, NSET=N_BOLTENDS, FREQUENCY={FREQ}", "U"] + stab_out
                + ["*NODE FILE", "U", "*CONTACT FILE", "CDIS, CSTR"])

    def dloads(items, factor):
        out = []
        for rows, val in items:
            if isinstance(val, tuple):   # gravity
                gm, gv = val
                out += [f"{rows[0][0]}, GRAV, {_g(gm * factor)}, " + ", ".join(_g(c) for c in gv)]
            else:
                out += [f"{e}, P{s}, {_g(val * factor)}" for e, s in rows]
        return out

    def cloads(d, factor):
        out = []
        for n in sorted(d):
            out += [f"{n + 1}, {k + 1}, {_g(d[n][k] * factor)}" for k in range(3) if d[n][k] != 0]
        return out
    # step 1: the preload (and the loads that do not change)
    # a small first increment: the contact is found at a twentieth of the preload
    L += ["*STEP, INC=1000", "*STATIC", "0.05, 1., 1e-05, 0.25", "*BOUNDARY"] + bnd
    L += ["*CLOAD"] + [f"{npt}, 1, {_g(P)}" for _, npt, _, P in pret] + cloads(cload_const, 1.0)
    if dload_const:
        L += ["*DLOAD"] + dloads(dload_const, 1.0)
    L += outputs() + ["*END STEP"]
    plan["steps"].append({"step": 1, "what": "preload", "factor": 0.0})
    # steps 2..n+1: the bolts held at their preload length, the swept load in equal steps
    for k in range(1, nsteps + 1):
        f = k / nsteps
        # each step adds 1/n of the load: one increment (CalculiX cuts it back if contact needs it)
        L += ["*STEP, INC=1000", "*STATIC", "1., 1., 1e-08, 1.", "*BOUNDARY, FIXED", "N_PRETENSION, 1, 1"]
        cl = cloads(cload_const, 1.0) + cloads(swept_cload, f)
        L += ["*CLOAD, OP=NEW"] + (cl or [f"{springs[0][2]}, 1, 0."])
        if k == 1:   # the gasket's friction at its full stick slope once it is pressed
            for g in plan["gaskets"]:
                L += [f"*CHANGE FRICTION, INTERACTION={g['interaction']}"] + g["friction"]
        if swept_dload:
            L += ["*DLOAD"] + dloads(swept_dload, f)
        L += outputs() + ["*END STEP"]
        plan["steps"].append({"step": k + 1, "what": "swept load", "factor": f})
    return "\n".join(L) + "\n", plan


# ---- results -------------------------------------------------------------------------------------
def frd_steps(text: str) -> list[dict]:
    """Every nodal result block of an ASCII .frd file, in order: {step, name, components, data {node: values}};
    the step comes from the block's 1PSTEP line."""
    out, step, cur, comps = [], None, None, []
    for line in text.splitlines():
        if "1PSTEP" in line:
            toks = line.split()
            try:
                step = int(toks[-1])
            except (ValueError, IndexError):
                pass
        elif line.startswith(" -4"):
            cur = {"step": step, "name": line[3:].split()[0].upper(), "components": [], "data": {}}
        elif line.startswith(" -5") and cur is not None:
            toks = line[3:].split()
            if toks:
                cur["components"].append(toks[0].upper())
        elif line.startswith(" -1") and cur is not None:
            try:
                node = int(line[3:13])
                body = line[13:].rstrip()
                cur["data"][node] = [float(body[i:i + 12]) for i in range(0, len(body), 12)]
            except ValueError:
                continue
        elif line.startswith(" -3") and cur is not None:
            out.append(cur)
            cur = None
    return out


def contact_pressure_by_step(blocks: list[dict]) -> dict[int, dict[int, float]]:
    """{step: {node (1-based): contact pressure}} from the last CONTACT block of each step."""
    out = {}
    for b in blocks:
        if not b["name"].startswith("CONTACT"):
            continue
        comps = b["components"]
        k = comps.index("CPRESS") if "CPRESS" in comps else 3
        out[b["step"]] = {n: (v[k] if len(v) > k else 0.0) for n, v in b["data"].items()}
    return out


def gasket_graph(parts_nodes: np.ndarray, tets: np.ndarray, faces: list[tuple[int, int]]):
    """The gasket's slave nodes (1-based), their adjacency, their areas, and its boundary edges' node loops."""
    quad = tets.shape[1] == 10
    adj, area, edges = {}, {}, {}

    def link(a, b):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    for e, s in faces:
        el = tets[e]
        c = [int(el[i]) for i in FACES[s - 1]]
        if quad:
            m = [int(el[EDGE_MID[frozenset((FACES[s - 1][i], FACES[s - 1][(i + 1) % 3]))]]) for i in range(3)]
            subs = [(c[0], m[0], m[2]), (m[0], c[1], m[1]), (m[2], m[1], c[2]), (m[0], m[1], m[2])]
        else:
            m = [None, None, None]
            subs = [tuple(c)]
        for t in subs:
            P = parts_nodes[list(t)]
            a = 0.5 * float(np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0])))
            for q in t:
                area[q + 1] = area.get(q + 1, 0.0) + a / 3
            for i in range(3):
                link(t[i] + 1, t[(i + 1) % 3] + 1)
        for i in range(3):
            key = tuple(sorted((c[i], c[(i + 1) % 3])))
            edges.setdefault(key, []).append(m[i])
    bound = [(a, b, mids[0]) for (a, b), mids in edges.items() if len(mids) == 1]
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, _ in bound:
        parent[find(a)] = find(b)
    loops = {}
    for a, b, m in bound:
        r = find(a)
        loops.setdefault(r, set()).update(q + 1 for q in (a, b, m) if q is not None)
    return adj, area, list(loops.values())


def ring_stress(adj: dict, p: dict, inner: set, outer: set) -> float:
    """The largest stress t such that a ring of nodes with stress >= t separates the inner edge from the outer
    edges: the minimax path value from inner to outer (Dijkstra on the largest stress met along the path).
    inf when there is no outer edge (nothing to leak to)."""
    if not outer:
        return math.inf
    best = {}
    h = [(p.get(n, 0.0), n) for n in inner]
    heapq.heapify(h)
    for v, n in h:
        best[n] = min(best.get(n, math.inf), v)
    while h:
        v, n = heapq.heappop(h)
        if v > best.get(n, math.inf):
            continue
        if n in outer:
            return v
        for q in adj.get(n, ()):
            w = max(v, p.get(q, 0.0))
            if w < best.get(q, math.inf):
                best[q] = w
                heapq.heappush(h, (w, q))
    return math.inf


def _crossing(xs: list[float], gs: list[float], floor: float | None = None) -> float | None:
    """The first x where g falls to <= 0, linearly interpolated (None if it never does). floor: the value g cannot go
    below (a contact stress cannot be negative): a crossing step at the floor is extrapolated from the two steps
    before it, since the true zero lies somewhere between them and it."""
    for i in range(len(gs)):
        if gs[i] <= 0:
            if i == 0:
                return xs[0]
            x0, x1, g0, g1 = xs[i - 1], xs[i], gs[i - 1], gs[i]
            if floor is not None and g1 <= floor + 1e-12 and i >= 2 and gs[i - 2] > g0:
                xm, gm = xs[i - 2], gs[i - 2]
                x = x0 + g0 * (x0 - xm) / (gm - g0)
                return min(max(x, x0), x1)
            return x0 + (x1 - x0) * g0 / (g0 - g1) if g0 != g1 else x1
    return None


def evaluate(study: dict, parts: list[dict], plan: dict, dat_text: str, frd_text: str) -> dict:
    """Bolt forces, gasket stresses at the preload and at every step of the swept load, leak and lift-off loads."""
    X = all_nodes(parts)
    tets = np.concatenate([p["mesh"].tets + p["n0"] for p in parts])
    dat = results.parse_dat(dat_text)
    ub = [b for b in dat["blocks"] if b["set"] == "N_BOLTENDS" and b["what"].startswith("displacements")]
    blocks = frd_steps(frd_text)
    cp = contact_pressure_by_step(blocks)
    sw = plan.get("sweep") or {}
    nsteps = len(plan["steps"])
    loads = [sw.get("max", 0.0) * s["factor"] for s in plan["steps"]]
    unit = "MPa" if sw.get("kind") == "pressure" else "N"
    res = {"analysis": "joint", "swept": {"bc": sw.get("bc"), "kind": sw.get("kind"), "max": sw.get("max"), "unit": unit},
           "bolts": {}, "gaskets": {}, "steps": [], "leak": {}, "lift_off": {}}
    # bolt forces: each spring's stiffness times its stretch (its ends' distance change along the axis); in step 1
    # it must equal the preload (bolt_preload_error)
    blist = [(rec, b_) for rec in plan["bolts"] for b_ in rec["bolts"]]
    forces = np.full((nsteps, len(blist)), np.nan)
    U = [{int(r[0]): np.asarray(r[1:4], float) for r in blk["rows"] if len(r) >= 4} for blk in ub]

    def stretch(s, rec, b_):
        n1, n2 = b_["ends"]
        if s >= len(U) or n1 not in U[s] or n2 not in U[s]:
            return None
        return float((U[s][n2] - U[s][n1]) @ np.asarray(rec["axis"], float))
    for i, (rec, b_) in enumerate(blist):
        for s in range(nsteps):
            e = stretch(s, rec, b_)
            if e is not None:
                forces[s, i] = b_["k_N_mm"] * e
    pre = [rec["preload_N"] for rec, _ in blist]
    if blist and np.isfinite(forces[0]).all():
        err = float(np.max(np.abs(forces[0] - pre) / np.maximum(pre, 1e-9)))
        res_pre_err = round(err, 6)
    else:
        res_pre_err = None
    # gaskets
    graphs = {}
    for g in plan["gaskets"]:
        adj, area, loops = gasket_graph(X, tets, [(e, s) for e, s in g["slave_faces"]])
        if g.get("inner_at") is not None:
            ref = np.asarray(g["inner_at"], float)
        elif sw.get("center"):
            ref = np.asarray(sw["center"], float)
        else:
            ref = np.mean([X[n - 1] for n in area], axis=0)
        dist = [float(np.mean([np.linalg.norm(X[n - 1] - ref) for n in lp])) for lp in loops]
        inner = loops[int(np.argmin(dist))] if loops else set()
        outer = set().union(*[lp for lp in loops if lp is not inner]) if len(loops) > 1 else set()
        graphs[g["name"]] = (adj, area, inner, outer)
        res["gaskets"][g["name"]] = {"area_mm2": round(g["area"], 4), "edges": len(loops),
                                     "seal": g["seal"], "slave_nodes": len(area)}
        if len(loops) < 2:
            res.setdefault("warnings", []).append(
                f"gasket {g['name']}: one edge only (nothing inside it to seal): lift-off is when all of it opens; "
                "no leak pressure")
    for s in range(nsteps):
        rec = {"step": s + 1, "load": round(loads[s], 9), "bolts": {}, "gaskets": {}}
        i0 = 0
        for brec in plan["bolts"]:
            n = len(brec["bolts"])
            fv = forces[s, i0:i0 + n]
            i0 += n
            if np.isfinite(fv).any():
                rec["bolts"][brec["name"]] = {"min_N": round(float(np.nanmin(fv)), 3), "max_N": round(float(np.nanmax(fv)), 3),
                                              "mean_N": round(float(np.nanmean(fv)), 3)}
        pmap = cp.get(s + 1)
        for g in plan["gaskets"]:
            adj, area, inner, outer = graphs[g["name"]]
            if pmap is None:
                continue
            pv = {n: max(0.0, float(pmap.get(n, 0.0))) for n in area}
            A = sum(area.values())
            touching = sum(a for n, a in area.items() if pv[n] > 0)
            force = sum(pv[n] * a for n, a in area.items())
            vals = np.array(list(pv.values()))
            # a gasket with an inner and outer edge: the ring stress; with one edge (no cavity inside it): the
            # largest stress (the joint lifts off when all of it opens; it cannot leak)
            ring = ring_stress(adj, pv, inner, outer) if outer else float(vals.max())
            seal = g["seal"]
            thr = max(float(seal.get("stress", 0.0)), float(seal.get("m", 0.0)) * (loads[s] if unit == "MPa" else 0.0))
            if not outer:
                thr = 0.0
            rec["gaskets"][g["name"]] = {"ring_MPa": None if math.isinf(ring) else round(ring, 6),
                                         "min_MPa": round(float(vals.min()), 6), "max_MPa": round(float(vals.max()), 6),
                                         "mean_MPa": round(force / A, 6), "contact_fraction": round(touching / A, 6),
                                         "force_N": round(force, 3), "seal_MPa": round(thr, 6),
                                         "leak": bool(not math.isinf(ring) and ring < thr),
                                         "open": bool(not math.isinf(ring) and ring <= 1e-9)}
        res["steps"].append(rec)
    # bolts summary
    i0 = 0
    for brec in plan["bolts"]:
        n = len(brec["bolts"])
        res["bolts"][brec["name"]] = {"size": brec["size"], "preload_N": round(brec["preload_N"], 3), "count": n,
                                      "stiffness_N_mm": round(brec["k_bolt_N_mm"], 3),
                                      "positions": [b_["B1"] for b_ in brec["bolts"]],
                                      "max_N": round(float(np.nanmax(forces[:, i0:i0 + n])), 3) if np.isfinite(forces[:, i0:i0 + n]).any() else None}
        i0 += n
    # leak and lift-off loads, by gasket
    for g in plan["gaskets"]:
        steps = [r for r in res["steps"] if g["name"] in r["gaskets"]]
        if not steps or steps[0]["gaskets"][g["name"]]["ring_MPa"] is None:
            res["leak"][g["name"]] = res["lift_off"][g["name"]] = None
            continue
        xs = [r["load"] for r in steps]
        ring = [r["gaskets"][g["name"]]["ring_MPa"] for r in steps]
        seal = [r["gaskets"][g["name"]]["seal_MPa"] for r in steps]
        leak = _crossing(xs, [a - b for a, b in zip(ring, seal)]) if any(seal) else None
        lift = _crossing(xs, ring, floor=0.0)
        top = xs[-1]
        res["leak"][g["name"]] = (round(leak, 6) if leak is not None else {"above": top}) if any(seal) else None
        res["lift_off"][g["name"]] = round(lift, 6) if lift is not None else {"above": top}
        pre = steps[0]["gaskets"][g["name"]]
        res["gaskets"][g["name"]]["preload"] = {k: pre[k] for k in ("ring_MPa", "min_MPa", "max_MPa", "mean_MPa",
                                                                     "contact_fraction", "force_N")}
        if pre["ring_MPa"] is not None and pre["seal_MPa"] and pre["ring_MPa"] < pre["seal_MPa"]:
            res.setdefault("warnings", []).append(
                f"gasket {g['name']}: already below its sealing stress at the preload (ring {pre['ring_MPa']:.4g} MPa "
                f"< {pre['seal_MPa']:.4g} MPa): more preload, more bolts, or a narrower gasket")
    res["bolt_preload_error"] = res_pre_err
    # stabilisation springs: what they carry (k u at each spring node) against the preload
    stp = plan.get("stabilisation") or {}
    if stp.get("springs"):
        us = [b for b in dat["blocks"] if b["set"] == "N_STAB" and b["what"].startswith("displacements")]
        tot_pre = sum(rec["preload_N"] * len(rec["bolts"]) for rec in plan["bolts"])
        by_step = []
        for blk in us[:nsteps]:
            f = [stp["k_N_mm"] * float(np.linalg.norm(r[1:4])) for r in blk["rows"] if len(r) >= 4]
            by_step.append(round(float(np.sum(f)), 6))
        mx = max(by_step) if by_step else None
        res["stabilisation"] = {"springs": stp["springs"], "k_N_mm": stp["k_N_mm"],
                                "parts": {k: len(v) for k, v in stp["parts"].items()},
                                "force_N_by_step": by_step, "max_force_N": mx,
                                "fraction_of_preload": round(mx / tot_pre, 8) if mx is not None and tot_pre else None}
        fr = res["stabilisation"]["fraction_of_preload"]
        if fr is not None and fr > STAB_WARN:
            res.setdefault("warnings", []).append(
                f"the stabilisation springs carry {100 * fr:.3g} % of the preload ({mx:.4g} N): a part is not held "
                f"by its bolts and gaskets alone ({', '.join(stp['parts'])}); check the gasket faces touch and the bolts' faces")
    for g in plan["gaskets"]:
        res["gaskets"][g["name"]]["gap_mm"] = (g.get("gap") or {}).get("median_mm")
        if gap_warning(g):
            res.setdefault("warnings", []).append(gap_warning(g))
        law = g.get("law") or {}
        res["gaskets"][g["name"]]["law"] = law
        if law.get("beyond_curve"):
            res.setdefault("warnings", []).append(
                f"gasket {g['name']}: the mean preload stress ({law['at_MPa']:.4g} MPa) is beyond its curve's last point: "
                "the secant uses the curve's last segment extended; give the curve up to that stress")
        # the secant reproduces the preload's mean stress; check what the gasket took against it
        pre_mean = res["gaskets"][g["name"]].get("preload", {}).get("mean_MPa")
        if law.get("kind") == "secant" and pre_mean and abs(pre_mean - law["at_MPa"]) > 0.2 * law["at_MPa"]:
            res.setdefault("warnings", []).append(
                f"gasket {g['name']}: its mean stress at the preload ({pre_mean:.4g} MPa) differs from the stress its "
                f"secant was taken at ({law['at_MPa']:.4g} MPa): other bolts or loads press it; give 'stiffness' instead")
    if res_pre_err is not None and res_pre_err > 0.01:
        res.setdefault("warnings", []).append(
            f"bolt forces at the preload differ from the preload by {100 * res_pre_err:.3g} %: check the bolt results")
    # checks: the gasket force balances the bolts at the preload (no other load across the joint then)
    tot_b = sum(r["preload_N"] * r["count"] for r in res["bolts"].values())
    tot_g = sum(v.get("preload", {}).get("force_N", 0.0) for v in res["gaskets"].values())
    if tot_b > 0 and tot_g > 0:
        res["preload_balance_error"] = round(abs(tot_g - tot_b) / tot_b, 6)
    return res


# ---- the state for pictures (render.py) ------------------------------------------------------------
def save_state(folder: Path, study: dict, parts: list[dict], plan: dict, res: dict, frd_text: str) -> None:
    """state.npz / state.json as render.py reads them, for the whole assembly: the merged nodes, elements and
    boundary triangles (face tags made unique per part), and per step (1 = the preload) the displacements U_steps,
    the nodal stress tensors S_steps (from the displacements, each part with its material) and the gasket stress
    cp_steps (contact pressure at the gasket nodes, 0 elsewhere); the bcs, gaskets and bolts for the setup view."""
    from types import SimpleNamespace

    from .render import nodal_stress
    from .render import save_state as save
    X = all_nodes(parts)
    N = len(X)
    tets = np.concatenate([p["mesh"].tets + p["n0"] for p in parts])
    surfaces = {}
    for i, p in enumerate(parts):
        for tg, tri in p["mesh"].surfaces.items():
            surfaces[(i + 1) * TAG_STRIDE + int(tg)] = np.asarray(tri, np.int64).reshape(-1, 3) + p["n0"]
    merged = SimpleNamespace(nodes=X, tets=tets, surfaces=surfaces,
                             size=float(np.linalg.norm(X.max(axis=0) - X.min(axis=0))))
    blocks = frd_steps(frd_text)
    nsteps = len(plan["steps"])
    U = np.zeros((nsteps, N, 3))
    for b in blocks:
        if b["name"].startswith("DISP") and b["step"] and 1 <= b["step"] <= nsteps:
            for n, v in b["data"].items():
                if 1 <= n <= N:
                    U[b["step"] - 1, n - 1] = v[:3]
    cp = np.zeros((nsteps, N))
    for s, pm in contact_pressure_by_step(blocks).items():
        if s and 1 <= s <= nsteps:
            for n, v in pm.items():
                if 1 <= n <= N:
                    cp[s - 1, n - 1] = max(0.0, v)
    S = np.zeros((nsteps, N, 6))
    for s in range(nsteps):
        for p in parts:
            sl = slice(p["n0"], p["n0"] + len(p["mesh"].nodes))
            S[s, sl] = nodal_stress(p["mesh"], U[s, sl], {"material": p["material"]}, {})
    fields = {"U_steps": U.astype(np.float32), "S_steps": S.astype(np.float32), "cp_steps": cp.astype(np.float32)}
    bcs = [{"name": n, "kind": plan["kinds"][n], "tags": plan.get("bc_tags", {}).get(n, [])} for n in plan["kinds"]]
    bcs += [{"name": g["name"], "kind": "gasket", "tags": g["tags"]} for g in plan["gaskets"]]
    sw = res.get("swept", {})
    meta = {"bcs": bcs, "joint": True,
            "steps": [{"step": s["step"], "load": r["load"]} for s, r in zip(plan["steps"], res["steps"])],
            "unit": sw.get("unit"), "swept": sw.get("bc"),
            "gaskets": {g["name"]: {"normal": g["normal"], "with_tags": g["with_tags"],
                                    "seal_MPa_steps": [max(float(g["seal"].get("stress", 0.0)), float(g["seal"].get("m", 0.0))
                                                           * (r["load"] if sw.get("unit") == "MPa" else 0.0))
                                                       for r in res["steps"]]}
                        for g in plan["gaskets"]},
            "bolts": [[b_["B1"], b_["B2"]] for rec in plan["bolts"] for b_ in rec["bolts"]],
            "parts": [{"name": p["name"], "tag_min": (i + 1) * TAG_STRIDE, "tag_max": (i + 2) * TAG_STRIDE - 1}
                      for i, p in enumerate(parts)]}
    save(folder, merged, fields, study, meta)


# ---- validation ----------------------------------------------------------------------------------
def _psel(sel, parts: set, where: str) -> list[str]:
    errs = []
    for s in (sel if isinstance(sel, list) else [sel]):
        if not isinstance(s, dict) or s.get("part") not in parts:
            errs.append(f"{where}: each selector needs 'part' (one of {', '.join(sorted(parts)) or 'the parts'})")
            continue
        bad = _selector({k: v for k, v in s.items() if k != "part"})
        if bad:
            errs.append(f"{where}: {bad}")
    return errs


def validate(st: dict) -> tuple[list[str], list[str]]:
    """Errors and warnings of a joint study."""
    errs, warns = [], []
    parts = st.get("parts")
    if not isinstance(parts, dict) or len(parts) < 2:
        return ["parts: {name: {part, material, mesh, move}} with at least two parts (a joint is between parts)"], []
    for pn, p in parts.items():
        w = f"parts.{pn}"
        if not re.fullmatch(r"[A-Za-z0-9_]+", pn):
            errs.append(f"{w}: a part name is a word (letters, digits, _)")
        if not isinstance(p, dict) or not isinstance(p.get("part"), dict):
            errs.append(f"{w}: needs part: {{step}} | {{goal}} | {{box}}")
            continue
        pp = p["part"]
        if not ("step" in pp or "goal" in pp or "box" in pp):
            errs.append(f"{w}.part: {{step: path}}, {{goal: build123d-translator goal}} or {{box: [lx, ly, lz]}}")
        m = p.get("material", "steel")
        if isinstance(m, str) and m not in MATERIALS:
            errs.append(f"{w}.material: unknown '{m}'")
        else:
            try:
                mat = material_of({"material": m})
                if not (_num(mat.get("E")) and _num(mat.get("nu"))):
                    errs.append(f"{w}.material: needs E and nu")
            except Exception:  # noqa: BLE001
                errs.append(f"{w}.material: a name or {{E, nu, ...}}")
        if "move" in p and not _point(p["move"]):
            errs.append(f"{w}.move: [dx, dy, dz]")
        extra = set(p) - {"part", "material", "mesh", "move"}
        if extra:
            errs.append(f"{w}: unknown keys {sorted(extra)}")
    names = set(parts)
    bcs = st.get("bcs", [])
    bnames, nsup = [], 0
    for i, bc in enumerate(bcs if isinstance(bcs, list) else []):
        k = bc.get("kind") if isinstance(bc, dict) else None
        w = f"bc {i + 1} ({k})"
        if k not in SUPPORTS + LOADS:
            errs.append(f"bc {i + 1}: in a joint, kind is one of {', '.join(SUPPORTS + LOADS)}")
            continue
        bnames.append(bc_name(i, bc))
        if k != "gravity":
            if "faces" not in bc:
                errs.append(f"{w}: needs faces")
            else:
                errs += _psel(bc["faces"], names, f"{w} faces")
        nsup += k in SUPPORTS
        if k == "pressure" and not _num(bc.get("p")):
            errs.append(f"{w}: 'p' (MPa; for the swept bc: the last step's value)")
        if k == "force" and not any(_num(bc.get(c)) for c in ("fx", "fy", "fz")):
            errs.append(f"{w}: fx, fy or fz (N)")
        if k == "displacement" and not ({"ux", "uy", "uz"} & set(bc)):
            errs.append(f"{w}: ux, uy or uz")
    if not nsup:
        errs.append("bcs: the assembly needs supports (fixed, slider or displacement) on at least one part")
    sw = st.get("sweep")
    if not isinstance(sw, dict) or sw.get("bc") not in bnames:
        errs.append("sweep: {bc: the name of the pressure (or force) bc to step up, steps: n (10)}")
    else:
        bc = next(b for i, b in enumerate(bcs) if bc_name(i, b) == sw["bc"])
        if bc.get("kind") not in ("pressure", "force"):
            errs.append("sweep.bc: a pressure or force bc")
        if "steps" in sw and (not _num(sw["steps"]) or int(sw["steps"]) != sw["steps"] or not 1 <= sw["steps"] <= 60):
            errs.append("sweep.steps: a whole number from 1 to 60")
        if set(sw) - {"bc", "steps"}:
            errs.append(f"sweep: unknown keys {sorted(set(sw) - {'bc', 'steps'})}")
    gaskets = st.get("gaskets", [])
    if not isinstance(gaskets, list) or not gaskets:
        errs.append("gaskets: a list of {name, faces, with, curve | stiffness, seal, friction}")
        gaskets = []
    for i, g in enumerate(gaskets):
        w = f"gasket {i + 1}"
        if not isinstance(g, dict):
            errs.append(f"{w}: an object")
            continue
        if not isinstance(g.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9_]+", g.get("name", "")):
            errs.append(f"{w}: 'name' (a word)")
        for key in ("faces", "with"):
            if key not in g:
                errs.append(f"{w}: needs '{key}' (the {'gasket side' if key == 'faces' else 'mating flange'} face)")
            else:
                errs += _psel(g[key], names, f"{w} {key}")
        if ("curve" in g) == ("stiffness" in g):
            errs.append(f"{w}: give 'curve' [[closure mm, stress MPa], ...] or 'stiffness' (MPa/mm)")
        if "curve" in g:
            c = g["curve"]
            ok = isinstance(c, list) and len(c) >= 2 and all(isinstance(q, (list, tuple)) and len(q) == 2 and all(_num(v) for v in q) for q in c)
            if not ok:
                errs.append(f"{w}.curve: at least two [closure mm, stress MPa] points")
            else:
                pts = sorted(c)
                if any(b[0] <= a[0] or b[1] < a[1] for a, b in zip(pts, pts[1:])) or pts[0][0] < 0 or pts[0][1] < 0:
                    errs.append(f"{w}.curve: closures increasing from >= 0, stresses not decreasing")
        if "stiffness" in g and (not _num(g["stiffness"]) or g["stiffness"] <= 0):
            errs.append(f"{w}.stiffness: MPa/mm, > 0")
        seal = g.get("seal")
        if not isinstance(seal, dict) or not ({"stress", "m"} & set(seal)):
            errs.append(f"{w}.seal: {{stress: minimum sealing stress MPa}} and/or {{m: gasket factor (seal at m x p)}}")
        elif any(k not in ("stress", "m") or not _num(v) or v < 0 for k, v in seal.items()):
            errs.append(f"{w}.seal: stress (MPa) and m, numbers >= 0")
        if "friction" in g and (not _num(g["friction"]) or not 0 <= g["friction"] <= 1):
            errs.append(f"{w}.friction: 0 to 1 (0.2)")
        if "inner_at" in g and not _point(g["inner_at"]):
            errs.append(f"{w}.inner_at: [x, y, z] near the gasket's pressurized edge")
        extra = set(g) - {"name", "faces", "with", "curve", "stiffness", "seal", "friction", "inner_at"}
        if extra:
            errs.append(f"{w}: unknown keys {sorted(extra)}")
    bolts = st.get("bolts", [])
    if not isinstance(bolts, list) or not bolts:
        errs.append("bolts: a list of {name, size, preload | torque, at | pattern, head, nut | thread}")
        bolts = []
    for i, b in enumerate(bolts):
        w = f"bolts {i + 1}"
        if not isinstance(b, dict):
            errs.append(f"{w}: an object")
            continue
        if not isinstance(b.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9_]+", b.get("name", "")):
            errs.append(f"{w}: 'name' (a word)")
        if b.get("size") not in BOLTS:
            errs.append(f"{w}.size: one of {', '.join(BOLTS)}")
        if ("preload" in b) == ("torque" in b):
            errs.append(f"{w}: give 'preload' (N) or 'torque' (N m, with k: the nut factor, 0.2)")
        for key in ("preload", "torque", "k", "washer_d", "hole_d"):
            if key in b and (not _num(b[key]) or b[key] <= 0):
                errs.append(f"{w}.{key}: a positive number")
        if ("at" in b) == ("pattern" in b):
            errs.append(f"{w}: give 'at' [[x, y, z], ...] (points on the bolt axes) or 'pattern' {{center, pcd, n, start_deg}}")
        if "at" in b and (not isinstance(b["at"], list) or not b["at"] or not all(_point(q) for q in b["at"])):
            errs.append(f"{w}.at: a list of [x, y, z]")
        if "pattern" in b:
            pt = b["pattern"]
            if not (isinstance(pt, dict) and _point(pt.get("center")) and _num(pt.get("pcd")) and pt["pcd"] > 0
                    and _num(pt.get("n")) and int(pt["n"]) == pt["n"] and pt["n"] >= 1):
                errs.append(f"{w}.pattern: {{center: [x, y, z], pcd: mm, n: count, start_deg}}")
        if "axis" in b and not (_point(b["axis"]) and any(b["axis"])):
            errs.append(f"{w}.axis: [x, y, z] from the head towards the nut")
        if "head" not in b:
            errs.append(f"{w}: needs 'head' (the head's bearing face, e.g. {{part: cover, nz: 1}})")
        else:
            errs += _psel(b["head"], names, f"{w} head")
        if ("nut" in b) == ("thread" in b):
            errs.append(f"{w}: give 'nut' (the nut's bearing face) or 'thread' {{part, depth}} (a tapped hole)")
        if "nut" in b:
            errs += _psel(b["nut"], names, f"{w} nut")
        if "thread" in b:
            th = b["thread"]
            if not (isinstance(th, dict) and th.get("part") in names and _num(th.get("depth")) and th["depth"] > 0):
                errs.append(f"{w}.thread: {{part, depth: engaged thread length mm}}")
        if "material" in b:
            try:
                mt = material_of({"material": b["material"]})
                if not _num(mt.get("E")):
                    errs.append(f"{w}.material: needs E")
            except Exception:  # noqa: BLE001
                errs.append(f"{w}.material: a name or {{E, nu}}")
        extra = set(b) - {"name", "size", "preload", "torque", "k", "at", "pattern", "axis", "head", "nut", "thread",
                          "washer_d", "hole_d", "material"}
        if extra:
            errs.append(f"{w}: unknown keys {sorted(extra)}")
    if st.get("converge"):
        errs.append("converge: not available for joint studies yet")
    extra = set(st) - JOINT_KEYS
    if extra:
        warns.append(f"unknown study keys ignored: {sorted(extra)}")
    return errs, warns


# ---- watching CalculiX: stop a run that cannot converge -----------------------------------------------
MAX_FAILED_ATTEMPTS = 3   # unconverged attempts of one increment before the run is stopped
CHATTER_ITERS = 16        # iterations of one attempt looked at for contact chatter
CHATTER_FLIPS = 6         # direction changes of the contact count among them
CHATTER_SPREAD = 0.05     # the contact count's swing, relative to its largest value


def read_sta(text: str) -> list[tuple[int, int, int, bool]]:
    """(step, increment, attempt, converged) of each line of a .sta file (an attempt like '2U' did not converge)."""
    out = []
    for line in text.splitlines():
        tk = line.split()
        if len(tk) >= 6 and tk[0].isdigit() and tk[1].isdigit() and re.fullmatch(r"\d+U?", tk[2]):
            out.append((int(tk[0]), int(tk[1]), int(tk[2].rstrip("U")), not tk[2].endswith("U")))
    return out


def read_cvg(text: str) -> list[tuple[int, int, int, int, int, float]]:
    """(step, increment, attempt, iteration, contact elements, residual force %) of each line of a .cvg file."""
    out = []
    for line in text.splitlines():
        tk = line.split()
        if len(tk) >= 6 and all(x.isdigit() for x in tk[:5]):
            try:
                out.append((int(tk[0]), int(tk[1]), int(tk[2]), int(tk[3]), int(tk[4]), float(tk[5])))
            except ValueError:
                continue
    return out


def _what(step: int, plan: dict | None) -> str:
    if step == 1:
        return "preload"
    st = (plan or {}).get("steps") or []
    f = st[step - 1]["factor"] if step - 1 < len(st) else None
    sw = (plan or {}).get("sweep") or {}
    return f"step {step}" + (f" ({sw.get('bc')} at {100 * f:.0f} %)" if f is not None else "")


def stall_reason(sta: str, cvg: str, plan: dict | None = None) -> tuple[str, int] | None:
    """Why the run should stop now (and the step it is in), or None: an increment that failed MAX_FAILED_ATTEMPTS
    times, or contact chatter: the number of contact elements swinging up and down in the current attempt."""
    rows = read_cvg(cvg)
    if rows:
        s, i, a = rows[-1][:3]
        cur = [r for r in rows if r[:3] == (s, i, a)]
        if len(cur) >= CHATTER_ITERS:
            c = np.array([r[4] for r in cur[-CHATTER_ITERS:]], float)
            d = np.sign(np.diff(c))
            d = d[d != 0]
            flips = int((d[1:] != d[:-1]).sum()) if len(d) > 1 else 0
            lo, hi = int(c.min()), int(c.max())
            if flips >= CHATTER_FLIPS and hi > 0 and (hi - lo) / hi >= CHATTER_SPREAD:
                return (f"{_what(s, plan)} did not converge: contact chatter ({hi - lo} contact elements switching, "
                        f"{lo} to {hi}; step {s}, increment {i}, attempt {a}, iteration {cur[-1][3]})"), s
    failed: dict = {}
    for s, i, a, ok in read_sta(sta):
        if not ok:
            failed[(s, i)] = failed.get((s, i), 0) + 1
    for (s, i), n in failed.items():
        if n >= MAX_FAILED_ATTEMPTS:
            return f"{_what(s, plan)} did not converge: increment {i} of step {s} failed {n} attempts", s
    return None


class Watch:
    """A monitor for runner.run_ccx: reads the .sta and .cvg files while CalculiX runs."""

    def __init__(self, inp: Path, plan: dict):
        self.inp, self.plan, self.step = Path(inp), plan, None

    def __call__(self) -> str | None:
        rd = lambda sfx: self.inp.with_suffix(sfx).read_text(errors="replace") if self.inp.with_suffix(sfx).is_file() else ""  # noqa: E731
        r = stall_reason(rd(".sta"), rd(".cvg"), self.plan)
        if r is None:
            return None
        self.step = r[1]
        return r[0]


def completed_steps(sta: str) -> int:
    """The last step whose final increment converged (step time 1)."""
    done = 0
    for line in sta.splitlines():
        tk = line.split()
        if len(tk) >= 7 and tk[0].isdigit() and tk[1].isdigit() and re.fullmatch(r"\d+", tk[2]):
            try:
                if abs(float(tk[5]) - 1.0) < 1e-6:
                    done = max(done, int(tk[0]))
            except ValueError:
                continue
    return done


# ---- one joint study end to end (runner.solve_once calls this) -----------------------------------
def solve(name: str, study: dict, folder: Path, solver: str, timeout: float, progress=None) -> dict:
    """Meshes the parts, writes and runs the deck (CalculiX: contact and pretension), evaluates the joint."""
    import time

    import subprocess

    from .runner import SolverError, Stopped, _record_timing, check_cancel, run_ccx
    if solver != "ccx":
        raise SolverError("a joint study needs CalculiX (contact, bolt pretension): the reference solver cannot run it")
    check_cancel()
    if progress:
        progress.stage("mesh")
    t0 = time.perf_counter()
    parts = mesh_parts(study, Path(folder))
    n_nodes = sum(len(p["mesh"].nodes) for p in parts)
    t1 = time.perf_counter()
    if progress:
        progress.stage("solve", nodes=n_nodes)
    text, plan = write(name, study, parts)
    inp = Path(folder) / f"{name}.inp"
    inp.write_text(text)
    check_cancel()
    watch = Watch(inp, plan)
    stopped = None
    try:
        run_ccx(inp, timeout, monitor=watch)
    except (Stopped, subprocess.TimeoutExpired) as exc:
        sta = inp.with_suffix(".sta")
        done = completed_steps(sta.read_text(errors="replace")) if sta.is_file() else 0
        why = str(exc) if isinstance(exc, Stopped) else f"CalculiX ran past the time limit ({timeout:g} s) in {_what(done + 1, plan)}"
        if done < 1 or not inp.with_suffix(".dat").is_file():   # not even the preload: nothing to report
            gaps = [w for w in (gap_warning(g) for g in plan["gaskets"]) if w]
            raise SolverError(why + ". " + (" ".join(gaps) + ". " if gaps else "")
                              + "Check that each gasket's faces touch (parts.move), that every part is held by "
                              "supports or bolts, and the start of the gasket curve; the stabilisation springs were on."
                              ) from exc
        stopped = (why, done)
        plan = dict(plan, steps=plan["steps"][:done])
    t2 = time.perf_counter()
    res = evaluate(study, parts, plan, inp.with_suffix(".dat").read_text(errors="replace"),
                   inp.with_suffix(".frd").read_text(errors="replace") if inp.with_suffix(".frd").is_file() else "")
    if stopped:
        res["stopped"] = {"why": stopped[0], "completed_steps": stopped[1]}
        res.setdefault("warnings", []).append(
            f"{stopped[0]}: the results stop at step {stopped[1]} ({res['steps'][-1]['load']:.4g} {res['swept']['unit']}); "
            "leak and lift-off beyond it are not known")
    minfo = {"nodes": n_nodes, "elements": sum(len(p["mesh"].tets) for p in parts), "order": parts[0]["mesh"].order,
             "parts": {p["name"]: {k: v for k, v in p["info"].items() if k in ("nodes", "elements", "size", "mesh_notes",
                                                                              "straightened_elements")} for p in parts}}
    _record_timing({"analysis": "joint"}, solver, minfo, t1 - t0, t2 - t1)
    res["selection"] = plan["selection"]
    res["mesh"] = minfo
    res["seconds"] = {"mesh": round(t1 - t0, 3), "solve": round(t2 - t1, 3), "total": round(time.perf_counter() - t0, 3)}
    res["files"] = {k: str(inp.with_suffix(s)) for k, s in (("inp", ".inp"), ("dat", ".dat"), ("frd", ".frd"))
                    if inp.with_suffix(s).is_file()}
    res["deck"] = {"steps": len(plan["steps"]), "bolts": sum(len(b["bolts"]) for b in plan["bolts"]),
                   "gaskets": len(plan["gaskets"])}
    try:   # the state for pictures (images, render_images): the results stand without it
        frd = inp.with_suffix(".frd")
        save_state(Path(folder), study, parts, plan, res, frd.read_text(errors="replace") if frd.is_file() else "")
        res["files"]["state"] = str(Path(folder) / "state.npz")
    except Exception as exc:  # noqa: BLE001
        res.setdefault("warnings", []).append(f"could not save the state for pictures: {type(exc).__name__}: {exc}"[:300])
    return res
