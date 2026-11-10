"""joint.py — assemblies of parts (analysis "assembly", and "joint" for bolted, gasketed joints; CalculiX only):
parts in contact (hard, penalty or bonded, with friction, clearance or interference), gaskets, bolts and nuts with
pretension, and a list of steps: the preload, static steps with their loads, a load swept up in steps (a joint: to
find where it leaks and lifts off), and modal steps on the preloaded, loaded state (*FREQUENCY, PERTURBATION).
A joint study is an assembly whose steps are [preload, sweep].
Version: 2026.11.10.3

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
from .meshdata import EDGE_MID, EDGES, FACES, Mesh, MeshError, fit_cylinder
from .study import G_SCALE, MATERIALS, _num, _point, _selector, bc_name, material_of, mesh_errors, to_deck_units

# ISO 261 coarse threads: nominal diameter d, pitch P (mm); ISO 273 medium clearance; ISO 7089 washer outer diameter
BOLTS = {"M3": (3, 0.5, 3.4, 7), "M4": (4, 0.7, 4.5, 9), "M5": (5, 0.8, 5.5, 10), "M6": (6, 1.0, 6.6, 12),
         "M8": (8, 1.25, 9.0, 16), "M10": (10, 1.5, 11.0, 20), "M12": (12, 1.75, 13.5, 24), "M14": (14, 2.0, 15.5, 28),
         "M16": (16, 2.0, 17.5, 30), "M18": (18, 2.5, 20.0, 34), "M20": (20, 2.5, 22.0, 37), "M22": (22, 2.5, 24.0, 39),
         "M24": (24, 3.0, 26.0, 44), "M27": (27, 3.0, 30.0, 50), "M30": (30, 3.5, 33.0, 56), "M33": (33, 3.5, 36.0, 60),
         "M36": (36, 4.0, 39.0, 66)}
JOINT_KEYS = {"parts", "analysis", "bcs", "bolts", "gaskets", "sweep", "mesh", "expect", "notes", "procs", "queue", "run",
              "equation_solver", "images"}
ASSEMBLY_KEYS = (JOINT_KEYS - {"sweep"}) | {"contacts", "nuts", "steps", "probes"}
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
        sub = {"part": spec["part"], "mesh": {**study.get("mesh", {}), **spec.get("mesh", {})}}   # the part's own over the study's
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
    else:   # a bolt: into the joint, against the head bearing face's outward normal; a nut: out of the face it
        # bears on, towards the thread it grips
        g = hp["mesh"].groups[htags[0]]
        if not g.planar:
            raise MeshError(f"{b.get('_kind', 'bolt')}s {b['name']}: the bearing face is not flat; give 'axis'")
        axis = (1.0 if b.get("_kind") == "nut" else -1.0) * _unit(g.normal[0])
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
STAB_FRACTION_E = 1e-9    # without bolts: each 1e-9 of E x the element size (N/mm)
STAB_NODES = 6            # per part that no support holds: 6 well-spread nodes, 3 springs each
STAB_WARN = 0.01          # warn when the springs carry more than 1 % of the preload
GAP_TOL = 0.01            # mm: gasket faces further apart than this (median) do not touch at the start
HARD_FACTOR = 10.0        # 'hard' contact: a linear penalty of 10 E / h (the softer part's E, the finer element size h)
NUT_STIFF = 10.0          # a nut on a meshed shaft: a connector of 10 x E A_s / d, stiff against the shaft it grips
STEP_KINDS = ("preload", "static", "sweep", "modal")
CONTACT_LAWS = ("hard", "bonded")


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


def connectors(study: dict) -> list[dict]:
    """The bolts and the nuts as connectors. A nut bears on a part ('bears_on') and grips a meshed shaft's thread:
    it pushes them apart (the clamp runs through the parts between the nut and the shaft's shoulder), with its own
    stiffness (NUT_STIFF x E A_s / d), since the shaft itself is meshed."""
    out = [dict(b, _kind="bolt") for b in study.get("bolts", []) or []]
    for n in study.get("nuts", []) or []:
        c = {k: v for k, v in n.items() if k != "bears_on"}
        c.update(head=n["bears_on"], _kind="nut")
        out.append(c)
    return out


def study_steps(study: dict) -> list[dict]:
    """The study's steps: a joint: the preload, then its sweep; an assembly: its 'steps', with a preload first when
    it has bolts or nuts."""
    if study.get("analysis") == "joint":
        sw = study.get("sweep", {}) or {}
        return [{"kind": "preload"}, {"kind": "sweep", "bc": sw.get("bc"), "steps": int(sw.get("steps", 10))}]
    steps = [dict(s) for s in study.get("steps", []) or [] if isinstance(s, dict)]
    if (study.get("bolts") or study.get("nuts")) and (not steps or steps[0].get("kind") != "preload"):
        steps = [{"kind": "preload"}] + steps
    return steps


def deck_steps(study: dict) -> list[dict]:
    """The CalculiX steps: {step, kind, what, factor (a sweep's), loads {bc: factor}, modes}. A load bc named by no
    static step and no sweep acts in every step (from the preload on); a static step adds the loads it names (x its
    scale), a sweep its bc in equal parts; loads do not carry over from one static step to the next."""
    load_names = [bc_name(i, b) for i, b in enumerate(study.get("bcs", []) or []) if isinstance(b, dict)
                  and b.get("kind") in LOADS]
    steps = study_steps(study)
    used = set()
    for s in steps:
        if s.get("kind") == "static":
            used |= set(s.get("loads", []) or [])
        elif s.get("kind") == "sweep":
            used.add(s.get("bc"))
    const = {n: 1.0 for n in load_names if n not in used}
    out = []
    for s in steps:
        k = s.get("kind")
        inc = {q: float(s[q]) for q in ("first_increment", "max_increment") if _num(s.get(q))}
        if k == "preload":
            out.append({"kind": "preload", "what": "preload", "factor": 0.0, "loads": dict(const), **inc})
        elif k == "static":
            f = float(s.get("scale", 1.0))
            out.append({"kind": "static", "what": s.get("name") or "static", "name": s.get("name"), "factor": None,
                        "loads": {**const, **{n: f for n in s.get("loads", []) or []}}, **inc})
        elif k == "sweep":
            n = int(s.get("steps", 10))
            out += [{"kind": "sweep", "what": "swept load", "factor": j / n, "loads": {**const, s["bc"]: j / n}}
                    for j in range(1, n + 1)]
        elif k == "modal":
            out.append({"kind": "modal", "what": f"modal ({int(s.get('modes', 6))} modes)", "factor": None,
                        "modes": int(s.get("modes", 6)), "loads": {}, "name": s.get("name")})
    k = 0
    for i, s in enumerate(out):
        s["step"] = i + 1
        if s["kind"] != "modal":   # modal steps are separate linear runs: the deck's own step numbers skip them
            k += 1
            s["ccx_step"] = k
        else:
            s["ccx_step"] = None
    return out


def contact_law(c: dict, parts: list[dict], slave, master) -> dict:
    """A contact's linear law: 'hard' (a penalty of HARD_FACTOR x E / h, E and h the softer and finer of its parts),
    'bonded' (the same stiffness, tied in every direction: CalculiX's TIED), or {stiffness: MPa/mm}."""
    law = c.get("law", "hard")
    if isinstance(law, dict):
        return {"kind": "penalty", "stiffness_MPa_mm": float(law["stiffness"])}
    ps = [p for p, _ in slave] + [p for p, _ in master]
    E = min(float(p["material"]["E"]) for p in ps)
    h = min(float(p["info"].get("size") or 1.0) for p in ps)
    return {"kind": law, "stiffness_MPa_mm": HARD_FACTOR * E / h, "E_MPa": E, "h_mm": h}


FIT_METHODS = ("expansion", "overlap")


def fit_geometry(parts: list[dict], c: dict, X: np.ndarray) -> dict | None:
    """A cylindrical interference fit's geometry: {inner (the part with the convex face), r, axis, point, t: (t0, t1)
    the axial span both faces share, elements (the inner part's, 0-based global, whose centroids lie in that span)},
    or None when the pair is not one convex and one concave cylinder of two different parts about one axis."""
    groups = [(p, p["mesh"].groups[t]) for key in ("faces", "with") for p, tg in select(parts, c[key]) for t in tg]
    if not groups or any(getattr(g, "concave", None) is None or not g.radius for _, g in groups):
        return None
    convex = [(p, g) for p, g in groups if not g.concave]
    concave = [(p, g) for p, g in groups if g.concave]
    inner = {p["name"] for p, _ in convex}
    if not convex or not concave or len(inner) != 1 or inner & {p["name"] for p, _ in concave}:
        return None
    pin = convex[0][0]
    nodes = np.unique(np.concatenate([g.nodes() + p["n0"] for p, g in convex]))
    fit = fit_cylinder(X[nodes], np.concatenate([g.normal for _, g in convex]))
    if fit is None:
        return None
    r, axis, point = float(np.mean([g.radius for _, g in convex])), np.asarray(fit[1], float), np.asarray(fit[2], float)
    axis = axis / np.linalg.norm(axis)

    def span(sel):
        t = (X[np.unique(np.concatenate([g.nodes() + p["n0"] for p, g in sel]))] - point) @ axis
        return float(t.min()), float(t.max())
    (a0, a1), (b0, b1) = span(convex), span(concave)
    t0, t1 = max(a0, b0), min(a1, b1)
    if t1 <= t0:
        return None
    tets = pin["mesh"].tets[:, :4] + pin["n0"]
    cen = X[tets].mean(axis=1)
    tt = (cen - point) @ axis
    rad = np.linalg.norm((cen - point) - np.outer(tt, axis), axis=1)
    h = float(pin["info"].get("size") or 1.0)
    pick = np.nonzero((tt >= t0) & (tt <= t1) & (rad <= r + 0.5 * h))[0]
    if not len(pick):
        return None
    return {"inner": pin["name"], "r": r, "axis": axis, "point": point, "t": (t0, t1),
            "elements": (pick + pin["e0"]).tolist(), "part": pin}


def eigenstrain(plan: dict, p: dict) -> np.ndarray | None:
    """(E, 6: xx, yy, zz, 2xy, 2xz, 2yz) strain a part's interference fits put into its elements (a fit's in-plane
    expansion eps (I - a a^T) about its axis a, whole at the end of the first step), or None."""
    fl = (plan.get("_eigen") or {}).get(p["name"])
    if not fl:
        return None
    out = np.zeros((len(p["mesh"].tets), 6))
    for f in fl:
        a = np.asarray(f["axis"], float)
        T = float(f["strain"]) * (np.eye(3) - np.outer(a, a))
        out[f["elements"]] = [T[0, 0], T[1, 1], T[2, 2], 2 * T[0, 1], 2 * T[0, 2], 2 * T[1, 2]]
    return out


def write(name: str, study: dict, parts: list[dict], modal: dict | None = None) -> tuple[str, dict]:
    """The deck of a joint or assembly study, and its plan (node sets, connectors, gaskets, contacts, steps) for
    reading the results. modal {modes, ties: {pair names}, tol}: instead, the linear frequency run of a modal step
    on the state its previous step left: the pairs in ties (bonded, and closed then) tied with *TIE, the others free,
    the bolts and nuts at their preloaded length."""
    order = parts[0]["mesh"].order
    etype = "C3D10" if order == 2 else "C3D4"
    an = study.get("analysis", "joint")
    L = [f"** {name}: written by fea-translator {__version__} ({an}: parts in contact, bolt pretension)",
         "** units: mm, N, s, t (tonne); stresses MPa", "*NODE, NSET=NALL"]
    # an interference fit (a contact's negative clearance): its slave face's nodes moved by the interference along
    # their outward normals, into the master, so the parts overlap; node-to-surface contact then resolves it.
    # CalculiX's face-to-face contact with an interference on curved quadratic faces settled on states with the
    # interference not enforced (closed 0-19 %, pressures near 0) or did not converge, whether made with *CLEARANCE
    # or in the geometry; node-to-surface took it in 8 increments, 1.4 % off Lame (tools/interference_matrix.py,
    # mochi 2026.11.05)
    # A cylindrical fit takes its interference as an expansion instead (fit_method 'expansion', the default where the
    # faces are one convex and one concave cylinder): the inner part's elements under the fit get an orthotropic
    # expansion, radial and hoop, of interference / r per degree, and a temperature ramped from 0 to 1 over the first
    # step, so the interference grows with the step's increments instead of being there whole at its start (S50 tight
    # hub fit, 424k nodes, -0.021 mm: hard contact failed after 3 attempts of its first increment, a 2e5 MPa/mm
    # penalty was still on attempt 3 after 5.5 h). A uniform in-plane expansion eps of a disc or ring moves its outer
    # face by eps x r, the interference; axially it is free (no expansion along the axis)
    Xw = all_nodes(parts).copy()
    tets_all = np.concatenate([p["mesh"].tets + p["n0"] for p in parts])
    moved, fits = {}, {}
    for c in study.get("contacts", []) or []:
        if float(c.get("clearance", 0.0) or 0.0) < 0:
            geo = (fit_geometry(parts, c, Xw) if c.get("fit_method", "expansion") == "expansion" and modal is None
                   else None)
            taken = {e for f in fits.values() for e in f["elements"]}
            if geo is not None and not taken & set(geo["elements"]):
                geo["delta"] = abs(float(c["clearance"]))
                fits[c["name"]] = geo
                continue
            if modal is not None and c.get("fit_method", "expansion") == "expansion":
                continue    # the frequency run starts from the undeformed parts: the fit's state is in its ties
            nn, nrm = face_node_normals(Xw, tets_all, _gfaces(select(parts, c["faces"])))
            Xw[nn] += abs(float(c["clearance"])) * nrm
            moved[c["name"]] = len(nn)
    for p in parts:
        L += [f"{p['n0'] + i + 1}, {_g(x)}, {_g(y)}, {_g(z)}"
              for i, (x, y, z) in enumerate(Xw[p["n0"]:p["n0"] + len(p["mesh"].nodes)])]
    n_next = sum(len(p["mesh"].nodes) for p in parts) + 1
    e_next = sum(len(p["mesh"].tets) for p in parts) + 1
    for p in parts:
        L.append(f"*ELEMENT, TYPE={etype}, ELSET=E_{_name(p['name'])}")
        L += [f"{p['e0'] + e + 1}, " + ", ".join(str(p["n0"] + n + 1) for n in row) for e, row in enumerate(p["mesh"].tets)]
    plan = {"analysis": an, "sets": {}, "supports": [], "support_parts": [], "part_names": [q["name"] for q in parts],
            "loads": [], "kinds": {}, "bolts": [], "gaskets": [],
            "contacts": [], "probes": {}, "steps": [], "sweep": None, "selection": {}, "n_nodes": n_next - 1,
            "n_elements": e_next - 1, "element": etype, "load_mag": {}}
    dsteps = deck_steps(study)
    sweep_step = next((s for s in study_steps(study) if s.get("kind") == "sweep"), None)
    swept = sweep_step.get("bc") if sweep_step else None
    # boundary conditions: supports in step 1; each load's lines, applied per step
    bnd, held, full_parts, LD = [], set(), set(), {}
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
            if k == "fixed" or (k == "displacement" and all(c in bc for c in ("ux", "uy", "uz"))):
                full_parts.update(p["name"] for p, _ in sel)     # held in every direction: no stabilisation
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
            plan.setdefault("support_parts", [])
            plan["support_parts"] = sorted(set(plan["support_parts"]) | {q["name"] for q, _ in sel})
        elif k == "pressure":
            rows = [(e + 1, s) for e, s in _gfaces(sel)]
            area = sum(p["mesh"].area_of(tg) for p, tg in sel)
            c = np.concatenate([(p["mesh"].groups[t].centroid * p["mesh"].groups[t].area[:, None]).sum(axis=0, keepdims=True)
                                / p["mesh"].groups[t].total_area for p, tg in sel for t in tg]).mean(axis=0)
            LD[bname] = {"kind": "dload", "rows": rows, "p": float(bc["p"])}
            plan["load_mag"][bname] = abs(float(bc["p"])) * area
            plan["loads"].append(bname)
            if bname == swept:
                plan["sweep"] = {"bc": bname, "kind": "pressure", "max": float(bc["p"]),
                                 "steps": int(sweep_step.get("steps", 10)), "center": c.tolist()}
        elif k == "force":
            f = np.array([float(bc.get(c, 0.0)) for c in ("fx", "fy", "fz")])
            area = sum(p["mesh"].area_of(tg) for p, tg in sel)
            nodes_f = {}
            for p, tg in sel:
                fp = f * p["mesh"].area_of(tg) / area
                for n, v in p["mesh"].force_nodes(tg, fp).items():
                    nodes_f[n + p["n0"]] = nodes_f.get(n + p["n0"], np.zeros(3)) + v
            LD[bname] = {"kind": "cload", "nodes": nodes_f}
            plan["load_mag"][bname] = float(np.linalg.norm(f))
            plan["loads"].append(bname)
            if bname == swept:
                plan["sweep"] = {"bc": bname, "kind": "force", "max": float(np.linalg.norm(f)),
                                 "steps": int(sweep_step.get("steps", 10)), "center": X[_gnodes(sel)].mean(axis=0).tolist()}
        elif k == "gravity":
            gv = np.array([float(bc.get("gx", 0.0)), float(bc.get("gy", 0.0)), float(bc.get("gz", -9.81))])
            gm = float(np.linalg.norm(gv))
            LD[bname] = {"kind": "grav", "items": [(f"E_{_name(p['name'])}", gm * G_SCALE, gv / gm) for p in parts]}
            plan["load_mag"][bname] = 0.0
            plan["loads"].append(bname)
    # materials and sections (a fit's elements: the part's material with the fit's expansion, in a cylindrical
    # orientation about the fit's axis: 1 radial, 2 hoop, 3 axial)
    for p in parts:
        m = to_deck_units(p["material"])
        pn = _name(p["name"])
        own = [(cn_, f) for cn_, f in fits.items() if f["inner"] == p["name"]]
        L += [f"*MATERIAL, NAME=M_{pn}", "*ELASTIC", f"{_g(m['E'])}, {_g(m['nu'])}"]
        if "density" in m:
            L += ["*DENSITY", _g(m["density"])]
        if not own:
            L.append(f"*SOLID SECTION, ELSET=E_{pn}, MATERIAL=M_{pn}")
            continue
        infit = {e for _, f in own for e in f["elements"]}
        rest = [p["e0"] + e + 1 for e in range(len(p["mesh"].tets)) if p["e0"] + e not in infit]
        if rest:
            L += [f"*ELSET, ELSET=E_{pn}_REST"] + _lines(rest) + [f"*SOLID SECTION, ELSET=E_{pn}_REST, MATERIAL=M_{pn}"]
        for cn_, f in own:
            fn = _name(cn_)
            eps = f["delta"] / f["r"]
            a, b = f["point"], f["point"] + f["axis"] * max(f["r"], 1.0)
            L += [f"*ELSET, ELSET=EF_{fn}"] + _lines([e + 1 for e in f["elements"]])
            L += [f"*MATERIAL, NAME=MF_{fn}", "*ELASTIC", f"{_g(m['E'])}, {_g(m['nu'])}"]
            if "density" in m:
                L += ["*DENSITY", _g(m["density"])]
            L += ["*EXPANSION, TYPE=ORTHO, ZERO=0.", f"{_g(eps)}, {_g(eps)}, 0."]
            L += [f"*ORIENTATION, NAME=OF_{fn}, SYSTEM=CYLINDRICAL",
                  ", ".join(_g(v) for v in (*a, *b))]
            L.append(f"*SOLID SECTION, ELSET=EF_{fn}, MATERIAL=MF_{fn}, ORIENTATION=OF_{fn}")
            fnodes = np.unique(tets_all[f["elements"]])
            L += [f"*NSET, NSET=NF_{fn}"] + _lines((fnodes + 1).tolist())
    plan["_eigen"] = {}   # the fits' strains, per part, for the stresses recomputed from the displacements
    for cn_, f in fits.items():
        plan["_eigen"].setdefault(f["inner"], []).append(
            {"elements": [e - f["part"]["e0"] for e in f["elements"]], "strain": f["delta"] / f["r"],
             "axis": f["axis"].tolist()})
    plan["fits"] = {cn_: {"method": "expansion", "inner": f["inner"], "r_mm": round(f["r"], 4),
                          "interference_mm": f["delta"], "strain": round(f["delta"] / f["r"], 9),
                          "elements": len(f["elements"]), "span_mm": round(f["t"][1] - f["t"][0], 4)}
                    for cn_, f in fits.items()}
    # connectors (bolts, nuts): a spring (SPRINGA) from a cut node C at the head (a nut: its bearing face) to the nut or
    # thread end B2, the pretension an equation across the cut: u_C = mean(u_head) - u_PT * axis (CalculiX's own
    # *PRE-TENSION SECTION on a beam deletes the beam and leaves a rigid link: the bolt would have no stretch)
    springs, eqs, pret = [], [], []
    eq_nodes: set = set()
    gasket_nodes = {g["name"]: _gnodes(select(parts, g["faces"])) for g in study.get("gaskets", []) or []}
    for b in connectors(study):
        bn = _name(b["name"])
        sz = bolt_size(b)
        P = preload_N(b)
        nut = b["_kind"] == "nut"
        mat = to_deck_units(material_of({"material": b.get("material", "steel")}))
        geo = bolt_geometry(parts, b)
        rec = {"name": b["name"], "kind": b["_kind"], "size": b["size"], "preload_N": P, "As": sz["As"], "bolts": [],
               "sign": -1.0 if nut else 1.0}
        for bg in geo:
            n2, nc, npt = n_next, n_next + 1, n_next + 2
            n_next += 3
            L += ["*NODE"] + [f"{n}, {_g(q[0])}, {_g(q[1])}, {_g(q[2])}" for n, q in
                              ((n2, bg["B2"]), (nc, bg["B1"]), (npt, bg["B1"]))]
            eid = e_next
            e_next += 1
            k = mat["E"] * sz["As"] * (NUT_STIFF / sz["d"] if nut else 1.0 / bg["grip"])
            springs.append((eid, nc, n2, bn, k))
            eq_nodes.update(int(q) for q in bg["head"])
            eq_nodes.update(int(q) for q in bg["nut"])
            head = [int(q) + 1 for q in bg["head"]]
            for d in (1, 2, 3):
                # the far end B2 follows the mean of its bearing nodes
                eqs += _equation(int(bg["nut"][0]) + 1, [int(q) + 1 for q in bg["nut"]], n2, d, held)
                # the cut C at the head: the mean of the head's bearing nodes, less the pretension along the axis,
                # n u_C - sum(u_head) + n axis_d u_PT = 0. Each dependent DOF (B2, C) is in no other equation
                n = float(len(head))
                terms = [(nc, d, n)] + [(q, d, -1.0) for q in head] + (
                    [(npt, 1, n * float(bg["axis"][d - 1]))] if abs(bg["axis"][d - 1]) > 1e-12 else [])
                eqs += ["*EQUATION", str(len(terms))] + [", ".join(f"{q}, {dd}, {_g(c)}" for q, dd, c in terms[j:j + 4])
                                                         for j in range(0, len(terms), 4)]
            pret.append((eid, npt, bg["axis"], P * rec["sign"]))   # a nut pushes its faces apart
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
    has_preload = any(s["kind"] == "preload" for s in dsteps)
    stick = PRELOAD_STICK if has_preload else 1.0        # friction soft until the preload has pressed the faces
    # gaskets: face-to-face contact. CalculiX ignores a TABULAR pressure-overclosure law in face-to-face contact (the
    # parts pass through each other: joint_two_bolt and an S50 cover, 2026.11.02): a curve becomes the LINEAR law of
    # its secant at the gasket's mean stress under the preload (the bolts nearest to it, over its area)
    for g in study.get("gaskets", []) or []:
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
        if modal is None:
            L += [f"*SURFACE INTERACTION, NAME=GI_{gn}", "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
                  _g(law["stiffness_MPa_mm"])] + friction_lines(law["stiffness_MPa_mm"], g, stick)
            L += [f"*CONTACT PAIR, INTERACTION=GI_{gn}, TYPE=SURFACE TO SURFACE", f"GS_{gn}, GM_{gn}"]
        elif g["name"] in modal["ties"]:
            L += [f"*TIE, NAME=T_{gn}, POSITION TOLERANCE={_g(modal['tol'])}", f"GS_{gn}, GM_{gn}"]
        nrm = sum((p["mesh"].groups[tg_].normal * p["mesh"].groups[tg_].area[:, None]).sum(axis=0)
                  for p, tgs in slave for tg_ in tgs)
        gap = face_gap(X, sn, _gnodes(master), _unit(nrm))
        plan["gaskets"].append({"name": g["name"], "slave_faces": sf, "seal": g.get("seal", {}),
                                "parts": sorted({q["name"] for q, _ in slave} | {q["name"] for q, _ in master}),
                                "inner_at": g.get("inner_at"), "tags": gtags(parts, slave),
                                "with_tags": gtags(parts, master), "normal": _unit(nrm).tolist(),
                                "law": law, "friction": friction_lines(law["stiffness_MPa_mm"], g),
                                "interaction": f"GI_{gn}", "gap": gap, "area": area})
        plan["selection"][f"gasket {g['name']}"] = [dict(p["mesh"].groups[t].describe(), part=p["name"]) for p, tg in slave for t in tg]
        plan["selection"][f"gasket {g['name']} (with)"] = [dict(p["mesh"].groups[t].describe(), part=p["name"]) for p, tg in master for t in tg]
    # contacts between parts: hard (a stiff penalty) or penalty contact with friction, or bonded (tied), with an
    # optional initial clearance (> 0) or interference (< 0)
    for c in study.get("contacts", []) or []:
        cn = _name(c["name"])
        slave, master = select(parts, c["faces"]), select(parts, c["with"])
        sf, mf = _gfaces(slave), _gfaces(master)
        law = contact_law(c, parts, slave, master)
        K = law["stiffness_MPa_mm"]
        mu = 0.0 if law["kind"] == "bonded" else float(c.get("friction", 0.2))
        L += [f"*SURFACE, NAME=CS_{cn}"] + [f"{e + 1}, S{s}" for e, s in sf]
        L += [f"*SURFACE, NAME=CM_{cn}"] + [f"{e + 1}, S{s}" for e, s in mf]
        fr = friction_lines(K, {"friction": mu}) if mu > 0 and law["kind"] != "bonded" else []
        if modal is not None:
            if c["name"] in modal["ties"]:
                only = (modal["ties"].get(c["name"]) if isinstance(modal["ties"], dict) else None)
                if only:   # tied where it was closed: those slave nodes, as a nodal surface
                    L += [f"*NSET, NSET=CTN_{cn}"] + _lines(sorted(int(q) for q in only))
                    L += [f"*SURFACE, NAME=CT_{cn}, TYPE=NODE", f"CTN_{cn}"]
                    L += [f"*TIE, NAME=T_{cn}, POSITION TOLERANCE={_g(modal['tol'])}", f"CT_{cn}, CM_{cn}"]
                else:
                    L += [f"*TIE, NAME=T_{cn}, POSITION TOLERANCE={_g(modal['tol'])}", f"CS_{cn}, CM_{cn}"]
        else:
            L += [f"*SURFACE INTERACTION, NAME=CI_{cn}"]
            if law["kind"] == "bonded":   # tied in every direction: CalculiX takes the tangential stiffness from
                # the stick slope, which TIED needs ('a stick slope using the *FRICTION card is mandatory')
                L += ["*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=TIED", _g(K), "*FRICTION", f"1., {_g(K)}"]
            else:
                L += ["*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR", _g(K)] + (
                    friction_lines(K, {"friction": mu}, stick) if mu > 0 else [])
            fit = float(c.get("clearance", 0.0) or 0.0) < 0
            L += [f"*CONTACT PAIR, INTERACTION=CI_{cn}, TYPE={'NODE' if fit else 'SURFACE'} TO SURFACE", f"CS_{cn}, CM_{cn}"]
            if "clearance" in c and not fit:
                L += [f"*CLEARANCE, MASTER=CM_{cn}, SLAVE=CS_{cn}, VALUE={_g(float(c['clearance']))}"]
        nrm = sum((p["mesh"].groups[tg_].normal * p["mesh"].groups[tg_].area[:, None]).sum(axis=0)
                  for p, tgs in slave for tg_ in tgs)
        plan["contacts"].append({"name": c["name"], "slave_faces": sf, "tags": gtags(parts, slave),
                                 "parts": sorted({q["name"] for q, _ in slave} | {q["name"] for q, _ in master}),
                                 "with_tags": gtags(parts, master), "normal": _unit(nrm).tolist(), "law": law,
                                 "friction_mu": mu, "friction": fr, "interaction": f"CI_{cn}",
                                 "clearance_mm": c.get("clearance"),
                                 "contact_type": "node to surface" if (c["name"] in moved or c["name"] in fits)
                                 else "surface to surface",
                                 "interference_nodes_moved": moved.get(c["name"]),
                                 "interference_method": ("expansion" if c["name"] in fits else "overlap" if c["name"]
                                                         in moved else None),
                                 "gap": face_gap(X, _gnodes(slave), _gnodes(master), _unit(nrm)),
                                 "area": sum(p["mesh"].area_of(tg) for p, tg in slave)})
        plan["selection"][f"contact {c['name']}"] = [dict(p["mesh"].groups[t].describe(), part=p["name"]) for p, tg in slave for t in tg]
        plan["selection"][f"contact {c['name']} (with)"] = [dict(p["mesh"].groups[t].describe(), part=p["name"]) for p, tg in master for t in tg]
    # probes: mean displacement of faces (read from the .frd)
    for pr in study.get("probes", []) or []:
        sel = select(parts, pr["faces"])
        plan["probes"][pr["name"]] = (_gnodes(sel) + 1).tolist()
        L += [f"*NSET, NSET=N_PR_{_name(pr['name'])}"] + _lines(plan["probes"][pr["name"]])
        plan["selection"][f"probe {pr['name']}"] = [dict(p["mesh"].groups[t].describe(), part=p["name"]) for p, tg in sel for t in tg]
    # stabilisation: weak springs to ground on every part that no support holds in every direction (before a gasket
    # or contact is pressed it has no stiffness and friction none: the part could slide, rock about a bolt line, spin)
    # their stiffness: 1e-6 of the softest bolt's (verified on the joint benchmarks), else 1e-9 E h (against a slender
    # part's own stiffness: a cantilever's tip stiffness is far below E h)
    ks = [q["k_N_mm"] for rec in plan["bolts"] if rec.get("kind", "bolt") == "bolt" for q in rec["bolts"]]
    k_stab = (STAB_FRACTION * min(ks) if ks else
              STAB_FRACTION_E * min(float(p["material"]["E"]) * float(p["info"].get("size") or 1.0) for p in parts))
    # a part bonded (tied) to a part held in every direction is held too
    bonded = [({p["name"] for p, _ in select(parts, c["faces"])}, {p["name"] for p, _ in select(parts, c["with"])})
              for c in study.get("contacts", []) or [] if c.get("law", "hard") == "bonded"]
    grew = True
    while grew:
        grew = False
        for a_, b_ in bonded:
            for x, y in ((a_, b_), (b_, a_)):
                if x & full_parts and not y <= full_parts:
                    full_parts |= y
                    grew = True
    stab, ground, stab_el = {}, [], []
    for p in parts:
        if p["name"] in full_parts:
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
    if pret:
        L += ["*NSET, NSET=N_PRETENSION"] + _lines(pnodes) + ["*NSET, NSET=N_BOLTENDS"] + _lines(ends)
    plan["pret_nodes"] = pnodes
    has_contact = bool(plan["gaskets"] or plan["contacts"]) and modal is None
    if modal is not None:   # the frequency run: supports, the bolts held at their preloaded length, nothing else
        L += ["*STEP", "*FREQUENCY", str(int(modal["modes"])), "*BOUNDARY"] + bnd + (["N_PRETENSION, 1, 1"] if pret else [])
        L += ["*NODE FILE", "U", "*END STEP"]
        plan["steps"] = [{"step": 1, "kind": "modal", "what": "frequencies", "modes": int(modal["modes"]),
                          "factor": None, "loads": {}, "ccx_step": 1}]
        return "\n".join(L) + "\n", plan
    frictional = [g for g in plan["gaskets"]] + [c for c in plan["contacts"] if c["friction"]]

    def outputs():   # bolt forces from the springs' ends; contact stresses from the contact output; no element
        # stresses (large: they are computed from the displacements)
        out = [f"*NODE PRINT, NSET=N_BOLTENDS, FREQUENCY={FREQ}", "U"] if pret else []
        out += [f"*NODE PRINT, NSET=N_STAB, FREQUENCY={FREQ}", "U"] if stab_el else []
        for pn in plan["probes"]:
            out += [f"*NODE PRINT, NSET=N_PR_{_name(pn)}, FREQUENCY={FREQ}", "U"]
        return out + ["*NODE FILE", "U"] + (["*CONTACT FILE", "CDIS, CSTR"] if has_contact else [])

    dl_ever = any(LD[n]["kind"] in ("dload", "grav") for s in dsteps for n in s["loads"] if n in LD)

    def step_loads(active: dict):
        cl, dl = {}, {}
        for bn_, f in active.items():
            ld = LD.get(bn_)
            if ld is None:
                continue
            if ld["kind"] == "cload":
                for n, v in ld["nodes"].items():
                    cl[n] = cl.get(n, np.zeros(3)) + v * f
            elif ld["kind"] == "dload":
                for e, s in ld["rows"]:
                    dl[f"{e}, P{s}"] = dl.get(f"{e}, P{s}", 0.0) + ld["p"] * f
            else:
                for elset, gm, gv in ld["items"]:
                    dl[f"{elset}, GRAV"] = f"{_g(gm * f)}, " + ", ".join(_g(c) for c in gv)
        cl_lines = [f"{n + 1}, {k + 1}, {_g(cl[n][k])}" for n in sorted(cl) for k in range(3) if cl[n][k] != 0]
        dl_lines = [f"{key}, {v}" if isinstance(v, str) else f"{key}, {_g(v)}" for key, v in dl.items()]
        return cl_lines, dl_lines

    dummy_dl = None
    for n_, ld in LD.items():
        if ld["kind"] == "dload" and ld["rows"]:
            dummy_dl = f"{ld['rows'][0][0]}, P{ld['rows'][0][1]}, 0."
            break
        if ld["kind"] == "grav":
            dummy_dl = f"{ld['items'][0][0]}, GRAV, 0., 0., 0., -1."
            break
    if fits:   # CalculiX wants a temperature at every node, the connectors' and springs' too (checktemp: 'no final
        # temperature defined in node 1496', assembly_press_fit on carbonio, 2026.11.10.2): 0 everywhere, 1 under a fit
        L += ["*NSET, NSET=N_TEMP, GENERATE", f"1, {n_next - 1}, 1",
              "*INITIAL CONDITIONS, TYPE=TEMPERATURE", "N_TEMP, 0."]
    prev_kind = None
    for s in dsteps:
        if s["kind"] == "modal":   # a separate linear run on the state the previous step left (solve, write(modal=))
            plan["steps"].append(s)
            continue
        first = s["ccx_step"] == 1
        # the first step from a small increment (the contact is found at a twentieth of the load); later steps add a
        # part of the load in one increment (CalculiX cuts it back if contact needs it); a step's first_increment and
        # max_increment (fractions of the step) override these
        d0, dmin, dmax = (0.05, 1e-05, 0.25) if first else (1.0, 1e-08, 1.0)
        d0 = float(s.get("first_increment") or d0)
        dmax = max(float(s.get("max_increment") or dmax), d0)
        L += ["*STEP, INC=1000", "*STATIC", f"{_g(d0)}, 1., {_g(min(dmin, d0))}, {_g(dmax)}"]
        if first:
            L += ["*BOUNDARY"] + bnd if bnd else []
            if fits:   # the fits' interference, grown with the step (a temperature of 1 is the whole interference)
                L += ["*TEMPERATURE", "N_TEMP, 0."] + [f"NF_{_name(cn_)}, 1." for cn_ in fits]
        elif pret:
            L += ["*BOUNDARY, FIXED", "N_PRETENSION, 1, 1"]
        cl, dl = step_loads(s["loads"])
        pre = [f"{npt}, 1, {_g(P)}" for _, npt, _, P in pret] if s["kind"] == "preload" else []
        if first:
            if pre + cl:
                L += ["*CLOAD"] + pre + cl
            if dl:
                L += ["*DLOAD"] + dl
        else:
            L += ["*CLOAD, OP=NEW"] + (cl or [f"{(springs[0][2] if springs else 1)}, 1, 0."])
            if dl_ever:
                L += ["*DLOAD, OP=NEW"] + (dl or [dummy_dl])
        if prev_kind == "preload":   # the friction at its full stick slope once the faces are pressed
            for g in frictional:
                L += [f"*CHANGE FRICTION, INTERACTION={g['interaction']}"] + g["friction"]
        L += outputs()
        L += ["*END STEP"]
        plan["steps"].append(s)
        prev_kind = s["kind"]
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


def contact_by_step(blocks: list[dict]) -> dict[int, dict[int, dict]]:
    """{step: {node (1-based): {copen, cslip1, cslip2, cpress, cshear1, cshear2}}} from the last CONTACT block of
    each step."""
    names = ("COPEN", "CSLIP1", "CSLIP2", "CPRESS", "CSHEAR1", "CSHEAR2")
    out = {}
    for b in blocks:
        if not b["name"].startswith("CONTACT"):
            continue
        comps = b["components"]
        idx = {n: (comps.index(n) if n in comps else i) for i, n in enumerate(names)}
        out[b["step"]] = {n: {k.lower(): (v[i] if len(v) > i else 0.0) for k, i in idx.items()} for n, v in b["data"].items()}
    return out


def contact_pressure_by_step(blocks: list[dict]) -> dict[int, dict[int, float]]:
    """{step: {node (1-based): contact pressure}} from the last CONTACT block of each step."""
    return {s: {n: c["cpress"] for n, c in d.items()} for s, d in contact_by_step(blocks).items()}


def disp_by_step(blocks: list[dict], steps: list[dict], N: int) -> dict[int, np.ndarray]:
    """{CalculiX step (1-based): U (N, 3)} of the deck's (static) steps."""
    out = {}
    for b in blocks:
        s = b["step"]
        if not b["name"].startswith("DISP") or not s:
            continue
        U = out.setdefault(s, np.zeros((N, 3)))
        for n, v in b["data"].items():
            if 1 <= n <= N:
                U[n - 1] = v[:3]
    return out


def face_node_normals(X: np.ndarray, tets: np.ndarray, faces) -> tuple[np.ndarray, np.ndarray]:
    """The nodes (0-based, sorted) of element faces [(element, face 1..4)] and each one's outward unit normal: the
    area-weighted mean of its faces' normals (on a cylinder, the radial direction at the node)."""
    acc: dict = {}
    for e, s in faces:
        el = tets[e]
        c = [int(el[i]) for i in FACES[s - 1]]
        nn = np.cross(X[c[1]] - X[c[0]], X[c[2]] - X[c[0]])          # twice the face's area, along its normal
        if nn @ (X[c].mean(axis=0) - X[el[:4]].mean(axis=0)) < 0:     # outward: away from the element
            nn = -nn
        ids = c + ([int(el[EDGE_MID[frozenset((FACES[s - 1][i], FACES[s - 1][(i + 1) % 3]))]]) for i in range(3)]
                   if len(el) == 10 else [])
        for q in ids:
            acc[q] = acc.get(q, 0.0) + nn
    nodes = np.array(sorted(acc), int)
    N = np.array([acc[q] for q in nodes], float).reshape(-1, 3)
    return nodes, N / np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]


def nodal_internal_forces(mesh, U: np.ndarray, material: dict, eigen: np.ndarray | None = None) -> np.ndarray:
    """The element forces at each node (N, 3): f_a = sum over elements of the integral of sigma . grad N_a, from the
    displacements (CalculiX's element formulation: 4 Gauss points for 10-node tetrahedra). At nodes with no load
    but a contact, their sum over a face is the contact force on it, exactly as the discretization balances it."""
    from .meshdata import EDGES, simplex_shape
    from .runner import GAUSS4
    E, nu = float(material["E"]), float(material["nu"])
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    quad = mesh.order == 2
    L = GAUSS4 if quad else np.full((1, 4), 0.25)
    w = 1.0 / (6 * len(L))
    _, dN = simplex_shape(L, quad, EDGES)
    X = mesh.nodes[mesh.tets]
    dNdxi = dN[:, :, 1:] - dN[:, :, :1]
    J = np.einsum("qnj,enk->eqjk", dNdxi, X)
    det = np.abs(np.linalg.det(J))
    G = np.einsum("eqkj,qnj->eqnk", np.linalg.inv(J), dNdxi)          # dN/dx (E, q, n, 3)
    gu = np.einsum("eqnk,enc->eqck", G, np.asarray(U, float)[mesh.tets])   # du_c/dx_k
    eps = 0.5 * (gu + np.swapaxes(gu, -1, -2))
    if eigen is not None:   # an interference fit's strain in these elements (eigenstrain: Voigt, engineering shears)
        e6 = np.asarray(eigen, float)
        T = np.zeros((len(e6), 3, 3))
        T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = e6[:, 0], e6[:, 1], e6[:, 2]
        T[:, 0, 1] = T[:, 1, 0] = e6[:, 3] / 2
        T[:, 0, 2] = T[:, 2, 0] = e6[:, 4] / 2
        T[:, 1, 2] = T[:, 2, 1] = e6[:, 5] / 2
        eps = eps - T[:, None]
    tr = np.trace(eps, axis1=-2, axis2=-1)
    sig = 2 * mu * eps + lam * tr[..., None, None] * np.eye(3)
    fe = np.einsum("eqnk,eqck,eq->enc", G, sig, det * w)                # (E, n, 3)
    out = np.zeros((len(mesh.nodes), 3))
    for a in range(mesh.tets.shape[1]):
        np.add.at(out, mesh.tets[:, a], fe[:, a])
    return out


def section_forces(dat_text: str) -> list[dict]:
    """CalculiX's *SECTION PRINT SOF blocks: {set, time, force [fx, fy, fz], normal (mean, area-weighted), area,
    normal_force (+ tension), shear_force} (printoutface.f: 'statistics for surface set S and time t', then the
    total force and moment, the centre of gravity and mean normal, the moment about it, and area, normal and shear
    force)."""
    out, cur, nums = [], None, []
    head = re.compile(r"statistics for surface set\s+(\S+)\s+and time\s+(\S+)", re.I)
    for line in dat_text.splitlines():
        m = head.search(line)
        if m:
            cur, nums = {"set": m.group(1).upper(), "time": float(m.group(2).replace("D", "E"))}, []
            out.append(cur)
            continue
        if cur is None:
            continue
        tk = line.split()
        try:
            vals = [float(x.replace("D", "E")) for x in tk] if tk else None
        except ValueError:
            vals = None
            if tk and not tk[0][0].isalpha() and "(" not in line:
                cur = None
            continue
        if vals:
            nums.append(vals)
            if len(nums) == 1 and len(vals) >= 3:
                cur["force"] = vals[:3]
            elif len(nums) == 2 and len(vals) >= 6:
                cur["normal"] = vals[3:6]
            elif len(nums) == 4 and len(vals) >= 3:
                cur["area"], cur["normal_force"], cur["shear_force"] = vals[:3]
                cur = None
    return out


def mode_shapes(blocks: list[dict], step: int, N: int) -> list[np.ndarray]:
    out = []
    for b in blocks:
        if b["name"].startswith("DISP") and b["step"] == step:
            U = np.zeros((N, 3))
            for n, v in b["data"].items():
                if 1 <= n <= N:
                    U[n - 1] = v[:3]
            out.append(U)
    return out


def _node_areas(X, tets, faces):
    _, area, _ = gasket_graph(X, tets, [(e, s) for e, s in faces])
    return area


def contact_state(c: dict, cmap: dict | None, area: dict) -> dict | None:
    """A contact pair at one step: its force, pressures, the fraction closed (pressure > 0), sticking (shear below
    mu x pressure) and the largest opening and slip."""
    if cmap is None:
        return None
    A = sum(area.values()) or 1.0
    p = {n: max(0.0, float(cmap.get(n, {}).get("cpress", 0.0))) for n in area}
    sh = {n: math.hypot(cmap.get(n, {}).get("cshear1", 0.0), cmap.get(n, {}).get("cshear2", 0.0)) for n in area}
    sl = [math.hypot(cmap.get(n, {}).get("cslip1", 0.0), cmap.get(n, {}).get("cslip2", 0.0)) for n in area]
    op = [float(cmap.get(n, {}).get("copen", 0.0)) for n in area]
    closed = sum(a for n, a in area.items() if p[n] > 0)
    out = {"_closed_nodes": sorted(n for n in area if p[n] > 0),   # for a modal step's ties (removed from results)
           "force_N": round(sum(p[n] * a for n, a in area.items()), 3), "max_pressure_MPa": round(max(p.values(), default=0.0), 6),
           "mean_pressure_MPa": round(sum(p[n] * a for n, a in area.items()) / A, 6),
           "closed_fraction": round(closed / A, 6), "max_opening_mm": round(max(op, default=0.0), 8),
           "max_slip_mm": round(max(sl, default=0.0), 8)}
    mu = c.get("friction_mu", 0.0)
    if c.get("law", {}).get("kind") == "bonded":
        out["stick_fraction"] = None
    elif mu > 0 and closed > 0:
        out["stick_fraction"] = round(sum(a for n, a in area.items() if p[n] > 0 and sh[n] < 0.999 * mu * p[n]) / closed, 6)
    else:
        out["stick_fraction"] = None
    return out


def evaluate(study: dict, parts: list[dict], plan: dict, dat_text: str, frd_text: str,
             modal_runs: dict | None = None) -> dict:
    """Bolt and nut forces, gasket stresses, contact states, probes and part stresses at every static step; the
    leak and lift-off loads of a sweep; the frequencies of modal steps."""
    X = all_nodes(parts)
    N = len(X)
    tets = np.concatenate([p["mesh"].tets + p["n0"] for p in parts])
    dat = results.parse_dat(dat_text)
    blocks = frd_steps(frd_text)
    cb = contact_by_step(blocks)
    steps = plan["steps"]
    solid = [i for i, s in enumerate(steps) if s["kind"] != "modal"]
    sw = plan.get("sweep") or {}
    unit = "MPa" if sw.get("kind") == "pressure" else "N"
    loads = [sw.get("max", 0.0) * s["factor"] if sw and s.get("factor") is not None else None for s in steps]
    assembly = plan.get("analysis") == "assembly"
    res = {"analysis": plan.get("analysis", "joint"), "bolts": {}, "gaskets": {}, "steps": [], "leak": {}, "lift_off": {}}
    if sw:
        res["swept"] = {"bc": sw.get("bc"), "kind": sw.get("kind"), "max": sw.get("max"), "unit": unit}
    # connector forces: each spring's stiffness times its stretch (its ends' distance change along the axis), a nut's
    # as a compression; at the preload it must equal the preload (bolt_preload_error). The .dat has one block per
    # static step, in order
    ub = [b for b in dat["blocks"] if b["set"] == "N_BOLTENDS" and b["what"].startswith("displacements")]
    Ub = {solid[j]: {int(r[0]): np.asarray(r[1:4], float) for r in blk["rows"] if len(r) >= 4}
          for j, blk in enumerate(ub[:len(solid)])}
    blist = [(rec, b_) for rec in plan["bolts"] for b_ in rec["bolts"]]
    forces = np.full((len(steps), len(blist)), np.nan)
    for i, (rec, b_) in enumerate(blist):
        n1, n2 = b_["ends"]
        for s, u in Ub.items():
            if n1 in u and n2 in u:
                forces[s, i] = rec.get("sign", 1.0) * b_["k_N_mm"] * float((u[n2] - u[n1]) @ np.asarray(rec["axis"], float))
    ipre = next((i for i, s in enumerate(steps) if s["kind"] == "preload"), None)
    res_pre_err = None
    if blist and ipre is not None and np.isfinite(forces[ipre]).all():
        pre = np.array([rec["preload_N"] for rec, _ in blist])
        res_pre_err = round(float(np.max(np.abs(forces[ipre] - pre) / np.maximum(pre, 1e-9))), 6)
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
    careas = {c["name"]: _node_areas(X, tets, c["slave_faces"]) for c in plan["contacts"]}
    # the slave faces' nodes of each contact (0-based, merged) and their part, for the contact forces
    snodes = {}
    for c in plan["contacts"]:
        nodes, nrm = face_node_normals(X, tets, c["slave_faces"]) if c["slave_faces"] else (np.zeros(0, int), np.zeros((0, 3)))
        part = next((p for p in parts if p["n0"] <= (nodes[0] if len(nodes) else -1) < p["n0"] + len(p["mesh"].nodes)), None)
        snodes[c["name"]] = (nodes, part, nrm)
    disp = disp_by_step(blocks, steps, N) if (assembly or plan.get("probes")) else {}
    fint: dict = {}
    # probes: from CalculiX's printout of their nodes (as the bolt ends), the .frd's displacements otherwise
    pblocks = {}
    for pn in plan.get("probes", {}):
        bl = [b for b in dat["blocks"] if b["set"] == f"N_PR_{_name(pn)}" and b["what"].startswith("displacements")]
        pblocks[pn] = {solid[j]: np.array([r[1:4] for r in blk["rows"] if len(r) >= 4], float)
                       for j, blk in enumerate(bl[:len(solid)])}
    for s in solid:
        st = steps[s]
        rec = {"step": s + 1, "kind": st["kind"], "what": st["what"], "load": None if loads[s] is None else round(loads[s], 9),
               "bolts": {}, "gaskets": {}}
        if st["kind"] == "static":
            rec["loads"] = {k: v for k, v in st["loads"].items()}
        i0 = 0
        for brec in plan["bolts"]:
            n = len(brec["bolts"])
            fv = forces[s, i0:i0 + n]
            i0 += n
            if np.isfinite(fv).any():
                rec["bolts"][brec["name"]] = {"min_N": round(float(np.nanmin(fv)), 3), "max_N": round(float(np.nanmax(fv)), 3),
                                              "mean_N": round(float(np.nanmean(fv)), 3)}
        ccx = st.get("ccx_step") or (s + 1)
        cmap = cb.get(ccx)
        for g in plan["gaskets"]:
            adj, area, inner, outer = graphs[g["name"]]
            if cmap is None:
                continue
            pv = {n: max(0.0, float(cmap.get(n, {}).get("cpress", 0.0))) for n in area}
            A = sum(area.values())
            touching = sum(a for n, a in area.items() if pv[n] > 0)
            force = sum(pv[n] * a for n, a in area.items())
            vals = np.array(list(pv.values()))
            # a gasket with an inner and outer edge: the ring stress; with one edge (no cavity inside it): the
            # largest stress (the joint lifts off when all of it opens; it cannot leak)
            ring = ring_stress(adj, pv, inner, outer) if outer else float(vals.max())
            seal = g["seal"]
            thr = max(float(seal.get("stress", 0.0)), float(seal.get("m", 0.0)) *
                      (loads[s] if unit == "MPa" and loads[s] is not None else 0.0))
            if not outer:
                thr = 0.0
            rec["gaskets"][g["name"]] = {"ring_MPa": None if math.isinf(ring) else round(ring, 6),
                                         "min_MPa": round(float(vals.min()), 6), "max_MPa": round(float(vals.max()), 6),
                                         "mean_MPa": round(force / A, 6), "contact_fraction": round(touching / A, 6),
                                         "force_N": round(force, 3), "seal_MPa": round(thr, 6),
                                         "leak": bool(not math.isinf(ring) and ring < thr),
                                         "open": bool(not math.isinf(ring) and ring <= 1e-9)}
        if plan["contacts"]:
            rec["contacts"] = {}
            for c in plan["contacts"]:
                cs = contact_state(c, cmap, careas[c["name"]])
                if cs is not None:
                    cs["pressure_force_N"] = cs["force_N"]
                    nodes, part, nrm = snodes[c["name"]]
                    if part is not None and len(nodes) and ccx in disp:
                        # the element forces at the slave face's nodes balance the contact forces on them exactly; each
                        # node's normal part is its contact pressure force (summing a stiff penalty's nodal pressures was
                        # up to 2x off; CalculiX's section force 7 %: assembly_clamp; on a press fit 0.74 and 378 MPa
                        # against 78.75: mochi 2026.11.04)
                        sl = slice(part["n0"], part["n0"] + len(part["mesh"].nodes))
                        key = (ccx, part["name"])
                        if key not in fint:
                            fint[key] = nodal_internal_forces(part["mesh"], disp[ccx][sl], part["material"],
                                                              eigenstrain(plan, part))
                        f = fint[key][nodes - part["n0"]]
                        fn = -np.einsum("ij,ij->i", f, nrm)                          # compression positive
                        ft = f + fn[:, None] * nrm
                        cs["mean_pressure_cpress_MPa"] = cs["mean_pressure_MPa"]
                        cs["force_N"] = round(float(fn.sum()), 3)
                        cs["mean_pressure_MPa"] = round(float(fn.sum()) / max(c["area"], 1e-300), 6)
                        cs["shear_N"] = round(float(np.linalg.norm(ft.sum(axis=0))), 3)
                        cs["force_from"] = "element forces"
                    else:
                        cs["force_from"] = "pressures"
                rec["contacts"][c["name"]] = cs
        if plan.get("probes") and (ccx in disp or any(s in pb for pb in pblocks.values())):
            rec["probes"] = {}
            for pn, nodes in plan["probes"].items():
                rows = pblocks.get(pn, {}).get(s)
                m = rows.mean(axis=0) if rows is not None and len(rows) else disp[ccx][np.asarray(nodes) - 1].mean(axis=0)
                rec["probes"][pn] = {"ux": round(float(m[0]), 9), "uy": round(float(m[1]), 9), "uz": round(float(m[2]), 9),
                                     "u": round(float(np.linalg.norm(m)), 9)}
        if assembly and ccx in disp:   # each part's peak von Mises (nodal, from the displacements)
            from .render import nodal_stress, von_mises
            U = disp[ccx]
            rec["parts"] = {}
            for p in parts:
                sl = slice(p["n0"], p["n0"] + len(p["mesh"].nodes))
                vm = von_mises(nodal_stress(p["mesh"], U[sl], {"material": p["material"]},
                                            {"eigenstrain": eigenstrain(plan, p)}))
                k = int(np.argmax(vm))
                rec["parts"][p["name"]] = {"peak_von_mises_MPa": round(float(vm[k]), 4),
                                           "at": np.round(p["mesh"].nodes[k], 3).tolist(),
                                           "max_displacement_mm": round(float(np.linalg.norm(U[sl], axis=1).max()), 9)}
        res["steps"].append(rec)
    # modal steps: each a linear frequency run (solve) on the state its previous step left
    modal = [s for s in steps if s["kind"] == "modal"]
    if modal:
        res["modal"] = []
        for s in modal:
            run = (modal_runs or {}).get(s["step"]) or {}
            base = next((r for r in reversed(res["steps"]) if r["step"] < s["step"]), None)
            allf = [round(float(e["hz"]), 6) for e in run.get("eigen", [])]
            floating = run.get("floating") or []
            rigid = [f for f in allf if f < RIGID_HZ][:max(run.get("extra_modes", 0), 0)] if floating else []
            fr = [f for f in allf if f not in rigid][:s["modes"]]
            rec_m = {"step": s["step"], "name": s.get("name"), "after_step": base["step"] if base else None,
                     "frequencies_Hz": fr, "tied": run.get("tied", []), "free": run.get("free", []),
                     "contacts": {k: {q: v[q] for q in ("closed_fraction", "stick_fraction", "force_N")}
                                  for k, v in (base or {}).get("contacts", {}).items() if v}}
            if run.get("tied_where_closed"):
                rec_m["tied_where_closed"] = {k: {"nodes": v[0], "closed_fraction": v[1]}
                                              for k, v in run["tied_where_closed"].items()}
            if floating:
                rec_m["floating_parts"] = floating
                rec_m["rigid_body_Hz"] = rigid
            res["modal"].append(rec_m)
            at = f"step {base['step']}" if base else "the start"
            for grp in floating:
                opened = [x["name"] for x in plan["contacts"] + plan["gaskets"]
                          if x["name"] in run.get("free", []) and set(x.get("parts", [])) & set(grp)]
                res.setdefault("warnings", []).append(
                    f"modal step {s['step']}: part(s) {', '.join(grp)} have no tied path to a support (their contacts "
                    f"{', '.join(opened) or '(none)'} were open at {at}); they float on feat's stabilisation springs"
                    + (f": their rigid-body modes ({', '.join(f'{f:.3g}' for f in rigid)} Hz) are left out of "
                       "frequencies_Hz" if rigid else "")
                    + "; the frequencies are not the assembly's: support or close those contacts (preload, clearance)")
            low = [f for f in fr if f < RIGID_HZ]
            if low:
                res.setdefault("warnings", []).append(
                    f"modal step {s['step']}: frequencies below {RIGID_HZ:g} Hz ({', '.join(f'{f:.3g}' for f in low)}): "
                    "a part or a mechanism is not held (rigid-body motion on feat's stabilisation springs); check the "
                    "supports and the contacts' states")
            for k, v in (rec_m.get("tied_where_closed") or {}).items():
                res.setdefault("warnings", []).append(
                    f"modal step {s['step']}: contact {k} was closed on {100 * v['closed_fraction']:.0f} % of its face at "
                    f"{at}: tied there only ({v['nodes']} nodes), free elsewhere")
            if run.get("error"):
                res.setdefault("warnings", []).append(f"modal step {s['step']}: {run['error']}")
            for k in run.get("slipping", []):
                res.setdefault("warnings", []).append(
                    f"modal step {s['step']}: contact {k} was closed but partly slipping at step "
                    f"{base['step'] if base else '?'}: tied for the frequencies, which are an upper bound there")
        res["frequencies"] = res["modal"][0]["frequencies_Hz"]
        if not res["frequencies"]:
            res.setdefault("warnings", []).append("modal step: no frequencies")
    # bolts summary
    i0 = 0
    for brec in plan["bolts"]:
        n = len(brec["bolts"])
        res["bolts"][brec["name"]] = {"kind": brec.get("kind", "bolt"), "size": brec["size"],
                                      "preload_N": round(brec["preload_N"], 3), "count": n,
                                      "stiffness_N_mm": round(brec["k_bolt_N_mm"], 3),
                                      "positions": [b_["B1"] for b_ in brec["bolts"]],
                                      "max_N": round(float(np.nanmax(forces[:, i0:i0 + n])), 3) if np.isfinite(forces[:, i0:i0 + n]).any() else None}
        i0 += n
    # leak and lift-off loads, by gasket: over the preload and the sweep
    for g in plan["gaskets"]:
        gsteps = [r for r in res["steps"] if g["name"] in r["gaskets"] and r["kind"] in ("preload", "sweep")]
        if sw and gsteps and gsteps[0]["gaskets"][g["name"]]["ring_MPa"] is not None:
            xs = [r["load"] for r in gsteps]
            ring = [r["gaskets"][g["name"]]["ring_MPa"] for r in gsteps]
            seal = [r["gaskets"][g["name"]]["seal_MPa"] for r in gsteps]
            leak = _crossing(xs, [a - b for a, b in zip(ring, seal)]) if any(seal) else None
            lift = _crossing(xs, ring, floor=0.0)
            top = xs[-1]
            res["leak"][g["name"]] = (round(leak, 6) if leak is not None else {"above": top}) if any(seal) else None
            res["lift_off"][g["name"]] = round(lift, 6) if lift is not None else {"above": top}
        else:
            res["leak"][g["name"]] = res["lift_off"][g["name"]] = None
        first = next((r for r in res["steps"] if g["name"] in r["gaskets"]), None)
        if first is None:
            continue
        pre = first["gaskets"][g["name"]]
        res["gaskets"][g["name"]]["preload"] = {k: pre[k] for k in ("ring_MPa", "min_MPa", "max_MPa", "mean_MPa",
                                                                     "contact_fraction", "force_N")}
        if pre["ring_MPa"] is not None and pre["seal_MPa"] and pre["ring_MPa"] < pre["seal_MPa"]:
            res.setdefault("warnings", []).append(
                f"gasket {g['name']}: already below its sealing stress at the preload (ring {pre['ring_MPa']:.4g} MPa "
                f"< {pre['seal_MPa']:.4g} MPa): more preload, more bolts, or a narrower gasket")
    if not plan["gaskets"] and not sw:
        res.pop("leak")
        res.pop("lift_off")
    if blist:
        res["bolt_preload_error"] = res_pre_err
    # contacts summary
    if plan["contacts"]:
        res["contacts"] = {}
        for c in plan["contacts"]:
            res["contacts"][c["name"]] = {"law": c["law"], "friction": c["friction_mu"], "area_mm2": round(c["area"], 4),
                                          "clearance_mm": c.get("clearance_mm"), "contact_type": c.get("contact_type"),
                                          "gap_mm": (c.get("gap") or {}).get("median_mm")}
            if c.get("interference_method"):   # how its interference was made: expansion (ramped) or overlap
                res["contacts"][c["name"]]["interference_method"] = c["interference_method"]
                if c["name"] in (plan.get("fits") or {}):
                    res["contacts"][c["name"]]["fit"] = plan["fits"][c["name"]]
            gp = (c.get("gap") or {}).get("median_mm")
            if gp is not None and gp > GAP_TOL and c.get("clearance_mm") is None:
                res.setdefault("warnings", []).append(
                    f"contact {c['name']}: its faces are {gp:.3g} mm apart (median) at the start"
                    + (": a bonded contact ties them across the gap" if c["law"]["kind"] == "bonded" else
                       ": they touch only once the parts move that far"))
    # stabilisation springs: what they carry (k u at each spring node) against the preload, or the largest load
    stp = plan.get("stabilisation") or {}
    if stp.get("springs"):
        us = [b for b in dat["blocks"] if b["set"] == "N_STAB" and b["what"].startswith("displacements")]
        tot_pre = sum(rec["preload_N"] * len(rec["bolts"]) for rec in plan["bolts"])
        by_step = []
        for blk in us[:len(solid)]:
            f = [stp["k_N_mm"] * float(np.linalg.norm(r[1:4])) for r in blk["rows"] if len(r) >= 4]
            by_step.append(round(float(np.sum(f)), 6))
        mx = max(by_step) if by_step else None
        ref = tot_pre or max((sum(plan.get("load_mag", {}).get(n, 0.0) * abs(f) for n, f in s["loads"].items())
                              for s in steps), default=0.0)
        what = "the preload" if tot_pre else "the largest load"
        res["stabilisation"] = {"springs": stp["springs"], "k_N_mm": stp["k_N_mm"],
                                "parts": {k: len(v) for k, v in stp["parts"].items()},
                                "force_N_by_step": by_step, "max_force_N": mx,
                                "fraction_of_preload" if tot_pre else "fraction_of_load":
                                    round(mx / ref, 8) if mx is not None and ref else None}
        fr = res["stabilisation"].get("fraction_of_preload", res["stabilisation"].get("fraction_of_load"))
        if fr is not None and fr > STAB_WARN:
            res.setdefault("warnings", []).append(
                f"the stabilisation springs carry {100 * fr:.3g} % of {what} ({mx:.4g} N): a part is not held by its "
                f"supports, connectors and contacts alone ({', '.join(stp['parts'])}); check the contact faces touch "
                "and every part is held")
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
        pre_mean = res["gaskets"][g["name"]].get("preload", {}).get("mean_MPa")
        if law.get("kind") == "secant" and pre_mean and abs(pre_mean - law["at_MPa"]) > 0.2 * law["at_MPa"]:
            res.setdefault("warnings", []).append(
                f"gasket {g['name']}: its mean stress at the preload ({pre_mean:.4g} MPa) differs from the stress its "
                f"secant was taken at ({law['at_MPa']:.4g} MPa): other bolts or loads press it; give 'stiffness' instead")
    if res_pre_err is not None and res_pre_err > 0.01:
        res.setdefault("warnings", []).append(
            f"bolt forces at the preload differ from the preload by {100 * res_pre_err:.3g} %: check the bolt results")
    # checks: the gasket force balances the bolts at the preload (no other load across the joint then)
    tot_b = sum(r["preload_N"] * r["count"] for r in res["bolts"].values() if r.get("kind", "bolt") == "bolt")
    tot_g = sum(v.get("preload", {}).get("force_N", 0.0) for v in res["gaskets"].values())
    if tot_b > 0 and tot_g > 0:
        res["preload_balance_error"] = round(abs(tot_g - tot_b) / tot_b, 6)
    return res


# ---- the state for pictures (render.py) ------------------------------------------------------------
def save_state(folder: Path, study: dict, parts: list[dict], plan: dict, res: dict, frd_text: str,
               modes_frd: str | None = None) -> None:
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
    steps = plan["steps"]
    nsteps = len(steps)
    disp = disp_by_step(blocks, steps, N)
    cbs = contact_pressure_by_step(blocks)
    U = np.zeros((nsteps, N, 3))
    cp = np.zeros((nsteps, N))
    for i, s in enumerate(steps):   # a modal step shows the state it started from
        if s["kind"] == "modal":
            if i:
                U[i], cp[i] = U[i - 1], cp[i - 1]
            continue
        if s.get("ccx_step", s["step"]) in disp:
            U[i] = disp[s.get("ccx_step", s["step"])]
        for n, v in (cbs.get(s.get("ccx_step", s["step"])) or {}).items():
            if 1 <= n <= N:
                cp[i, n - 1] = max(0.0, v)
    S = np.zeros((nsteps, N, 6))
    for s in range(nsteps):
        if s and steps[s]["kind"] == "modal":
            S[s] = S[s - 1]
            continue
        for p in parts:
            sl = slice(p["n0"], p["n0"] + len(p["mesh"].nodes))
            S[s, sl] = nodal_stress(p["mesh"], U[s, sl], {"material": p["material"]}, {"eigenstrain": eigenstrain(plan, p)})
    fields = {"U_steps": U.astype(np.float32), "S_steps": S.astype(np.float32), "cp_steps": cp.astype(np.float32)}
    first_modal = next((s for s in steps if s["kind"] == "modal"), None)
    if first_modal is not None and modes_frd:
        shapes = mode_shapes(frd_steps(modes_frd), 1, N)
        if shapes:
            fields["modes"] = np.array(shapes, np.float32)
    bcs = [{"name": n, "kind": plan["kinds"][n], "tags": plan.get("bc_tags", {}).get(n, [])} for n in plan["kinds"]]
    bcs += [{"name": g["name"], "kind": "gasket", "tags": g["tags"]} for g in plan["gaskets"]]
    bcs += [{"name": c["name"], "kind": "contact", "tags": c["tags"]} for c in plan.get("contacts", [])]
    sw = res.get("swept", {})
    by_step = {r["step"]: r for r in res["steps"]}
    load_of = [by_step.get(s["step"], {}).get("load") for s in steps]
    meta = {"bcs": bcs, "joint": True, "analysis": plan.get("analysis", "joint"),
            "steps": [{"step": s["step"], "load": load_of[i], "kind": s["kind"], "what": s["what"]}
                      for i, s in enumerate(steps)],
            "unit": sw.get("unit"), "swept": sw.get("bc"),
            "gaskets": {g["name"]: {"normal": g["normal"], "with_tags": g["with_tags"],
                                    "seal_MPa_steps": [max(float(g["seal"].get("stress", 0.0)), float(g["seal"].get("m", 0.0))
                                                           * (ld if sw.get("unit") == "MPa" and ld is not None else 0.0))
                                                       for ld in load_of]}
                        for g in plan["gaskets"]},
            "contacts": {c["name"]: {"normal": c["normal"], "with_tags": c["with_tags"]} for c in plan.get("contacts", [])},
            "bolts": [[b_["B1"], b_["B2"]] for rec in plan["bolts"] for b_ in rec["bolts"]],
            "parts": [{"name": p["name"], "tag_min": (i + 1) * TAG_STRIDE, "tag_max": (i + 2) * TAG_STRIDE - 1}
                      for i, p in enumerate(parts)]}
    if res.get("frequencies"):
        meta["mode_values"] = {"values": res["frequencies"], "unit": "Hz"}
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
    """Errors and warnings of a joint or assembly study."""
    errs, warns = [], []
    asm = st.get("analysis") == "assembly"
    parts = st.get("parts")
    if not isinstance(parts, dict) or len(parts) < 2:
        return ["parts: {name: {part, material, mesh, move}} with at least two parts"], []
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
        if "mesh" in p:
            e, w2 = mesh_errors(p["mesh"], f"{w}.mesh", "joint")
            errs += e
            warns += w2
        extra = set(p) - {"part", "material", "mesh", "move"}
        if extra:
            errs.append(f"{w}: unknown keys {sorted(extra)}")
    if "mesh" in st:
        e, w2 = mesh_errors(st["mesh"], "mesh", "joint")
        errs += e
        warns += w2
    names = set(parts)
    bcs = st.get("bcs", [])
    bnames, nsup = [], 0
    for i, bc in enumerate(bcs if isinstance(bcs, list) else []):
        k = bc.get("kind") if isinstance(bc, dict) else None
        w = f"bc {i + 1} ({k})"
        if k not in SUPPORTS + LOADS:
            errs.append(f"bc {i + 1}: in a{'n assembly' if asm else ' joint'}, kind is one of {', '.join(SUPPORTS + LOADS)}")
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
    if asm:
        if "sweep" in st:
            errs.append("sweep: in an assembly, a step {kind: sweep, bc, steps}")
        errs += _validate_steps(st, bcs, bnames, parts)
    elif not isinstance(sw, dict) or sw.get("bc") not in bnames:
        errs.append("sweep: {bc: the name of the pressure (or force) bc to step up, steps: n (10)}")
    else:
        bc = next(b for i, b in enumerate(bcs) if bc_name(i, b) == sw["bc"])
        if bc.get("kind") not in ("pressure", "force"):
            errs.append("sweep.bc: a pressure or force bc")
        if "steps" in sw and (not _num(sw["steps"]) or int(sw["steps"]) != sw["steps"] or not 1 <= sw["steps"] <= 60):
            errs.append("sweep.steps: a whole number from 1 to 60")
        if set(sw) - {"bc", "steps"}:
            errs.append(f"sweep: unknown keys {sorted(set(sw) - {'bc', 'steps'})}")
    gaskets = st.get("gaskets", [] if asm else None)
    if not isinstance(gaskets, list) or (not gaskets and not asm):
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
    bolts = st.get("bolts", [] if asm else None)
    if not isinstance(bolts, list) or (not bolts and not asm):
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
    if asm:
        e, w2 = _validate_assembly(st, names)
        errs += e
        warns += w2
        used = [x.get("name") for key in ("gaskets", "contacts", "bolts", "nuts", "probes")
                for x in (st.get(key) or []) if isinstance(x, dict)]
        dup = sorted({n for n in used if n and used.count(n) > 1})
        if dup:
            errs.append(f"names used twice among gaskets, contacts, bolts, nuts and probes: {dup}")
        if not any(st.get(k) for k in ("contacts", "gaskets", "bolts", "nuts")):
            errs.append("an assembly's parts meet only through contacts, gaskets, bolts or nuts: give at least one")
    if st.get("converge"):
        errs.append(f"converge: not available for {'assembly' if asm else 'joint'} studies yet")
    extra = set(st) - (ASSEMBLY_KEYS if asm else JOINT_KEYS)
    if extra:
        warns.append(f"unknown study keys ignored: {sorted(extra)}")
    return errs, warns


def _word(v) -> bool:
    return isinstance(v, str) and bool(re.fullmatch(r"[A-Za-z0-9_]+", v))


def _validate_steps(st: dict, bcs: list, bnames: list, parts: dict) -> list[str]:
    errs = []
    steps = st.get("steps")
    if not isinstance(steps, list) or not steps:
        return ["steps: a list, e.g. [{kind: preload}, {kind: static, loads: [side]}, {kind: modal, modes: 6}]"]
    kinds = {bc_name(i, b): b.get("kind") for i, b in enumerate(bcs) if isinstance(b, dict)}
    loads = [n for n in bnames if kinds.get(n) in LOADS]
    sweeps = 0
    for i, s in enumerate(steps):
        w = f"steps[{i}]"
        if not isinstance(s, dict) or s.get("kind") not in STEP_KINDS:
            errs.append(f"{w}: {{kind: {' | '.join(STEP_KINDS)}, ...}}")
            continue
        k = s["kind"]
        allowed = {"preload": {"kind", "first_increment", "max_increment"},
                   "static": {"kind", "loads", "scale", "name", "first_increment", "max_increment"},
                   "sweep": {"kind", "bc", "steps"}, "modal": {"kind", "modes", "name"}}[k]
        for q in ("first_increment", "max_increment"):
            if q in s and (not _num(s[q]) or not 0 < s[q] <= 1):
                errs.append(f"{w}.{q}: a fraction of the step, 0 < x <= 1")
        if set(s) - allowed:
            errs.append(f"{w} ({k}): unknown keys {sorted(set(s) - allowed)} (allowed: {', '.join(sorted(allowed))})")
        if "name" in s and not _word(s["name"]):
            errs.append(f"{w}.name: a word")
        if k == "preload":
            if i:
                errs.append(f"{w}: the preload is the first step")
            if not (st.get("bolts") or st.get("nuts")):
                errs.append(f"{w}: a preload step needs bolts or nuts")
        elif k == "static":
            bad = [n for n in (s.get("loads") or []) if n not in loads]
            if not isinstance(s.get("loads", []), list) or bad:
                errs.append(f"{w}.loads: names of load bcs ({', '.join(loads) or 'none'})" + (f"; not {bad}" if bad else ""))
            if "scale" in s and not _num(s["scale"]):
                errs.append(f"{w}.scale: a number (the loads' factor, 1)")
        elif k == "sweep":
            sweeps += 1
            if kinds.get(s.get("bc")) not in ("pressure", "force"):
                errs.append(f"{w}.bc: the name of a pressure or force bc to step up")
            n = s.get("steps", 10)
            if not _num(n) or int(n) != n or not 1 <= n <= 60:
                errs.append(f"{w}.steps: a whole number from 1 to 60")
        elif k == "modal":
            if i == 0 or not any(q.get("kind") != "modal" for q in steps[:i] if isinstance(q, dict)):
                errs.append(f"{w}: a modal step follows a static (or preload) step: its frequencies are those of that state")
            n = s.get("modes", 6)
            if not _num(n) or int(n) != n or not 1 <= n <= 100:
                errs.append(f"{w}.modes: a whole number from 1 to 100")
            for pn, p in parts.items():
                try:
                    if not _num(material_of({"material": (p or {}).get("material", "steel")}).get("density")):
                        errs.append(f"{w}: part {pn}'s material needs a density (kg/m3) for frequencies")
                except Exception:  # noqa: BLE001
                    pass
    if sweeps > 1:
        errs.append("steps: at most one sweep")
    return errs


def _validate_assembly(st: dict, names: set) -> tuple[list[str], list[str]]:
    """Errors and warnings of an assembly's contacts, nuts and probes."""
    errs, warns = [], []
    contacts = st.get("contacts", [])
    if not isinstance(contacts, list):
        errs.append("contacts: a list of {name, faces, with, law, friction, clearance, fit_method}")
        contacts = []
    for i, c in enumerate(contacts):
        w = f"contact {i + 1}"
        if not isinstance(c, dict):
            errs.append(f"{w}: an object")
            continue
        if not _word(c.get("name")):
            errs.append(f"{w}: 'name' (a word)")
        for key in ("faces", "with"):
            if key not in c:
                errs.append(f"{w}: needs '{key}' (the {'slave' if key == 'faces' else 'master'} side's faces)")
            else:
                errs += _psel(c[key], names, f"{w} {key}")
        law = c.get("law", "hard")
        if not (law in CONTACT_LAWS or (isinstance(law, dict) and set(law) == {"stiffness"} and _num(law["stiffness"])
                                        and law["stiffness"] > 0)):
            errs.append(f"{w}.law: hard (a stiff penalty), bonded (tied) or {{stiffness: MPa/mm}}")
        if "friction" in c:
            if not _num(c["friction"]) or not 0 <= c["friction"] <= 1:
                errs.append(f"{w}.friction: 0 to 1 (0.2; 0: frictionless)")
            elif law == "bonded":
                warns.append(f"{w}: a bonded contact has no friction (it is tied)")
        if "clearance" in c and not _num(c["clearance"]):
            errs.append(f"{w}.clearance: mm, > 0 an initial gap, < 0 an interference (a press fit)")
        if law == "bonded" and _num(c.get("clearance")) and c["clearance"] != 0:
            errs.append(f"{w}: a bonded contact takes no clearance")
        if "fit_method" in c:
            if c["fit_method"] not in FIT_METHODS:
                errs.append(f"{w}.fit_method: {' | '.join(FIT_METHODS)} (how an interference is made)")
            elif not (_num(c.get("clearance")) and c["clearance"] < 0):
                errs.append(f"{w}.fit_method: only with an interference (a negative clearance)")
        extra = set(c) - {"name", "faces", "with", "law", "friction", "clearance", "fit_method"}
        if extra:
            errs.append(f"{w}: unknown keys {sorted(extra)}")
    nuts = st.get("nuts", [])
    if not isinstance(nuts, list):
        errs.append("nuts: a list of {name, size, preload | torque, at, bears_on, thread {part, depth}}")
        nuts = []
    for i, n in enumerate(nuts):
        w = f"nuts {i + 1}"
        if not isinstance(n, dict):
            errs.append(f"{w}: an object")
            continue
        if not _word(n.get("name")):
            errs.append(f"{w}: 'name' (a word)")
        if n.get("size") not in BOLTS:
            errs.append(f"{w}.size: one of {', '.join(BOLTS)} (the shaft's thread)")
        if ("preload" in n) == ("torque" in n):
            errs.append(f"{w}: give 'preload' (N: the axial clamp force) or 'torque' (N m, with k: the nut factor, 0.2)")
        for key in ("preload", "torque", "k", "washer_d", "hole_d"):
            if key in n and (not _num(n[key]) or n[key] <= 0):
                errs.append(f"{w}.{key}: a positive number")
        if not (isinstance(n.get("at"), list) and n["at"] and all(_point(q) for q in n["at"])):
            errs.append(f"{w}.at: [[x, y, z]] a point on the shaft's axis")
        if "axis" in n and not (_point(n["axis"]) and any(n["axis"])):
            errs.append(f"{w}.axis: [x, y, z] from the face it bears on towards its thread")
        if "bears_on" not in n:
            errs.append(f"{w}: needs 'bears_on' (the face the nut presses, e.g. {{part: impeller, nz: 1}})")
        else:
            errs += _psel(n["bears_on"], names, f"{w} bears_on")
        th = n.get("thread")
        if not (isinstance(th, dict) and th.get("part") in names and _num(th.get("depth")) and th["depth"] > 0):
            errs.append(f"{w}.thread: {{part: the shaft, depth: the nut's engaged length mm}}")
        extra = set(n) - {"name", "size", "preload", "torque", "k", "at", "axis", "bears_on", "thread", "washer_d",
                          "hole_d", "material"}
        if extra:
            errs.append(f"{w}: unknown keys {sorted(extra)}")
    probes = st.get("probes", [])
    if not isinstance(probes, list):
        errs.append("probes: a list of {name, faces}")
        probes = []
    for i, pr in enumerate(probes):
        if not isinstance(pr, dict) or not _word(pr.get("name")) or "faces" not in pr or set(pr) - {"name", "faces"}:
            errs.append(f"probe {i + 1}: {{name, faces (part-qualified)}}")
        else:
            errs += _psel(pr["faces"], names, f"probe {i + 1} faces")
    return errs, warns


# ---- watching CalculiX: stop a run that cannot converge -----------------------------------------------
MAX_FAILED_ATTEMPTS = 3   # unconverged attempts of one increment before the run is stopped
CHATTER_ITERS = 16        # iterations of one attempt looked at for contact chatter
CHATTER_FLIPS = 6         # direction changes of the contact count among them
CHATTER_SPREAD = 0.05     # the contact count's swing, relative to its largest value
SETTLE_ITERS = 25         # iterations of a retried attempt after which the contact must have settled ...
SETTLE_RATIO = 3.0        # ... to within this factor (assembly_press_fit, mochi 2026.11.04: 213k down to 13 contact
                          # elements and back to 30k in one increment: a frictionless ring drifting off its shaft)


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
    """The deck's step `step` in words (its CalculiX step number)."""
    steps = (plan or {}).get("steps") or []
    st = next((s for i, s in enumerate(steps) if s.get("ccx_step", i + 1) == step), None)
    if st is None:
        return "preload" if step == 1 else f"step {step}"
    kind = st.get("kind") or ("preload" if step == 1 else "sweep" if st.get("factor") is not None else "static")
    if kind == "preload":
        return "preload"
    n = st.get("step", step)
    sw = (plan or {}).get("sweep") or {}
    f = st.get("factor")
    if kind == "sweep" and f is not None:
        return f"step {n} ({sw.get('bc')} at {100 * f:.0f} %)"
    return f"step {n} ({st.get('what') or kind})"


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
            # ... only once CalculiX has cut the increment back (attempt 2 on): a too large increment diverges like
            # this and its cutback then converges (assembly_press_fit, mochi 2026.11.04: stopped in attempt 1)
            if a >= 2 and len(cur) >= SETTLE_ITERS and hi > 100 and hi >= SETTLE_RATIO * max(lo, 1):
                return (f"{_what(s, plan)} did not converge: the contact does not settle ({lo} to {hi} contact "
                        f"elements over the last {CHATTER_ITERS} iterations; step {s}, increment {i}, attempt {a}, "
                        f"iteration {cur[-1][3]}): a part may be free to slide or turn on a frictionless contact; "
                        "friction, a support or a bonded contact can hold it"), s
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


TIE_CLOSED = 0.5      # a pair at least this closed at the base step is tied for its frequencies
STICK_WARN = 0.9      # ... and when less of it sticks, the frequencies are an upper bound (it was partly slipping)


def modal_ties(plan: dict, base: dict | None) -> tuple[dict, list[str], list[str], dict]:
    """(tied {pair: None (all of it) or its closed slave nodes}, free, slipping, partial {pair: [closed, all]}) for a
    modal step on the state `base` (a result step). Bonded contacts are tied whole; a contact closed anywhere is tied
    at its closed nodes (where it carries load), however little of it is closed; gaskets closed over TIE_CLOSED of
    their area are tied. (s50_hubfit_loose, 2026.11.09: sleeve_on_collar, closed on part of its face and 73 % stuck,
    was left free under a 50 % rule; the impeller and sleeve then floated: 0.22-0.53 Hz.)"""
    tied, free, slip, partial = {}, [], [], {}
    cst = (base or {}).get("contacts") or {}
    gst = (base or {}).get("gaskets") or {}
    for c in plan["contacts"]:
        st = cst.get(c["name"]) or {}
        closed = st.get("_closed_nodes") or []
        if c["law"]["kind"] == "bonded" or (st.get("closed_fraction") or 0) >= 0.999:
            tied[c["name"]] = None
        elif closed:
            tied[c["name"]] = closed
            partial[c["name"]] = [len(closed), round(float(st.get("closed_fraction") or 0), 4)]
        else:
            free.append(c["name"])
            continue
        if c["law"]["kind"] != "bonded" and st.get("stick_fraction") is not None and st["stick_fraction"] < STICK_WARN:
            slip.append(c["name"])
    for g in plan["gaskets"]:
        if (gst.get(g["name"]) or {}).get("contact_fraction", 0) >= TIE_CLOSED:
            tied[g["name"]] = None
        else:
            free.append(g["name"])
    return tied, free, slip, partial


def free_groups(plan: dict, tied) -> list[list[str]]:
    """Groups of parts with no tied path to a support in a modal run (each would move as a rigid body on feat's
    stabilisation springs): parts joined by tied pairs, without a part that a support holds. Bolts and nuts are axial
    springs: they do not hold a part in the other directions, so they do not join parts here."""
    names = list(plan.get("part_names") or [])
    up = {n: n for n in names}

    def root(n):
        while up[n] != n:
            up[n] = up[up[n]]
            n = up[n]
        return n
    for x in plan["contacts"] + plan["gaskets"]:
        ps = [q for q in x.get("parts", []) if q in up]
        if x["name"] in tied and len(ps) > 1:
            for q in ps[1:]:
                up[root(q)] = root(ps[0])
    held = {root(q) for q in plan.get("support_parts", []) if q in up}
    groups: dict = {}
    for n in names:
        groups.setdefault(root(n), []).append(n)
    return [sorted(g) for r, g in groups.items() if r not in held]


RIGID_HZ = 1.0   # modes below this are rigid-body motion on the stabilisation springs, or a mechanism


def modal_run(name: str, study: dict, parts: list[dict], plan: dict, res: dict, step: dict, folder: Path,
              timeout: float) -> dict:
    """The frequencies of a modal step: a linear CalculiX *FREQUENCY run of the assembly with the pairs that were
    closed (and the bonded ones) tied, the others free, the bolts and nuts at their preloaded length. (A PERTURBATION
    step after frictional contact kept no tangential stiffness: assembly_clamped_beam's first mode was 0 Hz.)"""
    from .runner import SolverError, run_ccx
    base = next((r for r in reversed(res["steps"]) if r["step"] < step["step"]), None)
    tied, free, slip, partial = modal_ties(plan, base)
    floating = free_groups(plan, tied)
    extra = 6 * len(floating)   # their rigid-body modes come first: ask for that many more
    h = min(float(p["info"].get("size") or 1.0) for p in parts)
    gaps = [max(float((x.get("gap") or {}).get("max_mm") or 0.0), abs(float(x.get("clearance_mm") or 0.0)))
            for x in plan["contacts"] + plan["gaskets"] if x["name"] in tied]
    tol = 0.25 * h + max(gaps, default=0.0)
    out = {"tied": sorted(tied), "free": free, "slipping": slip, "tied_where_closed": partial, "floating": floating,
           "extra_modes": extra}
    try:
        text, _ = write(f"{name}_modal{step['step']}", study, parts,
                        modal={"modes": int(step["modes"]) + extra, "ties": tied, "tol": tol})
        inp = folder / f"{name}_modal{step['step']}.inp"
        inp.write_text(text)
        from .executor import modal_timeout_s
        run_ccx(inp, modal_timeout_s(timeout), serial=True)
        out["eigen"] = results.parse_dat(inp.with_suffix(".dat").read_text(errors="replace")).get("eigen") or []
        out["frd"] = inp.with_suffix(".frd").read_text(errors="replace") if inp.with_suffix(".frd").is_file() else None
    except (SolverError, OSError, MeshError) as exc:
        out["error"] = str(exc)[:500]
    return out


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
        progress.stage("solve", nodes=n_nodes, ccx_steps=sum(1 for q in deck_steps(study) if q["kind"] != "modal"),
                       modal_runs=sum(1 for q in deck_steps(study) if q["kind"] == "modal"))
    text, plan = write(name, study, parts)
    inp = Path(folder) / f"{name}.inp"
    inp.write_text(text)
    check_cancel()
    watch = Watch(inp, plan)
    stopped = None
    try:
        run_ccx(inp, timeout, monitor=watch, nodes=n_nodes, method="direct")
    except (Stopped, subprocess.TimeoutExpired) as exc:
        sta = inp.with_suffix(".sta")
        done = completed_steps(sta.read_text(errors="replace")) if sta.is_file() else 0
        n_ccx = max((s.get("ccx_step") or 0 for s in plan["steps"]), default=0)
        why = str(exc) if isinstance(exc, Stopped) else (
            f"CalculiX ran past the time limit ({timeout:g} s) in {_what(done + 1, plan)}" if done < n_ccx else
            f"CalculiX ran past the time limit ({timeout:g} s) after its last step, while writing its results")
        if done < 1 or not inp.with_suffix(".dat").is_file():   # not even the first step: nothing to report
            gaps = [w for w in (gap_warning(g) for g in plan["gaskets"]) if w]
            raise SolverError(why + ". " + (" ".join(gaps) + ". " if gaps else "")
                              + "Check that the contact and gasket faces touch (parts.move, or a contact's clearance), "
                              "that every part is held by supports, bolts or contacts, and the start of a gasket curve; "
                              "the stabilisation springs were on.") from exc
        stopped = (why, done)
        cut = next((i for i, s in enumerate(plan["steps"]) if s.get("ccx_step") and s["ccx_step"] > done),
                   len(plan["steps"]))
        plan = dict(plan, steps=plan["steps"][:cut])
    dat_text = inp.with_suffix(".dat").read_text(errors="replace")
    frd_text = inp.with_suffix(".frd").read_text(errors="replace") if inp.with_suffix(".frd").is_file() else ""
    res = evaluate(study, parts, plan, dat_text, frd_text)
    # modal steps: a linear frequency run each, on the state its previous step left
    runs, modes_frd = {}, None
    for s in [q for q in plan["steps"] if q["kind"] == "modal"]:
        check_cancel()
        run = modal_run(name, study, parts, plan, res, s, Path(folder), timeout)
        runs[s["step"]] = run
        if modes_frd is None and run.get("frd"):
            modes_frd = run["frd"]
    if runs:
        res = evaluate(study, parts, plan, dat_text, frd_text, runs)
    t2 = time.perf_counter()
    if stopped:
        res["stopped"] = {"why": stopped[0], "completed_steps": stopped[1]}
        last = res["steps"][-1] if res["steps"] else {}
        at = (f"{last['load']:.4g} {res.get('swept', {}).get('unit', '')}".strip() if last.get("load") is not None
              else last.get("what", ""))
        res.setdefault("warnings", []).append(
            f"{stopped[0]}: the results stop at step {last.get('step', '?')} ({at}); the later steps are not known")
    minfo = {"nodes": n_nodes, "elements": sum(len(p["mesh"].tets) for p in parts), "order": parts[0]["mesh"].order,
             "parts": {p["name"]: {k: v for k, v in p["info"].items() if k in ("nodes", "elements", "size", "mesh_notes",
                                                                              "straightened_elements")} for p in parts}}
    _record_timing({"analysis": study.get("analysis", "joint")}, solver, minfo, t1 - t0, t2 - t1)
    res["selection"] = plan["selection"]
    res["mesh"] = minfo
    res["seconds"] = {"mesh": round(t1 - t0, 3), "solve": round(t2 - t1, 3), "total": round(time.perf_counter() - t0, 3)}
    res["files"] = {k: str(inp.with_suffix(s)) for k, s in (("inp", ".inp"), ("dat", ".dat"), ("frd", ".frd"))
                    if inp.with_suffix(s).is_file()}
    for r in res.get("steps", []):   # the closed nodes were for the modal ties
        for v in (r.get("contacts") or {}).values():
            if isinstance(v, dict):
                v.pop("_closed_nodes", None)
    res["deck"] = {"steps": len(plan["steps"]), "bolts": sum(len(b["bolts"]) for b in plan["bolts"]),
                   "gaskets": len(plan["gaskets"])}
    from .runner import CRASHES, MEMORY_NOTES, PEAK
    if CRASHES:   # a threaded run that died (out of memory, a crash) and was rerun single-threaded
        res["solver_crashes"] = list(CRASHES)
    if MEMORY_NOTES:   # threads cut to 1 for the meshed size, or a run that may not fit the job's memory
        res["memory_notes"] = list(MEMORY_NOTES)
    if PEAK["rss_mb"]:
        res["seconds"]["ccx_peak_rss_mb"] = round(PEAK["rss_mb"], 1)
        res["seconds"]["ccx_threads"] = PEAK["threads"]
    try:   # the state for pictures (images, render_images): the results stand without it
        frd = inp.with_suffix(".frd")
        save_state(Path(folder), study, parts, plan, res, frd.read_text(errors="replace") if frd.is_file() else "",
                   modes_frd)
        res["files"]["state"] = str(Path(folder) / "state.npz")
    except Exception as exc:  # noqa: BLE001
        res.setdefault("warnings", []).append(f"could not save the state for pictures: {type(exc).__name__}: {exc}"[:300])
    return res
