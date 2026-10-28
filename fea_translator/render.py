"""render.py — pictures of a solved study: a field on the part's surface (deformed or not), mode shapes, the
"setup" view (which faces each support and load took), and section cuts; plus the saved state they are drawn from.
Version: 2026.10.28.1

    render(folder, specs, name) -> ({name: {path, what, ...}}, warnings)    from <folder>/state.npz into <folder>/images
    default_images(state | study) -> specs (for "images": true)
    image_errors(images, study) -> validation errors (study.validate_study calls it)
    save_state(folder, mesh, fields, study, meta) / load_state(folder) -> state: re-render without solving
    field_values(state, field, mode) -> nodal values in the field's units

As cfd-translator 2026.10.28.1 (render.py, render_vtk.py): Matplotlib (Agg) draws by default and is the fallback;
VTK through PyVista (render_vtk, in a child process) draws "quality": "high" views, lit and smooth. Every run saves
state.npz (nodes, elements, the boundary triangles with their CAD-face tags, nodal fields) and state.json (the study,
each bc's faces, mode values, the peak), so render_images draws other views later without solving again.

Fields are computed here from the nodal solution, the same way for CalculiX and the reference solver: stresses
from the displacements (runner.recompute_stresses, CalculiX's element formulation, thermal strain included), the
element means averaged to the nodes by element volume; gradients (heat flux, electric field, current density) at
element centroids, averaged the same way. A uniform field (uniaxial stress) is exact at every node.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}
# labels without '|' (VTK's text has none): "displacement magnitude", not |u|
FIELDS = {"von_mises": "von Mises stress (MPa)", "displacement": "displacement magnitude (mm)",
          "ux": "displacement x (mm)", "uy": "displacement y (mm)", "uz": "displacement z (mm)",
          "max_principal": "max principal stress (MPa)", "safety_factor": "safety factor (yield / von Mises)",
          "temperature": "temperature (°C)", "heat_flux": "heat flux magnitude (W/m²)", "potential": "potential (V)",
          "current_density": "current density magnitude (A/mm²)", "e_field": "electric field magnitude (kV/mm)"}
MODE_LABEL = "mode shape: displacement magnitude (relative)"
CMAPS = {"von_mises": "jet", "max_principal": "coolwarm", "safety_factor": "RdYlGn", "temperature": "inferno",
         "heat_flux": "inferno", "potential": "viridis", "current_density": "plasma", "e_field": "plasma",
         "ux": "coolwarm", "uy": "coolwarm", "uz": "coolwarm", "displacement": "viridis"}
MECH = ("von_mises", "displacement", "ux", "uy", "uz", "max_principal", "safety_factor")
PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
           "#7f7f7f", "#393b79", "#ad494a")
PART_GREY = "#c8c8c8"
DEFORM_FRACTION = 0.05      # deformed: true scales the largest displacement to 5 % of the part's size
SIZE_IN, DPI = (9, 7), 130
SPEC_KEYS = ("name", "surface", "field", "deformed", "view", "clim", "quality", "mode", "setup", "slice")


# ---- which fields an analysis has ----------------------------------------------------------------
def fields_of(study: dict) -> tuple:
    """The fields a study's pictures can show (safety_factor needs the material's yield)."""
    an = study.get("analysis")
    mech = ("von_mises", "displacement", "ux", "uy", "uz", "max_principal")
    from .study import material_of
    try:
        has_yield = bool(material_of(study).get("yield"))
    except Exception:  # noqa: BLE001
        has_yield = False
    mech += ("safety_factor",) if has_yield else ()
    out = {"static": mech, "modal": (), "buckling": (), "thermal": ("temperature", "heat_flux"),
           "electric": ("potential", "current_density", "e_field"), "electrostatic": ("potential", "e_field"),
           "joule": ("temperature", "heat_flux", "potential", "current_density", "e_field")}.get(an, ())
    if study.get("stress") and an in ("thermal", "joule"):
        out += mech
    return out


def default_images(study: dict, modes: int | None = None) -> list[dict]:
    """static: von Mises on the deformed part, and the setup view; modal: the first 3 modes; buckling: mode 1;
    thermal: temperature (with stress: also von Mises, deformed); electric: potential and current density;
    electrostatic: potential and electric field; joule: temperature, potential and current density."""
    an = study.get("analysis")
    vm = {"name": "von_mises", "field": "von_mises", "deformed": True}
    if an == "static":
        return [vm, {"name": "setup", "setup": True}]
    if an == "modal":
        n = min(3, int(modes if modes is not None else study.get("modes", 6)))
        return [{"name": f"mode{k}", "mode": k} for k in range(1, n + 1)]
    if an == "buckling":
        return [{"name": "mode1", "mode": 1}]
    out = {"thermal": [{"name": "temperature", "field": "temperature"}],
           "electric": [{"name": "potential", "field": "potential"}, {"name": "current_density", "field": "current_density"}],
           "electrostatic": [{"name": "potential", "field": "potential"}, {"name": "e_field", "field": "e_field"}],
           "joule": [{"name": "temperature", "field": "temperature"}, {"name": "potential", "field": "potential"},
                     {"name": "current_density", "field": "current_density"}]}.get(an, [])
    if study.get("stress") and an in ("thermal", "joule"):
        out.append(vm)
    return out


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def image_errors(images, study: dict) -> list[str]:
    """Errors of a study's "images": true, or a list of views (validated against its analysis and bc names)."""
    if images is True or images is False:
        return []
    if not isinstance(images, list):
        return ["images: true (the default views) or a list of {name, field | mode | setup, surface, deformed, view, "
                "clim, quality, slice}"]
    from .study import bc_name
    an = study.get("analysis")
    bcs = [bc_name(i, b) for i, b in enumerate(study.get("bcs", []) or []) if isinstance(b, dict)]
    have = fields_of(study)
    modes = int(study.get("modes", 6 if an == "modal" else 3)) if _num(study.get("modes", 0)) else 6
    errs, names = [], []
    for i, sp in enumerate(images):
        w = f"images[{i}]"
        if not isinstance(sp, dict):
            errs.append(f"{w}: an object {{name, field | mode | setup, ...}}")
            continue
        for k in sp:
            if k not in SPEC_KEYS:
                errs.append(f"{w}: unknown key {k!r} (allowed: {', '.join(SPEC_KEYS)})")
        if "name" in sp:
            if not isinstance(sp["name"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", sp["name"]):
                errs.append(f"{w}.name: a word (letters, digits, _ . -)")
            elif sp["name"] in names:
                errs.append(f"{w}.name: {sp['name']!r} twice")
            names.append(sp.get("name"))
        what = [k for k in ("field", "mode", "setup") if k in sp]
        if len(what) > 1:
            errs.append(f"{w}: give one of field, mode or setup")
        if sp.get("setup") not in (None, True):
            errs.append(f"{w}.setup: true (the faces each support and load took)")
        if "mode" in sp:
            if an not in ("modal", "buckling"):
                errs.append(f"{w}.mode: for modal and buckling studies")
            elif not _num(sp["mode"]) or int(sp["mode"]) != sp["mode"] or not 1 <= sp["mode"] <= modes:
                errs.append(f"{w}.mode: a mode number from 1 to {modes} (the study's modes)")
        if "field" in sp and sp["field"] not in have:
            errs.append(f"{w}.field: {sp['field']!r} is not a field of this {an} study: "
                        + (", ".join(have) if have else "it has only mode shapes (mode: n)")
                        + ("" if sp["field"] != "safety_factor" or an not in ("static", "thermal", "joule")
                           else " (safety_factor needs the material's yield)"))
        if not what and not have:
            errs.append(f"{w}: a {an} study's pictures are mode shapes: give mode: n")
        if "surface" in sp and sp["surface"] != "part" and sp["surface"] not in bcs:
            errs.append(f"{w}.surface: part or a bc name ({', '.join(bcs) or 'none'})")
        if "setup" in sp and any(k in sp for k in ("deformed", "clim")):
            errs.append(f"{w}: the setup view takes no deformed or clim")
        if "deformed" in sp:
            d = sp["deformed"]
            mech = sp.get("field") in MECH or "mode" in sp
            if not (d is True or d is False or (_num(d) and d > 0)):
                errs.append(f"{w}.deformed: true (scaled to about 5 % of the part's size) or a scale factor > 0")
            elif d is not False and not mech:
                errs.append(f"{w}.deformed: for displacement and stress fields, and mode shapes")
        if "view" in sp and (not isinstance(sp["view"], (list, tuple)) or len(sp["view"]) != 2
                             or not all(_num(v) for v in sp["view"])):
            errs.append(f"{w}.view: [elevation, azimuth] in degrees")
        if "clim" in sp:
            c = sp["clim"]
            if not (c in ("full", "robust") or (isinstance(c, (list, tuple)) and len(c) == 2 and all(_num(v) for v in c)
                                               and c[0] < c[1])):
                errs.append(f"{w}.clim: [lo, hi] (lo < hi), full or robust")
        if "quality" in sp and sp["quality"] not in ("high", "basic"):
            errs.append(f"{w}.quality: high (VTK: lit, smooth) or basic (Matplotlib)")
        if "slice" in sp:
            s = sp["slice"]
            if not isinstance(s, dict) or s.get("axis") not in AXES or not _num(s.get("at")) or set(s) - {"axis", "at"}:
                errs.append(f"{w}.slice: {{axis: x | y | z, at: mm}} (keeps the part on the low side of the plane)")
            elif "setup" in sp:
                errs.append(f"{w}.slice: not for the setup view")
    return errs


# ---- fields from the nodal solution ------------------------------------------------------------
def _centroid_grads(mesh) -> np.ndarray:
    """dN/dx at each element's centroid: (E, n, 3)."""
    from .meshdata import EDGES, simplex_shape
    L = np.full((1, 4), 0.25)
    _, dN = simplex_shape(L, mesh.order == 2, EDGES)
    dNdxi = dN[:, :, 1:] - dN[:, :, :1]                           # (1, n, 3)
    X = mesh.nodes[mesh.tets]                                    # (E, n, 3)
    J = np.einsum("qnj,enk->eqjk", dNdxi, X)[:, 0]               # (E, 3, 3)
    return np.einsum("ekj,nj->enk", np.linalg.inv(J), dNdxi[0])  # (E, n, 3)


def to_nodes(mesh, per_element: np.ndarray) -> np.ndarray:
    """Element values (E, ...) averaged to the nodes, weighted by element volume (every node of an element)."""
    from .meshdata import tet_volumes
    v = np.abs(tet_volumes(mesh.nodes, mesh.tets))
    pe = per_element.reshape(len(mesh.tets), -1)
    acc = np.zeros((len(mesh.nodes), pe.shape[1]))
    wsum = np.zeros(len(mesh.nodes))
    for c in range(mesh.tets.shape[1]):
        np.add.at(acc, mesh.tets[:, c], pe * v[:, None])
        np.add.at(wsum, mesh.tets[:, c], v)
    out = acc / np.maximum(wsum, 1e-300)[:, None]
    return out.reshape((len(mesh.nodes),) + per_element.shape[1:])


def nodal_gradient(mesh, values: np.ndarray) -> np.ndarray:
    """Gradient (N, 3) of a nodal scalar, per unit mm: at element centroids, averaged to the nodes."""
    g = np.einsum("enk,en->ek", _centroid_grads(mesh), np.nan_to_num(values)[mesh.tets])
    return to_nodes(mesh, g)


def nodal_stress(mesh, U: np.ndarray, study: dict, plan: dict) -> np.ndarray:
    """Stress tensor (N, 6: sxx, syy, szz, sxy, sxz, syz) at the nodes: from the displacements at the integration
    points (runner.recompute_stresses), element means averaged to the nodes."""
    from .runner import recompute_stresses
    return to_nodes(mesh, recompute_stresses(mesh, np.nan_to_num(U), study, plan).mean(axis=1))


def von_mises(S: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, sxz, syz = (S[:, k] for k in range(6))
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2))


def max_principal(S: np.ndarray) -> np.ndarray:
    T = np.empty((len(S), 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = S[:, 0], S[:, 1], S[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = S[:, 3]
    T[:, 0, 2] = T[:, 2, 0] = S[:, 4]
    T[:, 1, 2] = T[:, 2, 1] = S[:, 5]
    return np.linalg.eigvalsh(T)[:, -1]


def field_values(state: dict, field: str | None, mode: int | None = None) -> np.ndarray:
    """Nodal values of a field (FIELDS units), or the relative displacement magnitude of a mode shape."""
    f = state["fields"]
    if mode is not None:
        modes = f.get("modes")
        if modes is None or len(modes) < mode:
            raise ValueError(f"no shape of mode {mode} was saved (the solver gave {0 if modes is None else len(modes)})")
        m = np.linalg.norm(modes[mode - 1], axis=1)
        return m / max(float(m.max()), 1e-300)
    need = {"von_mises": "S", "max_principal": "S", "safety_factor": "S", "displacement": "U", "ux": "U", "uy": "U",
            "uz": "U", "temperature": "T", "heat_flux": "q", "potential": "phi", "current_density": "J", "e_field": "E"}
    key = need.get(field)
    if key is None:
        raise ValueError(f"unknown field {field!r}")
    if f.get(key) is None:
        raise ValueError(f"field {field!r} is not in this study's results")
    v = f[key]
    if field == "von_mises":
        return von_mises(v)
    if field == "max_principal":
        return max_principal(v)
    if field == "safety_factor":
        y = state["meta"].get("yield")
        if not y:
            raise ValueError("safety_factor needs the material's yield")
        return float(y) / np.maximum(von_mises(v), 1e-9 * float(y))
    if field == "displacement":
        return np.linalg.norm(v, axis=1)
    if field in ("ux", "uy", "uz"):
        return v[:, "xyz".index(field[1])]
    if field in ("heat_flux", "current_density", "e_field"):
        return np.linalg.norm(v, axis=1)
    return np.asarray(v, float)


def displacement_of(state: dict, spec: dict) -> np.ndarray | None:
    """The displacement the picture deforms the part by (unscaled): the mode's shape, or U."""
    f = state["fields"]
    if spec.get("mode") is not None:
        return f["modes"][int(spec["mode"]) - 1] if f.get("modes") is not None and len(f["modes"]) >= spec["mode"] else None
    return f.get("U")


def deform_scale(state: dict, spec: dict) -> float:
    """0 (undeformed), the given factor, or (true, and always for modes) 5 % of the part's size over the largest
    displacement."""
    d = spec.get("deformed", True if spec.get("mode") is not None else False)
    if d is False:
        return 0.0
    U = displacement_of(state, spec)
    if U is None:
        return 0.0
    if _num(d) and d is not True:
        return float(d)
    umax = float(np.nanmax(np.linalg.norm(U, axis=1))) if len(U) else 0.0
    return DEFORM_FRACTION * state["meta"]["size"] / umax if umax > 0 else 0.0


def _limits(vals: np.ndarray, spec: dict, default: str) -> tuple[float, float, str]:
    """Colour range: clim [lo, hi], 'full' (min to max) or 'robust' (2nd to 98th percentile). Returns (lo, hi, extend)."""
    c = spec.get("clim", default)
    v = vals[np.isfinite(vals)]
    if isinstance(c, (list, tuple)):
        lo, hi = float(c[0]), float(c[1])
    elif c == "robust" and len(v):
        lo, hi = (float(x) for x in np.percentile(v, [2, 98]))
    else:
        lo, hi = (float(np.min(v)), float(np.max(v))) if len(v) else (0.0, 1.0)
    if not isinstance(c, (list, tuple)) and hi - lo <= 1e-6 * max(abs(lo), abs(hi)):
        # a uniform field (uniaxial stress): round-off must not look like a pattern; +-1 % around the value
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.01 * abs(mid), mid + 0.01 * abs(mid)
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-9, 1e-12)
    ext = {(True, True): "both", (True, False): "min", (False, True): "max"}.get(
        (bool(len(v) and v.min() < lo), bool(len(v) and v.max() > hi)), "neither")
    return lo, hi, ext


# ---- the saved state -----------------------------------------------------------------------------
def save_state(folder: Path, mesh, fields: dict, study: dict, meta: dict) -> None:
    """state.npz: nodes, tets, the boundary triangles (corner nodes) and their CAD-face tags, the nodal fields
    (U, S, T, q, phi, E, J, modes; those the analysis has); state.json: the study, each bc's tags and kind,
    mode values, the peak, the yield, the part's size."""
    folder = Path(folder)
    tags = sorted(mesh.surfaces)
    tri = np.concatenate([np.asarray(mesh.surfaces[t], np.int64).reshape(-1, 3) for t in tags]) if tags else np.zeros((0, 3), int)
    tri_tag = np.concatenate([np.full(len(np.asarray(mesh.surfaces[t]).reshape(-1, 3)), t) for t in tags]) if tags else np.zeros(0, int)
    arr = {"nodes": np.asarray(mesh.nodes, float), "tets": np.asarray(mesh.tets, np.int32), "tri": tri.astype(np.int32),
           "tri_tag": tri_tag.astype(np.int32)}
    for k, v in fields.items():
        if v is not None:
            arr[f"f_{k}"] = np.asarray(v, float)
    tmp = folder / ".state.npz.tmp.npz"
    np.savez_compressed(tmp, **arr)
    tmp.replace(folder / "state.npz")
    m = dict(meta, size=float(mesh.size), study=study)
    (folder / ".state.json.tmp").write_text(json.dumps(m, default=float))
    (folder / ".state.json.tmp").replace(folder / "state.json")


def load_state(folder: Path) -> dict:
    folder = Path(folder)
    z = np.load(folder / "state.npz")
    meta = json.loads((folder / "state.json").read_text())
    return {"nodes": z["nodes"], "tets": z["tets"], "tri": z["tri"], "tri_tag": z["tri_tag"],
            "fields": {k[2:]: z[k] for k in z.files if k.startswith("f_")}, "meta": meta, "folder": str(folder)}


def bc_faces(state: dict) -> list[dict]:
    """[{name, kind, tags, color}] of the supports and loads, in the study's order, each with its colour."""
    out = []
    for i, b in enumerate(state["meta"].get("bcs", [])):
        out.append(dict(b, color=PALETTE[i % len(PALETTE)]))
    return out


def surface_triangles(state: dict, spec: dict) -> np.ndarray:
    """Indices of the boundary triangles a picture shows: all (surface 'part') or a bc's faces."""
    which = spec.get("surface", "part")
    if which == "part":
        return np.arange(len(state["tri"]))
    b = next((b for b in state["meta"].get("bcs", []) if b["name"] == which), None)
    if b is None:
        raise ValueError(f"no bc named {which!r}")
    idx = np.nonzero(np.isin(state["tri_tag"], b["tags"]))[0]
    if not len(idx):
        raise ValueError(f"bc {which!r} took no faces")
    return idx


def peak_point(state: dict, spec: dict) -> dict | None:
    """The off-BC von Mises peak (point and value) for von Mises and safety-factor pictures."""
    if spec.get("field") not in ("von_mises", "safety_factor"):
        return None
    pk = state["meta"].get("peak")
    return pk if pk and pk.get("at") else None


def _nearest_node(nodes: np.ndarray, p) -> int:
    return int(np.argmin(np.linalg.norm(nodes - np.asarray(p, float), axis=1)))


# ---- section cuts (marching tetrahedra on the corner nodes) ----------------------------------------
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def section(nodes: np.ndarray, tets: np.ndarray, axis: int, at: float, values: np.ndarray, U: np.ndarray | None):
    """Polygons (list of (k, 3) arrays) where the plane {x_axis = at} cuts the elements, with the field and the
    displacement interpolated at their corners: (polys, values per polygon corner, displacements per corner)."""
    c = tets[:, :4]
    d = nodes[c, axis] - at
    cut = (d.min(axis=1) < 0) & (d.max(axis=1) > 0)
    polys, vals, disp = [], [], []
    for e in np.nonzero(cut)[0]:
        pts, vv, uu = [], [], []
        for a, b in _TET_EDGES:
            da, db = d[e, a], d[e, b]
            if (da < 0) != (db < 0):
                t = da / (da - db)
                na, nb = c[e, a], c[e, b]
                pts.append(nodes[na] + t * (nodes[nb] - nodes[na]))
                vv.append(values[na] + t * (values[nb] - values[na]))
                uu.append(U[na] + t * (U[nb] - U[na]) if U is not None else np.zeros(3))
        if len(pts) < 3:
            continue
        P = np.array(pts)
        ij = [k for k in range(3) if k != axis]
        cen = P.mean(axis=0)
        o = np.argsort(np.arctan2(P[:, ij[1]] - cen[ij[1]], P[:, ij[0]] - cen[ij[0]]))
        polys.append(P[o])
        vals.append(np.array(vv)[o])
        disp.append(np.array(uu)[o])
    return polys, vals, disp


# ---- Matplotlib ----------------------------------------------------------------------------------
def _light(view) -> np.ndarray:
    e, a = np.radians(view[0]), np.radians(view[1])
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _shade(tris: np.ndarray, view) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    return 0.55 + 0.45 * np.abs(n @ _light(view))


def _axes3d(fig, pts, view):
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False                    # markers drawn last, over the surface
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.02 * float(np.max(hi - lo) or 1.0)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    ax.set_box_aspect(np.maximum(hi - lo, 1e-3 * max(float(np.max(hi - lo)), 1e-9)))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    return ax


def draw_matplotlib(state: dict, spec: dict, title: str, path: Path) -> dict:
    """One picture with Matplotlib (Agg): the setup view, or a field / mode shape on the surface (with a section
    cut when slice is given)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    view = spec.get("view", [25, -60])
    X = state["nodes"]
    tri = state["tri"]
    fig = plt.figure(figsize=SIZE_IN, dpi=DPI)
    fig.patch.set_facecolor("white")
    if spec.get("setup"):
        T = X[tri]
        colors = np.tile(np.array(mcolors.to_rgba(PART_GREY)), (len(tri), 1))
        colors[:, :3] *= _shade(T, view)[:, None]
        handles, info = [], {}
        for b in bc_faces(state):
            m = np.isin(state["tri_tag"], b["tags"])
            if m.any():
                colors[m] = mcolors.to_rgba(b["color"])         # flat: the bc's own colour
            handles.append(Patch(color=b["color"], label=f"{b['name']} ({b['kind']})" + ("" if m.any() else ": no faces")))
            info[b["name"]] = {"color": b["color"], "triangles": int(m.sum())}
        ax = _axes3d(fig, X, view)
        ax.add_collection3d(Poly3DCollection(T, facecolors=colors, edgecolors="none"))
        if handles:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=9, framealpha=0.9)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        return {"bcs": info, "renderer": "matplotlib"}
    mode = spec.get("mode")
    vals = field_values(state, spec.get("field"), mode)
    U = displacement_of(state, spec)
    sc = deform_scale(state, spec)
    Xd = X + sc * np.nan_to_num(U) if sc and U is not None else X
    idx = surface_triangles(state, spec)
    sl = spec.get("slice")
    polys, pvals = [], []
    if sl:
        a, at = AXES[sl["axis"]], float(sl["at"])
        keep = X[tri[idx], a].mean(axis=1) <= at
        idx = idx[keep]
        P, V, D = section(X, state["tets"], a, at, vals, U)
        polys = [p + sc * d for p, d in zip(P, D)] if sc else P
        pvals = [float(np.mean(v)) for v in V]
        if not polys and not len(idx):
            raise ValueError(f"nothing of the part on the low side of {sl['axis']} = {at:g} mm")
    T = Xd[tri[idx]]
    tv = vals[tri[idx]].mean(axis=1)
    allv = np.concatenate([tv, np.array(pvals)]) if pvals else tv
    lo, hi, ext = _limits(allv, spec, "full")
    norm = mcolors.Normalize(vmin=lo, vmax=hi)
    cmap = matplotlib.colormaps["viridis" if mode is not None else CMAPS.get(spec.get("field"), "viridis")]
    fc = cmap(norm(tv))
    fc[:, :3] *= _shade(T, view)[:, None]
    pts = np.concatenate([T.reshape(-1, 3)] + ([np.concatenate(polys)] if polys else []))
    ax = _axes3d(fig, pts, view)
    if len(T):
        ax.add_collection3d(Poly3DCollection(T, facecolors=fc, edgecolors="none"))
    if polys:
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cmap(norm(np.array(pvals))), edgecolors="none"))
    info = {}
    pk = peak_point(state, spec)
    if pk and sl and float(pk["at"][AXES[sl["axis"]]]) > float(sl["at"]):
        pk = None                                   # on the side the cut takes away
    if pk:
        p = np.asarray(pk["at"], float)
        if sc and U is not None:
            p = p + sc * np.nan_to_num(U[_nearest_node(X, p)])
        ax.scatter([p[0]], [p[1]], [p[2]], marker="X", s=90, color="black", zorder=10, depthshade=False)
        ax.text(p[0], p[1], p[2], f"  peak off-BC {pk['value']:.4g} MPa", color="black", fontsize=9, zorder=11)
        info["peak_marked"] = [round(float(x), 4) for x in pk["at"]]
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, label=MODE_LABEL if mode is not None else FIELDS[spec["field"]], shrink=0.7, extend=ext,
                      format="%.4g")
    cb.ax.yaxis.get_major_formatter().set_useOffset(False) if hasattr(cb.ax.yaxis.get_major_formatter(), "set_useOffset") else None
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    info.update({"min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)), "clim": [lo, hi], "renderer": "matplotlib",
                 "triangles": int(len(T)) + len(polys)})
    if sc:
        info["deformation_scale"] = float(f"{sc:.6g}")
    return info


def title_of(state: dict, spec: dict, name: str) -> str:
    if spec.get("setup"):
        return f"{name}: supports and loads (the faces each bc took)"
    mv = state["meta"].get("mode_values") or {}
    if spec.get("mode") is not None:
        k = int(spec["mode"])
        vals = mv.get("values") or []
        v = f", {vals[k - 1]:.5g} {mv.get('unit', '')}".rstrip() if len(vals) >= k else ""
        return f"{name}: mode {k}{v}"
    s = f"{name}: {FIELDS[spec['field']]}"
    if spec.get("surface", "part") != "part":
        s += f" on {spec['surface']}"
    if spec.get("slice"):
        s += f", cut at {spec['slice']['axis']} = {float(spec['slice']['at']):g} mm"
    return s


def _state_for(folder: Path) -> dict:
    return load_state(folder)


def render(folder: Path, specs: list, name: str = "study") -> tuple[dict, list]:
    """PNGs in <folder>/images from <folder>/state.npz. quality 'high' views go to VTK (render_vtk, a child
    process); one VTK cannot draw falls back to Matplotlib with a warning. FEAT_RENDER=matplotlib: no VTK."""
    folder = Path(folder)
    out_dir = folder / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    state = _state_for(folder)
    if specs is True:
        modes = state["fields"].get("modes")
        specs = default_images(state["meta"]["study"], len(modes) if modes is not None else None)
    specs = [dict(sp, name=sp.get("name") or f"image{k + 1}") for k, sp in enumerate(specs)]
    out, warns = {}, []
    hq = [sp for sp in specs if sp.get("quality") == "high"]
    if hq:
        from . import render_vtk
        got, w = render_vtk.render(folder, hq, out_dir, name)
        out.update(got)
        warns += w
        for sp in hq:
            if sp["name"] not in got:
                warns.append(f"image {sp['name']!r}: drawn with Matplotlib instead")
    import importlib.util
    rest = [sp for sp in specs if sp["name"] not in out]
    if rest and importlib.util.find_spec("matplotlib") is None:
        return out, warns + ["images need matplotlib (pip install matplotlib)"]
    for sp in rest:
        nm = sp["name"]
        path = out_dir / f"{nm}.png"
        try:
            what = title_of(state, sp, name)
            info = draw_matplotlib(state, sp, what, path)
            out[nm] = {"path": str(path), "what": what, **{k: round(v, 6) if isinstance(v, float) else v for k, v in info.items()}}
        except Exception as exc:  # noqa: BLE001 (a picture must not fail the study)
            warns.append(f"image {nm!r}: {type(exc).__name__}: {exc}")
    return out, warns
