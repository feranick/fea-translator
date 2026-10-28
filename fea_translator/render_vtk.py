"""render_vtk.py — 3D pictures with VTK (through PyVista), offscreen: lit, smooth surfaces for an image's
"quality": "high" (fields, mode shapes, section cuts, the setup view). Matplotlib (render.py) stays the default
and the fallback when VTK cannot draw.
Version: 2026.10.28.1

    render(state_dir, specs, out_dir, name) -> ({name: {path, what, ...}}, warnings)    (in the caller's process)
    probe() -> {ok, window, prefix, why}: can this machine draw offscreen (cached per process)
    status() -> {vtk: version | None, offscreen}: for tool_status

As cfd-translator 2026.10.28.1. The drawing runs in a child process (python -m fea_translator.render_vtk <state dir>
<specs.json> <out dir> <name>, printing one JSON line {images, warnings}): an OpenGL crash cannot take down a server
or a job, and its memory goes back when it ends. Offscreen OpenGL comes from EGL (a GPU driver), OSMesa (libosmesa6)
or an X display; without any, the child runs under xvfb-run when it is installed. FEAT_RENDER=matplotlib turns VTK
off; FEAT_RENDER_TIMEOUT (s, default 900) bounds the child.

The tetrahedra map straight to a VTK UnstructuredGrid (tet4: VTK_TETRA; tet10: VTK_QUADRATIC_TETRA, whose mid-edge
order 01, 12, 20, 03, 13, 23 is CalculiX's); only its boundary surface is drawn (curved, from the quadratic
elements). A section cut clips the volume mesh (the low side of the plane stays), so the cut face shows the field
inside. Smooth shading with split_sharp_edges, a white background, a vertical scalar bar; VTK's text has no '|'.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_PROBE: dict = {}
SIZE = (1400, 900)


# ---- can this machine draw? ---------------------------------------------------------------------
def available() -> bool:
    return importlib.util.find_spec("pyvista") is not None and importlib.util.find_spec("vtk") is not None


_PROBE_CODE = ("import pyvista as pv\n"
               "p = pv.Plotter(off_screen=True, window_size=(64, 64))\n"
               "p.add_mesh(pv.Sphere())\n"
               "img = p.screenshot(return_img=True)\n"
               "print('VTKOK', p.ren_win.GetClassName(), int(img.std() > 0))\n")


def _try(prefix: list) -> tuple[bool, str]:
    try:
        r = subprocess.run(prefix + [sys.executable, "-c", _PROBE_CODE], capture_output=True, text=True, timeout=90,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("VTKOK")), None)
    if r.returncode == 0 and line and line.endswith(" 1"):
        return True, line.split()[1]
    lines = [ln for ln in (r.stderr or "").splitlines() if ln.strip()]
    err = [ln for ln in lines if "WARN" in ln or "rror" in ln] or lines[-1:]   # e.g. xvfb-run's mktemp: no space left
    return False, "; ".join(dict.fromkeys(e.split("|")[-1].strip() for e in err[-3:])) or f"exit {r.returncode}"


def probe() -> dict:
    """{ok, window (the VTK render window class), prefix (e.g. xvfb-run), why}. Cached per process."""
    if _PROBE:
        return _PROBE
    if os.environ.get("FEAT_RENDER", "").strip().lower() == "matplotlib":
        _PROBE.update(ok=False, why="FEAT_RENDER=matplotlib")
    elif not available():
        _PROBE.update(ok=False, why="PyVista/VTK not installed (pip install 'fea-translator[render]')")
    else:
        ok, how = _try([])
        if ok:
            _PROBE.update(ok=True, window=how, prefix=[])
        elif shutil.which("xvfb-run"):
            pre = ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24"]
            ok2, how2 = _try(pre)
            if not ok2:                                # xvfb-run -a can lose a race for a display number: once more
                ok2, how2 = _try(pre)
            _PROBE.update(ok=ok2, window=(how2 + " (xvfb-run)") if ok2 else None, prefix=pre if ok2 else None,
                          why=None if ok2 else f"no offscreen OpenGL ({how}); xvfb-run failed too ({how2})")
        else:
            _PROBE.update(ok=False, why=f"no offscreen OpenGL ({how}): install libosmesa6 or libegl1, or xvfb")
    return _PROBE


def status() -> dict:
    """For tool_status: the PyVista version, and how it draws once a picture has been asked for."""
    if not available():
        return {"vtk": None, "offscreen": None}
    from importlib.metadata import version
    try:
        v = version("pyvista")
    except Exception:  # noqa: BLE001
        v = "yes"
    return {"vtk": v, "offscreen": (_PROBE.get("window") or ("no: " + str(_PROBE.get("why")))) if _PROBE else "not checked"}


# ---- in the caller ------------------------------------------------------------------------------
def render(state_dir: Path, specs: list, out_dir: Path, name: str = "study") -> tuple[dict, list]:
    """Draws specs in a child process. Images it could not draw are missing from the answer, with a warning each
    (render.py then draws them with Matplotlib)."""
    state_dir, out_dir = Path(state_dir), Path(out_dir)
    names = [sp["name"] for sp in specs]
    pr = probe()
    if not pr.get("ok"):
        return {}, [f"image {n!r}: quality high needs VTK: {pr.get('why')}" for n in names]
    if not (state_dir / "state.npz").is_file():
        return {}, [f"image {n!r}: VTK reads the saved state; none in {state_dir}" for n in names]
    out_dir.mkdir(parents=True, exist_ok=True)
    sp_file = out_dir / f".vtk_specs_{os.getpid()}.json"
    sp_file.write_text(json.dumps(specs))
    env = dict(os.environ)
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = pr["prefix"] + [sys.executable, "-m", "fea_translator.render_vtk", str(state_dir), str(sp_file), str(out_dir), name]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL,
                           timeout=float(os.environ.get("FEAT_RENDER_TIMEOUT", "900") or 900))
    except subprocess.TimeoutExpired:
        return {}, [f"image {n!r}: VTK took longer than FEAT_RENDER_TIMEOUT" for n in names]
    finally:
        sp_file.unlink(missing_ok=True)
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("{")), None)
    if r.returncode != 0 or line is None:
        tail = " ".join((r.stderr or "").strip().splitlines()[-2:])[-300:]
        return {}, [f"image {n!r}: the VTK renderer failed (exit {r.returncode}): {tail}" for n in names]
    res = json.loads(line)
    return res["images"], res["warnings"]


# ---- in the child: drawing ----------------------------------------------------------------------
def _grid(state: dict):
    import pyvista as pv
    tets = state["tets"].astype(np.int64)
    k = tets.shape[1]
    ctype = pv.CellType.QUADRATIC_TETRA if k == 10 else pv.CellType.TETRA
    cells = np.column_stack([np.full(len(tets), k), tets]).ravel()
    return pv.UnstructuredGrid(cells, np.full(len(tets), ctype, np.uint8), np.asarray(state["nodes"], float))


def _tri_polydata(state: dict, idx: np.ndarray, points: np.ndarray):
    import pyvista as pv
    tri = state["tri"][idx].astype(np.int64)
    used, inv = np.unique(tri.ravel(), return_inverse=True)
    faces = np.column_stack([np.full(len(tri), 3), inv.reshape(-1, 3)]).ravel()
    return pv.PolyData(points[used], faces), used


def _camera(p, view, bounds):
    elev, azim = np.radians(view[0]), np.radians(view[1])
    b = np.asarray(bounds, float).reshape(3, 2)
    c = b.mean(axis=1)
    d = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    p.camera.focal_point = c
    p.camera.position = c + d * float(np.linalg.norm(b[:, 1] - b[:, 0])) * 2
    p.camera.up = (0, 0, 1) if abs(np.sin(elev)) < 0.99 else (0, 1, 0)
    p.reset_camera(bounds=b.ravel())
    p.camera.zoom(1.15)


def _plotter(title):
    import pyvista as pv
    p = pv.Plotter(off_screen=True, window_size=SIZE)
    p.set_background("white")
    p.add_text(title, position="upper_left", font_size=11, color="black")
    try:
        p.enable_anti_aliasing("fxaa")
    except Exception:  # noqa: BLE001
        pass
    return p


def _bar(label):
    return {"title": label, "color": "black", "vertical": True, "position_x": 0.88, "position_y": 0.15,
            "height": 0.7, "width": 0.05, "title_font_size": 14, "label_font_size": 12, "fmt": "%.4g"}


def draw_setup(state, spec, title, path) -> dict:
    from .render import PART_GREY, bc_faces
    import matplotlib.colors as mcolors
    m, _ = _tri_polydata(state, np.arange(len(state["tri"])), np.asarray(state["nodes"], float))
    rgb = np.tile(np.array(mcolors.to_rgb(PART_GREY)), (len(state["tri"]), 1))
    legend, info = [], {}
    for b in bc_faces(state):
        sel = np.isin(state["tri_tag"], b["tags"])
        rgb[sel] = mcolors.to_rgb(b["color"])
        legend.append([f"{b['name']} ({b['kind']})" + ("" if sel.any() else ": no faces"), b["color"]])
        info[b["name"]] = {"color": b["color"], "triangles": int(sel.sum())}
    m.cell_data["rgb"] = (rgb * 255).astype(np.uint8)
    p = _plotter(title)
    p.add_mesh(m, scalars="rgb", rgb=True, smooth_shading=True, split_sharp_edges=True, specular=0.15, ambient=0.45,
               diffuse=0.6)                                   # bright enough to tell the bc colours apart
    for i, (label, color) in enumerate(legend):   # one line per bc in its colour, under the title
        p.add_text(label, position=(18, SIZE[1] - 64 - 26 * i), font_size=12, color=color)
    _camera(p, spec.get("view", [25, -60]), m.bounds)
    p.screenshot(str(path))
    p.close()
    return {"bcs": info, "renderer": "vtk"}


def draw_field(state, spec, title, path) -> dict:
    from .render import AXES, CMAPS, FIELDS, MODE_LABEL, _limits, _nearest_node, deform_scale, displacement_of, \
        field_values, peak_point, surface_triangles
    mode = spec.get("mode")
    vals = field_values(state, spec.get("field"), mode)
    U = displacement_of(state, spec)
    sc = deform_scale(state, spec)
    X = np.asarray(state["nodes"], float)
    sl = spec.get("slice")
    lo, hi, _ = _limits(vals, spec, "full")
    if spec.get("surface", "part") == "part":
        g = _grid(state)
        g.point_data["v"] = vals
        if U is not None:
            g.point_data["u"] = np.nan_to_num(U)
        if sl:
            org = [0.0, 0.0, 0.0]
            org[AXES[sl["axis"]]] = float(sl["at"])
            g = g.clip(normal=sl["axis"], origin=org)          # the low side stays
            if g.n_cells == 0:
                raise ValueError(f"nothing of the part on the low side of {sl['axis']} = {float(sl['at']):g} mm")
        if sc and U is not None:
            g = g.warp_by_vector("u", factor=sc)
        m = g.extract_surface(nonlinear_subdivision=2)
    else:
        idx = surface_triangles(state, spec)
        pts = X + sc * np.nan_to_num(U) if sc and U is not None else X
        m, used = _tri_polydata(state, idx, pts)
        m.point_data["v"] = vals[used]
    p = _plotter(title)
    cmap = "viridis" if mode is not None else CMAPS.get(spec.get("field"), "viridis")
    p.add_mesh(m, scalars="v", cmap=cmap, clim=[lo, hi], smooth_shading=True, split_sharp_edges=True, specular=0.3,
               scalar_bar_args=_bar(MODE_LABEL if mode is not None else FIELDS[spec["field"]]))
    info = {}
    pk = peak_point(state, spec)
    if pk and sl and float(pk["at"][AXES[sl["axis"]]]) > float(sl["at"]):
        pk = None
    if pk:
        q = np.asarray(pk["at"], float)
        if sc and U is not None:
            q = q + sc * np.nan_to_num(U[_nearest_node(X, q)])
        p.add_point_labels(q[None, :], [f"peak off-BC {pk['value']:.4g} MPa"], point_color="black", point_size=14,
                           render_points_as_spheres=True, text_color="black", font_size=14, shape_opacity=0.6,
                           always_visible=True)
        info["peak_marked"] = [round(float(x), 4) for x in pk["at"]]
    _camera(p, spec.get("view", [25, -60]), m.bounds)
    p.screenshot(str(path))
    p.close()
    info.update({"min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)), "clim": [lo, hi], "renderer": "vtk",
                 "triangles": int(m.n_cells)})
    if sc:
        info["deformation_scale"] = float(f"{sc:.6g}")
    return info


def _child(state_dir: str, spec_file: str, out_dir: str, name: str) -> dict:
    from .render import load_state, title_of
    state = load_state(Path(state_dir))
    specs = json.loads(Path(spec_file).read_text())
    images, warns = {}, []
    for sp in specs:
        nm = sp["name"]
        path = Path(out_dir) / f"{nm}.png"
        try:
            what = title_of(state, sp, name)
            info = draw_setup(state, sp, what, path) if sp.get("setup") else draw_field(state, sp, what, path)
            images[nm] = {"path": str(path), "what": what, **info}
        except Exception as exc:  # noqa: BLE001 (a picture must not fail the study)
            warns.append(f"image {nm!r}: VTK: {type(exc).__name__}: {exc}")
    return {"images": images, "warnings": warns}


if __name__ == "__main__":
    print(json.dumps(_child(*sys.argv[1:5]), default=float))
