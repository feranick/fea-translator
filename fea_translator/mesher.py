"""mesher.py — the part as a tetrahedral mesh: Gmsh from a STEP file (or a build123d-translator goal).
Version: 2026.11.05.4

    mesh_part(study, workdir) -> (Mesh, info)

Gmsh imports the STEP with OpenCascade, meshes the one volume with 10-node (or 4-node) tetrahedra
and returns the boundary triangles of every geometric face; meshdata orders the nodes for CalculiX
and finds the faces' normals, areas and cylinders. part {"box": [lx, ly, lz]} gives a structured box
mesh without Gmsh (tests and benchmarks).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .meshdata import GMSH_TET10, Mesh, MeshError, box_mesh, order_known, straighten_inverted

TET = {4: (1, 4), 11: (2, 10)}     # Gmsh element type -> (order, nodes)
TRI = {2: 3, 9: 6}


def part_step(study: dict, workdir: Path) -> Path:
    """A STEP file of the part: the given one, or the build123d-translator goal built into workdir."""
    part = study["part"]
    if "step" in part:
        p = Path(part["step"]).expanduser()
        if not p.is_file():
            raise MeshError(f"no STEP file {p}")
        return p
    try:
        from build123d_translator import translate as T
    except ImportError as exc:
        raise MeshError("part is a goal but build123d-translator is not installed (pip install -e ../build123d-translator)") from exc
    solid, _ = T.build(part["goal"])
    Path(workdir).mkdir(parents=True, exist_ok=True)   # OCCT does not make the folder: 'Failed to write STEP file'
    base = Path(workdir) / "part"
    return Path(T.export(solid, base, ("step",))["step"])


_FAILED_SURFACE = re.compile(r"(?:mesh periodic surface|surface)\s+(\d+)", re.I)
SURFACE_ALGORITHMS = (1, 5, 6)               # MeshAdapt, Delaunay, Frontal-Delaunay: tried in turn on a failing surface
ALGORITHM_NAMES = ("MeshAdapt", "Delaunay", "Frontal-Delaunay")
VOLUME_ALGORITHMS = ((10, "HXT"), (4, "Frontal"), (1, "Delaunay"))   # then the whole volume with another 3D algorithm


def _has_mesh(gmsh, dim: int) -> bool:
    """Whether generate(dim) left elements of that dimension (Gmsh can return without an error and no mesh)."""
    try:
        types, _, _ = gmsh.model.mesh.getElements(dim)
        return len(types) > 0
    except Exception:  # noqa: BLE001
        return False


def generate(gmsh, dim: int, notes: list | None = None) -> None:
    """gmsh.model.mesh.generate(dim), with fallbacks, each noted: a surface Gmsh names ('Invalid boundary mesh
    (overlapping facets) on surface 80', 'Impossible to mesh periodic surface 126') is meshed again with MeshAdapt,
    then Delaunay, then Frontal-Delaunay (as cfd-translator 2026.11.01.2); when that does not help, or Gmsh names no
    surface, or it returns without a mesh, the volume is meshed with another 3D algorithm (HXT, Frontal, Delaunay).
    A MeshError says what was tried when nothing works."""
    tried: dict[int, int] = {}
    vol_k = 0
    last = None
    first_3d = None
    while True:
        try:
            gmsh.model.mesh.generate(dim)
            if _has_mesh(gmsh, dim):
                return
            raise RuntimeError(f"Gmsh returned without {('', 'lines', 'triangles', 'tetrahedra')[dim]}")
        except Exception as exc:  # noqa: BLE001 (Gmsh raises plain Exception)
            last = exc
            m = _FAILED_SURFACE.search(str(exc))
            sf = int(m.group(1)) if m else None
            if sf is not None and tried.get(sf, 0) < len(SURFACE_ALGORITHMS):
                k = tried.get(sf, 0)
                tried[sf] = k + 1
                gmsh.model.mesh.clear()
                gmsh.model.mesh.setAlgorithm(2, sf, SURFACE_ALGORITHMS[k])
                if notes is not None:
                    notes.append(f"Gmsh could not mesh surface {sf} (near {_where(gmsh, sf)} mm) ({str(exc)[:120]}): "
                                 f"meshed it again with {ALGORITHM_NAMES[k]}")
                continue
            if dim == 3 and vol_k < len(VOLUME_ALGORITHMS):
                if first_3d is None:
                    try:
                        first_3d = int(gmsh.option.getNumber("Mesh.Algorithm3D"))
                    except Exception:  # noqa: BLE001
                        first_3d = 1
                alg, aname = VOLUME_ALGORITHMS[vol_k]
                vol_k += 1
                if alg == first_3d:
                    continue                      # the one that just failed
                gmsh.model.mesh.clear()
                gmsh.option.setNumber("Mesh.Algorithm3D", alg)
                if notes is not None:
                    notes.append(f"Gmsh could not mesh the volume ({str(exc)[:120]}): meshed it again with the "
                                 f"{aname} 3D algorithm")
                continue
            where = (f" surface {sf} (near {_where(gmsh, sf)} mm)" if sf is not None else "")
            done = ([f"2D algorithms {', '.join(ALGORITHM_NAMES)} on surface{'s' if len(tried) > 1 else ''} "
                     f"{', '.join(map(str, tried))}"] if tried else []) + (
                [f"3D algorithms {', '.join(n for _, n in VOLUME_ALGORITHMS)}"] if dim == 3 else [])
            raise MeshError(f"Gmsh could not mesh{where or ' the part'} ({last}); tried {' and '.join(done) or 'nothing else'}. "
                            "It is often a thin sliver face or a seam meeting a tiny edge: change mesh.size a little, "
                            "refine there (mesh.refine), or simplify that face in the CAD") from exc


def _where(gmsh, sf: int) -> list:
    try:
        bb = gmsh.model.getBoundingBox(2, sf)
        return [round(0.5 * (bb[i] + bb[i + 3]), 1) for i in range(3)]
    except Exception:  # noqa: BLE001
        return []


def _gmsh_init(gmsh):
    try:
        gmsh.initialize(interruptible=False)   # Gmsh >= 4.11: no signal handler (works off the main thread)
    except TypeError:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)


CURVATURE = 12            # Gmsh's default here: elements per full circle on curved faces (0: sizes ignore curvature)


def mesh_step(step: Path, size: float | None = None, order: int = 2, refine=(), save: Path | None = None,
              factor: float = 1.0, min_size: float | None = None, curvature: int = CURVATURE) -> tuple[Mesh, dict]:
    """Mesh of a one-solid STEP: element size `size` (default 1/20 of the bounding-box diagonal) x factor.
    min_size: no element smaller (mm, x factor; default size/10, or a quarter of the finest refine); curvature:
    elements per full circle on curved faces (0 turns it off: small fillets and holes no longer shrink the elements
    around them). Edges shorter than min_size in the CAD still get one element each."""
    import gmsh

    _gmsh_init(gmsh)
    try:
        gmsh.model.add("part")
        gmsh.model.occ.importShapes(str(step))
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
        if len(vols) != 1:
            raise MeshError(f"the STEP has {len(vols)} solids; a study is one solid part")
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
        h = (float(size) if size else diag / 20) * factor
        refine = [dict(r, size=float(r['size']) * factor) for r in refine]
        hmin = (float(min_size) * factor if min_size else
                (min(h, *(r["size"] for r in refine)) / 4 if refine else h / 10))
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(hmin, h))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(curvature))   # elements per full circle (0: off)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)        # mid-edge nodes on the geometry
        if refine:
            fields = []
            for r in refine:
                f = gmsh.model.mesh.field.add("Ball")
                (x, y, z), rad = r["at"], float(r["r"])
                for k, v in (("XCenter", x), ("YCenter", y), ("ZCenter", z), ("Radius", rad),
                             ("Thickness", rad), ("VIn", float(r["size"])), ("VOut", h)):
                    gmsh.model.mesh.field.setNumber(f, k, float(v))
                fields.append(f)
            fmin = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
        notes: list = []
        generate(gmsh, 3, notes)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes_all = np.asarray(coords, float).reshape(-1, 3)
        tags = np.asarray(tags, np.int64)
        if not len(tags):
            raise MeshError("Gmsh produced no nodes (an empty mesh)")
        lut = -np.ones(int(tags.max()) + 1, np.int64)
        lut[tags] = np.arange(len(tags))
        etypes, _, enodes = gmsh.model.mesh.getElements(3)
        tets = None
        for et, en in zip(etypes, enodes):
            if int(et) not in TET:
                raise MeshError(f"unexpected volume element type {int(et)} (only tetrahedra)")
            n = TET[int(et)][1]
            tets = lut[np.asarray(en, np.int64).reshape(-1, n)]
        if tets is None or not len(tets):
            raise MeshError("Gmsh produced no tetrahedra")
        surfaces, info = {}, {}
        for _, s in gmsh.model.getEntities(2):
            st, _, sn = gmsh.model.mesh.getElements(2, s)
            for et, en in zip(st, sn):
                if int(et) in TRI:
                    tri = np.asarray(en, np.int64).reshape(-1, TRI[int(et)])[:, :3]
                    surfaces[int(s)] = lut[tri]
            info[int(s)] = {"kind": gmsh.model.getType(2, s)}
        if save:
            gmsh.write(str(save))
    finally:
        gmsh.finalize()
    # keep only nodes used by the tetrahedra, renumbered
    used = np.unique(tets)
    remap = -np.ones(len(nodes_all), np.int64)
    remap[used] = np.arange(len(used))
    nodes = nodes_all[used]
    tets = remap[tets]
    surfaces = {k: remap[v] for k, v in surfaces.items()}
    # Gmsh's node order is documented: use it (matching mid-edge nodes by distance fails on strongly
    # curved elements); then straighten the few elements Gmsh's curving left inverted
    if tets.shape[1] == 10:
        tets = tets[:, GMSH_TET10]
    tets = order_known(nodes, tets)
    mask: list = []
    nodes, straightened = straighten_inverted(nodes, tets, mask_out=mask)
    mesh = Mesh(nodes, tets, surfaces, info).build()
    mesh.straightened = np.nonzero(mask[0])[0] if mask else np.zeros(0, int)   # their stresses are approximate
    info = {"size": h, "min_size": round(min(hmin, h), 6), "curvature": int(curvature), "diag": diag,
            "nodes": len(nodes), "elements": len(tets), "order": order, "straightened_elements": straightened}
    if notes:
        info["mesh_notes"] = notes
    return mesh, info


def mesh_part(study: dict, workdir: Path, size_factor: float = 1.0) -> tuple[Mesh, dict]:
    m = study.get("mesh", {})
    order = int(m.get("order", 2))
    part = study["part"]
    if "box" in part:
        lx, ly, lz = (float(v) for v in part["box"])
        h = float(m.get("size") or np.linalg.norm([lx, ly, lz]) / 20) * size_factor
        div = part.get("divisions") or [max(1, round(L / h)) for L in (lx, ly, lz)]
        if size_factor != 1.0 and part.get("divisions"):
            div = [max(1, round(d / size_factor)) for d in div]
        org = part.get("origin", [-lx / 2, -ly / 2, 0.0])
        mesh = box_mesh(lx, ly, lz, *div, order=order, origin=org)
        return mesh, {"size": h, "nodes": len(mesh.nodes), "elements": len(mesh.tets), "order": order, "divisions": div}
    Path(workdir).mkdir(parents=True, exist_ok=True)   # mesh.msh (and a goal's part.step) go there
    step = part_step(study, workdir)
    mesh, info = mesh_step(step, m.get("size"), order, m.get("refine", []), Path(workdir) / "mesh.msh", size_factor,
                           m.get("min_size"), int(m.get("curvature", CURVATURE)))
    info["step"] = str(step)
    return mesh, info
