"""mesher.py — the part as a tetrahedral mesh: Gmsh from a STEP file (or a build123d-translator goal).
Version: 2026.10.16.6

    mesh_part(study, workdir) -> (Mesh, info)

Gmsh imports the STEP with OpenCascade, meshes the one volume with 10-node (or 4-node) tetrahedra
and returns the boundary triangles of every geometric face; meshdata orders the nodes for CalculiX
and finds the faces' normals, areas and cylinders. part {"box": [lx, ly, lz]} gives a structured box
mesh without Gmsh (tests and benchmarks).
"""
from __future__ import annotations

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
    base = Path(workdir) / "part"
    return Path(T.export(solid, base, ("step",))["step"])


def _gmsh_init(gmsh):
    try:
        gmsh.initialize(interruptible=False)   # Gmsh >= 4.11: no signal handler (works off the main thread)
    except TypeError:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)


def mesh_step(step: Path, size: float | None = None, order: int = 2, refine=(), save: Path | None = None,
              factor: float = 1.0) -> tuple[Mesh, dict]:
    """Mesh of a one-solid STEP: element size `size` (default 1/20 of the bounding-box diagonal) x factor."""
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
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(h, *(r["size"] for r in refine)) / 4 if refine else h / 10)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)   # elements per full circle on curved faces
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
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes_all = np.asarray(coords, float).reshape(-1, 3)
        tags = np.asarray(tags, np.int64)
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
    nodes, straightened = straighten_inverted(nodes, tets)
    mesh = Mesh(nodes, tets, surfaces, info).build()
    return mesh, {"size": h, "diag": diag, "nodes": len(nodes), "elements": len(tets), "order": order,
                  "straightened_elements": straightened}


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
    step = part_step(study, workdir)
    mesh, info = mesh_step(step, m.get("size"), order, m.get("refine", []), Path(workdir) / "mesh.msh", size_factor)
    info["step"] = str(step)
    return mesh, info
