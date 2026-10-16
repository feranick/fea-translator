"""meshdata.py — the mesh as CalculiX needs it: node ordering, boundary faces, face selection, loads.
Version: 2026.10.16.1

Pure NumPy (no Gmsh, no CalculiX), so every step from a mesh to the deck is testable on its own.

Element conventions (CalculiX = Abaqus): C3D4 corners 1-4 with a positive volume
det(p2 - p1, p3 - p1, p4 - p1) > 0; C3D10 adds the mid-edge nodes 5-10 on the edges 1-2, 2-3, 3-1,
1-4, 2-4, 3-4. Faces: S1 = 1-2-3, S2 = 1-4-2, S3 = 2-4-3, S4 = 3-4-1. Mid-edge nodes are put in
this order from their coordinates (each goes to the edge whose midpoint is nearest), so the
mesher's own numbering never matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))       # mid-node k + 4 sits on EDGES[k]
FACES = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))            # S1..S4, local corner indices
EDGE_MID = {frozenset(e): 4 + k for k, e in enumerate(EDGES)}


class MeshError(Exception):
    """A mesh that CalculiX cannot use, or a selector that matches no face."""


def order_elements(nodes: np.ndarray, corners: np.ndarray, mids: np.ndarray | None = None) -> np.ndarray:
    """Tetrahedra in CalculiX order: corners with positive volume, mid-edge nodes on EDGES.
    corners (M, 4) and mids (M, 6) are node indices in any order."""
    c = np.array(corners, dtype=np.int64)
    p = nodes[c]
    det = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    if np.any(np.abs(det) < 1e-300):
        raise MeshError("degenerate tetrahedra (zero volume)")
    neg = det < 0
    c[neg, 1], c[neg, 2] = c[neg, 2].copy(), c[neg, 1].copy()
    if mids is None:
        return c
    m = np.array(mids, dtype=np.int64)
    p = nodes[c]
    edge_mid = np.stack([(p[:, a] + p[:, b]) / 2 for a, b in EDGES], axis=1)      # (M, 6, 3)
    q = nodes[m]                                                                  # (M, 6, 3)
    d = np.linalg.norm(edge_mid[:, :, None, :] - q[:, None, :, :], axis=3)        # (M, edge, mid)
    pick = np.argmin(d, axis=2)
    if not np.all(np.sort(pick, axis=1) == np.arange(6)):
        raise MeshError("cannot match the mid-edge nodes to the edges (badly curved elements?)")
    return np.concatenate([c, np.take_along_axis(m, pick, axis=1)], axis=1)


def tet_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p = nodes[tets[:, :4]]
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0


def _keys(tri: np.ndarray, n: int) -> np.ndarray:
    s = np.sort(tri, axis=1).astype(np.int64)
    return (s[:, 0] * n + s[:, 1]) * n + s[:, 2]


def _point_triangle_distance(q: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance from point q to each triangle (a, b, c are (T, 3))."""
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", n, n)
    nn = np.where(nn > 0, nn, 1e-300)
    w = q - a
    # barycentric coordinates of the projection
    v = np.einsum("ij,ij->i", np.cross(w, ac), n) / nn
    u = np.einsum("ij,ij->i", np.cross(ab, w), n) / nn
    inside = (v >= 0) & (u >= 0) & (u + v <= 1)
    dplane = np.abs(np.einsum("ij,ij->i", w, n)) / np.sqrt(nn)

    def seg(p0, p1):
        d = p1 - p0
        t = np.clip(np.einsum("ij,ij->i", q - p0, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-300), 0, 1)
        return np.linalg.norm(q - (p0 + d * t[:, None]), axis=1)

    dedge = np.minimum(np.minimum(seg(a, b), seg(b, c)), seg(c, a))
    return np.where(inside, dplane, dedge)


def fit_cylinder(pts: np.ndarray, normals: np.ndarray, rtol: float = 2e-3):
    """(radius, axis direction) if the points lie on a cylinder (the normals are all normal to one
    axis and the points are on a circle around it), else None."""
    if len(pts) < 6:
        return None
    w, v = np.linalg.eigh(normals.T @ normals)
    if w[0] > 0.02 * w[1]:     # facet normals not (nearly) all normal to one direction: not a cylinder
        return None
    axis = v[:, 0]

    def frame(a):
        e1 = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        return e1, np.cross(a, e1)

    e1, e2 = frame(axis)
    u, s = pts @ e1, pts @ e2
    A = np.column_stack([2 * u, 2 * s, np.ones_like(u)])
    sol, *_ = np.linalg.lstsq(A, u * u + s * s, rcond=None)
    r0 = float(np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0)))
    if r0 <= 0:
        return None
    from scipy.optimize import least_squares

    def resid(x):  # x: tilt of the axis (2), centre in the frame (2), radius
        a = axis + x[0] * e1 + x[1] * e2
        a = a / np.linalg.norm(a)
        c = x[2] * e1 + x[3] * e2
        d = pts - c
        radial = d - np.outer(d @ a, a)
        return np.linalg.norm(radial, axis=1) - x[4]

    fit = least_squares(resid, [0.0, 0.0, sol[0], sol[1], r0], x_scale="jac")
    r = abs(float(fit.x[4]))
    a = axis + fit.x[0] * e1 + fit.x[1] * e2
    a /= np.linalg.norm(a)
    return (r, a) if np.abs(fit.fun).max() <= rtol * r else None


@dataclass
class FaceGroup:
    """The boundary triangles of one geometric face (a Gmsh surface entity)."""
    tag: int
    elem: np.ndarray        # (T,) element index of each triangle
    side: np.ndarray        # (T,) face number 1..4 (S1..S4)
    corners: np.ndarray     # (T, 3) node indices, in the element face's order
    mids: np.ndarray | None  # (T, 3) mid-edge nodes of edges (0,1), (1,2), (2,0) of the face
    area: np.ndarray        # (T,)
    normal: np.ndarray      # (T, 3) unit outward normals
    centroid: np.ndarray    # (T, 3)
    kind: str = "unknown"   # "Plane", "Cylinder", ... (from the geometry when known)
    radius: float | None = None

    @property
    def total_area(self) -> float:
        return float(self.area.sum())

    @property
    def center(self) -> np.ndarray:
        return (self.centroid * self.area[:, None]).sum(axis=0) / max(self.total_area, 1e-300)

    @property
    def planar(self) -> bool:
        n0 = self.normal[0]
        return bool(np.all(self.normal @ n0 > 1 - 1e-6))

    def nodes(self) -> np.ndarray:
        return np.unique(self.corners if self.mids is None else np.concatenate([self.corners, self.mids], axis=1))

    def distance(self, q) -> float:
        p = self._pts
        return float(_point_triangle_distance(np.asarray(q, float), p[0], p[1], p[2]).min())


@dataclass
class Mesh:
    nodes: np.ndarray                     # (N, 3) mm
    tets: np.ndarray                      # (M, 4) or (M, 10) node indices, CalculiX order
    surfaces: dict                        # tag -> (T, 3) corner node indices of its boundary triangles
    surface_info: dict = field(default_factory=dict)   # tag -> {"kind": ..., "radius": ...}
    groups: dict = field(default_factory=dict)

    @property
    def order(self) -> int:
        return 2 if self.tets.shape[1] == 10 else 1

    @property
    def size(self) -> float:
        lo, hi = self.nodes.min(axis=0), self.nodes.max(axis=0)
        return float(np.linalg.norm(hi - lo))

    def volume(self) -> float:
        return float(tet_volumes(self.nodes, self.tets).sum())

    def build(self) -> "Mesh":
        """Matches every surface triangle with its element face (S1..S4) and outward normal."""
        n = len(self.nodes)
        corner = self.tets[:, :4]
        allf = np.concatenate([corner[:, list(f)] for f in FACES])               # (4M, 3), face-major
        keys = _keys(allf, n)
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        M = len(self.tets)
        for tag, tri in self.surfaces.items():
            tri = np.asarray(tri, dtype=np.int64)
            tk = _keys(tri, n)
            pos = np.searchsorted(sk, tk)
            ok = (pos < len(sk)) & (sk[np.minimum(pos, len(sk) - 1)] == tk)
            if not np.all(ok):
                raise MeshError(f"surface {tag}: {int((~ok).sum())} triangles are not element faces")
            hit = order[pos]
            side_idx, elem = hit // M, hit % M
            # a face shared by two elements would be internal: check it is unique
            dup = (pos + 1 < len(sk)) & (sk[np.minimum(pos + 1, len(sk) - 1)] == tk)
            if np.any(dup):
                raise MeshError(f"surface {tag}: internal faces (the part has several volumes?)")
            loc = np.array(FACES)[side_idx]                                        # (T, 3) local corners
            cor = np.take_along_axis(self.tets[elem, :4], loc, axis=1)
            mids = None
            if self.order == 2:
                mloc = np.array([[EDGE_MID[frozenset((f[0], f[1]))], EDGE_MID[frozenset((f[1], f[2]))],
                                  EDGE_MID[frozenset((f[2], f[0]))]] for f in FACES])[side_idx]
                mids = np.take_along_axis(self.tets[elem], mloc, axis=1)
            pa, pb, pc = self.nodes[cor[:, 0]], self.nodes[cor[:, 1]], self.nodes[cor[:, 2]]
            nv = np.cross(pb - pa, pc - pa)
            fourth = np.array([[i for i in range(4) if i not in f][0] for f in FACES])[side_idx]
            pd = self.nodes[self.tets[elem, fourth]]
            flip = np.einsum("ij,ij->i", nv, pd - pa) > 0
            nv[flip] *= -1
            ln = np.linalg.norm(nv, axis=1)
            info = self.surface_info.get(tag, {})
            g = FaceGroup(tag, elem, side_idx + 1, cor, mids, ln / 2, nv / ln[:, None], (pa + pb + pc) / 3,
                          info.get("kind", "unknown"), info.get("radius"))
            g._pts = (pa, pb, pc)
            if not g.planar:  # STEP cylinders can come as B-spline surfaces: fit every curved face
                fit = fit_cylinder(self.nodes[np.unique(cor)], g.normal)
                if fit is not None:
                    g.kind, g.radius, g.axis = "Cylinder", fit[0], fit[1]
            self.groups[tag] = g
        return self

    # ---- selection ----
    def select(self, sel) -> list[int]:
        """Surface tags matched by a selector (see study.SELECTOR_DOC); a list is a union."""
        if isinstance(sel, list):
            out = []
            for s in sel:
                out += [t for t in self.select(s) if t not in out]
            return out
        groups = list(self.groups.values())
        if sel.get("all") is True:
            return [g.tag for g in groups]
        if "pts" in sel:
            tol = max(0.05, 0.02 * self.size)
            out = []
            for q in sel["pts"]:
                d = [(g.distance(q), g.tag) for g in groups]
                dm, tag = min(d)
                if dm > tol:
                    raise MeshError(f"no face within {tol:.3g} mm of {list(q)}")
                if tag not in out:
                    out.append(tag)
            return out
        if "cyl_r" in sel:
            r = float(sel["cyl_r"])
            cands = [g for g in groups if g.kind.lower().startswith("cyl") and g.radius and abs(g.radius - r) <= 0.01 * r]
            if not cands:
                radii = sorted({round(g.radius, 3) for g in groups if g.kind.lower().startswith("cyl") and g.radius})
                raise MeshError(f"no cylindrical face of radius {r:g}; radii here: {radii}")
            if "at" in sel:
                return [min(cands, key=lambda g: g.distance(sel["at"])).tag]
            return [g.tag for g in cands]
        d = np.array([float(sel.get(k, 0.0)) for k in ("nx", "ny", "nz")])
        d /= np.linalg.norm(d)
        cands = [g for g in groups if g.planar and g.normal[0] @ d > 0.999]
        if not cands:
            raise MeshError(f"no planar face with outward normal {np.round(d, 4).tolist()}")
        if "at" in sel:
            g = min(cands, key=lambda g: (round(g.distance(sel["at"]), 6), -round(g.total_area, 6)))
        else:
            g = max(cands, key=lambda g: (round(g.total_area, 6), float(g.center @ d)))
        return [g.tag]

    # ---- quantities on selected faces ----
    def nodes_of(self, tags) -> np.ndarray:
        return np.unique(np.concatenate([self.groups[t].nodes() for t in tags]))

    def area_of(self, tags) -> float:
        return float(sum(self.groups[t].total_area for t in tags))

    def faces_of(self, tags) -> list[tuple[int, int]]:
        """(element index, face number) of every triangle on the faces."""
        out = []
        for t in tags:
            g = self.groups[t]
            out += list(zip(g.elem.tolist(), g.side.tolist()))
        return out

    def force_nodes(self, tags, force) -> dict[int, np.ndarray]:
        """Nodal forces equivalent to the total `force` spread as a uniform traction over the faces:
        for 6-node triangles the corners take 0 and the mid-edge nodes A/3 each; for 3-node, A/3 each corner."""
        t = np.asarray(force, float) / self.area_of(tags)
        acc = np.zeros((len(self.nodes), 3))
        for tag in tags:
            g = self.groups[tag]
            share = (g.area / 3)[:, None] * t[None, :]
            tgt = g.mids if g.mids is not None else g.corners
            for j in range(3):
                np.add.at(acc, tgt[:, j], share)
        idx = np.nonzero(np.any(acc != 0, axis=1))[0]
        return {int(i): acc[i] for i in idx}

    def face_nodes_weights(self, tags):
        """Per triangle: its nodes (T, 6|3) and the integrals of the shape functions (T, 6|3) on it
        (flat triangles: quadratic corners 0, mid-edge nodes A/3; linear corners A/3)."""
        for tag in tags:
            g = self.groups[tag]
            if g.mids is not None:
                nodes = np.concatenate([g.corners, g.mids], axis=1)
                w = np.concatenate([np.zeros((len(g.area), 3)), np.repeat((g.area / 3)[:, None], 3, axis=1)], axis=1)
            else:
                nodes, w = g.corners, np.repeat((g.area / 3)[:, None], 3, axis=1)
            yield g, nodes, w

    def face_mass(self, g) -> np.ndarray:
        """Integrals of N_a N_b on each triangle (T, n, n), flat triangles."""
        if g.mids is not None:
            m = np.array([[6, -1, -1, 0, -4, 0], [-1, 6, -1, 0, 0, -4], [-1, -1, 6, -4, 0, 0],
                          [0, 0, -4, 32, 16, 16], [-4, 0, 0, 16, 32, 16], [0, -4, 0, 16, 16, 32]]) / 180.0
        else:
            m = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
        return g.area[:, None, None] * m[None]

    def pressure_nodes(self, tags, p: float) -> np.ndarray:
        """Consistent nodal forces (N, 3) of a uniform pressure p (positive into the faces)."""
        acc = np.zeros((len(self.nodes), 3))
        for g, nodes, w in self.face_nodes_weights(tags):
            for j in range(nodes.shape[1]):
                np.add.at(acc, nodes[:, j], -p * w[:, j:j + 1] * g.normal)
        return acc

    def flux_nodes(self, tags, q: float) -> np.ndarray:
        acc = np.zeros(len(self.nodes))
        for _, nodes, w in self.face_nodes_weights(tags):
            np.add.at(acc, nodes.ravel(), (q * w).ravel())
        return acc

    def film_nodes(self, tags, h: float, t_inf: float, T: np.ndarray) -> np.ndarray:
        """Nodal heat (N,) into the part through a film: integral of N h (T_inf - T)."""
        acc = np.zeros(len(self.nodes))
        for g, nodes, w in self.face_nodes_weights(tags):
            m = self.face_mass(g)
            val = h * (t_inf * w - np.einsum("tab,tb->ta", m, T[nodes]))
            np.add.at(acc, nodes.ravel(), val.ravel())
        return acc

    def body_nodes(self, accel_density: np.ndarray) -> np.ndarray:
        """Consistent nodal forces (N, 3) of a uniform body force per volume (straight-sided
        elements: 10-node corners -V/20, mid-edge nodes V/5; 4-node V/4)."""
        v = tet_volumes(self.nodes, self.tets)
        acc = np.zeros((len(self.nodes), 3))
        if self.order == 2:
            wts = np.array([-1 / 20] * 4 + [1 / 5] * 6)
        else:
            wts = np.full(4, 1 / 4)
        for j, wj in enumerate(wts):
            np.add.at(acc, self.tets[:, j], (wj * v)[:, None] * accel_density[None, :])
        return acc

    def pressure_resultant(self, tags, p: float) -> np.ndarray:
        r = np.zeros(3)
        for tag in tags:
            g = self.groups[tag]
            r -= p * (g.area[:, None] * g.normal).sum(axis=0)
        return r

    def axis_normal(self, tags) -> int:
        """Axis (0, 1, 2) every selected face is normal to (for sliders)."""
        axes = set()
        for tag in tags:
            g = self.groups[tag]
            if not g.planar:
                raise MeshError("a slider needs planar faces")
            k = int(np.argmax(np.abs(g.normal[0])))
            if abs(abs(g.normal[0][k]) - 1) > 1e-6:
                raise MeshError("a slider needs faces normal to X, Y or Z")
            axes.add(k)
        if len(axes) != 1:
            raise MeshError("the faces of one slider must all be normal to the same axis; use one slider per axis")
        return axes.pop()


def box_mesh(lx: float, ly: float, lz: float, nx: int, ny: int, nz: int, order: int = 2,
             origin=(0.0, 0.0, 0.0)) -> Mesh:
    """A structured tetrahedral mesh of a box (6 tetrahedra per cell), with its six faces as
    surfaces 1..6 (-X, +X, -Y, +Y, -Z, +Z). For tests and benchmarks without Gmsh."""
    o = np.asarray(origin, float)
    xs, ys, zs = (np.linspace(0, L, n + 1) for L, n in ((lx, nx), (ly, ny), (lz, nz)))
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3) + o
    idx = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = [idx(i + a, j + b, k + c) for a, b, c in
                     ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
                # Kuhn split: one tetrahedron per path 000 -> 111 along the cell's edges (conforming)
                for t in ((0, 1, 2, 6), (0, 1, 5, 6), (0, 3, 2, 6), (0, 3, 7, 6), (0, 4, 5, 6), (0, 4, 7, 6)):
                    tets.append([v[q] for q in t])
    tets = np.array(tets)
    nodes = grid
    mids = None
    if order == 2:
        edge_ids, extra = {}, []
        mids = np.zeros((len(tets), 6), dtype=np.int64)
        for e, t in enumerate(tets):
            for k, (a, b) in enumerate(EDGES):
                key = (min(t[a], t[b]), max(t[a], t[b]))
                if key not in edge_ids:
                    edge_ids[key] = len(nodes) + len(extra)
                    extra.append((nodes[key[0]] + nodes[key[1]]) / 2)
                mids[e, k] = edge_ids[key]
        nodes = np.vstack([nodes, np.array(extra)])
        mids = mids[:, ::-1]  # scrambled on purpose: order_elements must sort them back
    tets = order_elements(nodes, tets, mids)
    # boundary triangles per box face
    corner = tets[:, :4]
    allf = np.concatenate([corner[:, list(f)] for f in FACES])
    s = np.sort(allf, axis=1)
    uniq, inv, cnt = np.unique(s, axis=0, return_inverse=True, return_counts=True)
    bnd = allf[cnt[inv.ravel()] == 1]
    c = nodes[bnd].mean(axis=1) - o
    eps = 1e-9 * max(lx, ly, lz)
    surf = {1: bnd[np.abs(c[:, 0]) < eps], 2: bnd[np.abs(c[:, 0] - lx) < eps],
            3: bnd[np.abs(c[:, 1]) < eps], 4: bnd[np.abs(c[:, 1] - ly) < eps],
            5: bnd[np.abs(c[:, 2]) < eps], 6: bnd[np.abs(c[:, 2] - lz) < eps]}
    info = {t: {"kind": "Plane"} for t in surf}
    return Mesh(nodes, tets, surf, info).build()
