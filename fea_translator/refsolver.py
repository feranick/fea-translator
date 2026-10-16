"""refsolver.py — a small reference solver for the decks this package writes (not CalculiX).
Version: 2026.10.16.1

    run(inp_path) -> writes <job>.dat in CalculiX's .dat layout (the parts results.py reads)

Reads the subset of the CalculiX/Abaqus input that deck.py writes (*NODE, *ELEMENT C3D4/C3D10,
*NSET, *MATERIAL, *BOUNDARY, *CLOAD, *DLOAD P/GRAV, *DFLUX S, *FILM F, *STATIC, *FREQUENCY,
*HEAT TRANSFER) and solves linear statics, natural frequencies and steady conduction with its own
element code (isoparametric tetrahedra, collapsed Gauss-Legendre quadrature, SciPy sparse solvers).
It exists to check the decks and the result handling independently of CalculiX, and as a fallback
for small studies when CalculiX is not installed. Buckling needs CalculiX.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
FACES = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))


class RefError(Exception):
    pass


# ---------------------------------------------------------------- quadrature and shape functions

def tet_rule(n: int = 4):
    """Points (barycentric L1..L4) and weights (summing to 1/6) on the unit tetrahedron (Duffy map)."""
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = (x + 1) / 2, w / 2
    pts, wts = [], []
    for a, wa in zip(x, w):
        for b, wb in zip(x, w):
            for c, wc in zip(x, w):
                xi, eta, zeta = a, b * (1 - a), c * (1 - a) * (1 - b)
                pts.append((xi, eta, zeta))
                wts.append(wa * wb * wc * (1 - a) ** 2 * (1 - b))
    p = np.array(pts)
    L = np.column_stack([1 - p.sum(axis=1), p])
    return L, np.array(wts)


def tri_rule(n: int = 4):
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = (x + 1) / 2, w / 2
    pts, wts = [], []
    for a, wa in zip(x, w):
        for b, wb in zip(x, w):
            pts.append((a, b * (1 - a)))
            wts.append(wa * wb * (1 - a))
    p = np.array(pts)
    return np.column_stack([1 - p.sum(axis=1), p]), np.array(wts)


def tet_shape(L: np.ndarray, quad: bool):
    """N (q, n) and dN/dL (q, n, 4) at barycentric points."""
    q = len(L)
    if not quad:
        return L.copy(), np.broadcast_to(np.eye(4), (q, 4, 4)).copy()
    N = np.zeros((q, 10))
    dN = np.zeros((q, 10, 4))
    for i in range(4):
        N[:, i] = L[:, i] * (2 * L[:, i] - 1)
        dN[:, i, i] = 4 * L[:, i] - 1
    for k, (a, b) in enumerate(EDGES):
        N[:, 4 + k] = 4 * L[:, a] * L[:, b]
        dN[:, 4 + k, a] = 4 * L[:, b]
        dN[:, 4 + k, b] = 4 * L[:, a]
    return N, dN


def tri_shape(L: np.ndarray, quad: bool):
    q = len(L)
    if not quad:
        return L.copy(), np.broadcast_to(np.eye(3), (q, 3, 3)).copy()
    N = np.zeros((q, 6))
    dN = np.zeros((q, 6, 3))
    for i in range(3):
        N[:, i] = L[:, i] * (2 * L[:, i] - 1)
        dN[:, i, i] = 4 * L[:, i] - 1
    for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        N[:, 3 + k] = 4 * L[:, a] * L[:, b]
        dN[:, 3 + k, a] = 4 * L[:, b]
        dN[:, 3 + k, b] = 4 * L[:, a]
    return N, dN


def _grad(X: np.ndarray, dNdL: np.ndarray):
    """Physical gradients (E, q, n, 3) and |J| (E, q) for elements X (E, n, 3)."""
    # natural coordinates xi, eta, zeta = L2, L3, L4; dN/dxi_j = dN/dL_{j+1} - dN/dL_1
    dNdxi = dNdL[:, :, 1:] - dNdL[:, :, :1]                     # (q, n, 3)
    J = np.einsum("qnj,enk->eqjk", dNdxi, X)                    # (E, q, 3, 3): d x_k / d xi_j
    det = np.linalg.det(J)
    if np.any(det <= 0):
        raise RefError("an element has a non-positive Jacobian")
    # dN/dxi_j = sum_k J[j, k] dN/dx_k, so dN/dx_k = sum_j inv(J)[k, j] dN/dxi_j
    G = np.einsum("eqkj,qnj->eqnk", np.linalg.inv(J), dNdxi)
    return G, det


# ---------------------------------------------------------------- reading the deck

def read_deck(text: str) -> dict:
    d = {"nodes": {}, "elems": [], "etype": None, "nsets": {}, "mat": {}, "bnd": [], "cload": [], "dload": [],
         "dflux": [], "film": [], "step": None, "modes": 6, "init_T": 0.0}
    kw, opts, nset = None, {}, None
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("**")]
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        i += 1
        if ln.startswith("*"):
            parts = [p.strip() for p in ln[1:].split(",")]
            kw = parts[0].upper()
            opts = {p.split("=")[0].strip().upper(): (p.split("=")[1].strip() if "=" in p else True) for p in parts[1:]}
            if kw == "ELEMENT":
                d["etype"] = opts.get("TYPE", "").upper()
            elif kw == "NSET":
                nset = opts["NSET"].upper()
                d["nsets"][nset] = []
            elif kw in ("STATIC", "FREQUENCY", "HEAT TRANSFER", "BUCKLE"):
                d["step"] = kw
                if kw in ("FREQUENCY", "BUCKLE") and i < len(lines) and not lines[i].startswith("*"):
                    d["modes"] = int(float(lines[i].split(",")[0]))
                    i += 1
                elif kw == "HEAT TRANSFER" and i < len(lines) and not lines[i].startswith("*"):
                    i += 1
            elif kw in ("ELASTIC", "DENSITY", "CONDUCTIVITY", "EXPANSION") and i < len(lines):
                d["mat"][kw] = [float(x) for x in lines[i].split(",") if x.strip()]
                i += 1
            continue
        f = [x.strip() for x in ln.split(",") if x.strip()]
        if kw == "NODE":
            d["nodes"][int(f[0])] = [float(x) for x in f[1:4]]
        elif kw == "ELEMENT":
            d["elems"].append([int(x) for x in f])
        elif kw == "NSET":
            d["nsets"][nset] += [int(x) for x in f]
        elif kw == "BOUNDARY":
            d["bnd"].append((f[0].upper(), int(f[1]), int(f[2]), float(f[3]) if len(f) > 3 else 0.0))
        elif kw == "CLOAD":
            d["cload"].append((int(f[0]), int(f[1]), float(f[2])))
        elif kw == "DLOAD":
            d["dload"].append(f)
        elif kw == "DFLUX":
            d["dflux"].append((int(f[0]), int(f[1][1:]), float(f[2])))
        elif kw == "FILM":
            d["film"].append((int(f[0]), int(f[1][1:]), float(f[2]), float(f[3])))
        elif kw == "INITIAL CONDITIONS":
            d["init_T"] = float(f[1])
    return d


# ---------------------------------------------------------------- the solver

class Model:
    def __init__(self, d: dict):
        ids = sorted(d["nodes"])
        self.ids = np.array(ids)
        self.pos = {nid: k for k, nid in enumerate(ids)}
        self.X = np.array([d["nodes"][n] for n in ids])
        el = np.array(d["elems"])
        self.eids = el[:, 0]
        self.conn = np.vectorize(self.pos.get)(el[:, 1:])
        self.quad = self.conn.shape[1] == 10
        self.d = d
        self.L, self.w = tet_rule(4)
        self.N, self.dN = tet_shape(self.L, self.quad)
        self.G, self.det = _grad(self.X[self.conn], self.dN)
        self.epos = {e: k for k, e in enumerate(self.eids)}

    def set_nodes(self, name: str) -> np.ndarray:
        if name.upper() == "NALL":
            return np.arange(len(self.ids))
        return np.array([self.pos[n] for n in self.d["nsets"][name.upper()]])

    def face(self, elem: int, side: int):
        """Nodes (local face order: corners then mids) and the outward direction test point."""
        k = self.epos[elem]
        f = FACES[side - 1]
        c = self.conn[k]
        nodes = [c[f[0]], c[f[1]], c[f[2]]]
        if self.quad:
            mid = {frozenset(e): 4 + j for j, e in enumerate(EDGES)}
            nodes += [c[mid[frozenset((f[0], f[1]))]], c[mid[frozenset((f[1], f[2]))]], c[mid[frozenset((f[2], f[0]))]]]
        fourth = [q for q in range(4) if q not in f][0]
        return np.array(nodes), self.X[c[fourth]]

    def face_integrals(self, elem: int, side: int):
        """(nodes, integral of N (n,), integral of N N^T (n, n), integral of N n_out dA (n, 3))."""
        nodes, inner = self.face(elem, side)
        L, w = tri_rule(4)
        N, dN = tri_shape(L, self.quad)
        P = self.X[nodes]
        ds = np.einsum("qn,nk->qk", dN[:, :, 1] - dN[:, :, 0], P)
        dt = np.einsum("qn,nk->qk", dN[:, :, 2] - dN[:, :, 0], P)
        nv = np.cross(ds, dt)
        x = N @ P
        if np.mean(np.einsum("qk,qk->q", nv, inner - x)) > 0:
            nv = -nv
        dA = np.linalg.norm(nv, axis=1)
        return nodes, (N * (w * dA)[:, None]).sum(0), np.einsum("q,qa,qb->ab", w * dA, N, N), \
            np.einsum("q,qa,qk->ak", w, N, nv)

    # -- assembly --
    def _assemble(self, ke: np.ndarray, dof: int):
        E, n = self.conn.shape
        idx = (self.conn[:, :, None] * dof + np.arange(dof)).reshape(E, n * dof)
        rows = np.repeat(idx, n * dof, axis=1).ravel()
        cols = np.tile(idx, (1, n * dof)).ravel()
        N = len(self.X) * dof
        return sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(N, N))

    def stiffness(self, Emod: float, nu: float):
        lam, mu = Emod * nu / ((1 + nu) * (1 - 2 * nu)), Emod / (2 * (1 + nu))
        D = np.zeros((6, 6))
        D[:3, :3] = lam
        D[range(3), range(3)] += 2 * mu
        D[range(3, 6), range(3, 6)] = mu
        self.D = D
        B = self._B(self.G)
        DB = np.einsum("ij,eqjb->eqib", D, B, optimize=True)
        ke = np.einsum("eq,eqia,eqib->eab", self.det * self.w, B, DB, optimize=True)
        return self._assemble(ke, 3)

    @staticmethod
    def _B(G):
        E, q, n, _ = G.shape
        B = np.zeros((E, q, 6, 3 * n))
        for a in range(n):
            gx, gy, gz = G[:, :, a, 0], G[:, :, a, 1], G[:, :, a, 2]
            c = 3 * a
            B[:, :, 0, c], B[:, :, 1, c + 1], B[:, :, 2, c + 2] = gx, gy, gz
            B[:, :, 3, c], B[:, :, 3, c + 1] = gy, gx           # xy
            B[:, :, 4, c], B[:, :, 4, c + 2] = gz, gx           # xz
            B[:, :, 5, c + 1], B[:, :, 5, c + 2] = gz, gy       # yz
        return B

    def mass(self, rho: float):
        me = np.einsum("eq,qa,qb->eab", self.det * self.w, self.N, self.N, optimize=True) * rho
        E, n = self.conn.shape
        m3 = np.zeros((E, 3 * n, 3 * n))
        for k in range(3):
            m3[:, k::3, k::3] = me
        return self._assemble(m3, 3)

    def conduction(self, k: float):
        ke = np.einsum("eq,eqak,eqbk->eab", self.det * self.w, self.G, self.G, optimize=True) * k
        return self._assemble(ke, 1)

    def body(self, g: np.ndarray, rho: float):
        fe = np.einsum("eq,qa->ea", self.det * self.w, self.N) * rho
        f = np.zeros(len(self.X) * 3)
        for k in range(3):
            np.add.at(f, self.conn * 3 + k, fe * g[k])
        return f


def _solve(K, f, fixed: dict, ndof: int):
    u = np.zeros(ndof)
    fx = np.array(sorted(fixed), dtype=np.int64)
    u[fx] = [fixed[i] for i in fx]
    free = np.setdiff1d(np.arange(ndof), fx)
    rhs = f[free] - K[free][:, fx] @ u[fx]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return u


def _fmt(v: float) -> str:
    return f"{v: .6E}"


def run(inp: Path) -> Path:
    inp = Path(inp)
    d = read_deck(inp.read_text())
    if d["etype"] not in ("C3D10", "C3D4"):
        raise RefError(f"element type {d['etype']} is not supported")
    m = Model(d)
    out = [""]
    n = len(m.X)
    if d["step"] in ("STATIC", "FREQUENCY"):
        Emod, nu = d["mat"]["ELASTIC"][:2]
        K = m.stiffness(Emod, nu)
        fixed = {}
        for name, a, b, val in d["bnd"]:
            for nd in m.set_nodes(name):
                for dof in range(a, b + 1):
                    fixed[3 * nd + dof - 1] = val
        if d["step"] == "FREQUENCY":
            M = m.mass(d["mat"]["DENSITY"][0])
            free = np.setdiff1d(np.arange(3 * n), np.array(sorted(fixed), dtype=np.int64))
            k = min(d["modes"], len(free) - 2)
            vals = spla.eigsh(K[free][:, free].tocsc(), k=k, M=M[free][:, free].tocsc(), sigma=0, which="LM",
                              return_eigenvectors=False)
            vals = np.sort(vals)
            out += ["     E I G E N V A L U E   O U T P U T", "",
                    " MODE NO    EIGENVALUE                       FREQUENCY",
                    "                                     REAL PART            IMAGINARY PART",
                    "                          (RAD/TIME)      (CYCLES/TIME     (RAD/TIME)", ""]
            for j, lam in enumerate(vals):
                om = np.sqrt(max(lam, 0))
                out.append(f"{j + 1:7d}  {_fmt(lam)}  {_fmt(om)}  {_fmt(om / 2 / np.pi)}  {_fmt(0.0)}")
        else:
            f = np.zeros(3 * n)
            for nid, dof, val in d["cload"]:
                f[3 * m.pos[nid] + dof - 1] += val
            for rec in d["dload"]:
                if rec[1].upper() == "GRAV":
                    g = float(rec[2]) * np.array([float(x) for x in rec[3:6]])
                    f += m.body(g, d["mat"]["DENSITY"][0])
                else:
                    nodes, _, _, nint = m.face_integrals(int(rec[0]), int(rec[1][1:]))
                    p = float(rec[2])
                    for a, nd in enumerate(nodes):
                        f[3 * nd:3 * nd + 3] -= p * nint[a]
            u = _solve(K, f, fixed, 3 * n)
            r = K @ u  # as CalculiX's RF: the external force at the node (reaction plus applied loads)
            U = u.reshape(-1, 3)
            out += [" displacements (vx,vy,vz) for set NALL and time  0.1000000E+01", ""]
            out += [f"{m.ids[i]:10d} {_fmt(U[i, 0])} {_fmt(U[i, 1])} {_fmt(U[i, 2])}" for i in range(n)]
            # stresses at the 4 Gauss points of C3D10 (1 for C3D4), as CalculiX prints them
            if m.quad:
                a_, b_ = 0.5854101966249685, 0.1381966011250105
                Lg = np.array([[a_, b_, b_, b_], [b_, a_, b_, b_], [b_, b_, a_, b_], [b_, b_, b_, a_]])
            else:
                Lg = np.full((1, 4), 0.25)
            _, dNg = tet_shape(Lg, m.quad)
            Gg, _ = _grad(m.X[m.conn], dNg)
            Bg = Model._B(Gg)
            ue = U[m.conn].reshape(len(m.conn), -1)
            eps = np.einsum("eqij,ej->eqi", Bg, ue)
            sig = np.einsum("ij,eqj->eqi", m.D, eps)
            out += ["", " stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EALL and time  0.1000000E+01", ""]
            for e in range(len(m.conn)):
                for q in range(len(Lg)):
                    s = sig[e, q]
                    out.append(f"{m.eids[e]:10d}{q + 1:4d} " + " ".join(_fmt(s[c]) for c in (0, 1, 2, 3, 4, 5)))
            R = r.reshape(-1, 3)
            for name in d["nsets"]:
                if any(b[0] == name for b in d["bnd"]):
                    out += ["", f" forces (fx,fy,fz) for set {name} and time  0.1000000E+01", ""]
                    out += [f"{m.ids[i]:10d} {_fmt(R[i, 0])} {_fmt(R[i, 1])} {_fmt(R[i, 2])}" for i in m.set_nodes(name)]
    elif d["step"] == "HEAT TRANSFER":
        Kc = m.conduction(d["mat"]["CONDUCTIVITY"][0])
        K = Kc.tolil()
        f = np.zeros(n)
        for e, s, q in d["dflux"]:
            nodes, nint, _, _ = m.face_integrals(e, s)
            np.add.at(f, nodes, q * nint)
        for e, s, tinf, h in d["film"]:
            nodes, nint, nn, _ = m.face_integrals(e, s)
            for a, na in enumerate(nodes):
                for b, nb in enumerate(nodes):
                    K[na, nb] += h * nn[a, b]
            np.add.at(f, nodes, h * tinf * nint)
        K = K.tocsr()
        fixed = {}
        for name, a, b, val in d["bnd"]:
            if a == 11:
                for nd in m.set_nodes(name):
                    fixed[int(nd)] = val
        T = _solve(K, f, fixed, n)
        r = Kc @ T  # as CalculiX's RFL (assumed): conduction heat at the node, films and fluxes there included
        out += [" temperatures for set NALL and time  0.1000000E+01", ""]
        out += [f"{m.ids[i]:10d} {_fmt(T[i])}" for i in range(n)]
        for name in d["nsets"]:
            if any(b[0] == name for b in d["bnd"]):
                out += ["", f" heat generation (external concentrated heat sources) for set {name} and time  0.1000000E+01", ""]
                out += [f"{m.ids[i]:10d} {_fmt(r[i])}" for i in m.set_nodes(name)]
    else:
        raise RefError(f"the reference solver does not do {d['step']} (use CalculiX)")
    dat = inp.with_suffix(".dat")
    dat.write_text("\n".join(out) + "\n")
    return dat
