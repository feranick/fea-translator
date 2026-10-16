"""bench.py — benchmark studies with closed-form answers: the translator's checker.
Version: 2026.10.16.4

    run(solver="ccx", names=None, folder=...) -> {case: {...results, checks, passed}}

Every case is an ordinary study with an `expect` list. Box parts need no Gmsh; the others are
build123d-translator goals meshed by Gmsh. Cases (E = 210 GPa, nu = 0.3 unless noted):

  axial_bar       box 100x10x10 on three symmetry sliders, 1 kN pull: d = FL/EA, s = F/A (exact)
  cantilever      box 100x10x10, clamped, 100 N tip load: Timoshenko tip deflection (3 %)
  cantilever_modal  first bending frequency, Euler-Bernoulli (3 %)
  euler_column    box 200x5x5 fixed-free, unit axial load: P = pi^2 EI / 4L^2 (4 %; CalculiX only)
  gravity_block   box 100x10x10 clamped at the bottom under its own weight: reaction = m g (exact)
  pressure_cantilever  0.1 MPa on the top of the cantilever: reaction = p A (exact; the face touches the clamp)
  slab            conduction 100 mm, 100 -> 0 C, k 50: Q = k A dT / L (exact)
  fin_end         conduction + end convection h 1000: Q = dT / (L/kA + 1/hA) (exact)
  fin_sides       square pin fin 100 mm, sides and tip convecting (h 50): 1-D fin with a convecting tip (1 %)
  lame_ring       quarter ring (a 90-degree revolve) ri 20, ro 40, 10 MPa inside, plane stress: Lame bore displacement
                  and von Mises (2 %; Gmsh)
  plate_hole      quarter plate 200x50x2 with a 10 mm hole, net-section Kt = 2 + (1 - d/W)^3 (5 %; Gmsh, CalculiX)
"""
from __future__ import annotations

import math
from pathlib import Path

E, NU, RHO = 210000.0, 0.3, 7850e-12
G = E / 2 / (1 + NU)


def _box(lx, ly, lz, div, origin=None):
    p = {"box": [lx, ly, lz], "divisions": div}
    if origin is not None:
        p["origin"] = origin
    return p


def cases(solver: str = "ccx") -> dict:
    """The benchmark studies; peak stresses are checked at nodes with CalculiX (its .frd), else at
    integration points (a little lower near a steep gradient)."""
    L, b, h = 100.0, 10.0, 10.0
    I, A = b * h ** 3 / 12, b * h
    tip = 100 * L ** 3 / (3 * E * I) + 100 * L / (5 / 6 * G * A)
    f1 = 1.875104 ** 2 / (2 * math.pi) * math.sqrt(E * I / (RHO * A * L ** 4))
    Lc, a = 200.0, 5.0
    Pcr = math.pi ** 2 * E * a ** 4 / 12 / (4 * Lc ** 2)
    ri, ro, p = 20.0, 40.0, 10.0
    ur = p * ri ** 2 / (E * (ro ** 2 - ri ** 2)) * ((1 - NU) * ri + (1 + NU) * ro ** 2 / ri)
    s_t, s_r = p * (ro ** 2 + ri ** 2) / (ro ** 2 - ri ** 2), -p
    vm_bore = math.sqrt(s_t ** 2 + s_r ** 2 - s_t * s_r)
    W, d, t, Lp = 100.0, 10.0, 2.0, 400.0  # full plate; the model is a quarter
    kt = 2 + (1 - d / W) ** 3
    s_net = 1.0 * W * t / ((W - d) * t)  # 1 MPa remote stress
    fixed = {"kind": "fixed", "name": "wall", "faces": {"nx": -1}}
    sl = lambda n, ax: {"kind": "slider", "name": n, "faces": {ax: -1}}  # noqa: E731
    hf, kf, af = 50.0, 50.0, 0.01  # fin: h W/(m2 K), k W/(m K), side m
    mf = math.sqrt(hf * 4 * af / (kf * af * af))
    Mf = math.sqrt(hf * 4 * af * kf * af * af) * 100.0
    ratio = hf / (mf * kf)
    q_fin = Mf * (math.sinh(mf * 0.1) + ratio * math.cosh(mf * 0.1)) / (math.cosh(mf * 0.1) + ratio * math.sinh(mf * 0.1))
    weight = 7850 * (100 * 10 * 10 * 1e-9) * 9.81
    out = {
        "axial_bar": {"part": _box(100, 10, 10, [10, 2, 2], [0, 0, 0]), "material": "steel", "analysis": "static",
                      "bcs": [sl("sx", "nx"), sl("sy", "ny"), sl("sz", "nz"),
                              {"kind": "force", "name": "pull", "faces": {"nx": 1}, "fx": 1000}],
                      "expect": [{"quantity": "faces.pull.mean_u.ux", "value": 1000 * 100 / (E * 100), "tol": 1e-4},
                                 {"quantity": "max_von_mises", "value": 10.0, "tol": 1e-4}]},
        "cantilever": {"part": _box(L, b, h, [20, 4, 4], [0, -5, -5]), "material": "steel", "analysis": "static",
                       "bcs": [fixed, {"kind": "force", "name": "tip", "faces": {"nx": 1}, "fz": -100}],
                       "expect": [{"quantity": "faces.tip.mean_u.uz", "value": -tip, "tol": 0.03},
                                  {"quantity": "reactions.wall.fz", "value": 100.0, "tol": 1e-6}]},
        "cantilever_modal": {"part": _box(L, b, h, [20, 4, 4], [0, -5, -5]), "material": "steel", "analysis": "modal",
                             "modes": 4, "bcs": [fixed],
                             "expect": [{"quantity": "frequencies.0", "value": f1, "tol": 0.03},
                                        {"quantity": "frequencies.1", "value": f1, "tol": 0.03}]},
        "euler_column": {"part": _box(a, a, Lc, [2, 2, 40], [-a / 2, -a / 2, 0]), "material": "steel", "analysis": "buckling",
                         "modes": 2, "bcs": [{"kind": "fixed", "name": "base", "faces": {"nz": -1}},
                                             {"kind": "force", "name": "top", "faces": {"nz": 1}, "fz": -1}],
                         "expect": [{"quantity": "critical_factor", "value": Pcr, "tol": 0.04}]},
        "gravity_block": {"part": _box(10, 10, 100, [2, 2, 10], [-5, -5, 0]), "material": "steel", "analysis": "static",
                          "bcs": [{"kind": "fixed", "name": "base", "faces": {"nz": -1}}, {"kind": "gravity", "name": "g"}],
                          "expect": [{"quantity": "reactions.base.fz", "value": weight, "tol": 1e-6}]},
        "pressure_cantilever": {"part": _box(L, b, h, [20, 4, 4], [0, -5, -5]), "material": "steel", "analysis": "static",
                                "bcs": [fixed, {"kind": "pressure", "name": "top", "faces": {"nz": 1}, "p": 0.1}],
                                "expect": [{"quantity": "reactions.wall.fz", "value": 0.1 * L * b, "tol": 1e-6}]},
        "slab": {"part": _box(100, 10, 10, [10, 2, 2], [0, 0, 0]), "material": {"name": "steel", "k": 50}, "analysis": "thermal",
                 "bcs": [{"kind": "temperature", "name": "hot", "faces": {"nx": -1}, "T": 100},
                         {"kind": "temperature", "name": "cold", "faces": {"nx": 1}, "T": 0}],
                 "expect": [{"quantity": "heat_flow.hot", "value": 50 * 1e-4 * 100 / 0.1, "tol": 1e-6}]},
        "fin_end": {"part": _box(100, 10, 10, [10, 2, 2], [0, 0, 0]), "material": {"name": "steel", "k": 50}, "analysis": "thermal",
                    "bcs": [{"kind": "temperature", "name": "base", "faces": {"nx": -1}, "T": 100},
                            {"kind": "convection", "name": "end", "faces": {"nx": 1}, "h": 1000, "T_inf": 0}],
                    "expect": [{"quantity": "heat_flow.base", "value": 100 / (0.1 / (50 * 1e-4) + 1 / (1000 * 1e-4)), "tol": 1e-6}]},
        "fin_sides": {"part": _box(100, 10, 10, [20, 2, 2], [0, 0, 0]), "material": {"name": "steel", "k": kf},
                      "analysis": "thermal",
                      "bcs": [{"kind": "temperature", "name": "base", "faces": {"nx": -1}, "T": 100},
                              {"kind": "convection", "name": "air", "faces": [{"nx": 1}, {"ny": 1}, {"ny": -1}, {"nz": 1}, {"nz": -1}],
                               "h": hf, "T_inf": 0}],
                      "expect": [{"quantity": "heat_flow.base", "value": q_fin, "tol": 0.01}]},
        # quarter ring: the section (r 20..40, z 0..10) revolved 90 degrees about Z, from +X to +Y
        "lame_ring": {"part": {"goal": {"features": [
                          {"kind": "profile_revolve", "params": {
                              "outline": {"start": [ri, 0], "segs": [["L", ro, 0], ["L", ro, 10], ["L", ri, 10], ["L", ri, 0]]},
                              "ax": 0, "ay": 0, "az": 1, "nx": 0, "ny": 1, "nz": 0, "off": 0, "x": 0, "y": 0, "z": 0,
                              "angle": 90}}]}},
                      "material": "steel", "analysis": "static", "mesh": {"size": 2.5},
                      "bcs": [{"kind": "slider", "name": "sym_x", "faces": {"nx": -1}},
                              {"kind": "slider", "name": "sym_y", "faces": {"ny": -1}},
                              {"kind": "slider", "name": "base", "faces": {"nz": -1}},
                              {"kind": "pressure", "name": "bore", "faces": {"cyl_r": ri, "side": "inner"}, "p": p}],
                      "probes": [{"name": "bore", "faces": {"cyl_r": ri, "side": "inner"}}],
                      "expect": [{"quantity": "max_displacement", "value": math.hypot(ur, NU / E * (s_t + s_r) * 10), "tol": 0.02},
                                 {"quantity": "max_von_mises_nodal", "value": vm_bore, "tol": 0.02} if solver == "ccx" else
                                 {"quantity": "max_von_mises", "value": vm_bore, "tol": 0.04}]},
        "plate_hole": {"part": {"goal": {"features": [
                           {"kind": "base_box", "params": {"w": Lp / 2, "d": W / 2, "h": t}},
                           {"kind": "hole", "params": {"r": d / 2, "x": -Lp / 4, "y": -W / 4}}]}},
                       "material": "steel", "analysis": "static",
                       "mesh": {"size": 4, "refine": [{"at": [-Lp / 4, -W / 4 + d / 2, t / 2], "r": 6, "size": 0.4}]},
                       "bcs": [{"kind": "slider", "name": "sym_x", "faces": {"nx": -1}},
                               {"kind": "slider", "name": "sym_y", "faces": {"ny": -1}},
                               {"kind": "slider", "name": "base", "faces": {"nz": -1}},
                               {"kind": "force", "name": "pull", "faces": {"nx": 1}, "fx": 1.0 * W / 2 * t}],
                       "expect": [{"quantity": "max_von_mises_nodal", "value": kt * s_net, "tol": 0.05}]},
    }
    # plate_hole: the box is the quarter x in [-100, 100], y in [-25, 25]; the hole is centred on its
    # -X/-Y corner, so the two symmetry planes pass through the hole centre; the peak is at (-100, -20).
    return out


NEEDS = {"euler_column": {"ccx"}, "plate_hole": {"ccx", "gmsh"}, "lame_ring": {"gmsh"}}


def run(solver: str = "ccx", names=None, folder: Path | None = None) -> dict:
    from .runner import run_study

    folder = Path(folder or Path.home() / "feat_runs" / "bench")
    out = {}
    try:
        import gmsh  # noqa: F401
        have_gmsh = True
    except ImportError:
        have_gmsh = False
    for name, st in cases(solver).items():
        if names and name not in names:
            continue
        need = NEEDS.get(name, set())
        if "ccx" in need and solver != "ccx":
            out[name] = {"skipped": "needs CalculiX"}
            continue
        if "gmsh" in need and not have_gmsh:
            out[name] = {"skipped": "needs Gmsh"}
            continue
        r = run_study(name, st, folder / name, solver=solver)
        out[name] = {k: r.get(k) for k in ("ok", "passed", "checks", "error", "seconds", "mesh") if k in r}
    return out
