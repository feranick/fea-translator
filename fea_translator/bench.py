"""bench.py — benchmark studies with closed-form answers: the translator's checker.
Version: 2026.11.01.3

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
  bar_resistance  copper bar 100x10x10, 10 mV across: R = rho L / A, I = V / R, J = I / A (exact)
  bar_current     the same bar fed 100 A, other end grounded: terminal voltage I R (exact)
  plate_capacitor FR-4 plate 100x100x1, 1 kV: C = eps0 eps_r A / d, E = V / d (exact)
  coax_capacitor  PTFE ring ri 5, ro 15, length 20: C = 2 pi eps L / ln(ro/ri) (0.5 %; Gmsh)
  joule_bar       steel bar, 50 mV across, both ends at 20 C: P = V^2 / R, T_max = T0 + q L^2 / 8k (exact)
  thermal_stress_bar  steel bar at a uniform 120 C between two walls: sigma = E alpha dT (exact)
  plate_hole      quarter plate 200x50x2 with a 10 mm hole, net-section Kt = 2 + (1 - d/W)^3 (5 %; Gmsh, CalculiX)
  joint_pull      two 60x60x30 blocks, a full-face gasket (50 MPa/mm), one M12 through bolt at 20 kN, the top block
                  pulled off: gasket force at the preload = 20 kN (3 %), lift-off at F = P / (1 - C),
                  C = k_bolt / (k_bolt + k_joint) (10 %; CalculiX)
  joint_cover     a 40 mm cover on a flange ring (ri 40, ro 80), annular gasket (500 MPa/mm, seal 5 MPa), 8 x M12 at
                  20 kN, internal pressure: gasket force = 8 P (3 %), leak and lift-off pressures from the joint diagram
                  (15 %: the cover bends; Gmsh, CalculiX)
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
    EPS0_F_M = 8.8541878128e-12
    r_cu = 1.72e-8 * 0.1 / 1e-4                      # ohm: copper bar, L 0.1 m, A 1e-4 m2
    r_st = 1.6e-7 * 0.1 / 1e-4                       # steel bar
    q_st = (0.05 ** 2 / r_st) / (100 * 10 * 10)      # W/mm3 of Joule heat
    out = {
        "bar_resistance": {"part": _box(100, 10, 10, [10, 2, 2], [0, 0, 0]), "material": "copper", "analysis": "electric",
                           "bcs": [{"kind": "voltage", "name": "plus", "faces": {"nx": -1}, "V": 0.01},
                                   {"kind": "voltage", "name": "minus", "faces": {"nx": 1}, "V": 0.0}],
                           "expect": [{"quantity": "resistance_ohm", "value": r_cu, "tol": 1e-6},
                                      {"quantity": "terminals.plus.I_A", "value": 0.01 / r_cu, "tol": 1e-6},
                                      {"quantity": "max_current_density_A_mm2", "value": 0.01 / r_cu / 100, "tol": 1e-6}]},
        "bar_current": {"part": _box(100, 10, 10, [10, 2, 2], [0, 0, 0]), "material": "copper", "analysis": "electric",
                        "bcs": [{"kind": "current", "name": "feed", "faces": {"nx": -1}, "I": 100},
                                {"kind": "voltage", "name": "ground", "faces": {"nx": 1}, "V": 0.0}],
                        "expect": [{"quantity": "terminals.feed.V", "value": 100 * r_cu, "tol": 1e-6},
                                   {"quantity": "resistance_ohm", "value": r_cu, "tol": 1e-6},
                                   {"quantity": "power_W", "value": 100 ** 2 * r_cu, "tol": 1e-6}]},
        "plate_capacitor": {"part": _box(100, 100, 1, [10, 10, 2], [-50, -50, 0]), "material": "fr4",
                            "analysis": "electrostatic",
                            "bcs": [{"kind": "voltage", "name": "hv", "faces": {"nz": 1}, "V": 1000},
                                    {"kind": "voltage", "name": "ground", "faces": {"nz": -1}, "V": 0}],
                            "expect": [{"quantity": "capacitance_pF", "value": EPS0_F_M * 4.4 * 0.01 / 0.001 * 1e12, "tol": 1e-6},
                                       {"quantity": "max_field_kV_mm", "value": 1.0, "tol": 1e-6},
                                       {"quantity": "breakdown_safety_factor", "value": 20.0, "tol": 1e-6}]},
        "coax_capacitor": {"part": {"goal": {"features": [{"kind": "base_ring", "params": {"ri": 5, "ro": 15, "h": 20}}]}},
                           "material": "ptfe", "analysis": "electrostatic", "mesh": {"size": 1.5},
                           "bcs": [{"kind": "voltage", "name": "inner", "faces": {"cyl_r": 5, "side": "inner"}, "V": 1000},
                                   {"kind": "voltage", "name": "outer", "faces": {"cyl_r": 15, "side": "outer"}, "V": 0}],
                           "expect": [{"quantity": "capacitance_pF",
                                       "value": 2 * math.pi * EPS0_F_M * 2.1 * 0.02 / math.log(3) * 1e12, "tol": 0.005}]},
        "joule_bar": {"part": _box(100, 10, 10, [10, 2, 2], [0, 0, 0]), "material": "steel", "analysis": "joule",
                      "bcs": [{"kind": "voltage", "name": "plus", "faces": {"nx": -1}, "V": 0.05},
                              {"kind": "voltage", "name": "minus", "faces": {"nx": 1}, "V": 0.0},
                              {"kind": "temperature", "name": "cold_l", "faces": {"nx": -1}, "T": 20},
                              {"kind": "temperature", "name": "cold_r", "faces": {"nx": 1}, "T": 20}],
                      "expect": [{"quantity": "joule_heat_W", "value": 0.05 ** 2 / r_st, "tol": 1e-6},
                                 {"quantity": "T_max", "value": 20 + q_st * 100 ** 2 / (8 * 0.05), "tol": 1e-6},
                                 {"quantity": "electric.resistance_ohm", "value": r_st, "tol": 1e-6}]},
        "thermal_stress_bar": {"part": _box(100, 10, 10, [10, 2, 2], [0, 0, 0]), "material": "steel", "analysis": "thermal",
                               "stress": True, "T_ref": 20,
                               "bcs": [{"kind": "temperature", "name": "hot_l", "faces": {"nx": -1}, "T": 120},
                                       {"kind": "temperature", "name": "hot_r", "faces": {"nx": 1}, "T": 120},
                                       {"kind": "slider", "name": "wall_l", "faces": {"nx": -1}},
                                       {"kind": "slider", "name": "wall_r", "faces": {"nx": 1}},
                                       {"kind": "slider", "name": "sym_y", "faces": {"ny": -1}},
                                       {"kind": "slider", "name": "sym_z", "faces": {"nz": -1}}],
                               "expect": [{"quantity": "stress.peak_von_mises", "value": 210000 * 1.2e-5 * 100, "tol": 1e-6},
                                          {"quantity": "stress.reactions.wall_l.fx", "value": 210000 * 1.2e-5 * 100 * 100, "tol": 1e-6}]},
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
    out.update(_joint_cases())
    # plate_hole: the box is the quarter x in [-100, 100], y in [-25, 25]; the hole is centred on its
    # -X/-Y corner, so the two symmetry planes pass through the hole centre; the peak is at (-100, -20).
    return out


NEEDS = {"euler_column": {"ccx"}, "plate_hole": {"ccx", "gmsh"}, "lame_ring": {"gmsh"}, "coax_capacitor": {"gmsh"},
         "joint_pull": {"ccx"}, "joint_cover": {"ccx", "gmsh"}}


def _joint_cases() -> dict:
    """Bolted, gasketed joints (analysis "joint"; CalculiX only), against the joint diagram: the bolt and the clamped
    members (gasket and flanges in series) share the external load as C = k_b / (k_b + k_j); the clamp force falls
    by (1 - C) F, so the joint lifts off at F = n P / (1 - C) and leaks when the gasket stress (clamp force over the
    gasket area) falls to the sealing stress."""
    As = math.pi / 4 * (12 - 0.9382 * 1.75) ** 2                      # M12 tensile stress area
    P = 20000.0
    # joint_pull: blocks 60 x 60, 30 mm each; bolt grip 60; gasket 50 MPa/mm over 3600 mm2
    A, K, L = 3600.0, 50.0, 60.0
    kb = E * As / L
    kj = 1 / (1 / (K * A) + L / (E * A))
    C = kb / (kb + kj)
    pull = {"analysis": "joint",
            "parts": {"base": {"part": {"box": [60, 60, 30], "origin": [-30, -30, 0], "divisions": [12, 12, 6]}},
                      "top": {"part": {"box": [60, 60, 30], "origin": [-30, -30, 30], "divisions": [12, 12, 6]}}},
            "bcs": [{"kind": "fixed", "name": "side", "faces": {"part": "base", "nx": -1}},
                    {"kind": "force", "name": "pull", "faces": {"part": "top", "nz": 1}, "fz": 80000.0}],
            "gaskets": [{"name": "g", "faces": {"part": "top", "nz": -1}, "with": {"part": "base", "nz": 1},
                         "stiffness": K, "seal": {"stress": 1.0}}],
            "bolts": [{"name": "b", "size": "M12", "preload": P, "at": [[0, 0, 60]], "head": {"part": "top", "nz": 1},
                       "nut": {"part": "base", "nz": -1}}],
            "sweep": {"bc": "pull", "steps": 16},
            "expect": [{"quantity": "gaskets.g.preload.force_N", "value": P, "tol": 0.03},
                       {"quantity": "lift_off.g", "value": P / (1 - C), "tol": 0.10}]}
    # joint_cover: ring ri 40 ro 80 h 30; cover r 80 h 40 with a spigot r 39 into the bore; grip 70; 8 bolts on pcd 120
    Ag, Kg, Lc, n, sig = math.pi * (80 ** 2 - 40 ** 2), 500.0, 70.0, 8, 5.0
    kb8 = n * E * As / Lc
    kj8 = 1 / (1 / (Kg * Ag) + 40 / (E * Ag) + 30 / (E * Ag))
    C8 = kb8 / (kb8 + kj8)
    Ap = math.pi * 39 ** 2                                              # the spigot's bottom: the separating area
    cover = {"analysis": "joint", "mesh": {"size": 6},
             "parts": {"ring": {"part": {"goal": {"features": [{"kind": "base_ring", "params": {"ri": 40, "ro": 80, "h": 30}}]}}},
                       "cover": {"part": {"goal": {"features": [
                           {"kind": "base_cyl", "params": {"r": 80, "h": 40}},
                           {"kind": "boss_cyl", "params": {"r": 39, "x": 0, "y": 0, "h": 5, "nx": 0, "ny": 0, "nz": -1}}]}},
                                 "move": [0, 0, 30]}},
             "bcs": [{"kind": "fixed", "name": "outside", "faces": {"part": "ring", "cyl_r": 80, "side": "outer"}},
                     {"kind": "pressure", "name": "inside", "p": 60.0,
                      "faces": [{"part": "cover", "nz": -1, "at": [0, 0, 25]}, {"part": "cover", "cyl_r": 39},
                                {"part": "ring", "cyl_r": 40, "side": "inner"}]}],
             "gaskets": [{"name": "g", "faces": {"part": "cover", "nz": -1, "at": [60, 0, 30]},
                          "with": {"part": "ring", "nz": 1}, "stiffness": Kg, "seal": {"stress": sig}}],
             "bolts": [{"name": "cover", "size": "M12", "preload": P, "pattern": {"center": [0, 0, 70], "pcd": 120, "n": n},
                        "head": {"part": "cover", "nz": 1}, "nut": {"part": "ring", "nz": -1}}],
             "sweep": {"bc": "inside", "steps": 12},
             "expect": [{"quantity": "gaskets.g.preload.force_N", "value": n * P, "tol": 0.03},
                        {"quantity": "leak.g", "value": (n * P - sig * Ag) / ((1 - C8) * Ap), "tol": 0.15},
                        {"quantity": "lift_off.g", "value": n * P / ((1 - C8) * Ap), "tol": 0.15}]}
    return {"joint_pull": pull, "joint_cover": cover}


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
