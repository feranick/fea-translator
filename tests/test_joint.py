"""Bolted, gasketed joints (joint.py): the deck (contact, bolt connectors, pretension, steps), the gasket's ring
stress, reading multi-step .frd/.dat output, leak and lift-off from synthetic results, validation. Offline: the
CalculiX runs are the joint_pull and joint_cover benchmarks (test_fea.test_benchmark_calculix).
Version: 2026.11.02.9
"""
import copy
import math
import re

import numpy as np
import pytest

from fea_translator import joint
from fea_translator.bench import cases
from fea_translator.meshdata import MeshError
from fea_translator.study import validate_study


@pytest.fixture(scope="module")
def pull(tmp_path_factory):
    st = copy.deepcopy(cases()["joint_pull"])
    st["sweep"]["steps"] = 4
    parts = joint.mesh_parts(st, tmp_path_factory.mktemp("pull"))
    text, plan = joint.write("pull", st, parts)
    return st, parts, text, plan


def test_bolt_sizes_and_preload():
    b = joint.bolt_size({"size": "M12"})
    assert b["As"] == pytest.approx(84.27, abs=0.05)              # ISO 898-1 tensile stress area
    assert b["hole_d"] == 13.5 and b["washer_d"] == 24
    assert joint.preload_N({"size": "M12", "torque": 60}) == pytest.approx(60e3 / (0.2 * 12))
    assert joint.preload_N({"size": "M12", "preload": 1234}) == 1234
    pts = joint.bolt_positions({"pattern": {"center": [0, 0, 5], "pcd": 100, "n": 4}}, np.array([0, 0, -1.0]))
    assert len(pts) == 4 and all(np.linalg.norm(p[:2]) == pytest.approx(50) and p[2] == 5 for p in pts)


def test_deck_structure(pull, tmp_path):
    st, parts, text, plan = pull
    up = text.upper()
    count = lambda k: len(re.findall(rf"^\*{k}\b", up, re.M))    # noqa: E731
    assert count("CONTACT PAIR") == 1 and count("PRE-TENSION SECTION") == 0 and count("SPRING") == 2   # bolt, stabilisation
    assert "TYPE=SPRINGA" in up and "B31" not in up
    assert count("STEP") == 1 + 4 and count("END STEP") == 1 + 4
    assert count("EQUATION") == 6                                   # 3 directions x (nut end, the cut at the head)
    b0 = plan["bolts"][0]["bolts"][0]
    nc, n2, npt = b0["ends"][0], b0["ends"][1], b0["pret_node"]
    nh = b0["head_nodes"]
    lines = text.splitlines()
    for d in (1, 2, 3):   # n u_C - sum(u_head) + n axis_d u_PT = 0, axis -z: the PT term only in z
        i = next(k for k, ln in enumerate(lines) if ln.startswith(f"{nc}, {d}, {nh}, "))
        n_terms = int(lines[i - 1])
        body = ", ".join(lines[i:i + (n_terms + 3) // 4]).split(", ")
        trip = [(int(body[k]), int(body[k + 1]), float(body[k + 2])) for k in range(0, len(body), 3)]
        assert trip[0] == (nc, d, nh) and all(c == -1.0 and dd == d for _, dd, c in trip[1:1 + nh])
        assert trip[1 + nh:] == ([(npt, 1, -float(nh))] if d == 3 else [])
    assert b0["k_N_mm"] == pytest.approx(210000 * 84.266 / 60, rel=1e-4)
    assert up.count("*BOUNDARY, FIXED") == 4 and up.count("*CLOAD, OP=NEW") == 4
    assert "*EL FILE" not in up and up.count("*NODE PRINT, NSET=N_BOLTENDS") == 5 and up.count("*NODE PRINT, NSET=N_STAB") == 5
    # the preload starts small; the friction's stick slope is low there and back to full from step 2
    steps = up.split("*STEP")
    assert "0.05, 1., 1E-05, 0.25" in steps[1] and "CHANGE FRICTION" not in steps[1]
    assert "*CHANGE FRICTION, INTERACTION=GI_G\n*FRICTION\n0.2, 0.5\n" in steps[2] and "CHANGE FRICTION" not in steps[3]
    model = steps[0]
    assert "*FRICTION\n0.2, 0.005\n" in model                       # 50 MPa/mm / 100 x 0.01
    # stabilisation: the top block (no support) gets 6 well-spread nodes x 3 springs to fixed ground nodes
    stp = plan["stabilisation"]
    assert list(stp["parts"]) == ["top"] and len(stp["parts"]["top"]) == 6 and stp["springs"] == 18
    assert stp["k_N_mm"] == pytest.approx(1e-6 * plan["bolts"][0]["bolts"][0]["k_N_mm"])
    assert "N_STABGROUND, 1, 3" in steps[1]
    X = joint.all_nodes(parts)
    P = X[np.array(stp["parts"]["top"]) - 1]
    assert P[:, 2].min() >= 30 - 1e-9 and np.ptp(P[:, 0]) == pytest.approx(60) and np.ptp(P[:, 1]) == pytest.approx(60)
    assert "PRESSURE-OVERCLOSURE=LINEAR" in up and "*FRICTION" in up and "*CONTACT FILE" in up
    assert [s["factor"] for s in plan["steps"]] == [0.0, 0.25, 0.5, 0.75, 1.0]
    b = plan["bolts"][0]
    assert b["preload_N"] == 20000 and len(b["bolts"]) == 1
    assert b["bolts"][0]["grip_mm"] == pytest.approx(60, abs=1e-6)     # top face to bottom face
    assert plan["gaskets"][0]["area"] == pytest.approx(3600, rel=1e-6)
    assert plan["gaskets"][0]["gap"] == {"median_mm": 0.0, "max_mm": 0.0}       # the blocks touch
    moved = copy.deepcopy(st)
    moved["parts"]["top"]["move"] = [0, 0, 0.2]
    pm = joint.mesh_parts(moved, tmp_path)
    g2 = joint.write("m", moved, pm)[1]["gaskets"][0]
    assert g2["gap"]["median_mm"] == pytest.approx(0.2) and "0.2 mm apart" in joint.gap_warning(g2)
    # the beam's end nodes and the pretension node come after the mesh nodes, the beam after the tets
    n_mesh = sum(len(p["mesh"].nodes) for p in parts)
    assert min(plan["pret_nodes"]) > n_mesh
    # every equation's dependent (first) term is one of the bolt's own nodes: never held, never a mesh node
    eq = [ln for ln in text.splitlines()]
    deps = []
    for i, ln in enumerate(eq):
        if ln.upper().startswith("*EQUATION"):
            nterms = int(eq[i + 1])
            first = eq[i + 2].split(",")
            deps.append((int(first[0]), int(first[1])))
            assert int(first[0]) > n_mesh and nterms >= 2
    assert len(deps) == len(set(deps)) == 6                          # each DOF dependent in one equation only
    # no chains: a dependent DOF is never a term of another equation
    terms = set()
    for i, ln in enumerate(eq):
        if ln.upper().startswith("*EQUATION"):
            n_t = int(eq[i + 1])
            body = ", ".join(eq[i + 2:i + 2 + (n_t + 3) // 4]).split(", ")
            terms |= {(int(body[k]), int(body[k + 1])) for k in range(3, len(body), 3)}
    assert not set(deps) & terms


def test_gasket_curve_becomes_its_secant_at_the_preload():
    """CalculiX ignores a TABULAR law in face-to-face contact: a curve is the LINEAR law of its secant at the mean
    preload stress."""
    g = {"curve": [[0.2, 30], [0.05, 2], [0.1, 8]], "friction": 0.3}
    assert joint.closure_at(g, 5.0) == pytest.approx(0.075) and joint.closure_at(g, 8.0) == pytest.approx(0.1)
    law = joint.gasket_linear(g, 5.0)
    assert law["kind"] == "secant" and law["stiffness_MPa_mm"] == pytest.approx(5 / 0.075) and law["closure_mm"] == 0.075
    assert not law["beyond_curve"] and joint.gasket_linear(g, 45.0)["closure_mm"] == pytest.approx(0.2 + 15 / 220, abs=1e-6)
    assert joint.gasket_linear(g, 45.0)["beyond_curve"]
    assert joint.gasket_linear({"stiffness": 50}, 5.0) == {"kind": "stiffness", "stiffness_MPa_mm": 50.0}
    assert joint.friction_lines(50.0, g, 0.01) == ["*FRICTION", "0.3, 0.005"]


def _annulus(nr=4, nt=24, ri=10.0, ro=20.0):
    """A planar annulus of tri3 faces as a graph: nodes (1-based), adjacency, inner and outer rings."""
    idx = lambda r, t: 1 + r * nt + (t % nt)                        # noqa: E731
    adj = {}
    for r in range(nr):
        for t in range(nt):
            nb = [(r, t + 1), (r + 1, t), (r + 1, t + 1)] if r + 1 < nr else [(r, t + 1)]
            for q in nb:
                a, b = idx(r, t), idx(*q)
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
    inner = {idx(0, t) for t in range(nt)}
    outer = {idx(nr - 1, t) for t in range(nt)}
    return adj, inner, outer, idx


def test_ring_stress_finds_the_weakest_ring():
    adj, inner, outer, idx = _annulus()
    p = {n: 10.0 for n in adj}
    assert joint.ring_stress(adj, p, inner, outer) == 10.0
    p = {n: 1.0 for n in adj}
    for t in range(24):                                              # the best closed ring is at 3 MPa: the seal is 3
        p[idx(2, t)] = 3.0
    assert joint.ring_stress(adj, p, inner, outer) == 3.0
    p[idx(2, 5)] = 2.0                                               # one weaker node in it: the ring is as weak
    assert joint.ring_stress(adj, p, inner, outer) == 2.0
    p2 = {n: 10.0 for n in adj}
    for r in range(4):                                               # a radial channel at 0.5: a leak path
        p2[idx(r, 7)] = 0.5
    assert joint.ring_stress(adj, p2, inner, outer) == 0.5
    p2[idx(1, 7)] = 4.0                                              # the channel blocked by a 4 MPa node
    assert joint.ring_stress(adj, p2, inner, outer) == pytest.approx(4.0)
    assert joint.ring_stress(adj, p, inner, set()) == math.inf      # nothing outside to leak to


def test_crossing():
    assert joint._crossing([0, 1, 2, 3], [4, 3, 2, 1]) is None
    assert joint._crossing([0, 1, 2], [2, 1, -1]) == pytest.approx(1.5)
    # a contact stress clipped at zero: the zero lies between the last two positive steps' trend and the step
    assert joint._crossing([0, 10, 20, 30], [30, 20, 10, 0], floor=0.0) == pytest.approx(30)
    assert joint._crossing([0, 10, 20, 30], [30, 25, 20, 0], floor=0.0) == pytest.approx(30)
    assert joint._crossing([0, 10, 20, 30], [30, 20, 15, 0], floor=0.0) == pytest.approx(30)
    assert joint._crossing([0, 10, 20, 30], [40, 30, 20, 0], floor=0.0) == pytest.approx(30)
    assert joint._crossing([0, 10, 20, 30], [50, 30, 10, 0], floor=0.0) == pytest.approx(25)


def _frd(steps, disp=None):
    """An ASCII .frd with one CONTACT block per step: steps = [{node: cpress}]; disp = [{node: (ux, uy, uz)}]
    adds a DISP block per step."""
    out = ["    1C", "    1UUSER"]
    for k, pm in enumerate(steps, start=1):
        if disp:
            out += [f"    1PSTEP              {k}           1           {k}",
                    f"  100CL  101 1.000000000           {len(disp[k - 1])}                     0    {k}           1",
                    " -4  DISP        4    1", " -5  D1          1    2    1    0", " -5  D2          1    2    2    0",
                    " -5  D3          1    2    3    0", " -5  ALL         1    2    0    0    1ALL"]
            for n, u in sorted(disp[k - 1].items()):
                out.append(f" -1{n:10d}" + "".join(f"{x:12.5E}" for x in u))
            out.append(" -3")
        out += [f"    1PSTEP              {k}           1           {k}",
                f"  100CL  101 1.000000000           {len(pm)}                     0    {k}           1",
                " -4  CONTACT     6    1",
                " -5  COPEN       1    1    1    0", " -5  CSLIP1      1    1    2    0",
                " -5  CSLIP2      1    1    3    0", " -5  CPRESS      1    1    4    0",
                " -5  CSHEAR1     1    1    5    0", " -5  CSHEAR2     1    1    6    0"]
        for n, v in sorted(pm.items()):
            out.append(f" -1{n:10d}" + "".join(f"{x:12.5E}" for x in (0.0, 0.0, 0.0, v, 0.0, 0.0)))
        out.append(" -3")
    out.append(" 9999")
    return "\n".join(out) + "\n"


def _dat(plan, forces):
    """The springs' end displacements per step that give these bolt forces: forces = [[f per bolt] per step]."""
    out = []
    bolts = [(rec, b) for rec in plan["bolts"] for b in rec["bolts"]]
    for k, fs in enumerate(forces, start=1):
        out += ["", f" displacements (vx,vy,vz) for set N_BOLTENDS and time  {k:.7E}", ""]
        for (rec, b), f in zip(bolts, fs):
            a = np.asarray(rec["axis"], float)
            u2 = 0.01 * np.ones(3) + a * f / b["k_N_mm"]
            for n, u in zip(b["ends"], (0.01 * np.ones(3), u2)):
                out.append(f"{n:10d}" + "".join(f" {x:13.6E}" for x in u))
    return "\n".join(out) + "\n"


def test_frd_steps_and_contact_pressure():
    text = _frd([{1: 5.0, 2: 6.0}, {1: 1.0, 2: 0.0}])
    blocks = joint.frd_steps(text)
    assert [b["step"] for b in blocks] == [1, 2] and blocks[0]["components"][3] == "CPRESS"
    assert joint.contact_pressure_by_step(blocks) == {1: {1: 5.0, 2: 6.0}, 2: {1: 1.0, 2: 0.0}}


def test_evaluate_lift_off_on_a_full_face_gasket(pull):
    """The pull gasket covers the whole face (one edge): ring = the largest stress; it lifts off when that is 0."""
    st, parts, text, plan = pull
    g = plan["gaskets"][0]
    X = joint.all_nodes(parts)
    tets = np.concatenate([p["mesh"].tets + p["n0"] for p in parts])
    _, area, loops = joint.gasket_graph(X, tets, [tuple(f) for f in g["slave_faces"]])
    assert len(loops) == 1 and sum(area.values()) == pytest.approx(3600, rel=1e-6)
    nodes = sorted(area)
    # uniform gasket stress, falling linearly with the pull: 20000 N preload at 0, zero at 60000 N
    loads = [s["factor"] * 80000 for s in plan["steps"]]
    stress = [max(0.0, 20000 / 3600 * (1 - f / 60000)) for f in loads]
    frd = _frd([{n: s for n in nodes} for s in stress])
    bolt = [[20000.0]] + [[20000 + 0.2 * f] for f in loads[1:]]
    res = joint.evaluate(st, parts, plan, _dat(plan, bolt), frd)
    assert res["gaskets"]["g"]["preload"]["force_N"] == pytest.approx(20000, rel=1e-6)
    assert res["preload_balance_error"] < 1e-5
    assert res["bolt_preload_error"] < 1e-5
    assert res["lift_off"]["g"] == pytest.approx(60000, rel=1e-6)
    assert res["leak"]["g"] is None                                  # one edge: nothing to leak from
    assert any("one edge only" in w for w in res["warnings"])
    assert [s["bolts"]["b"]["max_N"] for s in res["steps"]] == pytest.approx([b[0] for b in bolt])
    assert res["bolts"]["b"]["max_N"] == pytest.approx(36000)
    assert res["swept"] == {"bc": "pull", "kind": "force", "max": 80000.0, "unit": "N"}


def test_evaluate_not_reached(pull):
    st, parts, text, plan = pull
    g = plan["gaskets"][0]
    X = joint.all_nodes(parts)
    tets = np.concatenate([p["mesh"].tets + p["n0"] for p in parts])
    _, area, _ = joint.gasket_graph(X, tets, [tuple(f) for f in g["slave_faces"]])
    frd = _frd([{n: 5.0 - k for n in area} for k in range(5)])
    res = joint.evaluate(st, parts, plan, _dat(plan, [[20000.0]] * 5), frd)
    assert res["lift_off"]["g"] == {"above": 80000.0}


def test_benchmarks_validate():
    for name in ("joint_pull", "joint_cover"):
        r = validate_study(cases()[name])
        assert r["ok"], (name, r["errors"])


def test_validation_errors():
    base = cases()["joint_pull"]

    def errs(mut):
        st = copy.deepcopy(base)
        mut(st)
        return " | ".join(validate_study(st)["errors"])

    def drop(path):
        def f(st):
            d = st
            for k in path[:-1]:
                d = d[k]
            del d[path[-1]]
        return f
    assert "at least two parts" in errs(lambda s: s["parts"].pop("top"))
    assert "needs 'part'" in errs(lambda s: s["gaskets"][0]["faces"].pop("part"))
    assert "each selector needs 'part'" in errs(lambda s: s["bcs"][0]["faces"].pop("part"))
    assert "give 'curve'" in errs(drop(["gaskets", 0, "stiffness"]))
    assert "closures increasing" in errs(lambda s: s["gaskets"][0].update(curve=[[0.1, 5], [0.05, 9]], stiffness=None) or s["gaskets"][0].pop("stiffness"))
    assert ".seal:" in errs(drop(["gaskets", 0, "seal"]))
    assert ".size: one of" in errs(lambda s: s["bolts"][0].update(size="M13"))
    assert "'preload' (N) or 'torque'" in errs(lambda s: s["bolts"][0].update(torque=50))
    assert "'nut'" in errs(lambda s: s["bolts"][0].update(thread={"part": "base", "depth": 12}))
    assert "sweep:" in errs(lambda s: s["sweep"].update(bc="nope"))
    assert "a pressure or force bc" in errs(lambda s: s["sweep"].update(bc="side"))
    assert "sweep.steps" in errs(lambda s: s["sweep"].update(steps=0))
    assert "needs supports" in errs(lambda s: s["bcs"].pop(0) and s["sweep"].update(bc="pull"))
    assert "kind is one of" in errs(lambda s: s["bcs"].append({"kind": "temperature", "faces": {"part": "top", "nz": 1}}))
    assert "converge" in errs(lambda s: s.update(converge=True))
    assert "images[0]" in errs(lambda s: s.update(images=[{"field": "temperature"}]))


def test_thread_engagement_needs_a_bore(tmp_path):
    """A tapped hole: the nut side is the bore over the thread depth; the plain box has no bore there."""
    st = copy.deepcopy(cases()["joint_pull"])
    st["bolts"][0].pop("nut")
    st["bolts"][0]["thread"] = {"part": "base", "depth": 12}
    parts = joint.mesh_parts(st, tmp_path)
    with pytest.raises(MeshError, match="bore nodes"):
        joint.bolt_geometry(parts, st["bolts"][0])


def test_reference_solver_refuses(tmp_path):
    from fea_translator.runner import SolverError
    with pytest.raises(SolverError, match="CalculiX"):
        joint.solve("x", cases()["joint_pull"], tmp_path, "ref", 60)


def test_thread_engagement_in_a_tapped_hole(tmp_path):
    """A cap screw into a tapped hole: the nut side is the bore over the engaged depth below the entry."""
    pytest.importorskip("gmsh")
    pytest.importorskip("build123d_translator")
    box = lambda r, z: {"part": {"goal": {"features": [{"kind": "base_box", "params": {"w": 60, "d": 60, "h": 30}},  # noqa: E731
                                                       {"kind": "hole", "params": {"r": r, "x": 0, "y": 0}}]}},
                        "move": [0, 0, z]}
    st = copy.deepcopy(cases()["joint_pull"])
    st["parts"] = {"base": box(5.125, 0), "top": box(6.75, 30)}
    st["mesh"] = {"size": 4}
    st["bolts"][0].pop("nut")
    st["bolts"][0]["thread"] = {"part": "base", "depth": 12}
    parts = joint.mesh_parts(st, tmp_path)
    zb = [float(joint.all_nodes(parts)[:, 2].min()), float(joint.all_nodes(parts)[:, 2].max())]
    (geo,) = joint.bolt_geometry(parts, st["bolts"][0])
    X = joint.all_nodes(parts)
    z_nut = X[geo["nut"], 2]
    top_of_base = zb[0] + 30
    assert np.all(z_nut <= top_of_base + 1e-6) and np.all(z_nut >= top_of_base - 12 - 1e-6)
    assert np.allclose(np.linalg.norm(X[geo["nut"], :2], axis=1), 5.125, atol=0.05)
    assert 30 < geo["grip"] < 42


def test_joint_pictures(pull, tmp_path, monkeypatch):
    """The state of a joint study (every step's displacements, stresses and gasket stress) and its default
    pictures: the setup with the bolts, the gasket at the preload and at the last step, the deformed assembly."""
    from fea_translator import render
    monkeypatch.setenv("FEAT_RENDER", "matplotlib")
    st, parts, text, plan = pull
    X = joint.all_nodes(parts)
    g = plan["gaskets"][0]
    tets = np.concatenate([p["mesh"].tets + p["n0"] for p in parts])
    _, area, _ = joint.gasket_graph(X, tets, [tuple(f) for f in g["slave_faces"]])
    nodes = sorted(area)
    n_top0 = parts[1]["n0"]
    # the gasket stress falls from the bolt outwards as the pull rises; the top block rises
    cps = [{n: max(0.0, 8.0 - k - 0.1 * float(np.linalg.norm(X[n - 1, :2]))) for n in nodes} for k in range(5)]
    disp = [{n: (0.0, 0.0, 0.002 * k * (n > n_top0)) for n in range(1, len(X) + 1)} for k in range(5)]
    frd = _frd(cps, disp)
    res = joint.evaluate(st, parts, plan, _dat(plan, [[20000.0]] * 5), frd)
    joint.save_state(tmp_path, st, parts, plan, res, frd)
    state = render.load_state(tmp_path)
    assert state["fields"]["U_steps"].shape == (5, len(X), 3) and state["fields"]["cp_steps"].shape == (5, len(X))
    assert state["fields"]["U_steps"][4, -1, 2] == pytest.approx(0.008, rel=1e-5)
    assert {b["name"]: b["kind"] for b in state["meta"]["bcs"]} == {"side": "fixed", "pull": "force", "g": "gasket"}
    assert len(state["meta"]["bolts"]) == 1 and state["meta"]["gaskets"]["g"]["normal"] == pytest.approx([0, 0, -1])
    specs = render.default_images(st)
    assert [sp["name"] for sp in specs] == ["setup", "g_preload", "g_last", "displacement"]
    assert validate_study(dict(st, images=True))["ok"] and validate_study(dict(st, images=specs))["ok"]
    out, warns = render.render(tmp_path, True, "pull")
    assert not [w for w in warns if "VTK" not in w and "Matplotlib" not in w], warns
    assert set(out) == {"setup", "g_preload", "g_last", "displacement"}
    assert out["setup"]["bolts"] == 1 and out["setup"]["bcs"]["g"]["triangles"] > 0
    assert out["g_preload"]["max"] == pytest.approx(max(cps[0].values()), rel=1e-5)
    assert out["g_preload"]["clim"] == pytest.approx([0, max(cps[0].values())], rel=1e-5)   # from 0 (open)
    assert out["g_last"]["max"] == pytest.approx(max(cps[4].values()), rel=1e-5)
    assert out["g_preload"]["seal_MPa"] == 1.0                     # the pull gasket's seal.stress, on the colour bar
    assert "step 1 (preload)" in out["g_preload"]["what"] and "pull = 8e+04 N" in out["g_last"]["what"]
    import matplotlib.image as mpimg
    for k in out:
        assert mpimg.imread(out[k]["path"]).std() > 0.02
    # a gasket picture is viewed face on: from below for this downward-facing gasket face
    _, sp = render.at_step(state, {"field": "gasket_stress"})
    assert sp["surface"] == "g" and sp["step"] == 5 and sp["view"] == [-90.0, -90.0]


def test_joint_image_validation():
    st = cases()["joint_pull"]

    def errs(images):
        return " | ".join(validate_study(dict(st, images=images))["errors"])
    assert errs([{"field": "gasket_stress", "step": 3}]) == ""
    assert "step: 1 (the preload) to 9" in errs([{"field": "displacement", "step": 10}])
    assert "drawn on a gasket" in errs([{"field": "gasket_stress", "surface": "side"}])
    assert "not a field of this joint study" in errs([{"field": "temperature"}])
    assert "for modal and buckling" in errs([{"mode": 1}])
    assert "step: for joint studies" in " | ".join(
        validate_study(dict(cases()["cantilever"], images=[{"field": "von_mises", "step": 1}]))["errors"])


def test_gasket_curve_gets_a_minimum_slope():
    pts, raised = joint.gasket_curve({"curve": [[0.02, 0.0], [0.05, 1.0], [0.1, 4.0], [0.3, 35.0]]})
    smin = 0.02 * 35.0 / 0.3
    assert raised and pts[0] == (0.0, 0.0) and pts[1] == (0.02, pytest.approx(0.02 * smin))
    assert all((s2 - s1) / (c2 - c1) >= smin - 1e-9 for (c1, s1), (c2, s2) in zip(pts, pts[1:]))
    assert pts[-1] == (0.3, 35.0)
    assert joint.gasket_curve({"curve": [[0.1, 5.0], [0.2, 20.0]]}) == ([(0.0, 0.0), (0.1, 5.0), (0.2, 20.0)], False)


def _cvg(conts, step=1, inc=1, att=1):
    head = [" SUMMARY OF C0NVERGENCE INFORMATION", "  STEP   INC   ATT  ITER     CONT.   RESID.        CORR.",
            "                             EL.     FORCE         DISP", ""]
    return "\n".join(head + [f"{step:6d}{inc:6d}{att:6d}{i + 1:6d}{c:10d}  0.1000E+08  0.1000E+03  0.0000E+00  0.0000E+00"
                              for i, c in enumerate(conts)]) + "\n"


def test_stall_reason_contact_chatter_and_failed_attempts():
    chatter = [96964 if i % 2 else 66000 + 10 * i for i in range(20)]
    why, step = joint.stall_reason("", _cvg(chatter))
    assert step == 1 and why.startswith("preload did not converge: contact chatter (")
    assert "contact elements switching" in why and "96964" in why and "iteration 20" in why
    settling = [96964 - 2000 * min(i, 10) for i in range(30)]            # contact found, then steady: no chatter
    assert joint.stall_reason("", _cvg(settling)) is None
    assert joint.stall_reason("", _cvg(chatter[:10])) is None            # too few iterations to judge
    sta = ("SUMMARY OF JOB INFORMATION\n  STEP      INC     ATT  ITRS     TOT TIME     STEP TIME      INC TIME\n"
           "     1          1     1U   60  0.000000E+00  0.000000E+00  0.500000E-01\n"
           "     1          1     2U   60  0.000000E+00  0.000000E+00  0.125000E-01\n")
    assert joint.stall_reason(sta, "") is None
    sta3 = sta + "     1          1     3U   60  0.000000E+00  0.000000E+00  0.312500E-02\n"
    why, _ = joint.stall_reason(sta3, "")
    assert why == "preload did not converge: increment 1 of step 1 failed 3 attempts"
    plan = {"steps": [{"factor": 0.0}, {"factor": 0.5}, {"factor": 1.0}], "sweep": {"bc": "inside"}}
    why, step = joint.stall_reason("", _cvg(chatter, step=3), plan)
    assert step == 3 and why.startswith("step 3 (inside at 100 %) did not converge")
    done = ("     1          1     1    2  0.500000E-01  0.500000E-01  0.500000E-01\n"
            "     1          9     1    3  0.100000E+01  0.100000E+01  0.250000E+00\n"
            "     2          1     1U   60  0.100000E+01  0.000000E+00  0.100000E+01\n")
    assert joint.completed_steps(done) == 1 and joint.completed_steps(sta3) == 0


def test_stabilisation_forces_in_the_results(pull):
    st, parts, text, plan = pull
    g = plan["gaskets"][0]
    X = joint.all_nodes(parts)
    tets = np.concatenate([p["mesh"].tets + p["n0"] for p in parts])
    _, area, _ = joint.gasket_graph(X, tets, [tuple(f) for f in g["slave_faces"]])
    frd = _frd([{n: 20000 / 3600 for n in area}] * 5)
    nodes = plan["stabilisation"]["parts"]["top"]
    k = plan["stabilisation"]["k_N_mm"]

    def dat(u):
        out = _dat(plan, [[20000.0]] * 5)
        for s in range(1, 6):
            out += f"\n displacements (vx,vy,vz) for set N_STAB and time  {s:.7E}\n\n"
            out += "\n".join(f"{n:10d}  0.000000E+00  0.000000E+00 {u:13.6E}" for n in nodes) + "\n"
        return out
    res = joint.evaluate(st, parts, plan, dat(0.1), frd)
    stb = res["stabilisation"]
    assert stb["springs"] == 18 and stb["parts"] == {"top": 6}
    assert stb["max_force_N"] == pytest.approx(6 * k * 0.1, rel=1e-4) and stb["fraction_of_preload"] < 1e-4
    assert not any("stabilisation springs carry" in w for w in res.get("warnings", []))
    big = joint.evaluate(st, parts, plan, dat(0.1 * 0.02 * 20000 / (6 * k * 0.1)), frd)   # 2 % of the preload
    assert big["stabilisation"]["fraction_of_preload"] == pytest.approx(0.02, rel=1e-4)
    assert any("stabilisation springs carry 2 % of the preload" in w for w in big["warnings"])


def _fake_ccx(tmp_path, monkeypatch, body: str):
    exe = tmp_path / "fake_ccx"
    exe.write_text("#!/bin/sh\n# fake ccx: -i <stem>\nstem=$2\n" + body + "\n")
    exe.chmod(0o755)
    monkeypatch.setenv("FEAT_CCX", str(exe))
    from fea_translator import runner
    monkeypatch.setattr(runner, "MONITOR_EVERY", 0.2)


def test_a_chattering_preload_stops_early_with_its_reason(tmp_path, monkeypatch):
    import time
    from fea_translator.runner import SolverError
    chatter = _cvg([96964 if i % 2 else 66000 for i in range(20)])
    (tmp_path / "cvg.txt").write_text(chatter)
    _fake_ccx(tmp_path, monkeypatch, f"cp {tmp_path}/cvg.txt $stem.cvg\necho '1 1 1 0' > $stem.sta\nsleep 60")
    st = copy.deepcopy(cases()["joint_pull"])
    st["sweep"]["steps"] = 2
    t0 = time.time()
    with pytest.raises(SolverError, match=r"preload did not converge: contact chatter .*stabilisation springs were on"):
        joint.solve("pull", st, tmp_path / "run", "ccx", 120)
    assert time.time() - t0 < 30


def test_a_stall_after_the_preload_keeps_the_steps_done(tmp_path, monkeypatch):
    """CalculiX stopped in step 3 (chatter): the results of steps 1 and 2, with the reason."""
    st = copy.deepcopy(cases()["joint_pull"])
    st["sweep"]["steps"] = 4
    parts = joint.mesh_parts(st, tmp_path / "pre")
    _, plan = joint.write("pull", st, parts)
    g = plan["gaskets"][0]
    X = joint.all_nodes(parts)
    tets = np.concatenate([p["mesh"].tets + p["n0"] for p in parts])
    _, area, _ = joint.gasket_graph(X, tets, [tuple(f) for f in g["slave_faces"]])
    src = tmp_path / "src"
    src.mkdir()
    (src / "dat").write_text(_dat(plan, [[20000.0], [24000.0]]))
    (src / "frd").write_text(_frd([{n: 5.0 for n in area}, {n: 3.0 for n in area}]))
    (src / "sta").write_text("     1          5     1    2  0.100000E+01  0.100000E+01  0.250000E+00\n"
                             "     2          1     1    3  0.200000E+01  0.100000E+01  0.100000E+01\n")
    (src / "cvg").write_text(_cvg([96964 if i % 2 else 66000 for i in range(20)], step=3))
    _fake_ccx(tmp_path, monkeypatch, f"for x in dat frd sta cvg; do cp {src}/$x $stem.$x; done\nsleep 60")
    res = joint.solve("pull", st, tmp_path / "run", "ccx", 120)
    assert res["stopped"]["completed_steps"] == 2 and "step 3" in res["stopped"]["why"]
    assert [s["step"] for s in res["steps"]] == [1, 2] and res["steps"][1]["bolts"]["b"]["max_N"] == pytest.approx(24000)
    assert any("the results stop at step 2" in w for w in res["warnings"])


def test_a_surface_gmsh_cannot_mesh_is_tried_with_other_algorithms():
    """A casing-to-head joint (carbonio, 2026.11.01): "Invalid boundary mesh (overlapping facets) on surface 80";
    with 2026.11.02.6 the retry returned an empty mesh (ValueError at the nodes): every result is checked, and the
    volume gets the other 3D algorithms when the surface's 2D algorithms do not help."""
    from fea_translator.mesher import SURFACE_ALGORITHMS, VOLUME_ALGORITHMS, generate

    class FakeGmsh:
        """generate fails `fails` times with msg, then succeeds; empty: the success leaves no mesh."""
        def __init__(self, fails, msg="Invalid boundary mesh (overlapping facets) on surface 80", empty=False):
            self.fails, self.msg, self.empty = fails, msg, empty
            self.algs, self.opts, self.calls, self.meshed = [], [], 0, False
            self.model = self.mesh = self.option = self

        def generate(self, dim):
            self.calls += 1
            if self.fails:
                self.fails -= 1
                raise Exception(self.msg)
            self.meshed = not self.empty

        def getElements(self, dim):
            return ([4], [[1]], [[1, 2, 3, 4]]) if self.meshed else ([], [], [])

        def clear(self):
            self.meshed = False

        def setAlgorithm(self, dim, tag, val):
            self.algs.append((dim, tag, val))

        def getNumber(self, name):
            return 1

        def setNumber(self, name, val):
            self.opts.append((name, val))

        def getBoundingBox(self, dim, tag):
            return [0, 0, 0, 2, 4, 6]
    g, notes = FakeGmsh(2), []
    generate(g, 3, notes)
    assert g.calls == 3 and g.algs == [(2, 80, 1), (2, 80, 5)] and g.opts == []
    assert "surface 80 (near [1.0, 2.0, 3.0] mm)" in notes[0] and "MeshAdapt" in notes[0] and "Delaunay" in notes[1]
    # the surface's three 2D algorithms fail too: the volume with HXT (then Frontal)
    g, notes = FakeGmsh(len(SURFACE_ALGORITHMS) + 1), []
    generate(g, 3, notes)
    assert g.opts == [("Mesh.Algorithm3D", 10)] and "HXT 3D algorithm" in notes[-1]
    # Gmsh returns without a mesh and no error: the next 3D algorithm, not an empty mesh
    g, notes = FakeGmsh(0, empty=True), []
    with pytest.raises(MeshError, match=r"Gmsh could not mesh the part \(Gmsh returned without tetrahedra\); tried 3D algorithms"):
        generate(g, 3, notes)
    assert [o[1] for o in g.opts] == [a for a, _ in VOLUME_ALGORITHMS if a != 1]
    with pytest.raises(MeshError, match=r"could not mesh surface 80 \(near \[1.0, 2.0, 3.0\] mm\) .*tried 2D algorithms"):
        generate(FakeGmsh(99), 3, [])

    class Other(FakeGmsh):
        def generate(self, dim):
            raise Exception("some other failure")
    with pytest.raises(MeshError, match="some other failure"):
        generate(Other(0), 3, [])



def test_a_nut_on_a_fixed_face(tmp_path):
    """joint_two_bolt's nuts bear on the body's fixed bottom face: every nut node is held; the equations still take
    the bolt's own end node as the dependent DOF (CalculiX: '*ERROR in cascade: the DOF corresponding to ...')."""
    st = cases()["joint_two_bolt"]
    parts = joint.mesh_parts(st, tmp_path)
    text, plan = joint.write("tb", st, parts)
    held = set(plan["sets"]["feet"]["nodes"])
    eq = text.splitlines()
    deps = [tuple(int(v) for v in eq[i + 2].split(",")[:2]) for i, ln in enumerate(eq) if ln.startswith("*EQUATION")]
    assert len(deps) == 12 and len(set(deps)) == 12 and not {n for n, _ in deps} & held
    nut = [i for i, ln in enumerate(eq) if ln.startswith("*EQUATION")][0]   # bolt 1, direction 1, the nut's equation
    terms = " ".join(eq[nut + 2:nut + 2 + 4]).replace(" ", "").split(",")
    assert any(int(n) in held for n in terms[3::3] if n.isdigit())
    assert list(plan["stabilisation"]["parts"]) == ["cover"]
    # the soft-start curve as its secant at the preload: 2 x 12.5 kN over 7200 mm2
    law = plan["gaskets"][0]["law"]
    assert law["kind"] == "secant" and law["at_MPa"] == pytest.approx(25000 / 7200) and law["curve_raised"]
    assert law["stiffness_MPa_mm"] == pytest.approx(law["at_MPa"] / joint.closure_at(st["gaskets"][0], law["at_MPa"]))
    assert "TABULAR" not in text and f"LINEAR\n{joint._g(law['stiffness_MPa_mm'])}\n" in text


def test_an_unexpected_error_says_where(tmp_path, monkeypatch):
    """A bare ValueError (the S50 casing-to-head joint, 2026.11.02.6) comes back with where it happened and the stage."""
    from fea_translator import runner

    def boom(*a, **k):
        import numpy as np_
        return np_.zeros(0).max()
    monkeypatch.setattr(runner, "solve_once", boom)
    r = runner.run_study("x", cases()["joint_pull"], tmp_path)
    assert r["why"] == "ValueError" and "zero-size array" in r["error"]
    assert r["stage"] and any("test_joint.py" in w and "boom" in w for w in r["where"])


def test_mesh_min_size_and_curvature():
    """A detailed part meshed coarsely: min_size and curvature (0: off) for any study, per part in a joint."""
    from fea_translator.study import mesh_errors
    assert mesh_errors({"size": 10, "min_size": 4, "curvature": 0}) == ([], [])
    e, _ = mesh_errors({"size": 10, "min_size": 12, "curvature": 2.5, "fine": 1}, "parts.head.mesh")
    assert any("unknown keys ['fine']" in m for m in e) and any("above 'size'" in m for m in e)
    assert any("'curvature' is elements per full circle" in m for m in e) and all(m.startswith("parts.head.mesh") for m in e)
    st = copy.deepcopy(cases()["joint_cover"])
    st["parts"]["cover"]["mesh"] = {"min_size": 3}
    assert validate_study(st)["ok"]
    st["parts"]["cover"]["mesh"] = {"min_size": -1}
    assert "parts.cover.mesh: 'min_size' must be positive" in " | ".join(validate_study(st)["errors"])


def test_curvature_off_and_min_size_coarsen_a_filleted_part(tmp_path):
    pytest.importorskip("gmsh")
    pytest.importorskip("build123d_translator")
    from fea_translator.mesher import mesh_part
    goal = {"features": [{"kind": "base_box", "params": {"w": 80, "d": 60, "h": 20}},
                         {"kind": "hole", "params": {"r": 1.5, "x": 20, "y": 10}},
                         {"kind": "fillet_vertical", "params": {"r": 2}}]}
    fine, i1 = mesh_part({"part": {"goal": goal}, "mesh": {"size": 10}}, tmp_path / "a")
    coarse, i2 = mesh_part({"part": {"goal": goal}, "mesh": {"size": 10, "min_size": 5, "curvature": 0}}, tmp_path / "b")
    assert i2["curvature"] == 0 and i2["min_size"] == 5 and i1["curvature"] == 12
    assert len(coarse.nodes) < 0.7 * len(fine.nodes), (len(coarse.nodes), len(fine.nodes))


def test_get_deck_and_estimates_of_a_joint(tmp_path, monkeypatch):
    """get_deck meshed only single parts (KeyError: 'part' on a joint); the node estimate was unknown for joints."""
    from pathlib import Path

    from fea_translator import api, executor
    monkeypatch.setenv("FEAT_OUT", str(tmp_path))
    st = copy.deepcopy(cases()["joint_pull"])
    r = api.deck(st, "pull")
    assert "error" not in r, r
    parts = r["mesh"]["parts"]
    assert set(parts) == {"base", "top"} and r["mesh"]["nodes"] == parts["base"]["nodes"] + parts["top"]["nodes"]
    assert r["bolts"]["b"]["count"] == 1 and r["bolts"]["b"]["grip_mm"] == [60.0]
    assert r["gaskets"]["g"]["gap_mm"]["median_mm"] == 0.0 and r["gaskets"]["g"]["law"]["kind"] == "stiffness"
    assert r["stabilisation"]["parts"] == {"top": 6} and "gasket g" in r["selection"] and Path(r["path"]).is_file()
    assert executor.nodes_hint(st) == r["mesh"]["nodes"] == executor.estimate_nodes(st)   # box parts: exact
    st["parts"]["top"]["mesh"] = {"order": 1}
    assert executor.nodes_hint(st) < r["mesh"]["nodes"]                                     # the part's own mesh
