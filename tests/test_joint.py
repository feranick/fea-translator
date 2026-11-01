"""Bolted, gasketed joints (joint.py): the deck (contact, bolt connectors, pretension, steps), the gasket's ring
stress, reading multi-step .frd/.dat output, leak and lift-off from synthetic results, validation. Offline: the
CalculiX runs are the joint_pull and joint_cover benchmarks (test_fea.test_benchmark_calculix).
Version: 2026.11.01.4
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


def test_deck_structure(pull):
    st, parts, text, plan = pull
    up = text.upper()
    count = lambda k: len(re.findall(rf"^\*{k}\b", up, re.M))    # noqa: E731
    assert count("CONTACT PAIR") == 1 and count("PRE-TENSION SECTION") == 1 and count("BEAM SECTION") == 1
    assert count("STEP") == 1 + 4 and count("END STEP") == 1 + 4
    assert count("EQUATION") == 6                                   # 3 directions x (head, nut)
    assert up.count("*BOUNDARY, FIXED") == 4 and up.count("*CLOAD, OP=NEW") == 4
    assert "*EL FILE" not in up and up.count("*NODE PRINT, NSET=N_BOLTENDS") == 5
    assert "PRESSURE-OVERCLOSURE=LINEAR" in up and "*FRICTION" in up and "*CONTACT FILE" in up
    assert [s["factor"] for s in plan["steps"]] == [0.0, 0.25, 0.5, 0.75, 1.0]
    b = plan["bolts"][0]
    assert b["preload_N"] == 20000 and len(b["bolts"]) == 1
    assert b["bolts"][0]["grip_mm"] == pytest.approx(60, abs=1e-6)     # top face to bottom face
    assert plan["gaskets"][0]["area"] == pytest.approx(3600, rel=1e-6)
    # the beam's end nodes and the pretension node come after the mesh nodes, the beam after the tets
    n_mesh = sum(len(p["mesh"].nodes) for p in parts)
    assert min(plan["pret_nodes"]) > n_mesh
    # every equation line: terms of (node, dof, coefficient), the first one's node not held by a support
    held = set(plan["sets"]["side"]["nodes"])
    eq = [ln for ln in text.splitlines()]
    for i, ln in enumerate(eq):
        if ln.upper().startswith("*EQUATION"):
            nterms = int(eq[i + 1])
            first = eq[i + 2].split(",")
            assert int(first[0]) not in held and nterms >= 2


def test_gasket_law_tabular_and_linear():
    lines = joint.gasket_law({"curve": [[0.2, 30], [0.05, 2]], "friction": 0.3})
    up = "\n".join(lines).upper()
    assert "PRESSURE-OVERCLOSURE=TABULAR" in up
    rows = [ln for ln in lines if re.fullmatch(r"\s*[\d.eE+-]+\s*,\s*[\d.eE+-]+\s*", ln)]
    vals = [tuple(float(v) for v in r.split(",")) for r in rows]
    assert vals[:3] == [(0.0, 0.0), (2.0, 0.05), (30.0, 0.2)]           # (pressure, overclosure), sorted, from 0
    assert any(ln.strip().startswith("0.3") for ln in lines)
    assert "PRESSURE-OVERCLOSURE=LINEAR" in "\n".join(joint.gasket_law({"stiffness": 50})).upper()


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


def _frd(steps):
    """An ASCII .frd with one CONTACT block per step: steps = [{node: cpress}]."""
    out = ["    1C", "    1UUSER"]
    for k, pm in enumerate(steps, start=1):
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
    """The bolt ends' displacements per step that give these bolt forces: forces = [[f per bolt] per step]."""
    out = []
    bolts = [(rec, b) for rec in plan["bolts"] for b in rec["bolts"]]
    for k, fs in enumerate(forces, start=1):
        out += ["", f" displacements (vx,vy,vz) for set N_BOLTENDS and time  {k:.7E}", ""]
        for (rec, b), f in zip(bolts, fs):
            a = np.asarray(rec["axis"], float)
            u2 = 0.01 * np.ones(3) + a * (f - rec["preload_N"]) / b["k_N_mm"]
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
    assert "images" in errs(lambda s: s.update(images=True))


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
