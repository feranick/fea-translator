"""Assemblies (analysis "assembly", joint.py): steps and their loads, contacts (hard, penalty, bonded, clearance), nuts
on a meshed shaft, probes, part stresses, modal steps on the loaded state, validation, pictures. Offline: the CalculiX
runs are the assembly_* benchmarks (test_fea.test_benchmark_calculix, FEAT_SLOW_TESTS=1).
Version: 2026.11.09.2
"""
import copy
import math

import numpy as np
import pytest

from fea_translator import joint
from fea_translator.bench import cases
from fea_translator.study import validate_study


def _boxes(**extra):
    """Two boxes stacked (base fixed), a side force on the top, a contact between them."""
    st = {"analysis": "assembly",
          "parts": {"base": {"part": {"box": [40, 40, 20], "origin": [-20, -20, 0], "divisions": [4, 4, 2]}},
                    "top": {"part": {"box": [40, 40, 20], "origin": [-20, -20, 20], "divisions": [4, 4, 2]}}},
          "bcs": [{"kind": "fixed", "name": "root", "faces": {"part": "base", "nz": -1}},
                  {"kind": "force", "name": "side", "faces": {"part": "top", "nz": 1}, "fx": 100.0},
                  {"kind": "pressure", "name": "clamp", "faces": {"part": "top", "nz": 1}, "p": 5.0}],
          "contacts": [{"name": "c", "faces": {"part": "top", "nz": -1}, "with": {"part": "base", "nz": 1}}],
          "probes": [{"name": "tip", "faces": {"part": "top", "nz": 1}}],
          "steps": [{"kind": "static", "loads": ["side"]}, {"kind": "modal", "modes": 3}]}
    st.update(extra)
    return st


def test_steps_and_their_loads():
    st = _boxes(steps=[{"kind": "static", "name": "a", "loads": ["side"], "scale": 2}, {"kind": "static"},
                       {"kind": "sweep", "bc": "side", "steps": 2}, {"kind": "modal", "modes": 4}])
    ds = joint.deck_steps(st)
    assert [s["kind"] for s in ds] == ["static", "static", "sweep", "sweep", "modal"]
    assert ds[0]["loads"] == {"clamp": 1.0, "side": 2.0} and ds[1]["loads"] == {"clamp": 1.0}   # no carry-over
    assert ds[3]["loads"] == {"clamp": 1.0, "side": 1.0} and ds[4]["modes"] == 4 and ds[0]["what"] == "a"
    nut = dict(_boxes(), nuts=[{"name": "n"}])
    assert [s["kind"] for s in joint.deck_steps(nut)][:2] == ["preload", "static"]               # a preload first
    j = cases()["joint_pull"]
    assert [s["kind"] for s in joint.deck_steps(j)] == ["preload"] + ["sweep"] * 8


@pytest.fixture(scope="module")
def boxes(tmp_path_factory):
    st = _boxes()
    parts = joint.mesh_parts(st, tmp_path_factory.mktemp("asm"))
    text, plan = joint.write("asm", st, parts)
    return st, parts, text, plan


def test_assembly_deck(boxes):
    st, parts, text, plan = boxes
    up = text.upper()
    steps = up.split("*STEP")
    assert len(steps) == 2 and "*FREQUENCY" not in up and "PERTURBATION" not in up   # modal: a separate run
    assert [s["kind"] for s in plan["steps"]] == ["static", "modal"] and plan["steps"][1]["ccx_step"] is None
    assert "*SECTION PRINT" not in up and "*NODE PRINT, NSET=N_PR_TIP, FREQUENCY=1000000\nU\n" in steps[1]
    assert "PRESSURE-OVERCLOSURE=LINEAR" in steps[0] and "*CONTACT PAIR, INTERACTION=CI_C" in steps[0]
    law = plan["contacts"][0]["law"]
    assert law["kind"] == "hard" and law["stiffness_MPa_mm"] == pytest.approx(10 * 210000 / law["h_mm"])
    # no preload: the friction at its full stick slope from the start, never changed
    assert f"*FRICTION\n0.2, {joint._g(law['stiffness_MPa_mm'] / 100)}\n" in steps[0] and "CHANGE FRICTION" not in up
    assert "*CLOAD\n" in steps[1] and "*DLOAD\n" in steps[1] and "0.05, 1., 1E-05, 0.25" in steps[1]
    assert "N_PRETENSION" not in up and "N_BOLTENDS" not in up
    # the top box (no support) gets springs; the base (fixed) none
    assert list(plan["stabilisation"]["parts"]) == ["top"]
    assert plan["stabilisation"]["k_N_mm"] == pytest.approx(1e-9 * 210000 * parts[0]["info"]["size"])
    assert plan["probes"]["tip"] and plan["contacts"][0]["gap"]["median_mm"] == 0.0


def test_bonded_clearance_and_held_by_bonding(tmp_path):
    st = _boxes(contacts=[{"name": "c", "faces": {"part": "top", "nz": -1}, "with": {"part": "base", "nz": 1},
                           "law": "bonded"}])
    parts = joint.mesh_parts(st, tmp_path / "a")
    text, plan = joint.write("b", st, parts)
    assert "PRESSURE-OVERCLOSURE=TIED\n" in text and "*FRICTION\n1., " in text   # TIED needs a stick slope
    assert plan["stabilisation"]["springs"] == 0          # tied to the fixed base: held
    st2 = _boxes(contacts=[{"name": "c", "faces": {"part": "top", "nz": -1}, "with": {"part": "base", "nz": 1},
                            "law": {"stiffness": 1000}, "friction": 0.0, "clearance": 0.01}])
    text2, plan2 = joint.write("p", st2, joint.mesh_parts(st2, tmp_path / "b"))
    assert "*CLEARANCE, MASTER=CM_C, SLAVE=CS_C, VALUE=0.01" in text2 and "*FRICTION" not in text2
    assert plan2["contacts"][0]["law"] == {"kind": "penalty", "stiffness_MPa_mm": 1000.0}


def _frd(contact_steps, disp_steps, modes=None, modal_step=None):
    out = ["    1C"]
    for k, (cm, dm) in enumerate(zip(contact_steps, disp_steps), start=1):
        out += [f"    1PSTEP              {k}           1           {k}", " -4  DISP        4    1",
                " -5  D1          1    2    1    0", " -5  D2          1    2    2    0", " -5  D3          1    2    3    0"]
        out += [f" -1{n:10d}" + "".join(f"{x:12.5E}" for x in u) for n, u in sorted(dm.items())] + [" -3"]
        out += [f"    1PSTEP              {k}           1           {k}", " -4  CONTACT     6    1"]
        out += [f" -5  {c:<10s}1    1    {i + 1}    0" for i, c in enumerate(("COPEN", "CSLIP1", "CSLIP2", "CPRESS", "CSHEAR1", "CSHEAR2"))]
        out += [f" -1{n:10d}" + "".join(f"{x:12.5E}" for x in v) for n, v in sorted(cm.items())] + [" -3"]
    for j, mm in enumerate(modes or []):
        out += [f"    1PSTEP              {j + 1}           1           {modal_step}", " -4  DISP        4    1",
                " -5  D1          1    2    1    0", " -5  D2          1    2    2    0", " -5  D3          1    2    3    0"]
        out += [f" -1{n:10d}" + "".join(f"{x:12.5E}" for x in u) for n, u in sorted(mm.items())] + [" -3"]
    return "\n".join(out + [" 9999"]) + "\n"


EIGEN = """
                        S T E P       2


     E I G E N V A L U E   O U T P U T

 MODE NO    EIGENVALUE                       FREQUENCY
                                     REAL PART            IMAGINARY PART
                           (RAD/TIME)      (CYCLES/TIME     (RAD/TIME)

      1   0.1000000E+07   0.1000000E+04   0.1591549E+03   0.0000000E+00
      2   0.1100000E+07   0.1048809E+04   0.1669227E+03   0.0000000E+00
      3   0.4000000E+08   0.6324555E+04   0.1006584E+04   0.0000000E+00

     P A R T I C I P A T I O N   F A C T O R S
"""


def test_assembly_results(boxes):
    st, parts, text, plan = boxes
    X = joint.all_nodes(parts)
    tets = np.concatenate([p["mesh"].tets + p["n0"] for p in parts])
    area = joint._node_areas(X, tets, plan["contacts"][0]["slave_faces"])
    nodes = sorted(area)
    # contact closed with 5 MPa, half of the nodes slipping (shear at the friction limit), the rest sticking
    cm = {n: (0.0, 0.001 * (i % 2), 0.0, 5.0, (1.0 if i % 2 else 0.2), 0.0) for i, n in enumerate(nodes)}
    top0 = parts[1]["n0"]
    dm = {n: (0.01 * (n > top0), 0.0, 0.0) for n in range(1, len(X) + 1)}
    modes = [{n: (1.0 * (n > top0), 0.0, 0.0) for n in range(1, len(X) + 1)}] * 3
    frd = _frd([cm], [dm])
    sof = ("\n statistics for surface set CS_C and time  0.1000000E+01\n\n"
           "   total surface force (fx,fy,fz) and moment about the origin(mx,my,mz)\n\n"
           "   -1.000000E+01  0.000000E+00 -7.900000E+03  0.000000E+00  0.000000E+00  0.000000E+00\n\n"
           "   center of gravity and mean normal\n\n"
           "    0.000000E+00  0.000000E+00  2.000000E+01  0.000000E+00  0.000000E+00 -1.000000E+00\n\n"
           "   moment about the center of gravity(mx,my,mz)\n\n"
           "    0.000000E+00  0.000000E+00  0.000000E+00\n\n"
           "   area,  normal force (+ = tension) and shear force (size)\n\n"
           "    1.600000E+03 -7.900000E+03  1.000000E+01\n")
    from fea_translator import results as R
    eig = R.parse_dat(EIGEN)["eigen"]
    mfrd = _frd([], [], modes, modal_step=1)
    runs = {2: {"eigen": eig, "tied": ["c"], "free": [], "slipping": ["c"], "frd": mfrd}}
    res = joint.evaluate(st, parts, plan, EIGEN, frd, runs)
    s0 = res["steps"][0]
    c = s0["contacts"]["c"]
    assert c["pressure_force_N"] == pytest.approx(5.0 * 1600, rel=1e-6) and c["closed_fraction"] == 1.0
    assert c["force_from"] == "element forces" and abs(c["force_N"]) < 1e-6   # a rigid shift: no force through the face
    assert 0.3 < c["stick_fraction"] < 0.7 and c["max_slip_mm"] == pytest.approx(0.001)
    assert s0["probes"]["tip"]["ux"] == pytest.approx(0.01) and s0["kind"] == "static" and s0["loads"]["side"] == 1.0
    # probes from CalculiX's printout of their nodes when it is there (as the bolt ends)
    pr = "\n displacements (vx,vy,vz) for set N_PR_TIP and time  0.1000000E+01\n\n" + "\n".join(
        f"{n:10d}  5.000000E-01  0.000000E+00  0.000000E+00" for n in plan["probes"]["tip"]) + "\n"
    assert joint.evaluate(st, parts, plan, pr + EIGEN, frd, runs)["steps"][0]["probes"]["tip"]["ux"] == 0.5
    assert s0["parts"]["top"]["max_displacement_mm"] == pytest.approx(0.01) and s0["parts"]["top"]["peak_von_mises_MPa"] < 1e-6
    assert res["frequencies"] == pytest.approx([159.1549, 166.9227, 1006.584])
    m = res["modal"][0]
    assert m["step"] == 2 and m["after_step"] == 1 and m["contacts"]["c"]["closed_fraction"] == 1.0 and m["tied"] == ["c"]
    assert any("partly slipping" in w for w in res["warnings"])
    assert joint.modal_ties(plan, s0) == ({"c": None}, [], ["c"], {})            # closed all over: tied whole
    some = c["_closed_nodes"][:5]                                                 # closed on part of it: tied there
    t, f, sl, part = joint.modal_ties(plan, {"contacts": {"c": dict(c, closed_fraction=0.2, _closed_nodes=some)}})
    assert t == {"c": some} and f == [] and part == {"c": [5, 0.2]}
    assert joint.modal_ties(plan, {"contacts": {"c": dict(c, closed_fraction=0.0, _closed_nodes=[])}})[1] == ["c"]  # open: free
    # a part with no tied path to a support floats: its rigid-body modes are left out, with a warning
    assert joint.free_groups(plan, {"c": None}) == [] and joint.free_groups(plan, {}) == [["top"]]
    rb = {2: dict(runs[2], eigen=[{"hz": 0.22}, {"hz": 0.31}] + eig, tied=[], free=["c"], floating=[["top"]],
                  extra_modes=6)}
    r2 = joint.evaluate(st, parts, plan, EIGEN, frd, rb)
    assert r2["modal"][0]["rigid_body_Hz"] == [0.22, 0.31] and r2["frequencies"] == pytest.approx([159.1549, 166.9227, 1006.584])
    assert any("part(s) top have no tied path to a support (their contacts c were open at step 1)" in w for w in r2["warnings"])
    deck, _ = joint.write("m", st, parts, modal={"modes": 3, "ties": {"c": some}, "tol": 0.1})   # tied at those nodes
    assert "*SURFACE, NAME=CT_C, TYPE=NODE\nCTN_C\n*TIE, NAME=T_C, POSITION TOLERANCE=0.1\nCT_C, CM_C" in deck
    assert "*NSET, NSET=CTN_C\n" + ", ".join(str(q) for q in some) in deck
    lo = {2: dict(runs[2], eigen=[{"hz": 0.5}] + eig)}                            # not floating, yet below 1 Hz: said
    assert any("frequencies below 1 Hz (0.5)" in w for w in joint.evaluate(st, parts, plan, EIGEN, frd, lo)["warnings"])
    assert "leak" not in res and "bolt_preload_error" not in res and res["contacts"]["c"]["law"]["kind"] == "hard"
    # pictures: the state keeps the static step for the modal step, and the mode shapes
    from fea_translator import render
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    joint.save_state(d, st, parts, plan, res, frd, mfrd)
    state = render.load_state(d)
    assert state["fields"]["modes"].shape == (3, len(X), 3)
    assert np.allclose(state["fields"]["U_steps"][1], state["fields"]["U_steps"][0])
    specs = render.default_images(st)
    assert [s["name"] for s in specs] == ["setup", "c_pressure", "displacement", "mode1", "mode2", "mode3"]
    assert validate_study(dict(st, images=specs))["ok"], validate_study(dict(st, images=specs))["errors"]


def test_assembly_validation():
    def errs(st):
        return " | ".join(validate_study(st)["errors"])
    assert errs(_boxes()) == ""
    assert "steps: a list" in errs({k: v for k, v in _boxes().items() if k != "steps"})
    assert "kind: preload | static | sweep | modal" in errs(_boxes(steps=[{"kind": "dynamic"}]))
    assert "a modal step follows a static" in errs(_boxes(steps=[{"kind": "modal"}]))
    assert "a preload step needs bolts or nuts" in errs(_boxes(steps=[{"kind": "preload"}, {"kind": "static"}]))
    assert "names of load bcs" in errs(_boxes(steps=[{"kind": "static", "loads": ["root"]}]))
    assert ".law: hard" in errs(_boxes(contacts=[{"name": "c", "faces": {"part": "top", "nz": -1},
                                                 "with": {"part": "base", "nz": 1}, "law": "glued"}]))
    assert "bonded contact takes no clearance" in errs(_boxes(contacts=[{"name": "c", "faces": {"part": "top", "nz": -1},
                                                                        "with": {"part": "base", "nz": 1},
                                                                        "law": "bonded", "clearance": 0.1}]))
    assert "meet only through contacts" in errs(_boxes(contacts=[]))
    assert "names used twice" in errs(_boxes(probes=[{"name": "c", "faces": {"part": "top", "nz": 1}}]))
    assert "bears_on" in errs(_boxes(nuts=[{"name": "n", "size": "M20", "preload": 1000, "at": [[0, 0, 40]],
                                            "thread": {"part": "base", "depth": 10}}]))
    assert "sweep: in an assembly" in errs(_boxes(sweep={"bc": "side"}))
    no_rho = _boxes()
    no_rho["parts"]["top"]["material"] = {"E": 70000, "nu": 0.3}
    assert "needs a density" in errs(no_rho)
    for name in ("assembly_clamp", "assembly_press_fit", "assembly_bonded_beam", "assembly_clamped_beam",
                 "assembly_partial_tie"):
        assert validate_study(cases()[name])["ok"], name
    assert errs(_boxes(contacts=[{"name": "c", "faces": {"part": "top", "nz": -1}, "with": {"part": "base", "nz": 1},
                                  "clearance": 0.05}])) == ""


def test_a_nut_on_a_meshed_shaft(tmp_path):
    """assembly_clamp: the nut bears on the hub's top face and grips the shaft above it; it pushes them apart."""
    pytest.importorskip("gmsh")
    pytest.importorskip("build123d_translator")
    st = copy.deepcopy(cases()["assembly_clamp"])
    st["mesh"] = {"size": 8}
    parts = joint.mesh_parts(st, tmp_path)
    text, plan = joint.write("clamp", st, parts)
    rec = plan["bolts"][0]
    assert rec["kind"] == "nut" and rec["sign"] == -1.0 and rec["axis"] == pytest.approx([0, 0, 1])
    b = rec["bolts"][0]
    assert b["B1"][2] == pytest.approx(80, abs=1e-6) and 80 < b["B2"][2] < 95
    X = joint.all_nodes(parts)
    (geo,) = joint.bolt_geometry(parts, joint.connectors(st)[0])
    assert np.allclose(np.linalg.norm(X[geo["nut"], :2], axis=1), 15, atol=0.05)     # the shaft's surface
    assert f"\n{b['pret_node']}, 1, -40000\n" in text                                 # pushes the faces apart
    assert set(plan["stabilisation"]["parts"]) == {"sleeve", "hub"}


def test_the_modal_run_deck(boxes):
    """The frequency run: the closed pairs tied, the supports, no contact pairs, no loads."""
    st, parts, text, plan = boxes
    t, pl = joint.write("asm_modal2", st, parts, modal={"modes": 5, "ties": {"c"}, "tol": 1.0})
    up = t.upper()
    assert "*TIE, NAME=T_C, POSITION TOLERANCE=1\nCS_C, CM_C" in up and "*CONTACT PAIR" not in up
    assert up.count("*STEP") == 1 and "*FREQUENCY\n5\n*BOUNDARY\nN_ROOT, 1, 3\nN_STABGROUND, 1, 3" in up
    assert "*CLOAD" not in up and "*DLOAD" not in up and pl["steps"][0]["kind"] == "modal"
    t2, _ = joint.write("asm_modal_free", st, parts, modal={"modes": 5, "ties": set(), "tol": 1.0})
    assert "*TIE" not in t2


def test_section_forces_parse():
    txt = ("\n statistics for surface set CS_A and time  0.2000000E+01\n\n"
           "   total surface force (fx,fy,fz) and moment about the origin(mx,my,mz)\n\n"
           "    1.0E+00  2.0E+00  3.0E+00  0.0E+00  0.0E+00  0.0E+00\n\n"
           "   center of gravity and mean normal\n\n"
           "    0.0E+00  0.0E+00  0.0E+00  0.0E+00  1.0E+00  0.0E+00\n\n"
           "   moment about the center of gravity(mx,my,mz)\n\n    0.0E+00  0.0E+00  0.0E+00\n\n"
           "   area,  normal force (+ = tension) and shear force (size)\n\n    5.0E+01  2.0E+00  3.1E+00\n")
    (s,) = joint.section_forces(txt)
    assert s == {"set": "CS_A", "time": 2.0, "force": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0], "area": 50.0,
                 "normal_force": 2.0, "shear_force": 3.1}


def test_contact_force_from_element_forces():
    """The element forces at a face's nodes add up to the stress over it: a box stretched uniformly in z."""
    from fea_translator.meshdata import box_mesh
    m = box_mesh(40, 30, 20, 4, 3, 2, order=2, origin=[0, 0, 0])
    eps = 1e-4
    U = np.column_stack([np.zeros(len(m.nodes)), np.zeros(len(m.nodes)), eps * m.nodes[:, 2]])
    f = joint.nodal_internal_forces(m, U, {"E": 210000.0, "nu": 0.3})
    E, nu = 210000.0, 0.3
    szz = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)) * eps                 # (lambda + 2 mu) eps: no lateral strain
    top = np.nonzero(np.isclose(m.nodes[:, 2], 20))[0]
    assert f[top].sum(axis=0) == pytest.approx([0, 0, szz * 1200], rel=1e-9, abs=1e-6)
    lo, hi = m.nodes.min(axis=0) + 1e-9, m.nodes.max(axis=0) - 1e-9
    inner = np.nonzero(np.all((m.nodes > lo) & (m.nodes < hi), axis=1))[0]
    assert len(inner) and np.abs(f[inner]).max() < 1e-6 * szz * 1200     # interior nodes: in balance


def test_face_node_normals_point_outward():
    from fea_translator.meshdata import box_mesh
    m = box_mesh(40, 30, 20, 4, 3, 2, order=2, origin=[0, 0, 0])
    tags = m.select({"nz": 1})
    nodes, N = joint.face_node_normals(m.nodes, m.tets, m.faces_of(tags))
    assert np.allclose(m.nodes[nodes, 2], 20) and np.allclose(N, [0, 0, 1])
    nodes2, N2 = joint.face_node_normals(m.nodes, m.tets, m.faces_of(m.select({"nx": -1})))
    assert np.allclose(N2, [-1, 0, 0])


def test_an_interference_fit_overlaps_the_parts(tmp_path):
    """A negative clearance: the slave face's nodes moved by it along their outward normals (into the master),
    node-to-surface contact, no *CLEARANCE (face-to-face contact did not enforce interference fits in CalculiX:
    tools/interference_matrix.py, mochi 2026.11.05)."""
    st = _boxes(contacts=[{"name": "c", "faces": {"part": "top", "nz": -1}, "with": {"part": "base", "nz": 1},
                           "clearance": -0.01}])
    assert validate_study(st)["ok"]
    parts = joint.mesh_parts(st, tmp_path)
    text, plan = joint.write("fit", st, parts)
    assert "TYPE=NODE TO SURFACE" in text and "*CLEARANCE" not in text
    c = plan["contacts"][0]
    assert c["contact_type"] == "node to surface" and c["interference_nodes_moved"] > 0
    lines = text.splitlines()
    i = lines.index("*NODE, NSET=NALL")
    X = np.array([[float(v) for v in ln.split(",")[1:]] for ln in lines[i + 1:i + 1 + sum(len(p["mesh"].nodes) for p in parts)]])
    top = parts[1]
    Z = X[top["n0"]:top["n0"] + len(top["mesh"].nodes), 2]
    bottom = np.isclose(top["mesh"].nodes[:, 2], 20)
    assert np.allclose(Z[bottom], 20 - 0.01) and np.allclose(Z[~bottom], top["mesh"].nodes[~bottom, 2])
