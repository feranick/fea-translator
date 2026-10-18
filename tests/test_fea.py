"""Tests: mesh handling, deck writing, result parsing and validation (pure Python); the benchmark
studies with the reference solver (box parts); with Gmsh and CalculiX installed, every benchmark.
Version: 2026.10.18.4
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from fea_translator import bench, deck, results
from fea_translator.meshdata import EDGES, box_mesh, fit_cylinder, order_elements, tet_volumes
from fea_translator.runner import find_ccx, run_study
from fea_translator.study import study_format, validate_study

try:
    import gmsh  # noqa: F401
    HAVE_GMSH = True
except ImportError:
    HAVE_GMSH = False
HAVE_CCX = find_ccx() is not None


# ---- mesh handling

def test_box_mesh_ordering_faces_and_loads():
    m = box_mesh(100, 10, 10, 10, 2, 2, origin=(0, 0, 0))
    assert abs(m.volume() - 1e4) < 1e-9 and tet_volumes(m.nodes, m.tets).min() > 0
    for k, (a, b) in enumerate(EDGES):  # mid-edge nodes sorted back onto their edges
        assert np.allclose(m.nodes[m.tets[:, 4 + k]], (m.nodes[m.tets[:, a]] + m.nodes[m.tets[:, b]]) / 2)
    assert m.select({"nx": -1}) == [1] and m.select({"nz": 1}) == [6]
    assert m.select({"pts": [[50, 5, 10]]}) == [6] and len(m.select({"all": True})) == 6
    f = m.force_nodes([2], [0, 0, -100])
    assert np.allclose(sum(f.values()), [0, 0, -100])
    assert np.allclose(m.pressure_resultant([6], 2.0), [0, 0, -2000])
    assert np.allclose(m.pressure_nodes([6], 2.0).sum(axis=0), [0, 0, -2000])
    assert np.allclose(m.body_nodes(np.array([0, 0, -1.0])).sum(axis=0), [0, 0, -1e4])
    assert abs(m.flux_nodes([1, 2], 3.0).sum() - 600) < 1e-9


def test_order_elements_fixes_orientation_and_mid_nodes():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    mids = np.array([(nodes[a] + nodes[b]) / 2 for a, b in EDGES])
    allp = np.vstack([nodes, mids])
    t = order_elements(allp, [[0, 2, 1, 3]], [[9, 8, 7, 6, 5, 4]])  # negative volume, mids reversed
    assert tet_volumes(allp, t)[0] > 0
    for k, (a, b) in enumerate(EDGES):
        assert np.allclose(allp[t[0, 4 + k]], (allp[t[0, a]] + allp[t[0, b]]) / 2)


def test_cylinder_fit():
    rng = np.random.default_rng(0)
    th, z = rng.uniform(0, np.pi, 200), rng.uniform(0, 20, 200)
    p = np.column_stack([4 * np.cos(th) + 3, 4 * np.sin(th) - 2, z])
    n = np.column_stack([np.cos(th), np.sin(th), rng.normal(0, 0.02, 200)])
    r, axis, _ = fit_cylinder(p, n / np.linalg.norm(n, axis=1)[:, None])
    assert abs(r - 4) < 1e-9 and abs(abs(axis[2]) - 1) < 1e-9
    assert fit_cylinder(np.column_stack([th, z, z * 0]), np.tile([0, 0, 1.0], (200, 1))) is None


# ---- deck and parsing

def test_deck_contents():
    st = bench.cases("ref")["pressure_cantilever"]
    m = box_mesh(100, 10, 10, 20, 4, 4, origin=(0, -5, -5))
    text, plan = deck.write("pc", st, m)
    for kw in ("*NODE, NSET=NALL", "*ELEMENT, TYPE=C3D10, ELSET=EALL", "*NSET, NSET=N_WALL", "*MATERIAL, NAME=MAT",
               "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT", "*STATIC", "*BOUNDARY\nN_WALL, 1, 3", "*DLOAD",
               "*NODE PRINT, NSET=N_WALL\nRF", "*END STEP"):
        assert kw in text, kw
    assert all(len(ln) <= 132 for ln in text.splitlines())  # CalculiX's line limit
    assert np.allclose(plan["applied_force"], [0, 0, -100])
    th = bench.cases("ref")["fin_sides"]
    text, _ = deck.write("fin", th, box_mesh(100, 10, 10, 20, 2, 2, origin=(0, 0, 0)))
    for kw in ("*HEAT TRANSFER, STEADY STATE", "N_BASE, 11, 11, 100", "*FILM", "*INITIAL CONDITIONS, TYPE=TEMPERATURE"):
        assert kw in text, kw


DAT = """
 displacements (vx,vy,vz) for set NALL and time  0.1000000E+01

         1  0.000000E+00  0.000000E+00  0.000000E+00
         2  1.000000E-03 -2.000000E-03  3.000000E-03

 total force (fx,fy,fz) for set N_WALL and time  0.1000000E+01

        -1.000000E+03  2.000000E-12  0.000000E+00

     E I G E N V A L U E   O U T P U T

 MODE NO    EIGENVALUE                       FREQUENCY
                                     REAL PART            IMAGINARY PART
                          (RAD/TIME)      (CYCLES/TIME     (RAD/TIME)

      1   0.2751297E+08   0.5245281E+04   0.8348094E+03   0.0000000E+00
      2   0.2752000E+08   0.5245951E+04   0.8349160E+03   0.0000000E+00

     P A R T I C I P A T I O N   F A C T O R S

MODE NO.   X-COMPONENT     Y-COMPONENT     Z-COMPONENT
      1  -0.1000000E-01   0.2000000E+00   0.0000000E+00   0.0   0.0   0.0

     B U C K L I N G   F A C T O R   O U T P U T

 MODE NO       BUCKLING
                FACTOR

      1   0.6747000E+03
      2   0.6750000E+03
"""


def test_parse_dat_and_frd():
    d = results.parse_dat(DAT)
    assert results.block(d, "displacements", "NALL")["rows"][1] == [2, 1e-3, -2e-3, 3e-3]
    assert results.block(d, "total force", "N_WALL")["rows"] == [[-1000.0, 2e-12, 0.0]]
    assert [e["hz"] for e in d["eigen"]] == [834.8094, 834.916]
    assert [b["factor"] for b in d["buckling"]] == [674.7, 675.0]
    # CalculiX 2.21's heat-flux header has no space before the set name (mochi, 2026-10-18)
    hf = results.parse_dat(" heat flux (elem, integ.pnt.,qx,qy,qz) for setEALL and time  0.1000000E+01\n\n"
                           "         1   1  5.813953E+00 -0.000000E+00  1.260700E-14\n")
    assert results.block(hf, "heat flux", "EALL")["rows"] == [[1, 1, 5.813953, -0.0, 1.2607e-14]]
    frd = (" -4  STRESS      6    1\n -5  SXX         1    4    1    1\n"
           " -1         1 1.00000E+01 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00\n"
           " -1         2-2.00000E+01 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00\n -3\n")
    s = results.parse_frd(frd)["STRESS"]
    assert s[2][0] == -20.0 and results.von_mises(np.array(list(s.values()))).max() == 20.0


def test_validation():
    assert validate_study(bench.cases()["cantilever"])["ok"]
    bad = {"part": {"box": [1, 1, 1]}, "material": "unobtainium", "analysis": "static",
           "bcs": [{"kind": "force", "faces": {"nx": 0}, "fz": 0}, {"kind": "temperature", "faces": {"all": True}, "T": 1}]}
    errs = " ".join(validate_study(bad)["errors"])
    for text in ("unknown 'unobtainium'", "not all 0", "non-zero fx", "not used in a static", "needs supports"):
        assert text in errs, text
    fmt = study_format()
    assert set(fmt["bc_kinds"]) >= {"fixed", "slider", "force", "pressure", "gravity", "temperature", "convection"}


# ---- benchmarks: reference solver on box parts (always), every case with Gmsh and CalculiX

BOX = ["axial_bar", "cantilever", "cantilever_modal", "gravity_block", "pressure_cantilever", "slab", "fin_end", "fin_sides", "bar_resistance", "bar_current", "plate_capacitor", "joule_bar", "thermal_stress_bar"]


@pytest.mark.parametrize("name", BOX)
def test_benchmark_reference_solver(name, tmp_path):
    r = run_study(name, bench.cases("ref")[name], tmp_path, solver="ref")
    assert r["ok"], r.get("error")
    assert r["passed"], r["checks"]
    assert r.get("equilibrium_error", 0) < 1e-6 and r.get("heat_balance_error", 0) < 1e-6


@pytest.mark.skipif(not HAVE_CCX, reason="CalculiX (ccx) not installed")
@pytest.mark.parametrize("name", sorted(bench.cases()))
def test_benchmark_calculix(name, tmp_path):
    if name in bench.NEEDS and "gmsh" in bench.NEEDS[name] and not HAVE_GMSH:
        pytest.skip("Gmsh not installed")
    r = run_study(name, bench.cases("ccx")[name], tmp_path, solver="ccx")
    assert r["ok"], r.get("error")
    assert r["passed"], r["checks"]
    assert r.get("equilibrium_error", 0) < 1e-4 and r.get("heat_balance_error", 0) < 1e-3


@pytest.mark.skipif(not HAVE_GMSH, reason="Gmsh not installed")
def test_gmsh_mesh_of_a_goal(tmp_path):
    pytest.importorskip("build123d_translator")
    from fea_translator.mesher import mesh_part

    st = bench.cases()["lame_ring"]
    mesh, info = mesh_part(st, tmp_path)
    assert mesh.order == 2 and info["elements"] > 100
    assert abs(mesh.volume() - np.pi * (40 ** 2 - 20 ** 2) * 10 / 4) < 0.01 * mesh.volume()
    assert len(mesh.select({"cyl_r": 20})) == 1 and len(mesh.select({"cyl_r": 40})) == 1
    assert mesh.select({"nx": -1}) != mesh.select({"ny": -1})


def test_cli_status_runs():
    from fea_translator import api

    s = api.tool_status()
    assert "translator" in s and s["default_solver"] in ("ccx", "ref")
    assert shutil.which("python3") or True
    assert Path(__file__).is_file()


def test_curved_quarter_ring_reference_solver(tmp_path):
    """A box mesh bent into a quarter ring (curved 10-node elements, no Gmsh): Lame bore
    displacement, and equilibrium with pressure on a curved face whose edges touch the supports."""
    import math

    from fea_translator.meshdata import Mesh
    from fea_translator.runner import evaluate
    from fea_translator import refsolver

    b = box_mesh(20, 1.0, 10, 6, 12, 3, origin=(0, 0, 0))
    r, th = 20 + b.nodes[:, 0], b.nodes[:, 1] * math.pi / 2
    m = Mesh(np.column_stack([r * np.cos(th), r * np.sin(th), b.nodes[:, 2]]), b.tets, b.surfaces).build()
    assert abs(m.groups[1].radius - 20) < 1e-9 and abs(m.groups[1].total_area - 100 * math.pi) < 1e-3
    assert np.allclose(m.pressure_resultant([1], 10.0)[:2], [2000, 2000])
    st = dict(bench.cases("ref")["lame_ring"], part={"box": [1, 1, 1]})
    text, plan = deck.write("ring", st, m)
    inp = tmp_path / "ring.inp"
    inp.write_text(text)
    refsolver.run(inp)
    res = evaluate(st, m, plan, results.parse_dat(inp.with_suffix(".dat").read_text()), None)
    want = next(e["value"] for e in st["expect"] if e["quantity"] == "max_displacement")
    assert abs(res["max_displacement"] - want) < 1e-3 * want
    assert res["equilibrium_error"] < 1e-6


def _ring(ri=20.0, ro=40.0):
    import math

    from fea_translator.meshdata import Mesh

    b = box_mesh(ro - ri, 1.0, 10, 4, 8, 2, origin=(0, 0, 0))
    r, th = ri + b.nodes[:, 0], b.nodes[:, 1] * math.pi / 2
    return Mesh(np.column_stack([r * np.cos(th), r * np.sin(th), b.nodes[:, 2]]), b.tets, b.surfaces).build()


def test_cylinders_inner_outer_and_selection_report():
    from fea_translator.meshdata import MeshError, cylinder_side

    m = _ring()
    bore, outer = m.groups[1], m.groups[2]
    assert bore.concave and not outer.concave and abs(bore.arc - 90) < 10 and abs(outer.arc - 90) < 10
    assert m.select({"cyl_r": 20, "side": "inner"}) == [1]
    with pytest.raises(MeshError):
        m.select({"cyl_r": 20, "side": "outer"})
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    c = np.column_stack([np.cos(th), np.sin(th), 0 * th])
    assert cylinder_side(4 * c, -c, np.ones(72), np.array([0, 0, 1.0]), np.zeros(3)) == (True, 360.0)
    # a hole and convex fillets of one radius under one cyl_r: reported, with a warning
    w = deck._mixed({"cyl_r": 4}, [{"side": "inner"}, {"side": "outer"}, {"side": "outer"}])
    assert w and "side: 'inner'" in w
    assert deck._mixed({"cyl_r": 4, "side": "inner"}, [{"side": "inner"}]) is None
    st = dict(bench.cases("ref")["lame_ring"], part={"box": [1, 1, 1]})
    _, plan = deck.write("ring", st, m)
    assert plan["selection"]["bore"][0]["side"] == "inner" and plan["selection"]["bore"][0]["radius"] == 20.0
    assert validate_study(dict(st, bcs=[dict(st["bcs"][0], faces={"nx": -1, "side": "inner"})]))["ok"] is False


def test_background_run_status_and_peak_at_support(tmp_path, monkeypatch):
    from fea_translator import api

    monkeypatch.setenv("FEAT_OUT", str(tmp_path))
    c = bench.cases("ref")
    r = api.run({"cant": c["cantilever"], "bad": {"part": {"box": [1, 1, 1]}, "analysis": "static"}}, "ref", wait=0)
    assert r["studies"]["bad"]["why"] == "invalid study" and "request" in r
    r = api.status(r["request"], wait=120)
    assert r["done"] and r["studies"]["cant"]["ok"]
    cr = r["studies"]["cant"]
    assert cr["peak_at_bc"] == ["wall"] and cr["peak_von_mises_off_bc"] < cr["peak_von_mises"]
    assert cr["safety_factor"] == round(250 / cr["peak_von_mises"], 4)
    assert any("singular" in w for w in cr["warnings"])
    assert cr["selection"]["wall"][0]["kind"] == "plane"


@pytest.mark.skipif(not HAVE_GMSH, reason="Gmsh not installed")
def test_bracket_hole_vs_corner_fillets(tmp_path):
    """The user's bracket: an r 4 load hole and r 4 corner fillets. cyl_r 4 takes all five faces
    and warns; side: 'inner' takes only the hole."""
    pytest.importorskip("build123d_translator")
    from fea_translator.mesher import mesh_part

    goal = {"features": [
        {"kind": "base_box", "params": {"w": 120, "d": 40, "h": 8}},
        {"kind": "hole_cb", "params": {"r": 3.2, "cbr": 5.5, "cbd": 3.5, "x": -45, "y": 12}},
        {"kind": "mirror", "params": {"ax": 0, "ay": 1, "az": 0}},
        {"kind": "hole", "params": {"r": 4, "x": 45, "y": 0}},
        {"kind": "fillet_vertical", "params": {"r": 4}},
        {"kind": "chamfer_top", "params": {"size": 0.5}}]}
    st = {"part": {"goal": goal}, "material": "aluminium_6061_t6", "analysis": "static", "mesh": {"size": 3},
          "bcs": [{"kind": "fixed", "name": "bolts", "faces": {"cyl_r": 3.2, "side": "inner"}},
                  {"kind": "force", "name": "load", "faces": {"cyl_r": 4}, "fz": -500}]}
    mesh, _ = mesh_part(st, tmp_path)
    assert len(mesh.select({"cyl_r": 4})) == 5
    assert len(mesh.select({"cyl_r": 4, "side": "inner"})) == 1
    assert len(mesh.select({"cyl_r": 3.2, "side": "inner"})) == 2
    _, plan = deck.write("b", st, mesh)
    assert any("side: 'inner'" in w for w in plan["selection_warnings"])
    assert len(plan["selection"]["load"]) == 5


def test_gmsh_order_flip_and_straightening():
    """Gmsh's tet10 numbering (last two mid-edge nodes swapped), negative orientation, and an element
    inverted by a mid-edge node pushed through the opposite face (as curving does in thin ligaments)."""
    from fea_translator.meshdata import GMSH_TET10, min_jacobian, order_known, straighten_inverted

    m = box_mesh(10, 10, 10, 2, 2, 2, origin=(0, 0, 0))
    gm = m.tets[:, GMSH_TET10]                       # as Gmsh would number them
    t = order_known(m.nodes, gm[:, GMSH_TET10])
    assert np.array_equal(t, m.tets)
    flipped = m.tets[:, [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]]   # negative volume, consistent mid-edge nodes
    assert np.array_equal(order_known(m.nodes, flipped), m.tets)
    assert min_jacobian(m.nodes, m.tets).min() > 0.99
    nodes = m.nodes.copy()
    e = m.tets[0]
    nodes[e[4]] = nodes[e[3]] + 1.5 * (nodes[e[3]] - (nodes[e[0]] + nodes[e[1]]) / 2)  # tangle element 0
    assert min_jacobian(nodes, m.tets)[0] <= 0
    fixed, n = straighten_inverted(nodes, m.tets)
    assert n >= 1 and min_jacobian(fixed, m.tets).min() > 0.05
    assert np.allclose(fixed[e[4]], (fixed[e[0]] + fixed[e[1]]) / 2)


@pytest.mark.skipif(not HAVE_CCX, reason="CalculiX (ccx) not installed")
def test_buckling_concurrent_runs(tmp_path):
    """Four euler_column runs at once must all find P = pi^2 EI / 4L^2 (a load-dependent wrong factor,
    1 run in 4, was found in the pump test)."""
    from concurrent.futures import ThreadPoolExecutor

    st = bench.cases("ccx")["euler_column"]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda i: run_study(f"e{i}", st, tmp_path / f"e{i}", solver="ccx"), range(4)))
    assert all(r["ok"] and r["passed"] for r in res), [r.get("critical_factor") for r in res]


def test_progress_and_time_left(tmp_path, monkeypatch):
    """Pending studies report step of steps and minutes left (expected from this machine's past
    solves, at most from the time limit); long runs tell the caller to stop polling."""
    import json
    import time

    from fea_translator import api

    monkeypatch.setenv("FEAT_OUT", str(tmp_path))
    monkeypatch.setenv("FEAT_WORKERS", "2")
    # past solves here: 1,000 nodes, mesh 1 s, solve 10 s  ->  solve ~ 10 s x (nodes / 1000)^1.5
    (tmp_path / "timing.jsonl").write_text("".join(json.dumps(
        {"analysis": "static", "solver": "ccx", "nodes": 1000, "mesh_s": 1.0, "solve_s": 10.0}) + "\n" for _ in range(3)))
    st = {"analysis": "static"}
    d = tmp_path / "req" / "a"
    d.mkdir(parents=True)

    def at(nodes, stage="solve", step=2, steps=2, ago=5.0):
        now = time.time()
        (d / "progress.json").write_text(json.dumps({"stage": stage, "step": step, "steps": steps, "nodes": nodes,
                                                     "started": now - 60, "stage_started": now - ago,
                                                     "done_steps_s": [2.0] * (step - 1)}))
        return api._progress(d, dict(st, converge=steps == 4), "ccx", 900)

    p = at(8000)                                       # 10 x 8^1.5 = 226 s expected, 5 s done
    assert p["iteration"] == 2 and p["max_iterations"] == 2 and p["s_per_iteration"] == 2.0
    assert abs(p["eta_min"] - 221 / 60) < 0.1 and abs(p["eta_min_max"] - 895 / 60) < 0.1
    t, nxt = api._timing(tmp_path / "req", {"a": p})
    assert t["eta_min"] < 5 and "study_status" in nxt and "stop" not in nxt
    p = at(40000)                                      # 10 x 40^1.5 = 2530 s: 42 min
    t, nxt = api._timing(tmp_path / "req", {"a": p})
    assert t["eta_min"] > 40 and "tell the user" in nxt and t["expected_finish"] in nxt
    p = at(1000, stage="mesh", step=1, steps=4, ago=0.5)   # converge: mesh, solve, finer mesh, finer solve
    fine = 10 * (1000 / 0.6 ** 3 / 1000) ** 1.5
    assert p["max_iterations"] == 4 and abs(p["eta_min"] - (0.5 + 10 + 4.63 + fine) / 60) < 0.05
    assert abs(p["eta_min_max"] - (2 * 900 + 0.5 + 4.63) / 60) < 0.1     # two solves at most 900 s each, plus meshing
    q = api._progress(tmp_path / "req" / "b", st, "ccx", 900)    # no progress file yet: waiting for a worker
    assert q["stage"] == "queued" and q["eta_min_max"] == 15.0
    (tmp_path / "timing.jsonl").unlink()                            # no history: expected time unknown
    p = at(8000)
    assert p["eta_min"] is None and "note" in p
    t, nxt = api._timing(tmp_path / "req", {"a": p})
    assert t["expected_finish"] is None and "tell the user" in nxt   # judged on the upper bound (15 min)


def test_runs_record_their_timing(tmp_path, monkeypatch):
    import json

    from fea_translator import api

    monkeypatch.setenv("FEAT_OUT", str(tmp_path))
    r = api.run({"cant": bench.cases("ref")["cantilever"]}, "ref")
    assert r["done"] and r["studies"]["cant"]["ok"]
    recs = [json.loads(x) for x in (tmp_path / "timing.jsonl").read_text().splitlines()]
    assert recs[-1]["analysis"] == "static" and recs[-1]["nodes"] > 0 and recs[-1]["solve_s"] >= 0
    prog = json.loads((tmp_path / r["request"].split("/")[-1] / "cant" / "progress.json").read_text())
    assert prog["step"] == 2 and prog["steps"] == 2 and prog["stage"] == "solve"


def test_electrical_validation_and_format():
    fmt = study_format()
    assert {"electric", "electrostatic", "joule"} <= set(fmt["analyses"])
    assert {"voltage", "current"} <= set(fmt["bc_kinds"]) and "resistivity" in fmt["material_keys"]
    box = {"box": [10, 10, 10]}
    errs = lambda st: " ".join(validate_study(st)["errors"])  # noqa: E731
    assert "two terminals" in errs({"part": box, "material": "copper", "analysis": "electric",
                                    "bcs": [{"kind": "voltage", "faces": {"nx": 1}, "V": 0}]})
    assert "voltage terminal" in errs({"part": box, "material": "copper", "analysis": "electric",
                                       "bcs": [{"kind": "current", "faces": {"nx": 1}, "I": 1},
                                               {"kind": "current", "faces": {"nx": -1}, "I": -1}]})
    assert "two different voltages" in errs({"part": box, "material": "ptfe", "analysis": "electrostatic",
                                             "bcs": [{"kind": "voltage", "faces": {"nx": 1}, "V": 5}]})
    assert "needs 'permittivity'" in errs({"part": box, "material": "copper", "analysis": "electrostatic",
                                           "bcs": [{"kind": "voltage", "faces": {"nx": 1}, "V": 5},
                                                   {"kind": "voltage", "faces": {"nx": -1}, "V": 0}]})
    assert "need stress: true" in errs({"part": box, "material": "steel", "analysis": "thermal",
                                        "bcs": [{"kind": "temperature", "faces": {"nx": 1}, "T": 50},
                                                {"kind": "fixed", "faces": {"nx": -1}}]})
    assert "not used in a static" in errs({"part": box, "material": "steel", "analysis": "static",
                                           "bcs": [{"kind": "fixed", "faces": {"nx": -1}},
                                                   {"kind": "voltage", "faces": {"nx": 1}, "V": 1}]})
    for name in ("bar_resistance", "plate_capacitor", "joule_bar", "thermal_stress_bar", "coax_capacitor"):
        assert validate_study(bench.cases()[name])["ok"], name


def test_stresses_are_checked_against_the_displacements(tmp_path):
    """CalculiX's stress step once returned wrong stresses from correct displacements (mochi, 2026-10-18):
    every static result recomputes the integration-point stresses from the displacements; when they
    disagree, the recomputed ones are used, with a warning."""
    from fea_translator import refsolver
    from fea_translator.runner import evaluate

    for name in ("axial_bar", "thermal_stress_bar"):
        r = run_study(name, bench.cases("ref")[name], tmp_path / name, solver="ref")
        chk = r["stress_check"] if name == "axial_bar" else r["stress"]["stress_check"]
        assert chk["agree"] and chk["max_difference_MPa"] < 1e-4, (name, chk)
    st = bench.cases("ref")["axial_bar"]
    m = box_mesh(100, 10, 10, 10, 2, 2, origin=(0, 0, 0))
    text, plan = deck.write("bar", st, m)
    inp = tmp_path / "bar.inp"
    inp.write_text(text)
    refsolver.run(inp)
    dat = results.parse_dat(inp.with_suffix(".dat").read_text())
    b = results.block(dat, "stresses", "EALL")
    for row in b["rows"][40:80]:          # corrupt 10 elements' stresses, as the race did
        row[2] *= 4.7
        row[5] += 15.0
    res = evaluate(st, m, plan, dat, None)
    assert res["stress_check"]["elements_disagreeing"] == 10 and not res["stress_check"]["agree"]
    assert abs(res["max_von_mises"] - 10.0) < 1e-4      # the recomputed stresses (from 7-digit displacements)
