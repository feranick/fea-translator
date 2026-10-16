"""Tests: mesh handling, deck writing, result parsing and validation (pure Python); the benchmark
studies with the reference solver (box parts); with Gmsh and CalculiX installed, every benchmark.
Version: 2026.10.16.3
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
    r, axis = fit_cylinder(p, n / np.linalg.norm(n, axis=1)[:, None])
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

BOX = ["axial_bar", "cantilever", "cantilever_modal", "gravity_block", "pressure_cantilever", "slab", "fin_end", "fin_sides"]


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

    s = api.status()
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
