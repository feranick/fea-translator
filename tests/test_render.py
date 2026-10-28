"""Pictures (render.py, render_vtk.py; as cfd-translator 2026.10.28.1): fields from the saved state against hand
values, validation, the setup view's colours, VTK where it can draw (skipped otherwise), the Matplotlib fallback,
render_images and get_image.
Version: 2026.10.28.1
"""
import asyncio
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from fea_translator import api, bench, render, render_vtk
from fea_translator.runner import run_study
from fea_translator.study import validate_study

E, NU = 210000.0, 0.3


@pytest.fixture()
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("FEAT_OUT", str(tmp_path))
    render_vtk._PROBE.clear()
    yield tmp_path
    render_vtk._PROBE.clear()


def _run(out: Path, name: str, **extra) -> tuple[dict, Path]:
    st = dict(bench.cases("ref")[name], **extra)
    d = out / "run_t" / name
    r = run_study(name, st, d, "ref", 300)
    assert r["ok"], r
    return r, d


def _png(path) -> np.ndarray:
    import matplotlib.image as mpimg
    return mpimg.imread(str(path))[..., :3]


def test_fields_of_a_uniaxial_bar_against_hand_values(out):
    """axial_bar: 1 kN on 10 x 10 mm, symmetry sliders at x = y = z = 0: sigma = 10 MPa everywhere, u_x = sigma x / E,
    u_y = -nu sigma y / E, u_z = -nu sigma z / E at every node."""
    r, d = _run(out, "axial_bar")
    assert (d / "state.npz").is_file() and r["files"]["state"].endswith("state.npz")
    s = render.load_state(d)
    X = s["nodes"]
    eps = 10.0 / E
    want = eps * np.sqrt(X[:, 0] ** 2 + NU ** 2 * (X[:, 1] ** 2 + X[:, 2] ** 2))
    assert np.allclose(render.field_values(s, "displacement"), want, rtol=1e-6, atol=1e-12)
    assert np.allclose(render.field_values(s, "ux"), eps * X[:, 0], atol=1e-10)
    # stresses come from gradients of the displacements the .dat prints to 7 digits: 1e-5
    assert np.allclose(render.field_values(s, "von_mises"), 10.0, rtol=1e-5)
    assert np.allclose(render.field_values(s, "max_principal"), 10.0, rtol=1e-5)
    assert np.allclose(render.field_values(s, "safety_factor"), 250 / 10.0, rtol=1e-5)     # steel, yield 250 MPa
    assert [b["name"] for b in s["meta"]["bcs"]] == ["sx", "sy", "sz", "pull"]
    assert s["meta"]["peak"]["value"] == pytest.approx(10.0, rel=1e-6)


def test_thermal_and_electric_fields_against_hand_values(out):
    _, d = _run(out, "slab")                       # 100 -> 0 C over 100 mm, k 50: q = 50 x 100 / 0.1 = 5e4 W/m2
    s = render.load_state(d)
    assert np.allclose(render.field_values(s, "temperature"), 100 - s["nodes"][:, 0], atol=1e-6)
    assert np.allclose(render.field_values(s, "heat_flux"), 5e4, rtol=1e-5)
    _, d = _run(out, "bar_resistance")             # 10 mV over 100 mm of copper: E = 1e-4 V/mm, J = I / A
    s = render.load_state(d)
    r_cu = 1.72e-8 * 0.1 / 1e-4
    assert np.allclose(render.field_values(s, "e_field"), 1e-4 * 1e-3, rtol=1e-5)        # kV/mm
    assert np.allclose(render.field_values(s, "current_density"), 0.01 / r_cu / 100, rtol=1e-5)
    _, d = _run(out, "plate_capacitor")            # 1 kV across 1 mm: 1 kV/mm
    assert np.allclose(render.field_values(render.load_state(d), "e_field"), 1.0, rtol=1e-5)


def test_default_images_and_what_they_show(out):
    r, d = _run(out, "cantilever", images=True)
    assert set(r["images"]) == {"von_mises", "setup"} and not r.get("warnings", [""])[0].startswith("image")
    vm = r["images"]["von_mises"]
    assert vm["renderer"] == "matplotlib" and vm["deformation_scale"] > 0 and vm["peak_marked"]
    s = render.load_state(d)
    umax = float(np.linalg.norm(s["fields"]["U"], axis=1).max())          # 5 % of the part's size
    assert vm["deformation_scale"] * umax == pytest.approx(0.05 * s["meta"]["size"], rel=1e-5)
    for k in ("von_mises", "setup"):
        img = _png(r["images"][k]["path"])
        assert Path(r["images"][k]["path"]).parent == d / "images" and img.std() > 0.02
    r, d = _run(out, "cantilever_modal", images=True)
    assert list(r["images"]) == ["mode1", "mode2", "mode3"] and "Hz" in r["images"]["mode1"]["what"]
    s = render.load_state(d)
    assert s["fields"]["modes"].shape == (len(r["frequencies"]), len(s["nodes"]), 3)
    # the first two modes: bending in y and in z (a square section), unit-free shapes
    m1, m2 = (np.abs(s["fields"]["modes"][k]).sum(axis=0) for k in (0, 1))
    assert {int(np.argmax(m1)), int(np.argmax(m2))} == {1, 2}
    assert render.default_images({"analysis": "buckling"}) == [{"name": "mode1", "mode": 1}]
    assert [i["field"] for i in render.default_images({"analysis": "electric"})] == ["potential", "current_density"]
    assert [i["field"] for i in render.default_images({"analysis": "thermal"})] == ["temperature"]


def test_setup_view_has_one_colour_per_bc(out):
    _, d = _run(out, "axial_bar")
    res, w = render.render(d, [{"name": "setup", "setup": True, "view": [30, -120]}], "bar")
    info = res["setup"]["bcs"]
    assert list(info) == ["sx", "sy", "sz", "pull"] and len({v["color"] for v in info.values()}) == 4
    assert all(v["triangles"] > 0 for v in info.values()) and not w
    img = _png(res["setup"]["path"])
    import matplotlib.colors as mcolors
    for bc in ("sx", "sy"):                       # the two seen from this side: drawn flat in their own colour
        c = np.array(mcolors.to_rgb(info[bc]["color"]))
        assert (np.abs(img - c).max(axis=2) < 0.02).sum() > 200, bc


def test_slices_bc_surfaces_and_scales(out):
    _, d = _run(out, "cantilever")
    specs = [{"name": "cut", "field": "von_mises", "slice": {"axis": "x", "at": 50}},
             {"name": "wall", "field": "von_mises", "surface": "wall"},
             {"name": "big", "field": "displacement", "deformed": 100, "clim": [0, 1]}]
    res, w = render.render(d, specs, "c")
    assert not w, w
    assert res["cut"]["triangles"] > 0 and res["wall"]["triangles"] < res["cut"]["triangles"]
    assert res["big"]["deformation_scale"] == 100 and res["big"]["clim"] == [0, 1]
    res, w = render.render(d, [{"name": "far", "field": "von_mises", "slice": {"axis": "x", "at": -5}}], "c")
    assert "far" not in res and "nothing of the part" in w[0]


def test_image_validation():
    st = dict(bench.cases("ref")["cantilever"])

    def errs(images, **kw):
        return " | ".join(validate_study(dict(st, images=images, **kw))["errors"])

    assert errs(True) == "" and errs([{"name": "a", "field": "von_mises", "deformed": True, "view": [10, 20]}]) == ""
    cases = [([{"field": "temperature"}], "not a field of this static study"),
             ([{"mode": 1}], "mode: for modal and buckling"),
             ([{"field": "von_mises", "surface": "nowhere"}], "surface: part or a bc name (wall, tip)"),
             ([{"setup": True, "deformed": True}], "setup view takes no deformed"),
             ([{"field": "von_mises", "clim": [3, 1]}], "clim: [lo, hi]"),
             ([{"field": "von_mises", "quality": "ultra"}], "quality: high"),
             ([{"field": "von_mises", "slice": {"axis": "w", "at": 1}}], "slice: {axis"),
             ([{"field": "von_mises", "deformed": -1}], "deformed: true"),
             ([{"field": "von_mises", "shading": 1}], "unknown key 'shading'"),
             ([{"name": "a b", "field": "von_mises"}], "name: a word"),
             ([{"name": "a", "field": "von_mises"}, {"name": "a", "setup": True}], "'a' twice"),
             ("yes", "images: true")]
    for images, text in cases:
        assert text in errs(images), (text, errs(images))
    modal = dict(bench.cases("ref")["cantilever_modal"])
    assert "a mode number from 1 to 4" in " | ".join(validate_study(dict(modal, modes=4, images=[{"mode": 5}]))["errors"])
    assert "only mode shapes" in " | ".join(validate_study(dict(modal, images=[{"field": "von_mises"}]))["errors"])
    th = bench.cases("ref")["slab"]
    assert "deformed: for displacement" in " | ".join(validate_study(dict(th, images=[{"field": "temperature", "deformed": True}]))["errors"])
    noyield = dict(st, material={"E": 1000, "nu": 0.3})
    assert "needs the material's yield" in " | ".join(validate_study(dict(noyield, images=[{"field": "safety_factor"}]))["errors"])


def test_matplotlib_fallback_when_vtk_is_off(out, monkeypatch):
    _, d = _run(out, "axial_bar")
    monkeypatch.setenv("FEAT_RENDER", "matplotlib")
    res, w = render.render(d, [{"name": "hq", "field": "von_mises", "quality": "high"}], "bar")
    assert res["hq"]["renderer"] == "matplotlib"
    assert any("FEAT_RENDER=matplotlib" in x for x in w) and any("drawn with Matplotlib instead" in x for x in w)
    assert api.tool_status()["render"].keys() == {"vtk", "offscreen"}


def test_vtk_pictures_where_it_can_draw(out):
    if not render_vtk.available() or not render_vtk.probe().get("ok"):
        pytest.skip(f"VTK cannot draw here: {render_vtk.probe().get('why')}")
    _, d = _run(out, "cantilever")
    specs = [{"name": "vm", "field": "von_mises", "deformed": True, "quality": "high"},
             {"name": "cut", "field": "von_mises", "slice": {"axis": "x", "at": 50}, "quality": "high"},
             {"name": "setup", "setup": True, "quality": "high"}]
    res, w = render.render(d, specs, "c")
    assert not w, w
    for k in ("vm", "cut", "setup"):
        assert res[k]["renderer"] == "vtk"
        img = _png(res[k]["path"])
        assert img.shape[:2] == (render_vtk.SIZE[1], render_vtk.SIZE[0]) and img.std() > 0.02   # not blank
    assert res["vm"]["deformation_scale"] > 0 and res["vm"]["peak_marked"]
    assert res["cut"]["triangles"] < res["vm"]["triangles"]
    assert api.tool_status()["render"]["offscreen"].startswith("vtk")


def test_render_images_and_get_image(out):
    _, d = _run(out, "cantilever")
    r = api.render_images(str(d), [{"name": "ux", "field": "ux"}])
    assert Path(r["images"]["ux"]["path"]).is_file() and r["warnings"] == []
    assert "error" in api.render_images(str(out.parent), True)                 # outside the output folder
    assert api.render_images(str(d), [{"field": "temperature"}])["error"] == "invalid images"
    assert api.image_path(r["images"]["ux"]["path"]).suffix == ".png"
    with pytest.raises(ValueError):
        api.image_path(str(d / "cantilever.json"))
    # through the MCP tools: a worker process; 'running' until done, then the same call collects
    mcp = pytest.importorskip("mcp")  # noqa: F841
    from fea_translator import mcp_server as S
    first = asyncio.run(S.render_images(str(d), [{"name": "uz", "field": "uz"}], wait_s=0))
    t_end = time.time() + 120
    res = first
    while res.get("state") == "running" and time.time() < t_end:
        res = asyncio.run(S.render_images(str(d), [{"name": "uz", "field": "uz"}], wait_s=10))
    assert Path(res["images"]["uz"]["path"]).is_file(), res
    img = asyncio.run(S.get_image(res["images"]["uz"]["path"]))
    assert type(img).__name__ == "Image"


@pytest.mark.skipif(not __import__("fea_translator.runner", fromlist=["x"]).find_ccx(), reason="CalculiX (ccx) not installed")
def test_calculix_mode_shapes_for_pictures(out):
    """With CalculiX: the deck prints each mode's displacements to the .dat (NODE PRINT U), one block per mode."""
    for name in ("cantilever_modal", "euler_column"):
        st = dict(bench.cases("ccx")[name], images=True)
        d = out / "run_c" / name
        r = run_study(name, st, d, "ccx", 600)
        assert r["ok"], r.get("error")
        s = render.load_state(d)
        n = len(r.get("frequencies") or r.get("buckling_factors"))
        assert s["fields"]["modes"].shape[0] == n and "mode1" in r["images"], (name, r.get("warnings"))
        assert np.abs(s["fields"]["modes"][0]).max() > 0
