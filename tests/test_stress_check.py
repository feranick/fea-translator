"""The stress check against the .dat's rounding, and stress probes (peak and mean von Mises within a sphere).
An S50 casing (carbonio, 2026.11.04: 1.7 M nodes, 0.06 mm elements at a 0.25 mm fillet, 1.7 mm displacements)
'disagreed' in 36,457 elements by up to 955 MPa: 7 digits of the displacements were worth that in its smallest and
straightened elements, and the recomputed stresses then replaced CalculiX's.
Version: 2026.11.04.8
"""
import numpy as np
import pytest

from fea_translator.meshdata import box_mesh
from fea_translator.runner import check_stresses, recompute_stresses, stress_probes
from fea_translator.study import validate_study

STEEL = {"material": "steel"}


def _rows(mesh, sig):
    q = sig.shape[1]
    return np.column_stack([np.repeat(np.arange(1, len(mesh.tets) + 1), q), np.tile(np.arange(1, q + 1), len(mesh.tets)),
                            sig.reshape(-1, 6)])


def _round7(U):
    return np.array([[float(f"{v:.6E}") for v in row] for row in U])


def test_rounding_of_the_displacements_is_not_a_disagreement():
    """Tiny elements (0.05 mm), a 2 mm rigid shift plus a small strain: the recomputed stresses from 7-digit
    displacements differ from the exact ones by more than 0.1 % of the peak, but within the rounding bound."""
    m = box_mesh(0.2, 0.2, 0.2, 4, 4, 4, order=2, origin=[0, 0, 0])
    U = np.column_stack([2.0 + 1e-5 * m.nodes[:, 0], 1.5 + 0 * m.nodes[:, 1], -1.0 + 2e-5 * m.nodes[:, 2]])
    exact = recompute_stresses(m, U, STEEL, {})
    ours, err = recompute_stresses(m, _round7(U), STEEL, {}, bound=True)
    rows = _rows(m, exact)
    old = check_stresses(rows, ours)
    assert not old["agree"] and old["elements_disagreeing"] > 0          # without the bound: a false alarm
    new = check_stresses(rows, ours, err)
    assert new["agree"] and new["elements_disagreeing"] == 0 and new["elements_within_rounding"] > 0


def test_a_real_error_is_still_caught():
    """CalculiX's threaded stress step once returned wrong stresses at half the points (axial_bar, mochi): a 4x
    error in ordinary elements is far beyond the rounding bound."""
    m = box_mesh(100, 10, 10, 10, 2, 2, order=2, origin=[0, 0, 0])
    U = np.column_stack([1e-3 * m.nodes[:, 0], -3e-4 * m.nodes[:, 1], -3e-4 * m.nodes[:, 2]])
    sig = recompute_stresses(m, U, STEEL, {})
    bad = sig.copy()
    bad[:5] *= 4.0
    ours, err = recompute_stresses(m, _round7(U), STEEL, {}, bound=True)
    chk = check_stresses(_rows(m, bad), ours, err)
    assert not chk["agree"] and chk["elements_disagreeing"] == 5 and chk["max_difference_MPa"] > 100
    assert check_stresses(_rows(m, sig), ours, err)["agree"]


def test_stress_probes():
    """A bending-like field: exx = c z along a 100 x 10 x 10 bar; the sphere near the top face finds the peak
    there, its mean is lower, and the straightened elements can be left out."""
    m = box_mesh(100, 10, 10, 20, 2, 2, order=2, origin=[0, 0, -5])
    c = 1e-4
    x, z = m.nodes[:, 0], m.nodes[:, 2]
    U = np.column_stack([c * x * z, np.zeros(len(x)), -0.5 * c * x ** 2])     # exx = c z, pure bending
    sig = recompute_stresses(m, U, STEEL, {})
    rows = _rows(m, sig)
    cen = m.nodes[m.tets[:, :4]].mean(axis=1)
    m.straightened = np.argsort(np.linalg.norm(cen - [0, 0, 5], axis=1))[:2]   # the two elements at the corner
    out = stress_probes([{"name": "top", "at": [50, 5, 5], "r": 3}, {"name": "far", "at": [500, 0, 0], "r": 1},
                         {"name": "corner", "at": [0, 0, 5], "r": 6}], m, rows, None)
    top = out["top"]
    assert top["points"] > 0 and top["peak_at"][2] > 2 and top["mean_von_mises_MPa"] < top["peak_von_mises_MPa"]
    # no lateral strain: sxx - syy = 2 mu exx, the von Mises stress, 2 mu c z at the peak's z
    assert top["peak_von_mises_MPa"] == pytest.approx(210000 / 1.3 * c * top["peak_at"][2], rel=1e-4)   # rounded to 4 decimals
    assert out["far"] == {"points": 0, "note": "no integration point within 1 mm of [500, 0, 0]: a larger r"}
    cr = out["corner"]
    assert cr["straightened_elements"] >= 1 and cr["peak_von_mises_unstraightened_MPa"] <= cr["peak_von_mises_MPa"]


def test_stress_probe_validation():
    st = {"part": {"box": [10, 10, 10]}, "analysis": "static", "material": "steel",
          "bcs": [{"kind": "fixed", "faces": {"nx": -1}}, {"kind": "force", "faces": {"nx": 1}, "fx": 1}]}
    assert validate_study(dict(st, stress_probes=[{"name": "corner", "at": [59, 12, -33], "r": 3}]))["ok"]
    errs = " | ".join(validate_study(dict(st, stress_probes=[{"name": "c", "at": [0, 0], "r": 3},
                                                            {"name": "d", "at": [0, 0, 0], "r": -1}]))["errors"])
    assert "stress_probes[0]" in errs and "stress_probes[1]" in errs
    modal = dict(st, analysis="modal", stress_probes=[{"name": "c", "at": [0, 0, 0], "r": 1}])
    assert "for static studies" in " | ".join(validate_study(modal)["errors"])
