"""Time limits and progress of long runs: CalculiX's own progress (.sta, .cvg) for the time left, extending a running
study's time limit (api.extend), the SLURM time limit with the meshing and an assembly's modal runs.
Job 419 on spark2 (an S50 sleeve, 2026.11.05): the stage estimate said 'eta 63 min' while CalculiX was at step time
0.175 of the first of two steps after 10.5 h, at about 15 min per Newton iteration.
Version: 2026.11.09.4
"""
import json
from pathlib import Path
import time

import pytest

from fea_translator import api, runner

STA = """SUMMARY OF JOB INFORMATION
  STEP      INC     ATT  ITRS     TOT TIME     STEP TIME      INC TIME
     1          1     1     8  0.500000E-01  0.500000E-01  0.500000E-01
     1          2     1     9  0.100000E+00  0.100000E+00  0.500000E-01
     1          3     1    12  0.175000E+00  0.175000E+00  0.750000E-01
"""


def _cvg(n):
    head = " SUMMARY OF C0NVERGENCE INFORMATION\n  STEP   INC   ATT  ITER     CONT.   RESID.        CORR.\n\n"
    rows = [f"{1:6d}{(i // 10) + 1:6d}{1:6d}{(i % 10) + 1:6d}{5000:10d}  0.1000E+01  0.1000E+00  0.0000E+00  0.0000E+00"
            for i in range(n)]
    return head + "\n".join(rows) + "\n"


def test_calculix_progress_job_419(tmp_path):
    (tmp_path / "s50.sta").write_text(STA)
    (tmp_path / "s50.cvg").write_text(_cvg(42))
    now = time.time()
    p = {"stage": "solve", "stage_started": now - 10.5 * 3600, "ccx_steps": 2}
    cx = api.ccx_progress(tmp_path, p, now)
    assert cx["step"] == 1 and cx["of_steps"] == 2 and cx["step_time"] == 0.175 and cx["increment"] == 5
    assert cx["fraction_done"] == pytest.approx(0.0875) and cx["iterations"] == 42
    assert cx["s_per_iteration"] == pytest.approx(10.5 * 3600 / 42, rel=1e-3)          # 15 min
    assert cx["eta_min"] == pytest.approx(10.5 * 60 * (1 - 0.0875) / 0.0875, rel=1e-3)  # about 110 h, not 63 min
    # a finished first step: half done
    (tmp_path / "s50.sta").write_text(STA + "     1          4     1     3  0.100000E+01  0.100000E+01  0.825000E+00\n")
    (tmp_path / "s50.cvg").write_text(_cvg(42).replace("\n", "\n", 1) + f"{2:6d}{1:6d}{1:6d}{1:6d}{5000:10d}  0.1E+01  0.1E+00  0.0E+00  0.0E+00\n")
    cx = api.ccx_progress(tmp_path, p, now)
    assert cx["step"] == 2 and cx["fraction_done"] == pytest.approx(0.5)


def test_progress_uses_calculix_and_warns_past_the_limit(tmp_path):
    d = tmp_path
    (d / "s.sta").write_text(STA)
    (d / "s.cvg").write_text(_cvg(42))
    now = time.time()
    (d / "progress.json").write_text(json.dumps({"analysis": "assembly", "solver": "ccx", "steps": 2, "step": 2,
                                                 "stage": "solve", "timeout_s": 43200, "started": now - 10.6 * 3600,
                                                 "stage_started": now - 10.5 * 3600, "done_steps_s": [360],
                                                 "nodes": 900000, "ccx_steps": 2}))
    out = api._progress(d, {"analysis": "assembly", "bcs": [], "steps": [{"kind": "static"}]}, "ccx", 43200)
    assert out["eta_from"] == "calculix progress" and out["eta_min"] > 6000 and "past its time limit" in out["warning"]
    assert out["calculix"]["step_time"] == 0.175


def test_extension_reaches_a_running_calculix(tmp_path, monkeypatch):
    """A fake CalculiX that runs 4 s with a 2 s limit: an extend file written meanwhile lets it finish."""
    exe = tmp_path / "fake_ccx"
    exe.write_text("#!/bin/sh\nsleep 4\necho done > $2.dat\n")
    exe.chmod(0o755)
    monkeypatch.setenv("FEAT_CCX", str(exe))
    monkeypatch.setattr(runner, "MONITOR_EVERY", 0.3)
    study = tmp_path / "study"
    study.mkdir()
    monkeypatch.setitem(runner.CANCEL, "file", study / "cancel")
    runner.EXTENDED["s"] = 0.0
    (study / "extend").write_text("10")
    inp = study / "x.inp"
    inp.write_text("")
    runner.run_ccx(inp, 2.0)
    assert runner.EXTENDED["s"] == 10.0 and not (study / "extend").exists()
    assert json.loads((study / "extended.json").read_text())["total_s"] == 10.0
    assert list(study.glob("extend.applied.*"))
    import subprocess
    runner.EXTENDED["s"] = 0.0
    with pytest.raises(subprocess.TimeoutExpired):     # without one, the same run is cut off
        runner.run_ccx(inp, 1.0)


def test_api_extend(tmp_path, monkeypatch):
    monkeypatch.setenv("FEAT_OUT", str(tmp_path))
    req = tmp_path / "run_x"
    (req / "a").mkdir(parents=True)
    (req / "b").mkdir()
    (req / "request.json").write_text(json.dumps({"names": ["a", "b"], "slurm": []}))
    (req / "studies.json").write_text(json.dumps({"a": {"analysis": "static", "run": {"timeout_s": 3600}},
                                                  "b": {"analysis": "static"}}))
    (req / "b" / "b.json").write_text("{}")
    r = api.extend(str(req), 2)
    assert r["studies"]["a"]["added_s"] == 7200 and float((req / "a" / "extend").read_text()) == 7200
    assert r["studies"]["b"]["state"].startswith("finished")
    monkeypatch.setenv("FEAT_MAX_TIMEOUT", "5400")
    (req / "a" / "extend").unlink()
    r = api.extend(str(req), 2, ["a"])
    assert r["studies"]["a"]["added_s"] == 1800 and "FEAT_MAX_TIMEOUT" in r["studies"]["a"]["capped"]
    assert "error" in api.extend(str(req), -1) and "no studies" in api.extend(str(req), 1, ["zz"])["error"]


def test_status_gives_the_version_a_request_ran_with(tmp_path, monkeypatch):
    """A 2026.11.09.2 bench on mochi read '2026.11.09.3' in bench_status after the upgrade (2026.11.09.4)."""
    from fea_translator import __version__
    assert api.ran_with({}, {"a": {"translator": "2026.11.09.2"}}) == {"translator": "2026.11.09.2",
                                                                       "translator_now": __version__}
    assert api.ran_with({"translator": "2026.11.09.2"}, {}) == {"translator": "2026.11.09.2", "translator_now": __version__}
    assert api.ran_with({}, {"a": {"translator": "x"}, "b": {"translator": "y"}})["translator"] == ["x", "y"]
    old = api.ran_with({}, {})
    assert old["translator"] == __version__ and "not recorded" in old["translator_note"]
    assert api.ran_with({}, {"a": {"translator": __version__}}) == {"translator": __version__}
    monkeypatch.setenv("FEAT_OUT", str(tmp_path))
    monkeypatch.setenv("FEAT_EXECUTOR", "local")
    r = api.run({"s": {"analysis": "static"}}, "ref", wait=1)            # invalid: answered at once
    assert json.loads((tmp_path / Path(r["request"]).name / "request.json").read_text())["translator"] == __version__
