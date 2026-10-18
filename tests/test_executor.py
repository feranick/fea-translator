"""SLURM execution (executor.py, llm-translators docs/EXECUTION.md) with fake sbatch, squeue, scancel,
sacct and sinfo (tests/fake_slurm): routing, job scripts, submit -> running -> done collected by a new
process, queued, every failure state from sacct and from job.log, jobs(), cancel(), the status key.
Version: 2026.10.18.5
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fea_translator import api, bench, executor

FAKE = Path(__file__).parent / "fake_slurm"
ROOT = Path(__file__).parents[1]


@pytest.fixture()
def slurm(tmp_path, monkeypatch):
    monkeypatch.setenv("FEAT_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("FEAT_EXECUTOR", "slurm")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(tmp_path / "fake"))
    monkeypatch.setenv("PATH", f"{FAKE}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    for k in ("FAKE_SBATCH_FAIL", "FAKE_SBATCH_PENDING", "FAKE_SBATCH_DELAY", "FAKE_SACCT_FAIL", "FEAT_PROCS"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def _bar(**kw):
    return dict(bench.cases("ref")["axial_bar"], **kw)


def test_routing_and_job_script(slurm, monkeypatch):
    assert executor.use_slurm({"queue": True}, 0.1) and not executor.use_slurm({"queue": False}, 99)
    assert executor.use_slurm({}, None)                  # no estimate: SLURM
    assert not executor.use_slurm({}, 1.0) and executor.use_slurm({}, 6.0)   # FEAT_SLURM_MIN_MINUTES 5
    monkeypatch.setenv("FEAT_EXECUTOR", "local")
    assert not executor.use_slurm({"queue": True}, None)  # local always wins
    monkeypatch.setenv("FEAT_EXECUTOR", "slurm")
    monkeypatch.setenv("FEAT_PROCS", "6")
    req = slurm / "out" / "run_x"
    s = executor.job_script(req, "a", {"analysis": "static"})
    assert "#SBATCH --ntasks=1" in s and "#SBATCH --cpus-per-task=6" in s and "--job-name=feat:run_x:a" in s
    assert "-m fea_translator.cli run-study" in s and f"--output={req / 'a' / 'job.log'}" in s
    assert "--cpus-per-task=3" in executor.job_script(req, "a", {"analysis": "static", "procs": 3})
    assert "--cpus-per-task=1" in executor.job_script(req, "a", {"analysis": "buckling", "procs": 8})
    # time limit: every solve's timeout + 10 min (converge doubles the solves)
    assert executor.time_limit_min({"analysis": "static", "converge": True}) == 2 * 15 + 10
    st = api.tool_status()["executor"]
    assert st["kind"] == "slurm" and st["sbatch"].endswith("sbatch") and st["min_minutes"] == 5 and st["procs"] == 6


def test_estimate_from_timing_records(slurm):
    out = Path(os.environ["FEAT_OUT"])
    out.mkdir(parents=True)
    assert api.estimate_min({"analysis": "static"}, "ref") is None
    (out / "timing.jsonl").write_text(json.dumps({"analysis": "static", "solver": "ref", "nodes": 500,
                                                  "mesh_s": 0.5, "solve_s": 2.0}) + "\n")
    e = api.estimate_min({"analysis": "static"}, "ref")
    assert abs(e - 2.5 / 60) < 1e-9 and not executor.use_slurm({}, e)
    assert api.estimate_min({"analysis": "static", "converge": True}, "ref") > e


def test_submit_run_and_collect_from_a_new_process(slurm):
    r = api.run({"bar": _bar(queue=True)}, "ref", wait=0)
    req = Path(r["request"])
    assert r["pending"] == ["bar"] and r["progress"]["bar"]["state"] in ("queued", "running")
    jid = r["progress"]["bar"]["job_id"]
    info = json.loads((req / "bar" / "job.json").read_text())
    assert info["job_id"] == jid and info["executor"] == "slurm" and info["procs"] == 1
    assert "SLURM jobs on this machine" in r["next"] and jid in r["next"]
    code = (f"import json; from fea_translator import api; "
            f"print(json.dumps(api.status({str(req)!r}, wait=120)))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, env=os.environ)
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["done"], res
    bar = res["studies"]["bar"]
    assert bar["ok"] and bar["passed"], bar
    assert (req / "bar" / "job.log").is_file()
    j = api.jobs()
    assert j["requests"][0]["request"] == str(req) and j["requests"][0]["studies"]["bar"]["state"] == "done"
    assert j["machine"] == {"cpus": 8, "cpus_allocated": 2}


def test_queued_state_and_submission_failure(slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    r = api.run({"bar": _bar(queue=True)}, "ref", wait=0)
    pr = r["progress"]["bar"]
    assert pr["state"] == "queued" and pr["reason"] == "Priority" and pr["ahead"] == 0
    assert api.jobs(active=True)["requests"][0]["studies"]["bar"]["state"] == "queued"
    monkeypatch.delenv("FAKE_SBATCH_PENDING")
    monkeypatch.setenv("FAKE_SBATCH_FAIL", "1")
    r = api.run({"bar": _bar(queue=True)}, "ref", wait=0)
    assert r["done"] and r["studies"]["bar"]["ok"] is False
    assert r["studies"]["bar"]["why"].startswith("could not submit:")


def _ended(slurm, sacct_state=None, log="", sacct_fail=False, monkeypatch=None):
    """A request whose job has left the queue without a result."""
    fake = Path(os.environ["FAKE_SLURM_DIR"]) / "jobs"
    fake.mkdir(parents=True, exist_ok=True)
    jid = str(5000 + len(list(fake.glob("*.json"))))
    (fake / f"{jid}.json").write_text(json.dumps({"name": "feat:r:s", "cpus": 1, "t0": time.time(), "log": "x",
                                                  "pid": 999999, **({"sacct": sacct_state} if sacct_state else {})}))
    req = Path(os.environ["FEAT_OUT"]) / f"run_ended_{jid}"
    (req / "s").mkdir(parents=True)
    (req / "studies.json").write_text(json.dumps({"s": _bar()}))
    (req / "request.json").write_text(json.dumps({"names": ["s"], "invalid": {}, "solver": "ref", "started": time.time(),
                                                  "slurm": ["s"], "timeout": 900}))
    (req / "s" / "job.json").write_text(json.dumps({"executor": "slurm", "job_id": jid}))
    (req / "s" / "job.log").write_text(log)
    if sacct_fail:
        monkeypatch.setenv("FAKE_SACCT_FAIL", "1")
    res = api.status(str(req))
    monkeypatch.delenv("FAKE_SACCT_FAIL", raising=False)
    assert res["done"] and res["studies"]["s"]["ok"] is False
    return res["studies"]["s"]


@pytest.mark.parametrize("acct,state", [("TIMEOUT", "timed_out"), ("OUT_OF_MEMORY", "out_of_memory"),
                                        ("CANCELLED by 1000", "cancelled"), ("FAILED", "failed"),
                                        ("NODE_FAIL", "failed"), ("COMPLETED", "failed")])
def test_failure_states_from_sacct(slurm, monkeypatch, acct, state):
    r = _ended(slurm, sacct_state=acct, monkeypatch=monkeypatch)
    assert r["state"] == state, r
    if state == "out_of_memory":
        assert "--mem" in r["why"]
    if acct.startswith("CANCELLED"):
        assert "1000" in r["why"]


@pytest.mark.parametrize("log,state", [("slurmstepd: error: *** JOB 12 ON mochi CANCELLED AT 2026-10-18T10:00:00 DUE TO TIME LIMIT ***", "timed_out"),
                                       ("slurmstepd: error: *** JOB 12 ON mochi CANCELLED AT 2026-10-18T10:00:00 ***", "cancelled"),
                                       ("Traceback (most recent call last):\nValueError: boom", "failed")])
def test_failure_states_from_job_log_without_accounting(slurm, monkeypatch, log, state):
    r = _ended(slurm, log=log, sacct_fail=True, monkeypatch=monkeypatch)
    assert r["state"] == state, r
    if state == "failed":
        assert "ValueError: boom" in r["why"]


def test_cancel_a_running_job(slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_DELAY", "30")
    r = api.run({"bar": _bar(queue=True)}, "ref", wait=0)
    assert r["progress"]["bar"]["state"] == "running"
    c = api.cancel(r["request"])
    assert c["cancelled_jobs"] == [r["progress"]["bar"]["job_id"]] and c["states"]["bar"] == "cancelled"
    res = api.status(r["request"])
    assert res["done"] and res["studies"]["bar"]["state"] == "cancelled"


def test_local_studies_stay_local(slurm):
    r = api.run({"bar": _bar(queue=False)}, "ref", wait=120)
    assert r["done"] and r["studies"]["bar"]["ok"] and not (Path(r["request"]) / "bar" / "job.json").exists()


def test_queued_estimate_and_slurm_finish_times(slurm, monkeypatch):
    """A queued SLURM study: run time from its mesh size when known before meshing (box parts), else
    unknown; the request's finish is SLURM's expected start plus the run time."""
    out = Path(os.environ["FEAT_OUT"])
    out.mkdir(parents=True)
    (out / "timing.jsonl").write_text(json.dumps({"analysis": "static", "solver": "ref", "nodes": 1000,
                                                  "mesh_s": 1.0, "solve_s": 10.0}) + "\n")
    box = _bar()                                         # 10 x 2 x 2 cells, order 2: 21 x 5 x 5 nodes
    assert api._nodes_hint(box) == 21 * 5 * 5
    q = api._progress(out / "none", box, "ref", 900)
    assert q["stage"] == "queued" and 0 < q["eta_min"] < 1
    g = api._progress(out / "none", {"analysis": "static", "part": {"goal": {}}}, "ref", 900)
    assert g["eta_min"] is None and "unknown" in g["note"]
    start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 600))   # SLURM: starts in 10 min
    prog = {"a": dict(q, state="queued", expected_start=start, job_id="7")}
    t, nxt = api._timing_slurm(out, prog, {"a": "7"})
    assert 10 < t["eta_min"] < 11 and 24 < t["eta_min_max"] < 26 and t["includes_queue_wait"]
    assert f"around {t['expected_finish']} (at the latest {t['latest_finish']})" in nxt
    t, nxt = api._timing_slurm(out, {"a": dict(g, state="queued", job_id="7")}, {"a": "7"})
    assert t["expected_finish"] is None and "jobs ahead of them" in nxt


def test_job_memory(slurm, monkeypatch):
    """Where SLURM gives a job without --mem the whole node, only one runs at a time (mochi): jobs ask for
    memory from the node count when known (box parts), else 8G; FEAT_SLURM_MEM and FEAT_SLURM_ARGS override."""
    req = Path(os.environ["FEAT_OUT"]) / "run_m"
    box = {"analysis": "static", "part": {"box": [100, 10, 10], "divisions": [80, 16, 16]}}   # 175,329 nodes
    assert executor.nodes_hint(box) == 161 * 33 * 33
    assert executor.mem_for(box) == "8G"                          # 2 + 3e-5 x 175k = 7.3 -> 8
    assert executor.mem_for(dict(box, converge=True)) == "27G"    # the finer mesh: 4.6x the nodes
    assert executor.mem_for({"analysis": "static", "part": {"goal": {}}}) == "8G"
    assert "#SBATCH --mem=8G" in executor.job_script(req, "a", box)
    monkeypatch.setenv("FEAT_SLURM_MEM", "20G")
    assert "#SBATCH --mem=20G" in executor.job_script(req, "a", box)
    monkeypatch.setenv("FEAT_SLURM_MEM", "none")
    assert "--mem" not in executor.job_script(req, "a", box)
    monkeypatch.setenv("FEAT_SLURM_MEM", "auto")
    monkeypatch.setenv("FEAT_SLURM_ARGS", "--mem=64G --nice=10")
    assert "#SBATCH --mem" not in executor.job_script(req, "a", box)  # the args' --mem wins
