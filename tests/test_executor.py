"""SLURM execution (executor.py, llm-translators docs/EXECUTION.md) with fake sbatch, squeue, scancel,
sacct and sinfo (tests/fake_slurm): routing, job scripts, submit -> running -> done collected by a new
process, queued, every failure state from sacct and from job.log, jobs(), cancel(), the status key.
Version: 2026.10.20.3
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
    # a running study expected to end in 3 min frees its CPUs before SLURM's planned start (10 min)
    run = {"stage": "solve", "eta_min": 3.0, "eta_min_max": 14.0, "state": "running", "job_id": "6"}
    t, _ = api._timing_slurm(out, {"r": run, **prog}, {"r": "6", "a": "7"})
    assert 3 < t["eta_min"] < 4 and 24 < t["eta_min_max"] < 26
    t, nxt = api._timing_slurm(out, {"a": dict(g, state="queued", job_id="7")}, {"a": "7"})
    assert t["expected_finish"] is None and "jobs ahead of them" in nxt


def test_job_memory(slurm, monkeypatch):
    """Jobs ask for the memory CalculiX needs (measured on mochi): the iterative solver little, the direct one
    ~nodes^1.43 (half single-threaded); FEAT_SLURM_MEM and FEAT_SLURM_ARGS override."""
    req = Path(os.environ["FEAT_OUT"]) / "run_m"
    box = {"analysis": "static", "part": {"box": [100, 10, 10], "divisions": [80, 16, 16]}}   # 175,329 nodes
    assert executor.nodes_hint(box) == 161 * 33 * 33
    assert executor.method_of(box, 175329) == "iterative"
    assert executor.mem_for(box) == "2G"                          # iterative: 0.64 GB measured
    assert executor.mem_for(dict(box, converge=True)) == "6G"     # the finer mesh: 4.6x the nodes
    modal = dict(box, analysis="modal", procs=16)                 # eigenvalues need the direct solver
    assert executor.mem_for(modal) == "19G"                       # 14 GB measured at 19 threads
    assert executor.mem_for(dict(modal, procs=1)) == "10G"        # 6.6 GB measured single-threaded
    assert executor.mem_for({"analysis": "static", "part": {"goal": {"features": []}}}) == "8G"   # unknown size
    assert "#SBATCH --mem=2G" in executor.job_script(req, "a", box)
    monkeypatch.setenv("FEAT_SLURM_MEM", "20G")
    assert "#SBATCH --mem=20G" in executor.job_script(req, "a", box)
    monkeypatch.setenv("FEAT_SLURM_MEM", "none")
    assert "--mem" not in executor.job_script(req, "a", box)
    monkeypatch.setenv("FEAT_SLURM_MEM", "auto")
    monkeypatch.setenv("FEAT_SLURM_ARGS", "--mem=64G --nice=10")
    assert "#SBATCH --mem" not in executor.job_script(req, "a", box)  # the args' --mem wins


def test_threads_only_where_they_help(slurm):
    """Iterative solves (single-threaded in CalculiX) ask for 1 CPU; direct ones at most a thread per 10k nodes."""
    large = {"analysis": "static", "procs": 16, "part": {"box": [100, 10, 10], "divisions": [80, 16, 16]}}
    small = {"analysis": "static", "procs": 16, "part": {"box": [100, 10, 10], "divisions": [50, 10, 10]}}  # 44,541 nodes
    assert executor.procs_for(large) == 1                          # iterative above 50k nodes
    assert executor.procs_for(small) == 5                          # direct: ceil(44,541 / 10k)
    assert executor.procs_for(dict(large, analysis="modal")) == 16  # direct, 18 allowed
    assert executor.procs_for(dict(large, equation_solver="direct", procs=32)) == 18
    assert executor.procs_for({"analysis": "static", "procs": 6, "part": {"goal": {"features": []}}}) == 6  # unknown size
    assert executor.procs_for(dict(small, procs=None)) == 1         # FEAT_PROCS default 1
    assert "--cpus-per-task=1" in executor.job_script(Path(os.environ["FEAT_OUT"]) / "r", "a", large)


def test_size_estimate_from_a_goal(slurm):
    pytest.importorskip("build123d_translator")
    goal = {"features": [{"kind": "base_box", "params": {"w": 100, "d": 50, "h": 20}}]}
    st = {"analysis": "static", "part": {"goal": goal}, "mesh": {"size": 5}}
    n = executor.estimate_nodes(st)
    assert abs(n - 12 * 100 * 50 * 20 / 5 ** 3) <= 1                # 12 nodes per h^3 of volume (upper estimate)
    assert executor.estimate_nodes(dict(st, mesh={"size": 5, "refine": [{"at": [0, 0, 10], "r": 5, "size": 1}]})) > n


def test_benchmarks_return_at_once_and_bench_status_collects(slurm, monkeypatch):
    """run_benchmarks submits and returns within its wait (cfdt's waited for every job and timed out);
    bench_status collects them, here from SLURM jobs (fake) and from the local pool."""
    r = api.bench(["axial_bar", "slab", "euler_column"], "ref", wait=0)
    assert r["skipped"] == {"euler_column": "needs CalculiX"} and set(r["pending"]) == {"axial_bar", "slab"}
    assert "bench_status(bench=" in r["next"]
    assert any(x["request"] == r["bench"] for x in api.jobs()["requests"])
    s = api.bench_status(r["bench"], wait=120)
    assert s["done"] and sorted(s["passed"]) == ["axial_bar", "slab"] and not s["failed"]
    monkeypatch.setenv("FEAT_EXECUTOR", "local")
    s = api.bench(["axial_bar"], "ref", wait=120)
    assert s["done"] and s["passed"] == ["axial_bar"]


def _job(req: Path, n: str) -> tuple[str, list]:
    info = json.loads((req / n / "job.json").read_text())
    rec = json.loads((Path(os.environ["FAKE_SLURM_DIR"]) / "jobs" / f"{info['job_id']}.json").read_text())
    return (req / n / "job.sh").read_text(), rec["args"]


def test_per_study_run_settings(slurm, monkeypatch):
    """A study's "run" sets its solver timeout, the job's --time, --mem and threads (docs/EXECUTION.md,
    'Per-study run settings'); a sibling in the same request keeps the machine's defaults."""
    monkeypatch.setenv("FEAT_SLURM_ARGS", "--mem=64G --nice=100")
    run = {"timeout_s": 30000, "mem": "12G", "procs": 4}
    r = api.run({"big": _bar(queue=True, run=run), "sib": _bar(queue=True)}, "ref", wait=0)
    req = Path(r["request"])
    s, args = _job(req, "big")
    assert "#SBATCH --time=510" in s and "#SBATCH --mem=12G" in s     # ceil(30000 / 60) + 10
    assert "--nice=100" in args and not any(a.startswith("--mem") for a in args)
    assert json.loads((req / "big" / "job.json").read_text())["run"] == run
    s, args = _job(req, "sib")
    assert "#SBATCH --time=25" in s and "#SBATCH --mem" not in s      # FEAT_TIMEOUT 900; the args' --mem
    assert "--nice=100" in args and "--mem=64G" in args
    res = api.status(str(req), wait=120)
    assert res["done"], res
    lim = res["studies"]["big"]["execution"]["limits"]
    assert lim == {"timeout_s": 30000.0, "mem": "12288M", "from_study": ["mem", "procs", "timeout_s"]}
    assert res["studies"]["sib"]["execution"]["limits"] == {"timeout_s": 900.0, "mem": "65536M", "from_study": []}
    # threads: run.procs, as --cpus-per-task (feat's --ntasks), still under the size cap
    goal = {"analysis": "static", "part": {"goal": {"features": []}}}
    assert "#SBATCH --cpus-per-task=4" in executor.job_script(req, "g", dict(goal, run={"procs": 4}))
    assert executor.procs_for(dict(_bar(), run={"procs": 4})) == 1    # 99 nodes: one thread is plenty
    assert executor.queue_of({"run": {"queue": False}}) is False and executor.use_slurm({"run": {"queue": True}}, 0.1)


def test_run_settings_validation_and_caps(slurm, monkeypatch):
    monkeypatch.setenv("FEAT_MAX_TIMEOUT", "3600")
    monkeypatch.setenv("FEAT_MAX_MEM", "32G")
    monkeypatch.setenv("FEAT_MAX_PROCS", "8")

    def errs(**kw):
        return " | ".join(api.validate({"s": _bar(**kw)})["s"]["errors"])

    assert errs(run={"timeout_s": 3000, "mem": "500M", "procs": 8, "queue": True}) == ""
    assert "FEAT_MAX_TIMEOUT" in errs(run={"timeout_s": 30000})
    assert "FEAT_MAX_MEM" in errs(run={"mem": "33G"}) and errs(run={"mem": "32768M"}) == ""
    assert "FEAT_MAX_PROCS" in errs(run={"procs": 10}) and "FEAT_MAX_PROCS" in errs(procs=10)
    assert "unknown keys ['sbatch']" in errs(run={"sbatch": "--exclusive"})
    assert "not a SLURM memory size" in errs(run={"mem": "lots"}) and "not a SLURM memory size" in errs(run={"mem": 12})
    assert "disagrees with procs" in errs(procs=2, run={"procs": 4}) and errs(procs=4, run={"procs": 4}) == ""
    assert "disagrees with queue" in errs(queue=False, run={"queue": True})
    assert "run.timeout_s" in errs(run={"timeout_s": -1}) and "run: an object" in errs(run=5)
    st = api.study_format()["status"]["executor"]
    assert st["caps"] == {"timeout_s": 3600.0, "mem": "32G", "procs": 8} and st["timeout_s"] == 900.0


def test_run_by_hand_and_local_pool_record_limits(slurm, monkeypatch, tmp_path):
    from fea_translator.runner import run_study

    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    r = run_study("bar", _bar(), tmp_path / "hand", "ref", 321)
    assert r["execution"]["limits"] == {"timeout_s": 321.0, "mem": None, "from_study": []}
    monkeypatch.setenv("FEAT_EXECUTOR", "local")
    r = api.run({"bar": _bar(run={"timeout_s": 123})}, "ref", wait=120)
    assert r["done"] and r["studies"]["bar"]["execution"]["limits"]["timeout_s"] == 123.0
