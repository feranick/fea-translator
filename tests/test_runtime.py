"""feat on llmt-runtime (2026.11.09.6, phase 1; 2026.11.09.8, phase 2: the SLURM layer): the shared pieces are the
runtime's, configured for feat (runtime.RT and its hooks).
Version: 2026.11.09.8
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import llmt_runtime
import pytest
from llmt_runtime import jobs as J
from llmt_runtime import memory as M
from llmt_runtime import slurm as S
from llmt_runtime import workers as W
from llmt_runtime.testing import fake_slurm

from fea_translator import api, executor, runner, workers
from fea_translator.runtime import RT

ROOT = Path(__file__).resolve().parent.parent


def test_runtime_configured_for_feat(monkeypatch):
    assert (RT.prefix, RT.cli, RT.package, RT.kind) == ("FEAT", "feat", "fea_translator", "threads")
    monkeypatch.setenv("FEAT_OUT", "/tmp/feat-x")
    assert RT.out_root() == Path("/tmp/feat-x") == api.out_root()


def test_runtime_version_reported():
    assert api.tool_status()["runtime"] == llmt_runtime.__version__
    assert executor.status_key()["runtime"] == llmt_runtime.__version__


def test_shared_pieces_are_the_runtimes():
    assert workers.track is W.track and workers._CHILDREN is W._CHILDREN
    assert runner.oom_kills is M.oom_kills and runner.oom_scope is M.oom_scope
    assert runner.memory_limit_mb is M.memory_limit_mb and runner.node_oom_text is M.node_oom_text
    assert api.REQUEST_NAME.match("run_router_abc") and not api.REQUEST_NAME.match("../x")


def test_request_records_both_versions(tmp_path, monkeypatch):
    monkeypatch.setenv("FEAT_OUT", str(tmp_path))
    bad = {"x": {"analysis": "nope"}}
    r = api.run(bad, "ref", wait=1, request_name="run_versions")
    meta = json.loads((tmp_path / "run_versions" / "request.json").read_text())
    assert meta["translator"] == api.__version__ and meta["runtime"] == llmt_runtime.__version__
    assert r["translator"] == api.__version__


def test_ended_state_oom_kill_wording_with_feats_advice(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "job.json").write_text(json.dumps({"procs": 1, "mem": "8G"}))
    (d / "job.log").write_text("slurmstepd: error: Detected 1 oom_kill event in StepId=12.batch.\n")
    state, why = executor.ended_state("FAILED", d / "job.log")     # accounted as FAILED: the log still says oom_kill
    assert state == "out_of_memory" and why.startswith("SLURM killed the job for memory (8G with 1 thread)")
    state, why = executor.ended_state("TIMEOUT", d / "job.log")
    assert state == "timed_out" and "raise FEAT_TIMEOUT" in why and why.endswith("or coarsen the mesh / drop converge")


def test_sigterm_ends_a_workers_calculix(tmp_path):
    """Pool.terminate() sends SIGTERM: the worker now ends the CalculiX run it tracks (its own process group)."""
    code = ('import os, subprocess, sys, time\n'
            'from fea_translator import workers\n'
            'workers.watch_parent(os.getppid())\n'
            'c = subprocess.Popen(["sleep", "300"], start_new_session=True)\n'
            'workers.track(c)\n'
            'print(os.getpid(), c.pid, flush=True)\n'
            'time.sleep(300)\n')
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env)
    worker, child = (int(v) for v in p.stdout.readline().split())
    os.kill(worker, signal.SIGTERM)
    p.wait(timeout=15)
    t_end = time.time() + 10
    while time.time() < t_end:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        os.killpg(child, signal.SIGKILL)
        raise AssertionError("the worker's CalculiX run outlived its SIGTERM")


# ---- 2026.11.09.8, phase 2: the SLURM layer (settings, caps, time, job.sh, submission, states, queue, jobs, cancel,
# extend) is llmt-runtime's; feat's sizing comes in as hooks

FAKE_KNOBS = ("FAKE_SBATCH_FAIL", "FAKE_SBATCH_PENDING", "FAKE_PENDING", "FAKE_SBATCH_DELAY", "FAKE_SBATCH_SLEEP",
              "FAKE_SBATCH_SLOW_AFTER", "FAKE_MIN_JOB_AGE", "FAKE_PRIORITY", "FAKE_RUN_NODE", "FAKE_START",
              "FAKE_SACCT_FAIL", "FAKE_SQUEUE_FAIL", "FAKE_SCONTROL_NODES", "FAKE_SCONTROL_FAIL", "FAKE_SCONTROL_DENY",
              "FAKE_SINFO_FAIL", "FAKE_SINFO_PARTITIONS", "FAKE_SINFO_CPUS", "FAKE_TIME_LEFT")
BOX = {"analysis": "static", "part": {"box": [100, 10, 10], "divisions": [10, 2, 2]}}


@pytest.fixture()
def slurm(tmp_path, monkeypatch):
    """llmt-runtime's fake SLURM (one node fake1: 40 CPUs, 2 threads per core, 64000 MB); returns its state folder."""
    monkeypatch.setenv("FEAT_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("FEAT_EXECUTOR", "slurm")
    for k in FAKE_KNOBS:
        monkeypatch.delenv(k, raising=False)
    return fake_slurm(tmp_path, monkeypatch)


def _fake_job(state: Path, jid: str, **rec) -> None:
    """A job in the fake's queue (a record as its sbatch writes them; pid 999999: not running)."""
    d = state / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    base = {"name": "feat:r:s", "cpus": 1, "t0": time.time(), "log": "x", "pid": 999999, "node": "fake1"}
    (d / f"{jid}.json").write_text(json.dumps({**base, **rec}))


def _request(state: Path, jid: str, log: str | None = None) -> Path:
    """A request with one SLURM study 's' whose job.json names the job `jid`."""
    req = state.parent / "out" / f"run_{jid}"
    (req / "s").mkdir(parents=True)
    (req / "request.json").write_text(json.dumps({"names": ["s"], "slurm": ["s"], "solver": "ref", "timeout": 900,
                                                  "started": time.time()}))
    (req / "studies.json").write_text(json.dumps({"s": {"analysis": "static"}}))
    (req / "s" / "job.json").write_text(json.dumps({"executor": "slurm", "job_id": jid}))
    if log is not None:
        (req / "s" / "job.log").write_text(log)
    return req


def test_slurm_layer_is_the_runtimes():
    assert executor.deadline is S.deadline and executor.remaining is S.remaining and executor._CALL is S._CALL
    assert executor._NODES is S._NODES and executor._PART is S._PART and executor.FINAL == J.FINAL
    assert RT.time_aliases == {"estimate": "timeout"} and RT.slurm_time_default == "max"
    h = RT.hooks                                    # feat's sizing, through the hooks
    assert h.procs(BOX) == executor.procs_for(BOX) and h.time_need_min(BOX) == executor.time_limit_min(BOX)
    assert h.solves({"analysis": "joule", "converge": True}) == 4 and h.mem_auto(BOX) == executor.mem_auto(BOX)
    assert set(h.job_extra(BOX)) == {"nodes_estimate", "nodes_estimate_raw", "equation_solver", "memory"}
    assert h.solver_left_s(Path("/nonexistent"), {}) == 0.0


def test_slurm_time_estimate_keeps_its_feat_meaning(monkeypatch):
    """FEAT_SLURM_TIME=estimate meant the runs' timeouts, the meshing and 10 min in feat until 2026.11.09.7: the
    runtime's 'timeout'; RT.time_aliases keeps config/env files that say 'estimate' meaning that."""
    st = {"analysis": "static", "run": {"timeout_s": 600}}
    assert executor.time_limit_min(st) == 10 + 10 + 10
    for word in ("timeout", "estimate"):
        monkeypatch.setenv("FEAT_SLURM_TIME", word)
        t, info = executor.job_time(st)
        assert t == "30" and info["setting"] == "timeout" and info["time_from"] == "timeout"
        k = executor.status_key()
        assert k["slurm_time"] == k["time"] == "timeout" and k["runtime"] == llmt_runtime.__version__


def test_caps_and_job_memory_exact(slurm, monkeypatch):
    """The node's memory exactly (it read '62G', 63488 MB), --mem as sbatch takes it, auto never above the node's."""
    c = executor.caps()
    assert c == {"mem": "64000M", "mem_from": "a node's memory (SLURM RealMemory)", "mem_mb": 64000}
    assert executor.job_mem_mb() == 64000.0
    assert executor.given_mem_gb({}) == (64000 / 1024, "a node's memory (SLURM RealMemory) 64000M")
    assert executor.mem_for(dict(BOX, run={"mem": "8GB"})) == "8192M"       # sbatch refuses '8GB'
    monkeypatch.setattr(executor, "mem_auto", lambda st: "100G")            # the hook's answer, capped at the node
    assert executor.mem_for(BOX) == "64000M"
    assert "#SBATCH --mem=64000M" in executor.job_script(slurm.parent / "out" / "run_m", "a", BOX)
    assert executor.node_cores() == 20                                      # Sockets x CoresPerSocket


def test_queue_summary_counts_each_node_once(slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SINFO_PARTITIONS", "debug*|1:00:00;long|infinite")   # fake1 in two partitions
    _fake_job(slurm, "1001", state="PENDING", cpus=4)
    _fake_job(slurm, "1002", state="PENDING", name="cfdt:r:x")                      # another translator's
    assert executor.queue_summary() == {"queue": {"running": 0, "queued": 1, "cpus_in_use": 0},
                                        "machine": {"cpus": 8, "cpus_allocated": 2, "cpus_idle": 6}}


def test_queue_position_by_priority(slurm):
    _fake_job(slurm, "1001", state="PENDING", prio=10)
    _fake_job(slurm, "1002", state="PENDING", prio=2000)
    assert executor.ahead_of("1002") == 0 and executor.ahead_of("1001") == 1     # by priority, not by id
    assert sorted(executor.pending_ids()) == [1001, 1002]


def test_ended_job_still_in_squeue_is_not_running(slurm, monkeypatch):
    """squeue lists an ended job for MinJobAge (300 s) in its final state: feat 2026.11.09.7 read it as running."""
    monkeypatch.setenv("FAKE_MIN_JOB_AGE", "300")
    monkeypatch.setenv("FAKE_SACCT_FAIL", "1")
    _fake_job(slurm, "1001", ended=time.time(), final="FAILED")
    req = _request(slurm, "1001", log="Traceback (most recent call last):\nValueError: boom")
    assert executor.squeue(["1001"])["1001"]["state"] == "FAILED"
    st = executor.job_states(req, ["s"])["s"]
    assert st["state"] == "failed" and "ValueError: boom" in st["why"]


def test_no_job_log_and_no_accounting_is_cancelled(slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SACCT_FAIL", "1")
    _fake_job(slurm, "1001", vanished=True, pid=0)
    req = _request(slurm, "1001")                   # no job.log: SLURM writes it when a job starts
    st = executor.job_states(req, ["s"])["s"]
    assert st["state"] == "cancelled" and "without starting" in st["why"]


def test_job_json_keys_and_a_late_sbatch_adopted(slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")  # the jobs stay queued
    req = slurm.parent / "out" / "run_j"
    info = executor.submit(req, "a", BOX)
    old = {"executor", "job_id", "submitted", "procs", "time_min", "slurm_time", "slurm_time_info", "mem", "nodelist",
           "nodes_estimate", "nodes_estimate_raw", "equation_solver", "run", "command", "memory"}   # 2026.11.09.7's
    assert old <= set(info) and json.loads((req / "a" / "job.json").read_text()) == info
    assert info["translator"] == api.__version__ and info["runtime"] == llmt_runtime.__version__
    assert info["nodes_estimate"] == executor.nodes_hint(BOX) and info["job_id"] == "1001" and "adopted" not in info
    assert (req / "a" / "job.sh").read_text() == executor.job_script(req, "a", BOX)
    monkeypatch.setattr(J, "SBATCH_TIMEOUT", 1.0)
    monkeypatch.setenv("FAKE_SBATCH_SLOW_AFTER", "4")    # SLURM takes the job; sbatch answers after its timeout
    info = executor.submit(req, "b", BOX)
    assert info.get("job_id") == "1002" and "adopted" in info
    assert json.loads((req / "b" / "job.json").read_text())["adopted"]


def test_jobs_lists_a_running_study_with_feats_progress(slurm):
    _fake_job(slurm, "1001", pid=os.getpid())         # alive: RUNNING
    _request(slurm, "1001")
    j = api.jobs()
    s = j["requests"][0]["studies"]["s"]
    assert j["translator"] == api.__version__ and j["executor"] == "slurm"
    assert s["state"] == "running" and s["node"] == "fake1" and "eta_min_max" in s and "stage" in s
    assert j["queue"] == {"running": 1, "queued": 0, "cpus_in_use": 1}


def test_extend_through_the_runtime(slurm, monkeypatch):
    _fake_job(slurm, "1001", pid=os.getpid())
    req = _request(slurm, "1001")
    monkeypatch.setenv("FAKE_TIME_LEFT", "0:20:00")      # 20 min left: less than 1 h + 10 min
    r = api.extend(str(req), 1)["studies"]["s"]
    assert r["added_s"] == 3600 and r["slurm_job"] == "1001" and r["slurm_time_limit"] == "raised"
    assert json.loads((slurm / "jobs" / "1001.json").read_text())["updates"] == [["JobId=1001", "TimeLimit=+60"]]
    assert "CalculiX" in r["note"]
    # the running solve's side: llmt-runtime's take_extension (extend -> extend.applied.<t>; extended.json total_s)
    runner.EXTENDED["s"] = 0.0
    assert runner.take_extension(req / "s") == 3600 and runner.EXTENDED["s"] == 3600
    assert json.loads((req / "s" / "extended.json").read_text())["total_s"] == 3600
    assert not (req / "s" / "extend").exists() and list((req / "s").glob("extend.applied.*"))
    runner.EXTENDED["s"] = 0.0
