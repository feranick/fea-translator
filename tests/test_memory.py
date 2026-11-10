"""Out of memory, threads sized to memory, and CalculiX's time left from its increments.
carbonio job 5797 (an S50 sleeve assembly, 2026.11.05): 6 threads in a 32 GB job were killed at 33.4 GB after 53 min
(feat's estimate was 62 GB for 6 threads, 31 GB for 1); feat took it for a crash and restarted it on 1 thread without
saying so, and the step-time share then said 160 h for about 11 h.
Version: 2026.11.09.8
"""
import json
import time
from pathlib import Path

import llmt_runtime.slurm
import pytest

from fea_translator import api, executor, runner

BOX = {"analysis": "static", "material": "steel", "equation_solver": "direct", "procs": 6,
       "part": {"box": [100, 10, 10], "divisions": [80, 16, 16]},
       "bcs": [{"kind": "fixed", "name": "w", "faces": {"nx": -1}}, {"kind": "force", "name": "t", "faces": {"nx": 1}, "fz": -1}]}

FAKE = """#!/bin/sh
# killed (SIGKILL, as the OOM killer does) whenever it runs with more than one thread
printf '     1     1     1     1   5000  0.1000E+01  0.1000E+00  0.0  0.0\\n' > $2.cvg
if [ "$OMP_NUM_THREADS" -gt 1 ] || [ -n "$ALWAYS_KILL" ]; then kill -9 $$; fi
echo done > $2.dat
"""


@pytest.fixture
def fake_ccx(tmp_path, monkeypatch):
    exe = tmp_path / "fake_ccx"
    exe.write_text(FAKE)
    exe.chmod(0o755)
    monkeypatch.setenv("FEAT_CCX", str(exe))
    kills = {"n": 0, "hits": 0, "job": True}   # job: the job's limit was reached (else the node ran out)

    def oom_kills():   # the cgroup's counter goes up by one each time CalculiX is killed
        return {"memory.events": kills["n"]}

    def waited(*a, **k):
        code = real_wait(*a, **k)
        if code == -9:
            kills["n"] += 1
            kills["hits"] += 1 if kills["job"] else 0
        return code
    real_wait = runner._wait_rss
    monkeypatch.setattr(runner, "oom_kills", oom_kills)
    monkeypatch.setattr(runner, "oom_limit_hits", lambda: {"cg": kills["hits"]})
    monkeypatch.setattr(runner, "kernel_oom_constraint", lambda pid, since: None)
    monkeypatch.setattr(runner, "node_jobs_text", lambda: "")
    monkeypatch.setattr(runner, "KILLS", kills, raising=False)
    monkeypatch.setattr(runner, "_wait_rss", waited)
    monkeypatch.setattr(runner, "memory_limit_mb", lambda: 32768.0)
    runner.CRASHES.clear()
    d = tmp_path / "study"
    d.mkdir()
    inp = d / "s.inp"
    inp.write_text("")
    return inp


def test_out_of_memory_restart_is_said_at_once(fake_ccx, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    monkeypatch.setenv("CCX_NPROC_EQUATION_SOLVER", "6")
    runner.run_ccx(fake_ccx, 20)
    rec = runner.CRASHES[-1]
    assert rec["out_of_memory"] and rec["threads"] == 6 and rec["limit_mb"] == 32768.0 and rec["iterations"] == 1
    assert "out of memory: killed at" in rec["message"] and "of 32.0 GB with 6 threads" in rec["message"]
    assert "rerunning on 1 thread" in rec["message"]
    assert (fake_ccx.parent / "s.threaded.log").is_file()                 # the killed run's log is kept
    r = json.loads((fake_ccx.parent / runner.RESTART_FILE).read_text())
    assert r["solve"] == "s" and r["out_of_memory"] and r["at"] <= time.time()


def test_out_of_memory_on_one_thread_is_an_error(fake_ccx, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("CCX_NPROC_EQUATION_SOLVER", "1")
    monkeypatch.setenv("ALWAYS_KILL", "1")
    with pytest.raises(runner.SolverError, match="ran out of memory with 1 thread"):
        runner.run_ccx(fake_ccx, 20)


def test_out_of_memory_test():
    assert runner.out_of_memory(-9, {"a": 0}, {"a": 1}, 100, None)
    assert not runner.out_of_memory(-11, {"a": 0}, {"a": 0}, 100, 32768)          # a crash
    assert runner.out_of_memory(-9, {}, {}, 31000, 32768)                          # no counters: near the limit
    assert not runner.out_of_memory(-9, {}, {}, 1000, 32768)
    assert not runner.out_of_memory(0, {"a": 0}, {"a": 1}, 100, None)


@pytest.fixture
def machine(monkeypatch):
    monkeypatch.setattr(executor, "node_cores", lambda: 12)
    # the machine is llmt-runtime's (2026.11.09.8): no SLURM cluster where it asks (caps, node pinning)
    monkeypatch.setattr(llmt_runtime.slurm, "cluster_nodes", lambda: None)
    for v in ("FEAT_SLURM_MEM", "FEAT_MAX_MEM", "FEAT_SLURM_ARGS", "FEAT_MAX_PROCS"):
        monkeypatch.delenv(v, raising=False)


def test_threads_sized_to_memory(machine, tmp_path):
    many, one = executor.mem_need_gb(BOX, 6), executor.mem_need_gb(BOX, 1)
    assert one < 12 < many                                        # ~9.6 GB and ~18 GB for this box
    assert executor.procs_for(BOX) == 6                           # feat's own estimate: the threads get their memory
    small = dict(BOX, run={"mem": "12G"})
    assert executor.procs_for(small) == 1                         # 12 GB: 1 thread from the start, 1 CPU from SLURM
    assert "--cpus-per-task=1" in executor.job_script(tmp_path, "x", small)
    note = executor.memory_plan(small)["note"]
    assert note.startswith(f"run.mem 12G is below the {many:.0f} GB estimate for 6 threads: give {executor.ask_gb(many)}G")
    assert "1 thread fits in 12 GB" in note
    assert executor.procs_for(dict(BOX, run={"mem": "32G"})) == 6
    tiny = executor.memory_plan(dict(BOX, run={"mem": "8G"}))
    assert tiny["threads"] == 1 and "even 1 thread needs about" in tiny["note"]
    v = api.validate({"small": small})["small"]
    assert v["memory"]["estimate_gb"]["6"] == round(many, 1) and any("below the" in w for w in v["warnings"])
    assert set(v["estimate"]) == {"minutes", "procs", "mem_mb"} and v["estimate"]["procs"] == 1


DECK = "*STEP\n*STATIC\n0.05, 1., 1e-05, 0.25\n*END STEP\n*STEP\n*STATIC\n1., 1., 1e-08, 1.\n*END STEP\n"
HEAD = "SUMMARY OF JOB INFORMATION\n  STEP      INC     ATT  ITRS     TOT TIME     STEP TIME      INC TIME\n"


def _cvg(rows):
    head = " SUMMARY OF C0NVERGENCE INFORMATION\n  STEP   INC   ATT  ITER     CONT.   RESID.        CORR.\n\n"
    return head + "".join(f"{s:6d}{i:6d}{1:6d}{k:6d}{5000:10d}  0.1000E+01  0.1000E+00  0.0000E+00  0.0000E+00\n"
                          for s, i, k in rows)


def test_increment_model():
    st = api.static_increments(DECK)
    assert st == [{"initial": 0.05, "period": 1.0, "min": 1e-05, "max": 0.25, "fixed": False},
                  {"initial": 1.0, "period": 1.0, "min": 1e-08, "max": 1.0, "fixed": False}]
    # from 0.05 after one increment: 0.05 (no growth after the first), then 0.075, 0.1125, 0.169, 0.25, 0.25, rest
    assert api.increments_left(0.05, 0.05, st[0], 1) == 7
    assert api.increments_left(0.0, 1.0, st[1], 0) == 1


def test_time_left_carbonio_5797(tmp_path):
    """One increment in (16 iterations), the second at its 9th, 25 iterations in 2 h since the 1-thread restart, a
    modal run after: a range around ~9-10 h, not 160 h, and no 'extend or cancel' while the low end is in the limit."""
    d = tmp_path
    (d / "s.inp").write_text(DECK)
    (d / "s.sta").write_text(HEAD + "     1          1     1    16  0.500000E-01  0.500000E-01  0.500000E-01\n")
    (d / "s.cvg").write_text(_cvg([(1, 1, k) for k in range(1, 17)] + [(1, 2, k) for k in range(1, 10)]))
    now = time.time()
    (d / runner.RESTART_FILE).write_text(json.dumps({
        "solve": "s", "threads": 6, "out_of_memory": True, "peak_rss_mb": 34200, "limit_mb": 32768, "run_s": 3180,
        "iterations": 8, "s_per_iteration": 397.5, "at": now - 7200,
        "message": "out of memory: killed at 33.4 GB of 32.0 GB with 6 threads after 53 min; rerunning on 1 thread"}))
    p = {"stage_started": now - 3 * 3600, "ccx_steps": 2, "modal_runs": 1}
    cx = api.ccx_progress(d, p, now)
    spi = 7200 / 25
    assert cx["timed_from"] == "the single-threaded restart" and cx["s_per_iteration"] == pytest.approx(spi, rel=1e-3)
    assert cx["increments_left_est"] == 8 and cx["iterations_left_est"] == 8 * 16 - 9
    assert cx["eta_min"] == pytest.approx((119 + 3) * spi / 60, abs=0.2)
    assert cx["eta_min_low"] == pytest.approx((119 * 0.5 + 1) * spi / 60, abs=0.2)
    assert cx["eta_min_high"] == pytest.approx((119 * 2 + 10) * spi / 60, abs=0.2)
    assert cx["fraction_of_step_time"] == 0.025 and 0.15 < cx["fraction_done"] < 0.2
    # in progress: the restart said first; a limit between the range's ends is a note, not 'extend or cancel'
    (d / "progress.json").write_text(json.dumps(dict(p, analysis="assembly", solver="ccx", steps=2, step=2, stage="solve",
                                                     timeout_s=129600, started=now - 3 * 3600, done_steps_s=[16],
                                                     nodes=424303)))
    out = api._progress(d, {"analysis": "assembly", "bcs": [], "steps": [{"kind": "static"}]}, "ccx", 43200)   # 600 min left of the restart
    assert out["warning"].startswith("CalculiX out of memory: killed at 33.4 GB of 32.0 GB with 6 threads")
    assert "extend(request)" not in out["warning"] and out["eta_min_range"][0] < out["eta_min_max"] < out["eta_min_range"][1]
    assert "may run past its limit" in out["note"] and out["restarted"]["slower_than_before"] == pytest.approx(spi / 397.5, rel=1e-2)


def test_time_left_job_419_with_its_deck(tmp_path):
    d = tmp_path
    (d / "s.inp").write_text(DECK)
    (d / "s.sta").write_text(HEAD + "".join(f"     1          {i}     1    {k}  {t:.6E}  {t:.6E}  {dt:.6E}\n" for i, k, t, dt in
                                            ((1, 8, 0.05, 0.05), (2, 9, 0.1, 0.05), (3, 12, 0.175, 0.075))))
    (d / "s.cvg").write_text(_cvg([(1, 1, k) for k in range(1, 9)] + [(1, 2, k) for k in range(1, 10)]
                                  + [(1, 3, k) for k in range(1, 13)] + [(1, 4, 1), (1, 4, 2)]))
    now = time.time()
    cx = api.ccx_progress(d, {"stage_started": now - 31 * 900, "ccx_steps": 2}, now)
    # the rest of step 1 from 0.175: 0.1125, 0.169, 0.25, 0.25, 0.044 (5), step 2 in one; 10.5 iterations each
    assert cx["increments_left_est"] == 6 and cx["iterations_left_est"] == 63 - 2
    assert cx["eta_min"] == pytest.approx(61 * 900 / 60, rel=1e-3)      # ~15 h at 15 min per iteration, not 110 h


def test_crash_message():
    m = runner.crash_message({"threads": 2, "crash": "signal 11 (SIGSEGV)", "run_s": 600})
    assert m.startswith("crashed (signal 11 (SIGSEGV), no error message) with 2 threads after 10 min; rerunning on 1")


# ---- 2026.11.06.1: carbonio job 5807 (run.procs 6, run.mem 30G): 107,171 nodes estimated, 424,303 meshed; SLURM
# killed the whole batch step 7 min in ('2 oom_kill events in StepId=5807.batch'): no restart, no result

def test_threads_from_the_meshed_count(monkeypatch):
    monkeypatch.setattr(runner, "_own_rss_gb", lambda: 0.0)
    t, note = runner.threads_for_memory(424303, "direct", 6, 30 * 1024.0)
    assert t == 1 and note["estimate_gb"] == {"1": 31.6, "6": 62.2} and note["threads"] == 1
    assert note["message"].startswith("the meshed 424,303 nodes need about 62 GB with 6 threads and 32 GB with 1; "
                                      "the job has 30 GB: CalculiX runs on 1 thread; it may run out of memory: ask for 33G")
    assert runner.threads_for_memory(424303, "direct", 6, 64 * 1024.0) == (6, None)
    assert runner.threads_for_memory(424303, "direct", 6, None) == (6, None)


def test_run_ccx_cuts_threads_for_the_meshed_size(tmp_path, monkeypatch):
    exe = tmp_path / "fake_ccx"
    exe.write_text('#!/bin/sh\necho "$OMP_NUM_THREADS $CCX_NPROC_EQUATION_SOLVER" > $2.dat\n')
    exe.chmod(0o755)
    monkeypatch.setenv("FEAT_CCX", str(exe))
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    monkeypatch.setenv("CCX_NPROC_EQUATION_SOLVER", "6")
    monkeypatch.setattr(runner, "memory_limit_mb", lambda: 30 * 1024.0)
    monkeypatch.setattr(runner, "_own_rss_gb", lambda: 0.0)
    runner.MEMORY_NOTES.clear()
    inp = tmp_path / "a.inp"
    inp.write_text("")
    runner.run_ccx(inp, 20, nodes=424303)
    assert (tmp_path / "a.dat").read_text().split() == ["1", "1"]
    assert runner.MEMORY_NOTES[-1]["solve"] == "a" and json.loads((tmp_path / runner.MEMORY_FILE).read_text())["threads"] == 1
    runner.run_ccx(inp, 20, nodes=20000)                       # a small mesh keeps its threads
    assert (tmp_path / "a.dat").read_text().split() == ["6", "6"]


def test_slurm_oom_of_the_batch_step(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "job.json").write_text(json.dumps({"procs": 6, "mem": "30G"}))
    (d / "progress.json").write_text(json.dumps({"stage": "solve", "nodes": 424303}))
    (d / "job.log").write_text("[2026-11-06T12:49:50.382] error: Detected 2 oom_kill events in StepId=5807.batch. "
                               "Some of the step tasks have been OOM Killed.\n")
    state, why = executor.ended_state(None, d / "job.log")
    assert state == "out_of_memory"
    assert why == ("SLURM killed the job for memory (30G with 6 threads); for the meshed 424,303 nodes feat estimates "
                   "about 62 GB with 6 thread(s) and 32 GB with 1: run.mem 64G for 6 threads, or run.procs 1 with "
                   "run.mem 33G (or a coarser mesh)")


def test_node_estimate_calibrated_by_this_machines_meshes(tmp_path, monkeypatch):
    monkeypatch.setenv("FEAT_OUT", str(tmp_path))
    monkeypatch.setattr(executor, "estimate_nodes_raw", lambda st: 107171)
    monkeypatch.setattr(executor, "nodes_hint", lambda st: None)
    st = {"analysis": "assembly"}
    assert executor.estimate_nodes(st) == 107171                     # no records: as estimated
    recs = [{"analysis": "assembly", "nodes": 424303, "nodes_estimate_raw": 107171},
            {"analysis": "static", "nodes": 1000, "nodes_estimate_raw": 4000}]   # a smaller mesh never lowers it
    (tmp_path / "timing.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert executor.estimate_nodes(st) == pytest.approx(424303, abs=2)
    assert executor.estimate_nodes({"analysis": "static"}) == 107171


# ---- 2026.11.07.2: the node's memory is a cap (reported, enforced); request_name

def test_node_memory_is_a_cap(monkeypatch):
    from fea_translator.study import validate_study
    monkeypatch.setenv("FEAT_EXECUTOR", "slurm")
    for v in ("FEAT_MAX_MEM", "FEAT_SLURM_ARGS", "FEAT_SLURM_NODE"):
        monkeypatch.delenv(v, raising=False)
    # caps is llmt-runtime's (2026.11.09.8): its node memory comes from llmt_runtime.slurm's cluster_nodes/this_node
    monkeypatch.setattr(llmt_runtime.slurm, "cluster_nodes", lambda: [{"NodeName": "carbonio", "RealMemory": "64000"},
                                                                         {"NodeName": "x", "RealMemory": "1"}])
    monkeypatch.setattr(llmt_runtime.slurm, "this_node", lambda nodes: nodes[0] if nodes else None)
    c = executor.caps()
    # llmt-runtime fix (2026.11.09.8): the node's memory exactly, '64000M' / 64000 MB (it read '62G' / 63488, rounded
    # down to GB, and refused a run.mem of 64000M)
    assert c["mem"] == "64000M" and c["mem_mb"] == 64000 and c["mem_from"] == "a node's memory (SLURM RealMemory)"
    errs = " | ".join(validate_study(dict(BOX, run={"mem": "90G"}))["errors"])
    assert "run.mem: 90G is above 64000M, the most on this machine (a node's memory (SLURM RealMemory))" in errs
    assert validate_study(dict(BOX, run={"mem": "64000M"}))["ok"]     # the fix: the node's whole memory may be asked
    assert validate_study(dict(BOX, run={"mem": "60G"}))["ok"]
    monkeypatch.setenv("FEAT_MAX_MEM", "32G")                 # a lower setting stays, and says so
    c = executor.caps()
    assert c["mem"] == "32G" and c["mem_mb"] == 32768 and c["mem_from"] == "FEAT_MAX_MEM"
    monkeypatch.setenv("FEAT_EXECUTOR", "local")              # no SLURM: only the settings
    monkeypatch.delenv("FEAT_MAX_MEM")
    assert "mem" not in executor.caps()


def test_request_name(tmp_path, monkeypatch):
    monkeypatch.setenv("FEAT_OUT", str(tmp_path))
    monkeypatch.setenv("FEAT_EXECUTOR", "local")
    bad = {"s": {"analysis": "static"}}                         # invalid: answered at once, nothing runs
    r = api.run(bad, "ref", wait=1, request_name="run_router_abc123")
    assert r["request"] == str(tmp_path / "run_router_abc123") and r["done"]
    again = api.run(bad, "ref", wait=1, request_name="run_router_abc123")
    assert again["request"] == r["request"] and "submitted before" in again["note"]
    assert "error" in api.run(bad, "ref", wait=1, request_name="../x")
    (tmp_path / "run_half").mkdir()
    assert "holds no request" in api.run(bad, "ref", wait=1, request_name="run_half")["error"]



# ---- 2026.11.08.1: the node ran out, not the job (carbonio, 2026.11.07: killed at 20.1 and 13.9 GB in 34 GB jobs)

def test_oom_scope():
    sc = runner.oom_scope
    # the job's limit: a kill with a 'limit reached' count
    assert sc({"k": 0}, {"k": 1}, {"c": 0}, {"c": 1}, -9, 33000, 34816, 20000, 64000) == "job"
    # the node: a kill without one, far below the limit
    assert sc({"k": 0}, {"k": 1}, {"c": 0}, {"c": 0}, -9, 14234, 34816, 300, 64000) == "node"
    # the kernel's word wins
    assert sc({}, {}, {}, {}, -9, 14234, 34816, None, None, kernel="CONSTRAINT_NONE") == "node"
    assert sc({}, {}, {}, {}, -9, 14234, 34816, None, None, kernel="CONSTRAINT_MEMCG") == "job"
    # cgroup v1 (no 'limit reached' count): the sizes decide
    assert sc({"k": 0}, {"k": 1}, {}, {}, -9, 14234, 34816, 300, 64000) == "node"
    assert sc({"k": 0}, {"k": 1}, {}, {}, -9, 33000, 34816, 300, 64000) == "job"
    # no counters at all: SIGKILL near the limit, or with the node out of memory
    assert sc({}, {}, {}, {}, -9, 33000, 34816, None, None) == "job"
    assert sc({}, {}, {}, {}, -9, 14234, 34816, 500, 64000) == "node"
    assert sc({}, {}, {}, {}, -9, 14234, 34816, 30000, 64000) is None     # a plain kill
    assert sc({}, {}, {}, {}, -11, 14234, 34816, 500, 64000) is None      # a crash
    assert sc({"k": 0}, {"k": 1}, {"c": 0}, {"c": 1}, 0, 1, 1, 1, 1) is None


def test_node_out_of_memory_is_not_a_memory_request(fake_ccx, monkeypatch):
    runner.KILLS["job"] = False
    monkeypatch.setattr(runner, "meminfo_mb", lambda: (64000.0, 300.0))
    monkeypatch.setattr(runner, "node_jobs_text", lambda: "; running on carbonio now: 5811 (feat:..., 34G), 5812 (x, 40G)")
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    monkeypatch.setenv("CCX_NPROC_EQUATION_SOLVER", "6")
    runner.run_ccx(fake_ccx, 20)                                # threaded: restarted on 1 thread, said so
    rec = runner.CRASHES[-1]
    assert rec["oom_scope"] == "node" and rec["node_min_free_mb"] == 300.0 and rec["node_total_mb"] == 64000.0
    assert rec["message"].startswith("the node ran out of memory (other processes): killed at ")
    assert "below the job's 32.0 GB; the node's free memory fell to 0.3 GB of 62.5 GB; running on carbonio" in rec["message"]
    assert "more run.mem will not help" in rec["message"]
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("CCX_NPROC_EQUATION_SOLVER", "1")
    monkeypatch.setenv("ALWAYS_KILL", "1")
    with pytest.raises(runner.SolverError) as e:                 # one thread: an error that says what to do
        runner.run_ccx(fake_ccx, 20)
    assert "because the node ran out of memory (other processes)" in str(e.value)
    assert "Run it when the node is free (jobs(), machines()); more run.mem will not help" in str(e.value)
    assert "Ask for more memory" not in str(e.value)


def test_kernel_log(monkeypatch):
    import subprocess as sp
    log = ("Memory cgroup out of memory: Killed process 213447 (ccx)\n"
           "oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=task_0,mems_allowed=0,task=ccx,pid=213447,uid=1000\n"
           "oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,task=ccx,pid=5461,uid=1002\n")
    monkeypatch.setattr(runner.shutil, "which", lambda c: "/usr/bin/journalctl")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 0, log, ""))
    assert runner.kernel_oom_constraint(213447, 0) == "CONSTRAINT_MEMCG"
    assert runner.kernel_oom_constraint(5461, 0) == "CONSTRAINT_NONE"
    assert runner.kernel_oom_constraint(1, 0) is None
