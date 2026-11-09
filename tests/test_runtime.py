"""feat on llmt-runtime (2026.11.09.6, phase 1): the shared pieces are the runtime's, configured for feat.
Version: 2026.11.09.6
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import llmt_runtime
from llmt_runtime import memory as M
from llmt_runtime import workers as W

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
