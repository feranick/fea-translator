"""Worker processes end with the server (workers.py, api.shutdown_workers): on the stale-server exit (os._exit skips
atexit) and when the server is killed (kill -9). carbonio, 2026.11.02: orphaned spawn workers and a resource_tracker
from three releases kept them 'in use' for llmt gc.
Version: 2026.11.18.3
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = '''
import json, multiprocessing as mp, os, sys, time


def hold(ppid):   # a get_deck-like worker process
    from fea_translator.workers import watch_parent
    watch_parent(ppid)
    time.sleep(300)


if __name__ == "__main__":
    from fea_translator import api, executor, mcp_server
    api._POOL.start()
    api._aux_pool()
    executor._volume_pool().submit(os.getpid).result(timeout=120)
    d = mp.get_context("spawn").Process(target=hold, args=(os.getpid(),), daemon=True)
    d.start()
    api._DECK["proc"] = d
    time.sleep(2)                               # every worker up, its watcher running
    from multiprocessing import resource_tracker
    pids = [p.pid for p in mp.active_children()]
    rt = getattr(resource_tracker._resource_tracker, "_pid", None)
    roles = {**{str(w.pid): "study pool" for w in api._POOL.pool._pool},
             **{str(w.pid): "aux pool" for w in api._AUX["pool"]._pool},
             **{str(w.pid): "volume pool" for w in executor._VPOOL["ex"]._processes.values()}, str(d.pid): "deck"}
    print(json.dumps({"workers": pids, "tracker": rt, "roles": roles}), flush=True)
    if sys.argv[1] == "exit":
        sys.stdin.readline()                    # after the test has seen every worker alive (no race with the exit)
        mcp_server._exit_soon("test", 0.0)      # the stale-server exit: os._exit after shutdown_workers
    time.sleep(300)
'''


def _alive(pid: int) -> bool:
    """A process that exists and is not a zombie. Linux: /proc; elsewhere (macOS has no /proc, which this helper took
    for 'every process is gone' before 2026.11.09.6): kill(pid, 0), then ps for the zombie state."""
    if os.path.isdir("/proc"):
        try:
            with open(f"/proc/{pid}/stat") as fh:
                return fh.read().rsplit(")", 1)[-1].split()[0] != "Z"   # a zombie has ended
        except FileNotFoundError:
            return False
        except OSError:
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return True
    return bool(st.strip()) and not st.strip().startswith("Z")


def _start(tmp_path, mode):
    script = tmp_path / "server.py"
    script.write_text(SCRIPT)
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               FEAT_OUT=str(tmp_path / "out"), FEAT_WORKERS="2", FEAT_AUX_WORKERS="2")
    p = subprocess.Popen([sys.executable, str(script), mode], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, env=env, cwd=str(tmp_path))
    line = p.stdout.readline()
    if not line:
        p.kill()
        pytest.fail(f"the server script did not start: {p.stderr.read()[-2000:]}")
    info = json.loads(line)
    assert len(info["workers"]) >= 6, info     # 2 study + 2 aux + 1 volume + 1 deck
    dead = [w for w in info["workers"] if not _alive(w)]
    if dead:                                   # which worker, and what ps says of it (a start-up race, or a real exit)
        ps = {w: subprocess.run(["ps", "-o", "pid=,ppid=,stat=,etime=,command=", "-p", str(w)], capture_output=True,
                                text=True).stdout.strip() or "(gone)" for w in dead}
        p.kill()
        err = p.stderr.read()[-2000:] if p.stderr else ""
        pytest.fail(f"workers not alive at start: {[(w, info.get('roles', {}).get(str(w), '?'), ps[w]) for w in dead]}; "
                    f"server stderr: {err}")
    return p, info


def _all_gone(pids, within=15.0):
    t_end = time.time() + within
    while time.time() < t_end:
        left = [w for w in pids if w and _alive(w)]
        if not left:
            return []
        time.sleep(0.25)
    return left


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX processes")
def test_the_stale_exit_stops_every_worker(tmp_path):
    p, info = _start(tmp_path, "exit")
    p.stdin.write("go\n")                        # every worker seen alive: now the stale exit
    p.stdin.flush()
    p.wait(timeout=60)
    assert p.returncode == 0, p.stderr.read()[-2000:]
    assert _all_gone(info["workers"] + [info["tracker"]]) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX processes")
def test_workers_end_when_the_server_is_killed(tmp_path):
    p, info = _start(tmp_path, "kill")
    os.kill(p.pid, signal.SIGKILL)               # no cleanup at all: the workers' watchers end them
    p.wait(timeout=30)
    assert _all_gone(info["workers"] + [info["tracker"]]) == []


def test_watch_parent_ends_an_orphan_and_its_calculix(tmp_path):
    """The watcher's exit kills the process groups the worker started (a CalculiX run)."""
    code = ('import os, subprocess, sys, time\n'
            'from fea_translator import workers\n'
            'if os.fork() == 0:\n'                  # the worker: its parent is the short-lived middle process
            '    workers.PARENT_POLL_S = 0.2\n'
            '    workers.watch_parent(os.getppid())\n'
            '    c = subprocess.Popen(["sleep", "300"], start_new_session=True)\n'
            '    workers.track(c)\n'
            '    print(os.getpid(), c.pid, flush=True)\n'
            '    time.sleep(300)\n'
            'else:\n'
            '    time.sleep(300)\n')
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env)
    worker, child = (int(v) for v in p.stdout.readline().split())
    assert _alive(worker) and _alive(child)
    os.kill(p.pid, signal.SIGKILL)
    p.wait(timeout=30)
    assert _all_gone([worker, child]) == []


# ---- 2026.11.18.3: worker processes through llmt-runtime's make_pool / make_executor / make_process ----------------

def _refuse(*a, **k):
    from llmt_runtime.workers import WorkersNotPermitted
    raise WorkersNotPermitted(PermissionError(1, "Operation not permitted"))


def test_refused_workers_are_said_and_the_server_stays_up(tmp_path, monkeypatch):
    """The Gemini Mac app's Local MCP with a customized sandbox refuses multiprocessing's semaphores (b3dt,
    2026.11.18): a local run answers with the runtime's text, nothing is half-submitted, and the next call works."""
    import asyncio

    from fea_translator import api, bench, executor
    from fea_translator import mcp_server as M
    monkeypatch.setenv("FEAT_OUT", str(tmp_path))
    monkeypatch.setenv("FEAT_EXECUTOR", "local")
    monkeypatch.setattr(api, "make_pool", _refuse)
    monkeypatch.setattr(api._POOL, "pool", None)
    monkeypatch.setattr(M.GUARD, "stale", lambda: None)
    bar = bench.cases("ref")["axial_bar"]
    with pytest.raises(M.ToolError, match="worker processes are not permitted here"):
        asyncio.run(M.server.call_tool("run_studies", {"studies": {"bar": bar}, "solver": "ref", "wait_s": 1}))
    assert not list(tmp_path.glob("run_*/request.json"))               # nothing recorded or submitted
    r = asyncio.run(M.server.call_tool("validate_studies", {"studies": {"bar": bar}}))   # the server answers on
    assert '"ok": true' in str(r)
    monkeypatch.setattr(api, "make_process", _refuse)                     # get_deck's meshing process
    with pytest.raises(Exception, match="worker processes are not permitted here"):
        api.deck(bar, "bar", wait=5)
    monkeypatch.setattr(api, "make_pool", _refuse)                        # rendering's pool
    monkeypatch.setitem(api._AUX, "pool", None)
    with pytest.raises(Exception, match="worker processes are not permitted here"):
        api.in_worker("rendering", "render_images", {"study_folder": "x"}, 1)


def test_refused_volume_estimate_is_left_out(monkeypatch):
    """Validation's part-volume estimate is advice: refused workers leave it out with a warning; validation goes on."""
    from fea_translator import api, executor
    monkeypatch.setattr(executor, "make_executor", _refuse)
    monkeypatch.setitem(executor._VPOOL, "ex", None)
    monkeypatch.setitem(executor.VOLUME_REFUSED, "why", None)
    monkeypatch.setattr(executor, "remaining", lambda: 40.0)              # inside an MCP call
    monkeypatch.setattr(executor, "_VOLUMES", {})
    st = {"analysis": "static", "material": "steel", "part": {"step": "/nonexistent/part.step"},
          "bcs": [{"kind": "fixed", "faces": {"nx": -1}}, {"kind": "force", "faces": {"nx": 1}, "fx": 1}]}
    assert executor._part_volume(st["part"]) is None
    assert "worker processes are not permitted here" in executor.VOLUME_REFUSED["why"]
    v = api.validate({"s": st})["s"]
    assert any("no size estimate for parts given as goals or STEP files" in w for w in v.get("warnings", []))
