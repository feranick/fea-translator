"""Worker processes end with the server (workers.py, api.shutdown_workers): on the stale-server exit (os._exit skips
atexit) and when the server is killed (kill -9). carbonio, 2026.11.02: orphaned spawn workers and a resource_tracker
from three releases kept them 'in use' for llmt gc.
Version: 2026.11.03.1
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
    print(json.dumps({"workers": pids, "tracker": rt}), flush=True)
    if sys.argv[1] == "exit":
        mcp_server._exit_soon("test", 0.0)      # the stale-server exit: os._exit after shutdown_workers
    time.sleep(300)
'''


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[-1].split()[0] != "Z"   # a zombie has ended
    except FileNotFoundError:
        return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _start(tmp_path, mode):
    script = tmp_path / "server.py"
    script.write_text(SCRIPT)
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               FEAT_OUT=str(tmp_path / "out"), FEAT_WORKERS="2", FEAT_AUX_WORKERS="2")
    p = subprocess.Popen([sys.executable, str(script), mode], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=env, cwd=str(tmp_path))
    line = p.stdout.readline()
    if not line:
        p.kill()
        pytest.fail(f"the server script did not start: {p.stderr.read()[-2000:]}")
    info = json.loads(line)
    assert len(info["workers"]) >= 6, info     # 2 study + 2 aux + 1 volume + 1 deck
    assert all(_alive(w) for w in info["workers"])
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
