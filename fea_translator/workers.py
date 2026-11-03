"""workers.py — feat's worker processes end with the server that started them.
Version: 2026.11.03.1

    watch_parent(ppid)   initializer of every worker pool and worker process (study pool, aux pool, volume pool,
                         get_deck's process): a daemon thread ends the worker when its parent is gone
    track(p) / untrack(p)  the CalculiX runs a worker started (their own process groups): killed with it

A server that exits cleanly stops its pools (api.shutdown_workers). This is the backstop for a server that crashes
or is killed (kill -9): carbonio, 2026.11.02, had 7 orphaned spawn workers and a resource_tracker from three
releases, which kept them 'in use' for llmt gc. A spawn worker holds both ends of its task queue, so it never sees
EOF when the server dies and would wait forever.

Not PR_SET_PDEATHSIG: on Linux it fires when the *thread* that created the child exits, and feat's pools are
created from the MCP server's worker threads, which come and go; it would kill healthy workers in mid-study.
Every platform polls os.getppid() instead (PARENT_POLL_S): when the parent dies the worker is re-parented
(to init, launchd or a subreaper) and its parent pid changes.
"""
from __future__ import annotations

import os
import signal
import threading
import time

PARENT_POLL_S = 1.0
_CHILDREN: set = set()          # subprocess.Popen objects started in their own process group (CalculiX)
_STARTED = {"pid": None}        # the process the watcher runs in (a fork would inherit the flag)


def track(p) -> None:
    _CHILDREN.add(p)


def untrack(p) -> None:
    _CHILDREN.discard(p)


def _orphaned() -> None:
    for p in list(_CHILDREN):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    os._exit(0)


def watch_parent(ppid: int | None = None) -> None:
    """Starts (once per process) a daemon thread that ends this process, and the CalculiX runs it started, when
    its parent `ppid` (default: the parent now) is gone."""
    if _STARTED["pid"] == os.getpid():
        return
    _STARTED["pid"] = os.getpid()
    ppid = int(ppid or os.getppid())
    if os.getppid() != ppid:      # the parent died before the worker started
        _orphaned()

    def loop():
        while True:
            time.sleep(PARENT_POLL_S)
            if os.getppid() != ppid:
                _orphaned()
    threading.Thread(target=loop, daemon=True, name="feat-parent-watch").start()
