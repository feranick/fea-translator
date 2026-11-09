"""workers.py — feat's worker processes end with the server that started them (from llmt-runtime).
Version: 2026.11.09.6

    watch_parent(ppid)   initializer of every worker pool and worker process (study pool, aux pool, volume pool,
                         get_deck's process): a daemon thread ends the worker when its parent (or the server ppid) is
                         gone; SIGTERM (Pool.terminate) also ends the CalculiX runs it started
    track(p) / untrack(p)  the CalculiX runs a worker started (their own process groups): killed with it

The code is llmt-runtime's workers module (llmt_runtime.workers: polling os.getppid(), not PR_SET_PDEATHSIG, which
fires when the *thread* that created the child exits; carbonio, 2026.11.02: orphaned spawn workers kept three releases
'in use' for llmt gc). These names stay so that feat's callers and tests keep working.
"""
from __future__ import annotations

from llmt_runtime import workers as _W
from llmt_runtime.workers import end_children, on_orphan, track, untrack  # noqa: F401 (feat's names)

PARENT_POLL_S = _W.PARENT_POLL_S
_CHILDREN = _W._CHILDREN
_STARTED = _W._STARTED


def watch_parent(ppid: int | None = None) -> None:
    """Starts (once per process) the watcher that ends this worker, and the CalculiX runs it started, when its parent
    `ppid` (default: the parent now) is gone (llmt_runtime.workers.watch_parent)."""
    _W.PARENT_POLL_S = PARENT_POLL_S       # (a test may set a shorter poll here)
    _W.watch_parent(ppid, name="feat")
