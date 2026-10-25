"""The MCP server: stale servers after a deploy (llm-translators docs/SHARED.md) and every tool answering within
its time (docs/EXECUTION.md: the router counts on an answer within 50 s).
Version: 2026.10.25.1
"""
import asyncio
import json
import os
import time
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")
from fea_translator import mcp_server as S  # noqa: E402

FAKE = Path(__file__).parent / "fake_slurm"
ROOT = Path(__file__).parents[1]


def _release(prefix: Path, name: str, version: str) -> Path:
    """A shared-install release: <prefix>/releases/<name>/RELEASE.json and fea-translator/fea_translator/."""
    rel = prefix / "releases" / name
    pkg = rel / "fea-translator" / "fea_translator"
    pkg.mkdir(parents=True)
    (rel / "RELEASE.json").write_text(json.dumps({"release": name}))
    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n')
    return pkg


def test_stale_reason_shared_and_personal(tmp_path):
    old = _release(tmp_path, "r1", "1.0")
    _release(tmp_path, "r2", "2.0")
    (tmp_path / "current").symlink_to(tmp_path / "releases" / "r1")
    assert S.stale_reason(old, "1.0") == (None, True)
    (tmp_path / "current").unlink()
    (tmp_path / "current").symlink_to(tmp_path / "releases" / "r2")
    why, shared = S.stale_reason(old, "1.0")
    assert shared and "r2 is installed" in why
    pers = tmp_path / "checkout" / "fea_translator"
    pers.mkdir(parents=True)
    (pers / "__init__.py").write_text('__version__ = "3.0"\n')
    assert S.stale_reason(pers, "3.0") == (None, False)
    why, shared = S.stale_reason(pers, "2.9")
    assert not shared and "3.0 is installed" in why


@pytest.fixture()
def stale(tmp_path, monkeypatch):
    monkeypatch.setenv("FEAT_OUT", str(tmp_path))
    exits = []
    monkeypatch.setattr(S, "_stale", lambda: S._STATE["why"])
    monkeypatch.setattr(S, "_exit_soon", lambda why, delay: exits.append(why) or S._STATE.update(exiting=True))
    monkeypatch.setitem(S._STATE, "why", "release r2 is installed; this server runs r1 (1.0)")
    monkeypatch.setitem(S._STATE, "exiting", False)
    monkeypatch.setitem(S._STATE, "shared", True)
    monkeypatch.setitem(S._STATE, "serving", True)
    return exits


def test_stale_server_refuses_when_idle_and_answers_when_busy(stale, monkeypatch):
    """A stale shared-install server refuses a call with STALE_SERVER (the router repeats it on a new server) and
    exits; while another call runs it answers (old code) with a note; a personal checkout only notes it."""
    S._STATE["inflight"] += 1                        # another call running: answer, note, no exit
    try:
        out = asyncio.run(S.jobs(limit=1))
    finally:
        S._STATE["inflight"] -= 1
    assert "requests" in out and "restarts when idle" in out["server_stale"] and not stale
    with pytest.raises(S.ToolError, match=S.STALE_MARK):
        asyncio.run(S.jobs(limit=1))
    assert stale and S._STATE["inflight"] == 0
    monkeypatch.setitem(S._STATE, "exiting", False)
    monkeypatch.setitem(S._STATE, "shared", False)   # a personal checkout: answer, say restart
    out = asyncio.run(S.jobs(limit=1))
    assert "restart the MCP server" in out["server_stale"]
    monkeypatch.setenv("FEAT_EXIT_WHEN_STALE", "1")  # ... unless asked to exit there too
    with pytest.raises(S.ToolError, match=S.STALE_MARK):
        asyncio.run(S.jobs(limit=1))
    monkeypatch.setenv("FEAT_EXIT_WHEN_STALE", "0")  # never
    monkeypatch.setitem(S._STATE, "exiting", False)
    monkeypatch.setitem(S._STATE, "shared", True)
    assert "requests" in asyncio.run(S.jobs(limit=1))


def test_tests_and_scripts_in_a_stale_release_run_normally(stale, monkeypatch):
    """Tools called in-process (llmt deploy --test runs the tests before current is switched; run-study in a
    release that is not current yet) are never refused and never exit, however stale the install looks."""
    monkeypatch.setitem(S._STATE, "serving", False)
    monkeypatch.setenv("FEAT_EXIT_WHEN_STALE", "1")
    assert "requests" in asyncio.run(S.jobs(limit=1)) and not stale


def test_an_idle_stale_server_exits_by_itself(stale, monkeypatch):
    S._watch_install(period=0.2)
    t_end = time.time() + 5
    while time.time() < t_end and not stale:
        time.sleep(0.1)
    assert stale


def test_tool_errors_carry_their_text(monkeypatch):
    def boom(*a, **kw):
        raise KeyError("names")
    monkeypatch.setattr(S.api, "jobs", boom)
    with pytest.raises(S.ToolError, match="KeyError: 'names'"):
        asyncio.run(S.jobs(limit=1))


def test_every_tool_answers_within_the_call_budget(tmp_path, monkeypatch):
    """A queued SLURM job and a controller that hangs: study_status still answers within the budget (here 4 s
    instead of 48), with the job pending."""
    monkeypatch.setenv("FEAT_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("FEAT_EXECUTOR", "slurm")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(tmp_path / "fake"))
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    monkeypatch.setenv("PATH", f"{FAKE}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    from fea_translator import bench
    st = dict(bench.cases("ref")["axial_bar"], queue=True)
    out = asyncio.run(S.run_studies({"bar": st}, solver="ref", wait_s=0))
    assert out["pending"] == ["bar"]
    monkeypatch.setattr(S, "BUDGET_S", 4.0)
    slow = tmp_path / "slowbin"
    slow.mkdir()
    (slow / "squeue").write_text("#!/bin/sh\nsleep 30\n")      # a controller that does not answer
    (slow / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{slow}{os.pathsep}{os.environ['PATH']}")
    t0 = time.time()
    res = asyncio.run(S.study_status(out["request"], wait_s=50))
    assert time.time() - t0 < 5 and res["pending"] == ["bar"] and res["progress"]["bar"]["state"] == "unknown"
    t0 = time.time()
    res = asyncio.run(S.jobs(limit=5))
    assert time.time() - t0 < 5
