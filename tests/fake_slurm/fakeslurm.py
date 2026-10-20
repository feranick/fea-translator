#!/usr/bin/env python3
"""Fake SLURM commands for the tests (sbatch, squeue, scancel, sacct, sinfo): one script, called by the
name it is invoked as. State lives in $FAKE_SLURM_DIR. sbatch starts job.sh in the background (honouring
--chdir and --output from its #SBATCH lines), prints an id and records the pid; squeue reports RUNNING while
the pid lives (PENDING with FAKE_SBATCH_PENDING=1); scancel kills the job and writes SLURM's CANCELLED line to
its log; sacct answers from recorded states (FAKE_SACCT_FAIL=1: it fails, as without accounting).
Other knobs: FAKE_SBATCH_FAIL=1 (sbatch refuses), FAKE_SBATCH_DELAY=s (the job sleeps first)."""
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

D = Path(os.environ.get("FAKE_SLURM_DIR", "/tmp/fake_slurm"))
J = D / "jobs"


def _jobs():
    J.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in sorted(J.glob("*.json")):
        out[f.stem] = json.loads(f.read_text())
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie is not alive
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return True


def _state(rec: dict) -> str | None:
    if rec.get("state") == "PENDING":
        return "PENDING"
    if rec.get("state") in ("CANCELLED",):
        return None
    return "RUNNING" if _alive(rec["pid"]) else None


def sbatch(args):
    if os.environ.get("FAKE_SBATCH_FAIL"):
        print("sbatch: error: Batch job submission failed: Unable to contact slurm controller", file=sys.stderr)
        return 1
    script = Path([a for a in args if not a.startswith("-")][-1])
    text = script.read_text()
    opt = dict(re.findall(r"^#SBATCH --([\w-]+)=(.*)$", text, re.M))
    J.mkdir(parents=True, exist_ok=True)
    counter = D / "counter"
    jid = int(counter.read_text()) + 1 if counter.is_file() else 1001
    counter.write_text(str(jid))
    rec = {"name": opt.get("job-name", script.name), "cpus": int(opt.get("cpus-per-task", 1)), "t0": time.time(),
           "log": opt.get("output", str(script.parent / f"slurm-{jid}.out")), "pid": 0, "args": args}
    for a in args:   # command-line options override the script's, as in sbatch
        if a.startswith("--") and "=" in a:
            opt[a[2:].split("=", 1)[0]] = a.split("=", 1)[1]
    mem = opt.get("mem")
    if os.environ.get("FAKE_SBATCH_PENDING"):
        rec["state"] = "PENDING"
    else:
        delay = os.environ.get("FAKE_SBATCH_DELAY")
        cmd = ["bash", "-c", f"sleep {delay}; exec bash {script}"] if delay else ["bash", str(script)]
        log = open(rec["log"], "a")
        env = dict(os.environ, SLURM_JOB_ID=str(jid), SLURM_CPUS_PER_TASK=str(rec["cpus"]), SLURM_NTASKS="1")
        env.pop("SLURM_MEM_PER_NODE", None)
        if mem:   # SLURM gives the job's --mem in MB
            m = re.fullmatch(r"(\d+)([KMGT]?)", mem.upper())
            env["SLURM_MEM_PER_NODE"] = str(int(int(m.group(1)) * {"K": 1 / 1024, "": 1, "M": 1, "G": 1024, "T": 1024 ** 2}[m.group(2)]))
        p = subprocess.Popen(cmd, cwd=opt.get("chdir", str(script.parent)), stdout=log, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        rec["pid"] = p.pid
    (J / f"{jid}.json").write_text(json.dumps(rec))
    print(jid if "--parsable" in args else f"Submitted batch job {jid}")
    return 0


def _fmt(fmt: str, jid: str, rec: dict, state: str) -> str:
    el = int(time.time() - rec["t0"])
    vals = {"i": jid, "T": state, "r": "Priority" if state == "PENDING" else "None", "S": "N/A",
            "M": f"{el // 60}:{el % 60:02d}", "j": rec["name"], "C": str(rec["cpus"])}
    return re.sub(r"%(\w)", lambda m: vals.get(m.group(1), ""), fmt)


def squeue(args):
    fmt, ids, states = "%i %T", None, None
    for i, a in enumerate(args):
        if a == "-o":
            fmt = args[i + 1]
        elif a == "-j":
            ids = set(args[i + 1].split(","))
        elif a == "-t":
            states = set(args[i + 1].split(","))
    for jid, rec in _jobs().items():
        st = _state(rec)
        if st is None or (ids is not None and jid not in ids) or (states is not None and st not in states):
            continue
        print(_fmt(fmt, jid, rec, st))
    return 0


def scancel(args):
    for jid in args:
        f = J / f"{jid}.json"
        if not f.is_file():
            continue
        rec = json.loads(f.read_text())
        if rec.get("pid") and _alive(rec["pid"]):
            try:
                os.killpg(rec["pid"], signal.SIGTERM)
            except OSError:
                pass
        rec["state"] = "CANCELLED"
        f.write_text(json.dumps(rec))
        with open(rec["log"], "a") as fh:
            fh.write(f"slurmstepd: error: *** JOB {jid} ON fake CANCELLED AT {time.strftime('%Y-%m-%dT%H:%M:%S')} ***\n")
    return 0


def sacct(args):
    if os.environ.get("FAKE_SACCT_FAIL"):
        print("sacct: error: Slurm accounting storage is disabled", file=sys.stderr)
        return 1
    ids = None
    for i, a in enumerate(args):
        if a == "-j":
            ids = set(args[i + 1].split(","))
    for jid, rec in _jobs().items():
        if ids is not None and jid not in ids:
            continue
        st = rec.get("sacct") or rec.get("state") or ("RUNNING" if _alive(rec["pid"]) else "COMPLETED")
        print(f"{jid}|{st}")
    return 0


def sinfo(args):
    print("2/6/0/8")
    return 0


if __name__ == "__main__":
    cmd = Path(sys.argv[0]).name
    sys.exit({"sbatch": sbatch, "squeue": squeue, "scancel": scancel, "sacct": sacct, "sinfo": sinfo}[cmd](sys.argv[1:]))
