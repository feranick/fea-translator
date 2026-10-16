"""results.py — reading CalculiX output (.dat, .frd) and turning it into the study's results.
Version: 2026.10.16.1

parse_dat(text)    -> {"blocks": [{"what", "set", "time", "rows"}], "eigen": [...], "buckling": [...]}
parse_frd(text)    -> {"STRESS": {node: [sxx, syy, szz, sxy, syz, szx]}, "DISP": ..., ...} (last block of each)
von_mises(s)       -> von Mises stress of rows [sxx, syy, szz, sxy, sxz|syz, ...] (shear order does not matter)

The .dat blocks are printed by *NODE PRINT / *EL PRINT: a header line
"<what> (<components>) for set <SET> and time <t>" followed by rows of numbers.
"""
from __future__ import annotations

import re

import numpy as np

_HEADER = re.compile(r"^\s*(?P<what>[a-z][a-z .]*?)\s*(\((?P<comp>[^)]*)\))?\s+for set\s+(?P<set>\S+)\s+and time\s+(?P<t>\S+)", re.I)
_NUM = re.compile(r"^[\s\d.eE+-]+$")


def _floats(line: str):
    try:
        return [float(x) for x in line.split()]
    except ValueError:
        return None


def parse_dat(text: str) -> dict:
    blocks, eigen, buckling = [], [], []
    cur = None
    table, started = None, False
    for line in text.splitlines():
        u = line.upper()
        if "E I G E N V A L U E   O U T P U T" in u:
            table, started, cur = "eigen", False, None
            continue
        if "B U C K L I N G   F A C T O R   O U T P U T" in u:
            table, started, cur = "buckling", False, None
            continue
        m = _HEADER.match(line)
        if m:
            table = None
            cur = {"what": m.group("what").strip().lower(), "components": (m.group("comp") or "").lower(),
                   "set": m.group("set").upper(), "time": float(m.group("t")) if _floats(m.group("t")) else None,
                   "rows": []}
            blocks.append(cur)
            continue
        if not line.strip():
            continue
        if table:
            vals = _floats(line) if _NUM.match(line) else None
            if vals:
                started = True
                if table == "eigen" and len(vals) >= 4:
                    eigen.append({"mode": int(vals[0]), "eigenvalue": vals[1], "rad_s": vals[2], "hz": vals[3]})
                elif table == "buckling" and len(vals) >= 2:
                    buckling.append({"mode": int(vals[0]), "factor": vals[1]})
            elif started:
                table = None  # the next table (participation factors, ...) starts
            continue
        if cur is not None and _NUM.match(line):
            vals = _floats(line)
            if vals:
                cur["rows"].append(vals)
        elif cur is not None:
            cur = None
    return {"blocks": blocks, "eigen": eigen, "buckling": buckling}


def block(dat: dict, what: str, set_name: str | None = None, last: bool = True):
    """The (last) block whose description starts with `what` (and is for `set_name`)."""
    found = [b for b in dat["blocks"] if b["what"].startswith(what) and (set_name is None or b["set"] == set_name.upper())]
    return (found[-1] if last else found) if found else None


def parse_frd(text: str) -> dict:
    """Nodal result blocks of an ASCII .frd file: {name: {node: values}}; the last block of each name wins.
    Data lines: ' -1' + node (10 characters) + values (12 characters each)."""
    out, name, data = {}, None, None
    for line in text.splitlines():
        if line.startswith(" -4"):
            name = line[3:].split()[0].upper()
            data = {}
        elif line.startswith(" -1") and data is not None:
            try:
                node = int(line[3:13])
                body = line[13:].rstrip()
                vals = [float(body[i:i + 12]) for i in range(0, len(body), 12)]
            except ValueError:
                continue
            data[node] = vals
        elif line.startswith(" -3") and data is not None:
            out[name] = data
            name, data = None, None
    return out


def von_mises(s: np.ndarray) -> np.ndarray:
    s = np.atleast_2d(s)
    sxx, syy, szz, t1, t2, t3 = (s[:, i] for i in range(6))
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (t1 ** 2 + t2 ** 2 + t3 ** 2))
