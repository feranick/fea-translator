"""study.py — the study format: analyses, materials, boundary conditions, face selectors, validation.
Version: 2026.10.22.2

A study file is {name: study}; a study is one FEA run on one solid part:

    {"part": {"step": "bracket.step"} or {"goal": <build123d-translator goal>},
     "material": "steel" or {"E": ..., "nu": ..., ...},
     "analysis": "static" | "modal" | "buckling" | "thermal",
     "modes": 6,                                   (modal, buckling)
     "mesh": {"size": 3, "order": 2, "refine": [{"at": [x, y, z], "r": 5, "size": 0.5}]},
     "bcs": [{"kind": "fixed", "faces": {"nx": -1}}, {"kind": "force", "faces": {...}, "fz": -500}, ...],
     "probes": [{"name": "tip", "faces": {...}}],
     "expect": [{"quantity": "max_displacement", "value": 0.19, "tol": 0.03}]}

Units the caller writes: mm, N, MPa, kg/m3, W/(m K), W/(m2 K), W/m2, W, degrees C, m/s2.
The deck uses mm, N, s, t (tonne), mW: conversions are in `to_deck_units`.
"""
from __future__ import annotations

ANALYSES = ("static", "modal", "buckling", "thermal", "electric", "electrostatic", "joule")
MECH = ("static", "modal", "buckling", "thermal", "joule")      # analyses that take mechanical bcs (thermal, joule: with stress)
EPS0 = 8.8541878128e-15  # vacuum permittivity, F/mm

# typical room-temperature values; give your own for design work
MATERIALS = {
    "steel": {"E": 210000, "nu": 0.30, "density": 7850, "k": 50, "alpha": 1.2e-5, "yield": 250, "resistivity": 1.6e-7},
    "stainless_304": {"E": 193000, "nu": 0.29, "density": 8000, "k": 16.2, "alpha": 1.73e-5, "yield": 215, "resistivity": 7.2e-7},
    "aluminium_6061_t6": {"E": 68900, "nu": 0.33, "density": 2700, "k": 167, "alpha": 2.36e-5, "yield": 276, "resistivity": 3.99e-8},
    "titanium_ti6al4v": {"E": 113800, "nu": 0.342, "density": 4430, "k": 6.7, "alpha": 8.6e-6, "yield": 880, "resistivity": 1.78e-6},
    "copper": {"E": 117000, "nu": 0.34, "density": 8940, "k": 390, "alpha": 1.7e-5, "yield": 70, "resistivity": 1.72e-8},
    "brass": {"E": 100000, "nu": 0.34, "density": 8500, "k": 110, "alpha": 1.9e-5, "yield": 200, "resistivity": 6.4e-8},
    "cast_iron": {"E": 110000, "nu": 0.26, "density": 7200, "k": 52, "alpha": 1.1e-5, "yield": 200, "resistivity": 1.0e-6},
    "silver": {"E": 83000, "nu": 0.37, "density": 10490, "k": 429, "alpha": 1.89e-5, "resistivity": 1.59e-8},
    "gold": {"E": 79000, "nu": 0.42, "density": 19300, "k": 318, "alpha": 1.42e-5, "resistivity": 2.44e-8},
    "abs": {"E": 2300, "nu": 0.35, "density": 1050, "k": 0.17, "alpha": 9.0e-5, "yield": 40,
            "resistivity": 1e14, "permittivity": 3.0, "dielectric_strength": 16},
    "pla": {"E": 3500, "nu": 0.36, "density": 1240, "k": 0.13, "alpha": 6.8e-5, "yield": 50,
            "resistivity": 1e14, "permittivity": 3.0, "dielectric_strength": 15},
    "nylon_pa6": {"E": 2800, "nu": 0.39, "density": 1140, "k": 0.25, "alpha": 8.5e-5, "yield": 70,
                  "resistivity": 1e12, "permittivity": 3.8, "dielectric_strength": 20},
    "alumina_96": {"E": 300000, "nu": 0.21, "density": 3720, "k": 24, "alpha": 8.2e-6,
                   "resistivity": 1e14, "permittivity": 9.0, "dielectric_strength": 12},
    "fr4": {"E": 22000, "nu": 0.12, "density": 1850, "k": 0.3, "alpha": 1.4e-5,
            "resistivity": 1e12, "permittivity": 4.4, "dielectric_strength": 20},
    "ptfe": {"E": 500, "nu": 0.46, "density": 2200, "k": 0.25, "alpha": 1.2e-4, "yield": 10,
             "resistivity": 1e16, "permittivity": 2.1, "dielectric_strength": 20},
}
MATERIAL_KEYS = {"E": "Young's modulus, MPa", "nu": "Poisson's ratio", "density": "kg/m3 (modal, gravity)",
                 "k": "thermal conductivity, W/(m K) (thermal)", "alpha": "thermal expansion, 1/K",
                 "yield": "yield strength, MPa (safety factor)",
                 "resistivity": "electrical resistivity, ohm m (electric, joule)",
                 "permittivity": "relative permittivity (electrostatic)",
                 "dielectric_strength": "breakdown field, kV/mm (electrostatic: margin to breakdown)"}

# ---- face selectors ----------------------------------------------------------------------
SELECTOR_DOC = {
    "nx, ny, nz": "a planar face with this outward normal (any direction; the others default to 0): the largest, "
                  "then the farthest along the normal; with at = [x, y, z] the one nearest to that point",
    "pts": "[[x, y, z], ...]: the face nearest to each point (a point on the face, away from its edges)",
    "cyl_r": "every cylindrical face of this radius; side: 'inner' for holes and bores (concave), 'outer' for "
             "shafts, bosses and outside fillets (convex), omitted: both (a hole and corner fillets of the same "
             "radius are all taken); with at, only the nearest",
    "all": "true: every boundary face (convection everywhere)",
}

# ---- boundary conditions -----------------------------------------------------------------
# kind: (analyses, required, optional, description)
BC_KINDS = {
    "fixed": (("static", "modal", "buckling"), {"faces": ""}, {}, "All displacements zero on the faces (clamped)."),
    "slider": (("static", "modal", "buckling"), {"faces": "planar faces normal to X, Y or Z"}, {},
               "Displacement normal to the face zero, free in its plane (symmetry planes, rollers)."),
    "displacement": (("static", "modal", "buckling"), {"faces": ""},
                     {"ux": "mm", "uy": "mm", "uz": "mm (each given component is prescribed; others free)"},
                     "Prescribed displacement components."),
    "force": (("static", "buckling"), {"faces": ""}, {"fx": "N", "fy": "N", "fz": "N (total force, spread uniformly over the faces' area)"},
              "Total force on the faces (bearing loads, bolt loads, tip loads)."),
    "pressure": (("static", "buckling"), {"faces": "", "p": "MPa; positive pushes into the faces"}, {},
                 "Uniform pressure normal to the faces."),
    "gravity": (("static", "buckling"), {}, {"gx": "", "gy": "", "gz": "m/s2 (default 0, 0, -9.81)"},
                "Body weight (needs density)."),
    "temperature": (("thermal", "joule"), {"faces": "", "T": "degrees C"}, {}, "Fixed temperature on the faces."),
    "heat_flux": (("thermal", "joule"), {"faces": "", "q": "W/m2 into the part (negative: out)"}, {}, "Uniform heat flux."),
    "heat": (("thermal", "joule"), {"faces": "", "power": "W into the part, spread uniformly over the faces' area"}, {},
             "Total heat input on the faces (a heater, a chip)."),
    "convection": (("thermal", "joule"), {"faces": "", "h": "film coefficient, W/(m2 K)", "T_inf": "ambient, degrees C"}, {},
                   "Convection to an ambient temperature."),
    "voltage": (("electric", "electrostatic", "joule"), {"faces": "", "V": "volts"}, {},
                "Fixed potential on the faces: a terminal or electrode (0 V for ground)."),
    "current": (("electric", "joule"), {"faces": "", "I": "amperes into the part, spread uniformly over the faces' area"}, {},
                "Total current fed into the faces (a terminal driven by a current source)."),
}
# mechanical supports and loads also go with thermal and joule studies that ask for stress
for _k in ("fixed", "slider", "displacement", "force", "pressure", "gravity"):
    _a, _r, _o, _w = BC_KINDS[_k]
    BC_KINDS[_k] = (_a + ("thermal", "joule"), _r, _o, _w)

CONVENTIONS = [
    "Units: mm, N, MPa; density kg/m3; conductivity W/(m K); film coefficient W/(m2 K); heat flux W/m2; "
    "heat W; temperature degrees C; gravity m/s2. Coordinates are the part's (build123d-translator: Z up, base on z = 0).",
    "One study is one analysis of one solid part. The part is a STEP file (mm) or a build123d-translator goal.",
    "Faces are chosen like build123d-translator faces: by outward normal (nx, ny, nz; the largest such planar face, "
    "or the one nearest to at), by points on them (pts), by cylinder radius (cyl_r), or all. 'faces' may also be a "
    "list of selectors (their union).",
    "static and buckling need supports that stop every rigid-body motion (fixed, or sliders on three faces normal "
    "to X, Y and Z, ...). modal needs supports too (a free part has zero-frequency modes). thermal needs at least "
    "one temperature or convection.",
    "force is a total force spread uniformly over the faces' area; pressure acts normal to the faces (positive "
    "into the part); gravity uses the material density.",
    "buckling: the loads are a reference; the results are load factors (critical load = factor x reference load).",
    "Electrical (DC): electric solves current flow (resistivity; voltage and current terminals; at least one voltage, "
    "e.g. 0 V ground): current per terminal, resistance (two terminals), power, peak current density. electrostatic "
    "solves the field in a dielectric (permittivity; voltage electrodes): charge per electrode, capacitance (two "
    "electrodes), energy, peak field and the margin to dielectric_strength. Faces with no bc are insulated (electric) "
    "or carry no charge (electrostatic). joule: electric, then the Joule heat (resistivity x current density squared) "
    "into a thermal solve with the thermal bcs (temperature, convection, ...): the electric and thermal results and "
    "the Joule heat (= I^2 R). Resistivity is taken at its room-temperature value.",
    "stress: true on thermal or joule adds a static solve with the temperatures (thermal expansion alpha from T_ref, "
    "default 20 degrees C) and the mechanical bcs of the study (supports must stop rigid-body motion): stresses, "
    "displacements, reactions, safety factor, as in static.",
    "Results: static: maximum displacement and von Mises stress (at integration points and at nodes), reaction "
    "force of each support, mean displacement of each load face and probe, safety factor (with yield), "
    "equilibrium check. modal: frequencies in Hz. buckling: load factors. thermal: temperature range, heat "
    "flow through each temperature boundary, heat balance.",
    "mesh.size is the element size in mm (default: 1/20 of the part's bounding-box diagonal); order 2 "
    "(10-node tetrahedra, default) is accurate for bending and curved faces; refine adds finer spheres "
    "{at, r, size} at stress concentrations. Rerun with a smaller size to check convergence (converge: true does it).",
    "expect: [{quantity, value, tol}] turns the run into a check: quantity is a dotted path into the results "
    "(max_displacement, max_von_mises, frequencies.0, buckling_factors.0, reactions.<bc name>.fz, "
    "probes.<name>.mean_u.uz, T_max, heat_flow.<bc name>); tol is relative (default 0.02).",
]

EXAMPLE = {"bracket_static": {
    "part": {"goal": {"features": [
        {"kind": "base_box", "params": {"w": 100, "d": 20, "h": 10}},
        {"kind": "hole", "params": {"r": 4, "x": 35, "y": 0}}]}},
    "material": "aluminium_6061_t6",
    "analysis": "static",
    "mesh": {"size": 3},
    "bcs": [{"kind": "fixed", "name": "wall", "faces": {"nx": -1}},
            {"kind": "force", "name": "load", "faces": {"cyl_r": 4, "side": "inner"}, "fz": -200}],
    "probes": [{"name": "tip", "faces": {"nx": 1}}]}}


def study_format() -> dict:
    bcs = {k: {"analyses": list(a), "required": list(r), "optional": list(o) + ["name"], "what": w,
               "descriptions": {p: d for p, d in {**r, **o}.items() if d}}
           for k, (a, r, o, w) in BC_KINDS.items()}
    return {"format": "FEA study (fea-translator)", "analyses": list(ANALYSES), "conventions": CONVENTIONS,
            "study_keys": {"part": "{'step': path} or {'goal': build123d-translator goal}",
                           "material": f"one of {sorted(MATERIALS)} or {{{', '.join(MATERIAL_KEYS)}}} "
                                       "(a name plus overrides: {'name': 'steel', 'yield': 355})",
                           "analysis": "|".join(ANALYSES), "modes": "number of modes or buckling factors (6; 3)",
                           "stress": "thermal, joule: true adds the thermal-stress static solve",
                           "T_ref": "thermal, joule with stress: stress-free temperature, degrees C (20)",
                           "procs": "CalculiX threads for this study (default FEAT_PROCS, 1; modal and buckling always 1)",
                           "queue": "true: always a SLURM job; false: always local (with FEAT_EXECUTOR=slurm; "
                                    "otherwise studies estimated above FEAT_SLURM_MIN_MINUTES go to SLURM)",
                           "run": "{timeout_s, mem ('12G'), procs, queue}: this study's solver time limit, SLURM memory, "
                                  "threads and queue, overriding the machine's defaults within the caps in status",
                           "mesh": "{size, order (1|2), refine: [{at, r, size}]}", "bcs": "list of boundary conditions",
                           "probes": "[{name, faces}]: mean and largest displacement (or temperature) there",
                           "expect": "[{quantity, value, tol}]", "converge": "true: rerun at 0.6 x size and report the change",
                           "equation_solver": "auto (iterative above 150k nodes for static and thermal) | direct | iterative",
                           "stress_band": "mm: *_off_bc peaks leave out material this close to supports and loads (default: 2 element sizes, "
                                          "the same on the converge mesh); widen it if peak_von_mises_off_bc does not converge"},
            "material_keys": MATERIAL_KEYS, "materials": MATERIALS, "bc_kinds": bcs, "selectors": SELECTOR_DOC,
            "example": EXAMPLE}


# ---- units -------------------------------------------------------------------------------

def material_of(study: dict) -> dict:
    m = study.get("material", "steel")
    if isinstance(m, str):
        return dict(MATERIALS[m])
    base = dict(MATERIALS.get(m.get("name"), {})) if isinstance(m.get("name"), str) else {}
    base.update({k: v for k, v in m.items() if k != "name"})
    return base


def to_deck_units(mat: dict) -> dict:
    """Material in the deck's units (mm, N, s, t, mW; S/mm)."""
    out = {"E": float(mat["E"]), "nu": float(mat["nu"])} if "E" in mat and "nu" in mat else {}
    if "resistivity" in mat:
        out["sigma"] = 1e-3 / float(mat["resistivity"])  # ohm m -> S/mm
    if "permittivity" in mat:
        out["eps_r"] = float(mat["permittivity"])
    if "density" in mat:
        out["density"] = float(mat["density"]) * 1e-12  # kg/m3 -> t/mm3
    if "k" in mat:
        out["k"] = float(mat["k"])  # W/(m K) = mW/(mm K)
    if "alpha" in mat:
        out["alpha"] = float(mat["alpha"])
    return out


H_SCALE = 1e-3     # W/(m2 K) -> mW/(mm2 K)
Q_SCALE = 1e-3     # W/m2 -> mW/mm2
P_SCALE = 1e3      # W -> mW
G_SCALE = 1e3      # m/s2 -> mm/s2


# ---- validation --------------------------------------------------------------------------

def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _point(p) -> bool:
    return isinstance(p, (list, tuple)) and len(p) == 3 and all(_num(c) for c in p)


def _selector(sel) -> str | None:
    if isinstance(sel, list):
        if not sel:
            return "an empty list of selectors"
        for s in sel:
            bad = _selector(s)
            if bad:
                return bad
        return None
    if not isinstance(sel, dict) or not sel:
        return "a selector must be an object: {nx, ny, nz[, at]}, {pts}, {cyl_r[, at]} or {all: true}"
    keys = set(sel)
    if keys & {"nx", "ny", "nz"}:
        v = [sel.get(k, 0) for k in ("nx", "ny", "nz")]
        if not all(_num(c) for c in v) or sum(c * c for c in v) < 1e-12:
            return "nx, ny, nz must be numbers, not all 0"
    elif "pts" in keys:
        if not isinstance(sel["pts"], list) or not sel["pts"] or not all(_point(q) for q in sel["pts"]):
            return "'pts' must be a non-empty list of [x, y, z]"
    elif "cyl_r" in keys:
        if not _num(sel["cyl_r"]) or sel["cyl_r"] <= 0:
            return "'cyl_r' must be a positive number"
    elif sel.get("all") is not True:
        return "unknown selector: use {nx, ny, nz[, at]}, {pts}, {cyl_r[, at]} or {all: true}"
    if "at" in sel and not _point(sel["at"]):
        return "'at' must be [x, y, z]"
    if "side" in keys and ("cyl_r" not in keys or sel["side"] not in ("inner", "outer")):
        return "'side' goes with cyl_r and is 'inner' (holes) or 'outer' (shafts, fillets)"
    extra = keys - {"nx", "ny", "nz", "at", "pts", "cyl_r", "all", "side"}
    if extra:
        return f"unknown selector keys {sorted(extra)}"
    return None


def bc_name(i: int, bc: dict) -> str:
    return str(bc.get("name") or f"{bc.get('kind', 'bc')}{i + 1}")


RUN_KEYS = ("timeout_s", "mem", "procs", "queue")


def _run_errors(st: dict) -> list[str]:
    """The study's "run" settings (docs/EXECUTION.md, 'Per-study run settings') and the machine's caps
    FEAT_MAX_TIMEOUT, FEAT_MAX_MEM, FEAT_MAX_PROCS (unset: no limit; FEAT_MAX_PROCS bounds the top-level procs too)."""
    from . import executor

    errs, cap = [], executor.caps()
    if _num(st.get("procs")) and "procs" in cap and st["procs"] > cap["procs"]:
        errs.append(f"procs: {st['procs']} is above FEAT_MAX_PROCS={cap['procs']}")
    if "run" not in st:
        return errs
    r = st["run"]
    if not isinstance(r, dict):
        return errs + ["run: an object {timeout_s, mem, procs, queue} (all optional)"]
    extra = set(r) - set(RUN_KEYS)
    if extra:
        errs.append(f"run: unknown keys {sorted(extra)} (allowed: {', '.join(RUN_KEYS)})")
    if "timeout_s" in r:
        if not _num(r["timeout_s"]) or r["timeout_s"] <= 0:
            errs.append("run.timeout_s: seconds per solve, > 0")
        elif "timeout_s" in cap and r["timeout_s"] > cap["timeout_s"]:
            errs.append(f"run.timeout_s: {r['timeout_s']} s is above FEAT_MAX_TIMEOUT={cap['timeout_s']:g}")
    if "mem" in r:
        mb = executor.mem_mb(r["mem"]) if isinstance(r["mem"], str) else None
        if not mb:
            errs.append(f"run.mem: {r['mem']!r} is not a SLURM memory size such as '500M' or '12G'")
        elif "mem" in cap and executor.mem_mb(cap["mem"]) and mb > executor.mem_mb(cap["mem"]):
            errs.append(f"run.mem: {r['mem']} is above FEAT_MAX_MEM={cap['mem']}")
    if "procs" in r:
        p = r["procs"]
        if not _num(p) or int(p) != p or p < 1:
            errs.append("run.procs: a whole number of threads, at least 1")
        elif "procs" in cap and p > cap["procs"]:
            errs.append(f"run.procs: {p} is above FEAT_MAX_PROCS={cap['procs']}")
        elif "procs" in st and st["procs"] != p:
            errs.append(f"run.procs ({p}) disagrees with procs ({st['procs']}): set one")
    if "queue" in r:
        if not isinstance(r["queue"], bool):
            errs.append("run.queue: true or false")
        elif "queue" in st and st["queue"] != r["queue"]:
            errs.append(f"run.queue ({r['queue']}) disagrees with queue ({st['queue']}): set one")
    return errs


def validate_study(st: dict) -> dict:
    errors, warnings = [], []
    if not isinstance(st, dict):
        return {"ok": False, "errors": ["a study must be an object"], "warnings": []}
    part = st.get("part")
    if isinstance(part, dict) and "box" in part:
        if not (isinstance(part["box"], list) and len(part["box"]) == 3 and all(_num(v) and v > 0 for v in part["box"])):
            errors.append("part: 'box' must be [lx, ly, lz] (positive)")
    elif not isinstance(part, dict) or not (isinstance(part.get("step"), str) or isinstance(part.get("goal"), dict)):
        errors.append("part: needs {'step': path} or {'goal': build123d-translator goal}")
    elif "goal" in part:
        try:
            from build123d_translator.goalformat import validate_goal
            v = validate_goal(part["goal"])
            errors += [f"part goal: {e}" for e in v["errors"]]
        except ImportError:
            warnings.append("part goal: build123d-translator is not installed here; the goal cannot be built")
    an = st.get("analysis")
    if an not in ANALYSES:
        errors.append(f"analysis: one of {', '.join(ANALYSES)}")
    m = st.get("material", "steel")
    if isinstance(m, str):
        if m not in MATERIALS:
            errors.append(f"material: unknown '{m}'; known: {', '.join(sorted(MATERIALS))}, or give the values")
        mat = MATERIALS.get(m, {})
    elif isinstance(m, dict):
        if "name" in m and m["name"] not in MATERIALS:
            errors.append(f"material: unknown name '{m['name']}'")
        mat = material_of(st) if ("name" not in m or m["name"] in MATERIALS) else m
        for k, v in m.items():
            if k != "name" and (k not in MATERIAL_KEYS or not _num(v)):
                errors.append(f"material: '{k}' {'is not a material key' if k not in MATERIAL_KEYS else 'must be a number'}")
    else:
        errors.append("material: a name or an object")
        mat = {}
    need = {"static": ("E", "nu"), "modal": ("E", "nu", "density"), "buckling": ("E", "nu"), "thermal": ("k",),
            "electric": ("resistivity",), "electrostatic": ("permittivity",), "joule": ("resistivity", "k")}.get(an, ())
    if st.get("stress") and an in ("thermal", "joule"):
        need = need + ("E", "nu", "alpha")
    for k in need:
        if k not in mat:
            errors.append(f"material: a {an} study{' with stress' if st.get('stress') and k in ('E', 'nu', 'alpha') else ''} "
                          f"needs '{k}'")
    if _num(mat.get("nu")) and not (-1 < mat["nu"] < 0.5):
        errors.append("material: 'nu' must be between -1 and 0.5")
    for k in ("E", "density", "k", "yield", "resistivity", "permittivity", "dielectric_strength"):
        if _num(mat.get(k)) and mat[k] <= 0:
            errors.append(f"material: '{k}' must be positive")
    bcs = st.get("bcs", [])
    if not isinstance(bcs, list):
        errors.append("bcs: a list")
        bcs = []
    names = set()
    supports = loads = 0
    for i, bc in enumerate(bcs):
        if not isinstance(bc, dict) or bc.get("kind") not in BC_KINDS:
            errors.append(f"bc {i + 1}: 'kind' one of {', '.join(BC_KINDS)}")
            continue
        k = bc["kind"]
        where = f"bc {i + 1} ({k})"
        analyses, req, opt, _ = BC_KINDS[k]
        if an in ANALYSES and an not in analyses:
            errors.append(f"{where}: not used in a {an} analysis (only {', '.join(analyses)})")
        for r in req:
            if r not in bc:
                errors.append(f"{where}: missing '{r}'")
            elif r != "faces" and not _num(bc[r]):
                errors.append(f"{where}: '{r}' must be a number")
        for o in opt:
            if o in bc and not _num(bc[o]):
                errors.append(f"{where}: '{o}' must be a number")
        extra = set(bc) - set(req) - set(opt) - {"kind", "name"}
        if extra:
            errors.append(f"{where}: unknown keys {sorted(extra)}")
        if "faces" in bc:
            bad = _selector(bc["faces"])
            if bad:
                errors.append(f"{where}: faces: {bad}")
        n = bc_name(i, bc)
        if n in names:
            errors.append(f"{where}: duplicate name '{n}'")
        names.add(n)
        if k == "displacement" and not ({"ux", "uy", "uz"} & set(bc)):
            errors.append(f"{where}: give at least one of ux, uy, uz")
        if k == "force" and not any(_num(bc.get(c)) and bc[c] != 0 for c in ("fx", "fy", "fz")):
            errors.append(f"{where}: give a non-zero fx, fy or fz")
        if k == "convection" and _num(bc.get("h")) and bc["h"] <= 0:
            errors.append(f"{where}: 'h' must be positive")
        if k == "gravity" and not _num(mat.get("density")):
            errors.append(f"{where}: needs the material density")
        if k in ("fixed", "slider", "displacement", "force", "pressure", "gravity") and an in ("thermal", "joule") \
                and not st.get("stress"):
            errors.append(f"{where}: mechanical bcs in a {an} study need stress: true")
        supports += k in (("temperature", "convection") if an == "thermal" else ("fixed", "slider", "displacement"))
        loads += k in ("force", "pressure", "gravity", "heat_flux", "heat") or (k == "displacement" and any(
            _num(bc.get(c)) and bc[c] != 0 for c in ("ux", "uy", "uz")))
    if an in ("static", "buckling", "modal") and not supports:
        errors.append(f"{an}: needs supports (fixed, slider or displacement); a free part moves as a rigid body")
    if an in ("static", "buckling") and not loads:
        errors.append(f"{an}: needs a load (force, pressure, gravity or a non-zero displacement)")
    kinds = [b.get("kind") for b in bcs if isinstance(b, dict)]
    if an in ("thermal", "joule") and not ({"temperature", "convection"} & set(kinds)):
        errors.append(f"{an}: needs a temperature or convection boundary (otherwise the temperature is undetermined)")
    if an in ("electric", "electrostatic", "joule") and "voltage" not in kinds:
        errors.append(f"{an}: needs at least one voltage terminal (e.g. 0 V ground: it sets the potential's reference)")
    if an in ("electric", "joule") and len([q for q in kinds if q in ("voltage", "current")]) < 2:
        errors.append(f"{an}: needs two terminals at least (voltage or current) for a current to flow")
    if an == "electrostatic":
        vs = {float(b["V"]) for b in bcs if isinstance(b, dict) and b.get("kind") == "voltage" and _num(b.get("V"))}
        if len(vs) < 2:
            errors.append("electrostatic: needs electrodes at two different voltages at least")
    if an in ("thermal", "joule") and st.get("stress"):
        if not ({"fixed", "slider", "displacement"} & set(kinds)):
            errors.append(f"{an} with stress: needs supports (fixed, slider or displacement)")
    if "stress" in st and (not isinstance(st["stress"], bool) or an not in ("thermal", "joule")):
        errors.append("stress: true or false, only for thermal and joule studies")
    if "queue" in st and not isinstance(st["queue"], bool):
        errors.append("queue: true or false")
    if "procs" in st and (not _num(st["procs"]) or int(st["procs"]) != st["procs"] or st["procs"] < 1):
        errors.append("procs: a whole number of threads, at least 1")
    errors += _run_errors(st)
    if "T_ref" in st and not _num(st["T_ref"]):
        errors.append("T_ref: a temperature in degrees C")
    if an in ("static", "buckling", "modal") or st.get("stress"):
        sl = [b for b in bcs if isinstance(b, dict) and b.get("kind") == "slider"]
        if sl and not any(isinstance(b, dict) and b.get("kind") in ("fixed", "displacement") for b in bcs) and len(sl) < 3:
            warnings.append("sliders alone need three faces normal to X, Y and Z (and may still allow rotation)")
    if "modes" in st and (not _num(st["modes"]) or int(st["modes"]) != st["modes"] or not 1 <= st["modes"] <= 100):
        errors.append("modes: a whole number from 1 to 100")
    mesh = st.get("mesh", {})
    if not isinstance(mesh, dict):
        errors.append("mesh: an object {size, order, refine}")
    else:
        if "size" in mesh and (not _num(mesh["size"]) or mesh["size"] <= 0):
            errors.append("mesh: 'size' must be positive (mm)")
        if mesh.get("order", 2) not in (1, 2):
            errors.append("mesh: 'order' is 1 or 2")
        if mesh.get("order", 2) == 1 and an in ("static", "modal", "buckling"):
            warnings.append("mesh: 4-node tetrahedra are much too stiff in bending; use order 2")
        for j, rf in enumerate(mesh.get("refine", []) if isinstance(mesh.get("refine", []), list) else [None]):
            if not isinstance(rf, dict) or not _point(rf.get("at")) or not _num(rf.get("r")) or not _num(rf.get("size")) \
                    or rf["r"] <= 0 or rf["size"] <= 0:
                errors.append(f"mesh: refine {j + 1} must be {{at: [x, y, z], r: radius > 0, size > 0}}")
    for j, pr in enumerate(st.get("probes", [])):
        if not isinstance(pr, dict) or not isinstance(pr.get("name"), str) or "faces" not in pr:
            errors.append(f"probe {j + 1}: needs 'name' and 'faces'")
        else:
            bad = _selector(pr["faces"])
            if bad:
                errors.append(f"probe {j + 1}: faces: {bad}")
    for j, ex in enumerate(st.get("expect", [])):
        if not isinstance(ex, dict) or not isinstance(ex.get("quantity"), str) or not _num(ex.get("value")):
            errors.append(f"expect {j + 1}: needs 'quantity' (a results path) and a numeric 'value'")
        elif "tol" in ex and (not _num(ex["tol"]) or ex["tol"] <= 0):
            errors.append(f"expect {j + 1}: 'tol' must be positive")
    if st.get("equation_solver", "auto") not in ("auto", "direct", "iterative"):
        errors.append("equation_solver: auto, direct or iterative")
    if "stress_band" in st and (not _num(st["stress_band"]) or st["stress_band"] < 0):
        errors.append("stress_band: a distance in mm (>= 0)")
    extra = set(st) - {"part", "material", "analysis", "modes", "mesh", "bcs", "probes", "expect", "converge", "notes",
                       "equation_solver", "stress_band", "stress", "T_ref", "procs", "queue", "run"}
    if extra:
        warnings.append(f"unknown study keys ignored: {sorted(extra)}")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def validate(studies: dict) -> dict:
    return {name: validate_study(s) for name, s in studies.items()}
