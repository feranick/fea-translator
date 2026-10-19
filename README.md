# fea-translator

Turns an **FEA study** (one solid part, a material, an analysis, supports and loads on chosen faces) into a [Gmsh](https://gmsh.info) mesh, a [CalculiX](http://www.calculix.de) input deck and checked results, deterministically. It includes a command line (`feat`), a Python API and an MCP server (`feat-mcp`), so LLM-based tools can set up analyses as studies and get the numbers back with their checks.

It is the FEA counterpart of [build123d-translator](https://github.com/feranick/build123d-translator): an LLM writes the study (the flexible part), the translator computes everything else (which mesh faces, which nodes, consistent nodal loads, element face numbers, the deck), and a checker confirms the result: closed-form benchmarks, equilibrium and heat balance on every run, expected values, and mesh convergence. Parts are STEP files or build123d-translator goals, so the same face conventions select CAD features and boundary conditions.

Analyses: **linear static** (displacements, von Mises stress, reactions, safety factor), **modal** (natural frequencies), **linear buckling** (load factors), **steady heat transfer** (temperatures, heat flows; with `stress: true`, thermal stresses), **DC electric conduction** (terminal currents, resistance, power, current density), **electrostatics** (charges, capacitance, field, margin to breakdown) and **Joule heating** (electric, then thermal, optionally stresses).

## Install

```bash
cd ~/Desktop/fea-translator
python3 -m venv .venv && source .venv/bin/activate        # Python 3.10+
pip install -e ".[mesh,mcp,test]"                          # numpy, scipy, gmsh, MCP SDK, pytest
pip install -e ../build123d-translator                     # optional: parts as goals
```

On **Linux ARM (aarch64**, e.g. DGX Spark) PyPI has no Gmsh wheels, so the `mesh` extra skips it; use Ubuntu's (Gmsh 4.12 on 24.04) and link its Python module into the venv (same Python version as the system's):

```bash
sudo apt install -y python3-gmsh
ln -sf /usr/lib/python3/dist-packages/gmsh.py "$VIRTUAL_ENV/lib/python3.12/site-packages/gmsh.py"
```

CalculiX (`ccx`) is a separate program:

- macOS: a Homebrew tap packages current ccx for Apple Silicon (`costerwi/calculix`, see its README for the exact formula name), or build it from source; then `ccx -v` should print the version.
- Linux (Ubuntu/Debian): `sudo apt install calculix-ccx`.

The translator finds `ccx` (or `ccx_2.*`) on the PATH or in `/opt/homebrew/bin`; `FEAT_CCX=/path/to/ccx` overrides. Without CalculiX the package's own reference solver runs static, modal and thermal studies on small meshes (`--solver ref`); buckling needs CalculiX.

```bash
feat status                    # versions; ccx and gmsh found?
pytest -q                      # 67 tests: 47 run anywhere (17 with fake SLURM commands), 20 need ccx (and Gmsh)
feat bench                     # every benchmark, pass/fail against closed-form answers
```

## Use

```bash
feat format > study_format.json          # analyses, materials, bc kinds, selectors, conventions, example
feat validate examples/studies.json      # static checks
feat run examples/studies.json           # mesh, solve, check; files in ~/feat_runs/run_<time>/
feat status ~/feat_runs/run_<time>       # results of a request (finished and still running studies)
feat deck examples/studies.json bracket_static   # the CalculiX deck only
feat bench lame_ring plate_hole          # chosen benchmarks
```

Every command prints JSON; `-` reads the studies from stdin. `feat run` waits for all studies; the MCP `run_studies` returns within about 45 s and leaves long studies `pending` (see below). Output folder: `FEAT_OUT` (default `~/feat_runs`); `FEAT_WORKERS` (2) worker processes, `FEAT_TIMEOUT` (900 s per study), `FEAT_SOLVER` (auto | ccx | ref).

Per study the folder holds `part.step`, `mesh.msh`, `<name>.inp` (the deck; runs with `ccx -i <name>` on its own), `<name>.dat` and `<name>.frd` (CalculiX results; `.frd` opens in CalculiX GraphiX, FreeCAD, or ParaView via ccx2paraview) and `<name>.json` (the results below).

## A study

```json
{"bracket_static": {
  "part": {"goal": {"features": [
      {"kind": "base_box", "params": {"w": 100, "d": 20, "h": 10}},
      {"kind": "hole", "params": {"r": 4, "x": 35, "y": 0}}]}},
  "material": "aluminium_6061_t6",
  "analysis": "static",
  "mesh": {"size": 3},
  "bcs": [{"kind": "fixed", "name": "wall", "faces": {"nx": -1}},
          {"kind": "force", "name": "load", "faces": {"cyl_r": 4, "side": "inner"}, "fz": -200}],
  "probes": [{"name": "tip", "faces": {"nx": 1}}]}}
```

- **Units:** mm, N, MPa; density kg/m³; conductivity W/(m K); film coefficient W/(m² K); heat flux W/m²; heat W; °C; gravity m/s². The deck uses mm, N, s, t, mW; the conversions are in `study.py`.
- **Part:** `{"step": path}` (mm) or `{"goal": ...}` (built with build123d-translator); `{"box": [lx, ly, lz]}` is a structured box mesh without Gmsh (tests, quick checks).
- **Material:** a name (steel, stainless_304, aluminium_6061_t6, titanium_ti6al4v, copper, brass, cast_iron, abs, pla, nylon_pa6; typical values) or `{E, nu, density, k, alpha, yield}`, or a name with overrides `{"name": "steel", "yield": 355}`.
- **Faces** (as build123d-translator): `{nx, ny, nz}` the largest planar face with that outward normal (`at` for the nearest one), `{pts: [[x, y, z], ...]}` the face nearest to each point, `{cyl_r: r}` every cylindrical face of radius r (found by fitting the mesh, so it does not depend on how the STEP describes the surface; `side: "inner"` for holes and bores only, `"outer"` for shafts, bosses and outside fillets, which otherwise match too: a 4 mm hole and 4 mm corner fillets are all `cyl_r: 4`), `{all: true}`; a list is their union. Every result lists what each selector took (`selection`: kind, radius, side, arc, area, centre per face), with a warning when one `cyl_r` takes holes and convex faces together.
- **Boundary conditions:** `fixed`, `slider` (normal displacement zero: symmetry planes, rollers), `displacement` (prescribed components); loads `force` (a total force spread uniformly over the faces' area), `pressure` (normal, positive into the part), `gravity`; thermal `temperature`, `heat_flux`, `heat` (total watts on faces), `convection`; electrical `voltage` (a terminal or electrode; 0 V for ground) and `current` (total amperes into the faces).
- **Electrical** (2026.10.18.1): `electric` (resistivity) and `electrostatic` (permittivity) are solved by CalculiX's steady heat transfer, whose equation is the same (∇·(σ∇φ) = 0 with the potential for the temperature); `joule` runs the electric solve, puts σ|∇φ|² into each element as heat and solves the thermal problem with the study's thermal bcs. `stress: true` on `thermal` or `joule` adds a static solve with the temperatures (expansion from `T_ref`, default 20 °C) and the mechanical bcs. Electrical material data in the library: `resistivity` (Ω m), `permittivity` (relative), `dielectric_strength` (kV/mm); new materials silver, gold, alumina_96, fr4, ptfe. Resistivity is constant (room temperature).
- **Mesh:** `size` (mm; default 1/20 of the bounding-box diagonal), `order` 2 (10-node tetrahedra, default; 4-node ones are far too stiff in bending), `refine: [{at, r, size}]` finer spheres at stress concentrations. `converge: true` reruns at 0.6 × size and reports the change of the main results.
- **Large meshes:** `equation_solver`: `auto` (CalculiX's iterative Cholesky-preconditioned solver above 150,000 nodes for static and thermal, else the direct one), `direct` or `iterative`.
- **Checks:** `expect: [{quantity, value, tol}]` with a dotted path into the results (`max_displacement`, `max_von_mises_nodal`, `frequencies.0`, `critical_factor`, `reactions.wall.fz`, `faces.load.mean_u.uz`, `probes.tip.mean_u.uz`, `heat_flow.base`, `T_max`...).

### Results

| Analysis | Results |
|---|---|
| static | `max_displacement` (+ where, per component), `max_von_mises` (integration points) and `max_von_mises_nodal` (CalculiX's nodal values), with where; `peak_von_mises` (the larger of the two) and `safety_factor` (yield / that peak); `peak_von_mises_off_bc` and `safety_factor_off_bc` (leaving out the material within `stress_band` of supports and loads, where stresses are singular; default two element sizes of the nominal mesh, the same on the `converge` mesh so the value can converge; widen it if it does not); `peak_at_bc` (the supports or loads the peak sits on); `reactions` per support, `faces` and `probes` (area-weighted mean displacement, largest), `applied_force`, `equilibrium_error`, `mass_kg` ; `stress_check` (CalculiX's integration-point stresses against those recomputed from its displacements: `agree`, `max_difference_MPa`, `elements_disagreeing`) |
| modal | `frequencies` (Hz), `mass_kg` |
| buckling | `buckling_factors`, `critical_factor` (critical load = factor × the applied loads) |
| thermal | `T_max`, `T_min` (+ where), `heat_flow` per temperature and convection boundary (W, into the part), `heat_balance_error`, `faces` and `probes` (mean, max, min temperature) |
| electric | `terminals` (per terminal: `V`, `I_A` into the part), `resistance_ohm` (two terminals), `power_W`, `current_balance_error`, `max_current_density_A_mm2` (+ where) and `max_current_density_off_terminals_A_mm2`, `V_max`, `V_min`, `faces`/`probes` mean potential |
| electrostatic | `terminals` (`V`, `Q_C`), `capacitance_pF` (two voltage levels), `energy_J`, `charge_balance_error`, `max_field_kV_mm` (+ where) and `max_field_off_electrodes_kV_mm`, `breakdown_safety_factor` (dielectric strength / peak field, also off the electrode edges) |
| joule | the thermal results, `electric` (the electric results), `joule_heat_W` and `joule_check_error` (element-wise heat against the electric power) |
| thermal or joule with `stress: true` | `stress`: the static results (displacements, von Mises peaks, safety factors, reactions) of the thermal expansion |

Every run also returns `checks` (the `expect` list, pass or fail), `passed`, `warnings` (unbalanced reactions or heat; a result changing by more than 5 % on a finer mesh; a peak at a support or load; a peak that grows on a finer mesh while the displacement has converged, i.e. a singularity; a `cyl_r` that took holes and fillets together), mesh size and counts, times, and the files.

## The checker

`feat bench` runs studies whose answers are known in closed form, the FEA counterpart of the FreeCAD comparison in build123d-translator:

| Case | Answer | Tolerance |
|---|---|---|
| `axial_bar` | δ = FL/EA, σ = F/A | 1e-4 (exact for the elements) |
| `cantilever` | Timoshenko tip deflection | 3 % |
| `cantilever_modal` | first bending frequency (Euler-Bernoulli) | 3 % |
| `euler_column` | P = π²EI/4L² | 4 % (CalculiX) |
| `gravity_block` | reaction = weight | 1e-6 |
| `pressure_cantilever` | reaction = pA (the loaded face touches the clamp) | 1e-6 |
| `slab`, `fin_end` | conduction, conduction + end convection | 1e-6 (exact) |
| `fin_sides` | pin fin with convecting sides and tip (1-D fin) | 1 % |
| `lame_ring` | quarter ring under internal pressure: Lamé bore displacement and von Mises | 2 % (Gmsh) |
| `bar_resistance`, `bar_current` | copper bar: R = ρL/A, I = V/R, J = I/A; terminal voltage IR | 1e-6 (exact) |
| `plate_capacitor` | FR-4 plate: C = ε₀εᵣA/d, E = V/d, breakdown margin | 1e-6 (exact) |
| `coax_capacitor` | PTFE ring: C = 2πεL/ln(ro/ri) | 0.5 % (Gmsh) |
| `joule_bar` | steel bar, both ends at 20 °C: P = V²/R, T_max = T₀ + qL²/8k | 1e-6 (exact) |
| `thermal_stress_bar` | bar at a uniform 120 °C between walls: σ = EαΔT, wall reaction | 1e-6 (exact) |
| `plate_hole` | quarter plate with a hole: Kt = 2 + (1 − d/W)³ on the net section | 5 % (Gmsh, CalculiX) |

**Status (2026-10-16, carbonio: CalculiX from Ubuntu's calculix-ccx, Gmsh 4.15.2, Python 3.14):** with 2026.10.16.3 all 28 tests and all 11 benchmarks pass with CalculiX and Gmsh (`euler_column`, `lame_ring` and `plate_hole` included). On the first run 10 of 11 passed: `lame_ring` met its Lamé values but its equilibrium check was off by 0.1 %, because the loads on support nodes were integrated on flat triangles while CalculiX integrates the curved faces; the face integrals are now isoparametric. The reference solver passes the 9 cases without Gmsh or CalculiX (the exact ones to rounding; cantilever 1.0 %, frequency 0.13 %, pin fin 0.09 %) and a curved quarter ring (bore displacement 0.01 %, equilibrium 1e-8). Buckling was then found to depend on the machine's load (1 of 4 concurrent `euler_column` runs returned 10-21 instead of 674.7): with eigenvalue accuracy 1e-6 and serial buckling jobs (2026.10.16.8), 8 of 8 concurrent runs give 676.25, identical to every digit.

The first CalculiX run confirmed what the translator assumes about it:

- The `.dat` layout: headers `<what> (<components>) for set <SET> and time <t>` (`displacements`, `stresses`, `forces` for RF per node, `temperatures`, `heat generation` for RFL per node).
- Reactions: CalculiX's RF at a node is the external force there (reaction plus applied loads), so the translator subtracts its own equivalent nodal loads and counts each constrained direction of a node once (`gravity_block`, `pressure_cantilever`, `axial_bar` exact). RFL (heat) works the same way, films included (`slab`, `fin_end` exact, `fin_sides` 1 %).
- Face numbers for `*DLOAD P`, `*DFLUX S`, `*FILM F`: the Abaqus/CalculiX convention for tetrahedra (S1 = 1-2-3, S2 = 1-4-2, S3 = 2-4-3, S4 = 3-4-1).

## How it works

1. **Part:** the STEP (or the goal built to STEP).
2. **Mesh** (`mesher.py`): Gmsh (OpenCascade import, one volume, second-order tetrahedra with mid-edge nodes on the geometry, refinement balls); nodes renumbered, Gmsh's documented 10-node order mapped to CalculiX's (the last two mid-edge nodes swap), corners oriented to a positive volume; elements that Gmsh's curving left inverted or nearly so (thin ligaments relative to the element size) get straight edges, and the result says how many (`mesh.straightened_elements`, with a warning).
3. **Faces** (`meshdata.py`): each Gmsh surface's boundary triangles matched to their element faces (S1..S4), outward normals from the element, areas, centres, planarity, cylinder fits; selectors resolved on this data.
4. **Deck** (`deck.py`): nodes, elements, node sets per boundary condition, material, the step (`*STATIC`, `*FREQUENCY`, `*BUCKLE`, `*HEAT TRANSFER`), boundaries, loads (forces as consistent nodal loads of a uniform traction: 0 at triangle corners, A/3 at mid-edge nodes), outputs.
5. **Solve:** `ccx -i <name>` (or the reference solver).
6. **Results** (`results.py`, `runner.py`): the `.dat` and `.frd` read, results computed on the mesh (area-weighted face means are exact for the element interpolation), checks.

The reference solver (`refsolver.py`) is independent element code (isoparametric tetrahedra, collapsed Gauss quadrature, SciPy sparse solvers) that reads the decks this package writes; it checks the deck and result handling without CalculiX.

## For LLM tools: the MCP server

| Tool | What it does |
|---|---|
| `study_format` | The study format, and which solver and mesher are installed. Read it first. |
| `validate_studies` | Static checks of `{name: study}`. |
| `run_studies` | Meshes, solves and checks: results, selections, checks, warnings, files. Returns within `wait_s` (≤ 50 s, under the clients' call time limits); studies still running are listed in `pending` with the `request` path. |
| `study_status` | The finished and pending studies of a request; waits up to `wait_s` for the rest. Pending studies come with `progress` and the request with `time_left` and `next` (below). |
| `get_deck` | The CalculiX deck of one study, as text. |
| `run_benchmarks` | The closed-form benchmarks: is the toolchain right on this machine? |
| `get_file` | A written file as base64. |
| `jobs` | Recent requests with SLURM jobs (also from other conversations), each study's state, job id and expected time; the translator's queue and the machine's CPUs. `active=True`: only queued or running ones. |
| `cancel` | Stops a request's studies (or the named ones): `scancel` for SLURM jobs. |

**Long runs.** Polling costs the calling model tokens on every call, so pending studies say where they are and how long they still need, and long runs are not polled:

- `progress.<study>`: `stage` (`queued`, `mesh`, `solve`, and with `converge: true` `mesh_fine`, `solve_fine`), `iteration` of `max_iterations` (those steps), `s_per_iteration` (finished steps), `elapsed_min`, `nodes`, `eta_min` (expected minutes left) and `eta_min_max` (at most: the time limit `FEAT_TIMEOUT` of each remaining solve).
- `time_left`: the same for the whole request, with queued studies waiting for a worker; `expected_finish` and `latest_finish` as clock times.
- `next`: under about 5 minutes, call `study_status` again with a `wait_s` that covers it; over 5 minutes, tell the user when the run should finish and stop; collect it later with `study_status` (when the user asks, or from a task scheduled after that time). The server instructions say the same.

The expected time comes from this machine's own finished solves: each run appends its mesh size and mesh and solve times to `timing.jsonl` in the output folder, and a new solve is estimated as (median seconds per nodes^1.5) x nodes^1.5 (sparse direct solver on a 3D mesh); the finer mesh of `converge: true` has about 4.6x the nodes. Before the first finished solve on a machine `eta_min` is unknown and the advice goes by `eta_min_max`.

**SLURM** (2026.10.18.2, the llm-translators contract in `docs/EXECUTION.md`). With `FEAT_EXECUTOR=slurm`, a study estimated above `FEAT_SLURM_MIN_MINUTES` (default 5, from this machine's `timing.jsonl`; a study with no estimate counts as long), or marked `"queue": true`, becomes a SLURM job; the others run in the local pool (`"queue": false` keeps one local). Each job is `<request>/<name>/job.sh` (a complete script: `sbatch job.sh` reruns it by hand) submitted with `sbatch --parsable $FEAT_SLURM_ARGS`; it runs `feat run-study REQUEST NAME` with `--ntasks=1 --cpus-per-task=N`, N = the study's `procs` or `FEAT_PROCS` (default 1), and CalculiX gets `OMP_NUM_THREADS` from `SLURM_CPUS_PER_TASK`; buckling always asks for 1 CPU (its eigenvalues are repeatable only single-threaded). Memory: `FEAT_SLURM_MEM` (`auto`, default; a size such as `8G`; `none`): where SLURM gives a job that asks for no memory the whole node (`CR_Core_Memory` with `DefMemPerNode=UNLIMITED`, as on mochi), only one job runs at a time, so `auto` asks for what CalculiX needs: 2 GB + 30 kB per node when the node count is known before meshing (box parts; the finer mesh for `converge`), else 8 GB; a `--mem` in `FEAT_SLURM_ARGS` replaces it, and a job that runs out ends as `out_of_memory`. The time limit is every CalculiX solve's `FEAT_TIMEOUT` plus 10 minutes (`converge` doubles the solves, `joule` has 2-3), so feat's own timeout writes a clean result first. `job.json` records the job id; a failed `sbatch` is that study's result (`could not submit: ...`), never a silent local run.

A SLURM study's state comes from the files and the queue only, so `study_status` works from any server process, after restarts and from other conversations: `done` (`<name>.json`, written atomically), `queued` (`reason`, `expected_start`, `ahead`), `running` (the progress above plus `job_elapsed_min`), and the final `timed_out`, `out_of_memory` (add `--mem=` to `FEAT_SLURM_ARGS`), `cancelled` and `failed` (the last 30 lines of `job.log`), which move the study into the results with `ok: false`. One `squeue` call per status; `sacct` for jobs that left the queue without a result, else `job.log` (SLURM's "DUE TO TIME LIMIT" and "CANCELLED" lines). `tool_status` reports `executor: {kind, sbatch, min_minutes, procs, time_limit_min}`. Command line: `feat run-study REQUEST NAME`, `feat jobs [--limit N] [--active]`, `feat cancel REQUEST [NAMES...]`. Tested with fake `sbatch`, `squeue`, `scancel`, `sacct` and `sinfo` (`tests/fake_slurm`): submit, running, done collected by a new process, queued, every final state from `sacct` and from `job.log`, cancel, `jobs`.

In the Claude desktop app's `claude_desktop_config.json` (or LM Studio's `mcp.json`), next to `b3dt`:

```json
{"mcpServers": {"feat": {"command": "/Users/<you>/Desktop/fea-translator/.venv/bin/feat-mcp"}}}
```

`feat-mcp --http 8767 [--host 0.0.0.0]` serves streamable HTTP at `/mcp` (no authentication: trusted networks only). With both servers, an agent can design a part with `b3dt` and analyse the same goal with `feat`.

## Files

| Path | What |
|---|---|
| `fea_translator/study.py` | The study format, materials, units, static validation |
| `fea_translator/meshdata.py` | Element ordering, boundary faces, selectors, consistent loads (pure NumPy) |
| `fea_translator/mesher.py` | Gmsh meshing of a STEP or goal |
| `fea_translator/deck.py` | The CalculiX input deck |
| `fea_translator/results.py` | `.dat` and `.frd` readers, von Mises |
| `fea_translator/runner.py` | One study end to end, checks |
| `fea_translator/refsolver.py` | Reference solver (static, modal, thermal) |
| `fea_translator/bench.py` | Benchmark studies with closed-form answers |
| `fea_translator/api.py`, `cli.py`, `mcp_server.py` | JSON API, `feat`, `feat-mcp` |
| `examples/studies.json` | A bracket (static, modal), a finned plate (thermal), a column (buckling) |
| `tests/` | Tests |

## Versions

| Component | Version |
|---|---|
| Package, `runner.py`, `tools/measure_threads.py` (new) | 2026.10.19.1 (CalculiX runs are waited for with `wait4`, which records each run's peak memory: `seconds.ccx_peak_rss_mb` and `ccx_threads` in the result, `rss_mb`, `threads` and `equation_solver` in `timing.jsonl`; `tools/measure_threads.py` measures how CalculiX scales with threads on a SLURM machine (large and small box, direct and iterative solver, a pair side by side) to set the thread cap and the memory formula of SLURM jobs) |
| `runner.py` (before) | 2026.10.18.7 (CalculiX 2.21's multithreaded stiffness assembly gave wrong displacements near the hole of `plate_hole` (equilibrium error 1.4e-3; exact in 4 reruns) while two 16-CPU jobs ran on mochi: the assembly runs single-threaded (`CCX_NPROC_STIFFNESS=1`, like the stress step), and a solve whose balance check is off (direct solver: equilibrium above 1e-5, heat or current balance above 1e-4; iterative: 1e-3) is rerun once fully single-threaded, reported in `solver_reruns` and a warning) |
| `api.py` | 2026.10.18.6 (a queued SLURM study's expected start is when the request's first running study is expected to end, if sooner than SLURM's planned start, which comes from the time limits of the jobs ahead and is kept as the latest; mochi: expected finish 22:13 for a study that would start about 21:49) |
| `api.py` (before) | 2026.10.18.5 (from the SLURM tests on mochi: `FEAT_SLURM_MEM` (auto: each job asks for the memory CalculiX needs; with SLURM's default every job took the node's 60 GB and they ran one at a time); the expected time never exceeds the time limit (it showed 16.3 min against a 15 min limit); a study not yet started is estimated from its mesh size when that is known before meshing (box parts), else its expected time is unknown (it showed 0.0); with SLURM jobs the request's finish is SLURM's expected start of each queued job plus its run time, and `next` gives 'around HH:MM (at the latest HH:MM)'; joule and stress studies count all their solves in the upper bound) |
| `runner.py`, `deck.py` | 2026.10.18.4 (CalculiX 2.21's multithreaded stress step returned wrong stresses at 504 of 960 integration points of `axial_bar` (correct displacements) in a busy test run on mochi, and was right in 10 reruns: the stress step runs single-threaded (`CCX_NPROC_RESULTS=1`), and every static result recomputes the integration-point stresses from the displacements (`stress_check`); if CalculiX's disagree, the recomputed ones give the peaks and safety factors, with a warning) |
| `results.py`, `tests/conftest.py` | 2026.10.18.3 (CalculiX 2.21 writes the heat-flux header as `for setEALL`: the current density, field and Joule heat were not read on mochi; tests run in local mode even where the machine's config sets `FEAT_EXECUTOR=slurm`) |
| `executor.py` (new), `api.py`, `cli.py`, `mcp_server.py`, `runner.py`, `study.py` | 2026.10.18.2 (SLURM execution per llm-translators `docs/EXECUTION.md`: `FEAT_EXECUTOR`, `FEAT_SLURM_MIN_MINUTES`, `FEAT_SLURM_ARGS`, `FEAT_PROCS`; study keys `procs` and `queue`; one job per study with `--cpus-per-task` (1 for buckling); states from files and the queue; `feat run-study`, `jobs` and `cancel` (tools and commands); `executor` in `tool_status`; result files written atomically; tests with fake SLURM commands) |
| `study.py`, `deck.py`, `runner.py`, `meshdata.py`, `refsolver.py`, `bench.py`, `mcp_server.py` | 2026.10.18.1 (electrical: `electric` and `electrostatic` analyses solved as heat transfer, `joule` (electric, then thermal with the Joule heat per element), `stress: true` for thermal and joule (thermal expansion in a static solve); `voltage` and `current` bcs; resistivity, permittivity and dielectric strength in the material library, five new materials; reference solver: body heat, heat-flux output, thermal strain; six benchmarks, five exact; no false 'peak at a support' for a uniform stress, heat-balance floor for a uniform temperature) |
| Package (before), `pyproject.toml` | 2026.10.16.10 |
| `api.py`, `runner.py`, `mcp_server.py` | 2026.10.16.10 (long runs: `progress` per pending study, `time_left` and `next`; `timing.jsonl` of finished solves for the expected time; instructions: tell the user when to come back instead of polling runs over about 5 min) |
| Package (before) | 2026.10.16.9 (the `mesh` extra skips Gmsh on Linux ARM, where PyPI has no wheels; README: Ubuntu's python3-gmsh linked into the venv) |
| `runner.py`, `deck.py` | 2026.10.16.8 (buckling: eigenvalue accuracy 1e-6 instead of CalculiX's 0.01, and one thread per buckling job: with one BLAS thread only, 1 of 4 concurrent `euler_column` runs still returned a wrong factor (20.8); a test runs four at once) |
| `runner.py`, `deck.py` (before) | 2026.10.16.7 (one BLAS thread per CalculiX job: Ubuntu's ccx links OpenBLAS, and a threaded OpenBLAS called from CalculiX's OpenMP code can give wrong eigenvalues; the pump test got a buckling factor of 10.07 for `euler_column`'s 674.7 while several studies ran at once; buckling asks for at least 4 factors and reports the requested number) |
| `mesher.py`, `meshdata.py`, `runner.py` | 2026.10.16.6 (from the pump test: the DN250 nozzle, with 7-12 mm ligaments around 16 holes at 10-15 mm elements, failed with 'cannot match the mid-edge nodes' or CalculiX's 'nonpositive jacobian'; Gmsh's node order is now used directly instead of matching mid-edge nodes by distance, and inverted curved elements are straightened and reported) |
| `runner.py`, `study.py` | 2026.10.16.5 (default `stress_band` two element sizes instead of one: in the second bracket test a one-element band moved 14 % on refinement, a two-element band 0.3 %; the convergence warning for `*_off_bc` points to `stress_band`) |
| `api.py`, `cli.py`, `mcp_server.py`, `runner.py`, `deck.py`, `meshdata.py`, `study.py`, `bench.py` | 2026.10.16.4 (from the first agent test, a bracket designed with b3dt and analysed with feat: background runs, so fine meshes and `converge: true` no longer hit the MCP clients' call time limit (`run_studies` returns within ~45 s, `study_status` collects the rest; `feat status REQUEST`); `cyl_r` with `side: inner/outer` and a warning when one selector took a hole and corner fillets of the same radius, which it did silently; `selection` in every result; safety factor from the larger of the integration-point and nodal peaks (it used the nodal one, which can be lower); peaks and safety factor away from supports and loads, `peak_at_bc` and singularity warnings; `equation_solver` (iterative above 150k nodes)) |
| `meshdata.py`, `runner.py` (before) | 2026.10.16.3 (face integrals isoparametric, as CalculiX computes them on curved 6-node faces: consistent pressure, traction, flux and film loads, face areas and means; body loads integrated over curved elements. The reaction correction on supports that touch a curved loaded face is now exact: `lame_ring` equilibrium error 1e-3 before) |
| `bench.py` | 2026.10.16.2 (`lame_ring` as a 90-degree revolve; README: results of the first CalculiX run) |
| Other modules | 2026.10.16.1 (first version) |
| Tested with | CalculiX (Ubuntu calculix-ccx), Gmsh 4.15.2, Python 3.14 (carbonio); Python 3.10, NumPy 2.2, SciPy 1.15 (reference solver) |

Author: Nicola Ferralis <feranick@hotmail.com>
