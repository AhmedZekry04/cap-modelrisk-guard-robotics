# ModelRisk-Guard: a runtime guard for LLM-written robot procedures

**AgentVerse 2026 - Robotics track**

A local LLM writes the procedure a data-centre maintenance robot should run. Before the robot
moves, ModelRisk-Guard executes that exact code in a MuJoCo sandbox, measures what would happen,
feeds the measured evidence back to the model, and only executes a revision that passes, through
the same restricted skill API, logging every skill call against the policy id. Everything runs on
one laptop: Ollama (`qwen2.5-coder:7b`), MuJoCo, and a browser dashboard served by the Python
standard library. No cloud.

Two missions share the pipeline:

| Mission | Task typed in the dashboard | Hazard | Guard |
|---|---|---|---|
| Transceiver reseat | *Disengage the target optical transceiver in slot 1 without touching the neighbouring fibre cable, then reseat it.* | a fibre trunk routed across the rack front, 45 mm behind the gripper | 4 perturbed physics sandboxes + wrist RGB-D camera confirmation |
| Rack navigation | *Go to Rack A and inspect it.* | a maintenance cart moved into the aisle | sandbox rollout with simulated LiDAR + route validation |

## Launch

```bash
pip install -r requirements.txt          # mujoco, numpy, rich, pillow
ollama pull qwen2.5-coder:7b             # once, ~4.7 GB
ollama run qwen2.5-coder:7b "hi"         # warm the model before a demo (cold load ~90 s)
python -m demo.server                    # opens http://127.0.0.1:8000
```

Without Ollama: `python -m demo.server --stub` (deterministic stand-in policies, labelled `stub`).
`--record` saves the next genuine run to `demo/recorded_run.json`, which **Replay recorded run** plays
back under an amber "RECORDED RUN" banner.

## Demo script

1. Press **Fullscreen**. Mission **Transceiver reseat** is selected; the task is prefilled.
2. Press **Run**. The model streams the procedure (about 5 s). The ORIGINAL procedure runs directly
   in a sandbox: the gripper pulls 120 mm and sweeps into the cable (result card FAILED). The guard
   then tests the same frozen code in 4 perturbed sandboxes and shows one plain-English message
   (INTERCEPTED, peak force, first contact). The model's revision streams in with the diff
   (0.120 m -> 0.037 m); the guard re-tests it; the wrist camera confirms the cable; the approved
   code runs live and the GUARDED card turns COMPLETED. About 60 s with a warm model.
3. Optional setup toggles, applied identically to both paths: **Move cable toward the gripper (mm)**
   (the camera clearance and the simulator clearance both drop by that amount) and **Occlude the
   wrist camera** (the guard HOLDS the approved motion: "camera could not confirm the cable").
4. Switch to **Rack navigation**, tick **Move the maintenance cart into the aisle**, press **Run**.
   The model writes `follow_route("R1"); inspect_rack("Rack A")`. The ORIGINAL run drives into the
   aisle until the LiDAR sees the cart and stops 4.9 m short (FAILED). The guard's sandbox rollout
   reports the block and that route R2 validated clear; the revision takes R2; the robot reaches
   Rack A on the map and the inspection skill runs there (COMPLETED). Untick the cart and run again:
   the original procedure passes on R1 and the page says no revision was needed.
5. **Reset** clears the page. The **Code** panel's "Executed skills" tab lists every skill call the
   simulator ran, with the policy id and a check that the executed code is byte-identical to the
   approved code. **Technical details** holds the guard checks, per-sandbox tables, hashes and the
   timeline.

Exact prompts: `Disengage the target optical transceiver in slot 1 without touching the neighbouring
fibre cable, then reseat it.` and `Go to Rack A and inspect it.` Both are editable.

## What is measured, and how (transceiver)

See [docs/AUDIT_TRANSCEIVER.md](docs/AUDIT_TRANSCEIVER.md) for the line-by-line trace. In short:

* **Simulator clearance**: signed distance between the gripper palm (or plug) and the cable capsule,
  `mj_geomDistance`, minimum taken every 5 ms tick over the whole motion. Labelled "simulator
  clearance" because it comes from the simulator's geometry.
* **Camera clearance**: the wrist camera on the palm's rear face renders RGB and an aligned metric
  depth buffer; the orange cable is segmented in HSV and the median depth over its pixels is the
  distance from the palm's rear face to the cable. Confidence is `confident`, `uncertain` or `none`.
  The guarded motion is held when the camera is not confident before or during the pull.
* **Cable contact / peak force**: sum of the normal contact forces between the cable and the
  palm/plug from `mj_contactForce`, peak over the motion. Any contact fails the task.
* **20 N limit**: `config.CRITICAL_FORCE_LIMIT_N`, a **configured simulation threshold** standing in
  for the port's structural rating; it is not a measured hardware value.
* **Verdicts**: the transceiver outcome (disengaged >= 30 mm, reseated within 4 mm) and the cable
  outcome (no contact, force below the limit) are checked separately. A 20 mm pull is rejected by
  the task check only; a 120 mm pull is intercepted by the force check.

## How the comparison is kept fair

* One generation per run; the code is frozen and identified by its sha256 (the policy id).
* A pristine environment is built with the same setup for the original run, the guard sandboxes
  and the guarded run; their state hashes are compared before any policy code runs.
* The guard verifies the exact text that is later executed; executed skill calls carry the policy id
  and the page shows "executed code matches approved policy".
* If no revision passes, the robot does not move; the last rejected candidate is shown only as a
  labelled sandbox replay.

## Navigation mission

`sim/nav_env.py`: a known 12 x 10 m hall with six named racks, a dock, two routes to Rack A's
service point (R1 through the cold aisle, 7.6 m; R2 around the north corridor, 18 m) and a
maintenance cart that can be moved into the aisle. The robot base is kinematic (x, y, yaw written
each 50 ms control tick); the LiDAR is 36 MuJoCo rangefinder rays at 10 degree spacing, range 8 m,
cast against the real scene geometry including the cart. A return closer than 0.5 m inside a
+/-30 degree heading cone stops the robot ("blocked at (x, y)"). The pose shown on the map is the
simulator's ground truth; there is no localisation estimate. Arrival is read from that pose and the
mission completes only when `inspect_rack("Rack A")` succeeds within 0.6 m of the service point.

## Tests

```bash
python -m tests.test_sim          # scene, HAL, clearance sensor, guard; short pull rejected by the task
                                  # check only; wrist camera detection tracks a cable shift, occlusion -> uncertain
python -m tests.test_agent        # AST whitelist, repair cycle (Ollama if up, else offline fallback)
python -m tests.test_integration  # terminal pipeline incl. the mid-stroke disturbance reflex
python -m tests.test_demo         # identical policy + state for both paths; unsafe candidate withheld;
                                  # incomplete vs violating pulls rejected for different reasons; executed
                                  # skill trace matches the approved policy id; occluded camera holds motion
python -m tests.test_nav          # robot reaches Rack A on a clear route and only completes after inspect_rack;
                                  # cart in the aisle changes the LiDAR scan and the route verdict; blocked
                                  # route rejected, R2 revision executed with matching policy id
```

## Known limitations

* The system prompt carries a generic vendor extraction stroke (0.12 m). That ungrounded prior
  makes the raw failure reproducible; without it the 7B model sometimes writes a passing procedure,
  which the page reports honestly ("no revision needed").
* The repair prompt gives the model the measured admissible stroke window (metres). Without it the
  7B model reuses the contact distance or slips between millimetres and metres. Every revision is
  re-verified; a bad one is withheld.
* Cable detection is colour segmentation of a rendered image; it would need a learned detector on
  real camera data. Depth is the renderer's z-buffer, i.e. an ideal calibrated depth camera.
* The cable is one rigid capsule on a spring carrier; the gripper is driven kinematically; the arm
  is a visual two-link IK representation without joint dynamics. Peak forces (hundreds of N) prove the
  collision, they are not calibrated hardware forces.
* Navigation: kinematic base, no contact physics on the base, two fixed routes from the scenario
  config (the "replan" is the model choosing the other validated route, not a path planner), pose
  from the simulator. Full mobile navigation would need a localisation estimate (e.g. scan matching
  on the LiDAR), a planner over an occupancy grid, and base dynamics.
* Omega is reported as the paper's normalised variance; it is very large whenever sandboxes
  disagree about contact and behaves like a binary flag.
* Live execution is paced for the camera; the physics is identical to the headless sandboxes.
  A cold Ollama start takes ~90 s; the page shows a visible error and offers the recorded run.

## Repository layout

| Path | Purpose |
|---|---|
| `demo/server.py` | dashboard server (`python -m demo.server`), missions, SSE |
| `demo/comparison.py` | shared original-vs-guarded flow, transceiver mission, frame + wrist-camera streaming, recording |
| `demo/nav_runner.py` | navigation mission runner and LiDAR route-validation guard |
| `demo/static/index.html` | the dashboard (single file, no external assets) |
| `demo/EVENTS.md` | event contract between server and page |
| `sim/environment.py` | rack scene, arm, gripper HAL, wrist RGB-D camera, clearance and force measurement |
| `sim/nav_env.py` | hall map, kinematic mobile base, LiDAR, navigation skills |
| `sim/risk_engine.py` | perturbed-sandbox guard for the transceiver mission |
| `agent/` | local LLM client (streaming), prompts for both missions, static code check |
| `docs/AUDIT_TRANSCEIVER.md` | trace of every dashboard number to code and simulator data |
| `main.py` | terminal + MuJoCo viewer demo with the mid-stroke disturbance reflex |
| `tests/` | the five suites above |
