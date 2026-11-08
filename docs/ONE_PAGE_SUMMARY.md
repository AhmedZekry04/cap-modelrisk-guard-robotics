# ModelRisk-Guard: Runtime Verification and Self-Repair for LLM-Written Robot Procedures in Data-Centre Halls

**AgentVerse 2026, Robotics track.** Local LLM (Qwen2.5-Coder-7B via Ollama) + MuJoCo + browser dashboard. No cloud.

## 1. Who needs it and why

Unplanned data-centre downtime is estimated at $9,000 to $23,750 per minute (industry surveys), and a stalled 800G
transceiver can stop a synchronised training job across thousands of GPUs. Halls are remote, hot and RF-shielded,
so operators are trialling robots for hardware servicing. A manipulator that follows an ungrounded motion can snag the
optical trunk cable next to the port and cause the outage it was sent to prevent. Code-as-Policies (Liang et al.,
ICRA 2023) gives zero-training flexibility, but a 7B model writes physically ungrounded motions: in our rack it pulls a
transceiver 120 mm and sweeps into a fibre cable 45 mm behind the gripper. Static checks cannot see this.

## 2. What we built

* **Robot and sensors.** A UR5e arm with a Robotiq 2F-85 gripper (MuJoCo Menagerie) on a mobile platform. Sensors are
  MuJoCo sensor elements, not simulator shortcuts: a wrist RGB-D camera (colour segmentation + aligned metric depth gives
  "camera clearance" and a confidence state), a gripper touch sensor, joint encoders, and a 36-ray LiDAR on the platform.
  Geometry-derived numbers are labelled "simulator clearance" and shown separately.
* **Guard before motion.** The model's procedure is frozen (sha-256 policy id) and executed in four perturbed sandboxes
  (cable position +/-5 mm, friction). Checks: peak cable force < 20 N (a configured simulation threshold), zero contact,
  ensemble variance (Radic Webster et al. 2021), and task completion (disengaged >= 30 mm, lifted, reseated).
* **Evidence-driven repair.** The measured evidence (first contact 44 mm, peak 1.3 kN, admissible stroke window)
  goes back to the model, which rewrites the 13-line procedure (sensor pre-checks, grip, pull 0.120 -> 0.037 m,
  80 mm inspection lift, reseat, post-check). The revision is re-verified; only then does the exact approved text run
  through the restricted skill API. Every executed skill call is logged with the policy id; the page proves
  executed == approved. If nothing passes, the robot holds and says why.
* **Perception-driven correction mid-run.** During the live pull the camera clearance is checked every 5 ms physics
  tick. An unscripted cable shift (button) drops it below 4 mm: the robot holds, the model re-plans from the measured
  travel, the re-plan is verified from that held state, then resumes. An occluded camera holds motion before it starts.
* **Navigation mission, same stack.** "Go to Rack A and inspect it": the model writes `follow_route`/`inspect_rack`;
  the guard runs a sandbox rollout plus LiDAR validation of every route. The maintenance cart is draggable live on the
  map; when the LiDAR stops the robot, it re-plans from its current pose (R1 -> R2), verified, up to four times.
* **Judge-facing dashboard.** One main view, one code panel (original, revision diff, executed skills), one
  plain-English guard message, Original vs Guarded cards, Technical details. Both paths start from an identical frozen
  policy and hashed simulator state; every number on the page is derived from the run.

## 3. Architecture, placement and measured latency (one laptop, RTX-class GPU not required)

| Layer | Placement | Measured | Role |
|---|---|---|---|
| Cognition | Qwen2.5-Coder-7B, Ollama, local, 4.7 GB | 4-10 s per procedure, 5-9 s per revision | writes and repairs `policy(robot)` |
| Static gate | Python AST whitelist | < 1 ms | no imports; robot skills only |
| Sandbox guard | 4 headless MuJoCo instances, seeded | 0.1-0.3 s per verdict | perturbed physics, task checks, route validation |
| Perception | wrist RGB-D 192x144, touch, encoders; 36-ray LiDAR | 5 ms per observation | camera clearance, contact, scan |
| Execution | kinematic drive, real contacts at 200 Hz | 60-100 s per full comparison (rendered) | approved code only, skill trace |

## 4. Measured results (local model, today)

Original 120 mm pull: contact at 48 mm, peak 1,269 N, FAILED in 4/4 sandboxes. Revision 37 mm: 0 N, min clearance
6.7 mm in sandboxes and 8.0 mm live, camera 44.3 mm vs geometry 45.0 mm, COMPLETED with an 80 mm lift. A 20 mm pull is
rejected by the task check alone (no contact, never disengaged). Occluded camera: HELD, no motion. Navigation with the
cart dropped into the aisle mid-drive: LiDAR stop at (2.3, 0.0) with 0.44 m ahead, re-plan to R2 verified from that
pose, Rack A reached and inspected after one re-plan. 26 automated tests across five suites pass.

## 5. Limitations and safety boundaries

The 20 N limit is configured, not measured from hardware; contact forces prove the collision, they are not calibrated.
The arm follows the gripper by inverse kinematics (no joint dynamics); the cable is a rigid capsule on a spring
carrier; camera depth is the renderer's ideal depth; navigation pose is simulator ground truth and routes are fixed
polylines (a re-plan picks another validated route). The system prompt carries a vendor 0.12 m stroke prior that makes
the failure reproducible, and the 7B model needs the measured admissible window to repair reliably. Fail-safe by
construction: no code moves the robot until it has passed the sandbox, and a hold is always a valid outcome.
