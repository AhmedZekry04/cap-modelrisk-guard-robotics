# Audit: what the transceiver dashboard numbers are and where they come from

Every number below is traced to the code that produces it. File references are to this repository.

## Objects

| Name on the dashboard | MuJoCo geom | Role |
|---|---|---|
| Gripper | `gripper_geom` (box, half sizes 15 x 25 x 20 mm), body `gripper` | collision proxy for the palm; driven kinematically along x |
| Target transceiver | `plug_geom` (box, 20 x 15 x 15 mm half sizes), body `plug` | slides along x on joint `plug_slide`; moved with the palm once gripped |
| Neighbouring fibre cable | `wire_geom` (capsule, radius 7 mm, half length 330 mm), body `wire` | sits on a spring-mounted carrier (`wire_flex_x`, `wire_slide` joints, k = 80 N/m) |

All other geometry (rack, panels, patch cables, arm links, fingers, base) has `contype=0 conaffinity=0`
and never touches anything.

## Simulator clearance

* Function: `DataCenterRackEnv.get_lidar_distance()` in `sim/environment.py`.
* Measure: `mujoco.mj_geomDistance(model, data, geom_a, wire_geom, LIDAR_MAX_RANGE_M)` for
  `geom_a` in {`gripper_geom`, `plug_geom`}; the smaller of the two is returned. It is the signed
  distance between the closest points of the two convex geoms; negative means penetration.
* Minimum over the motion: `_update_extrema()` runs after every physics tick inside `step_sim()`
  and keeps `self.min_clearance = min(self.min_clearance, clearance)`. Nothing is sampled between
  ticks, so the minimum is over every 5 ms step of the whole run (approach, grip, pull, dwell, push).
* Reported as `min_clearance_mm` on the result cards and `min_clearance_m` in telemetry. On the
  dashboard it is labelled **simulator clearance** because it is computed from the simulator's own
  geometry, not from a sensor.

## Camera clearance (observation)

* Function: `DataCenterRackEnv.camera_observation()`.
* Source: the `wrist_cam` camera on the palm's rear face, looking back along -x. One render pass
  produces RGB and a metric depth buffer from the same camera, so the two are pixel-aligned.
* Detection: HSV threshold for the cable's orange (`CABLE_HSV`, hue 10-40 deg, S >= 0.45,
  V >= 0.30). Fewer than `CAM_MIN_PIXELS` (60) matching pixels = not detected.
* Distance: median depth over the detected pixels, i.e. distance from the camera plane (the palm's
  rear face) to the nearest cable surface. Confidence: `confident` when the p10-p90 depth spread is
  <= 20 mm and at least 180 pixels are detected; `uncertain` otherwise; `none` when not detected.
* The guarded live action is HELD when the observation is not `confident` before the pull starts,
  or when it becomes `uncertain`/`none` for 3 consecutive ticks during the pull.

## Cable contact and peak force

* Function: `_cable_force_now()`.
* Measure: for every active contact in `data.contact` whose two geoms are the cable and either the
  palm or the plug, `mujoco.mj_contactForce()` is evaluated and the absolute normal component
  (`c_forces[0]`) is summed. This is the constraint force the contact solver applied in that tick.
* Peak over the motion: `_update_extrema()` keeps `self.peak_contact_force` as the running maximum
  per tick. `first_contact_displacement` records the plug travel at the first non-zero force.
* "Cable contact" on the dashboard = `peak_contact_force > 0`. Any contact at all counts; the
  task is "without touching the neighbouring fibre cable".

## The 20 N limit

* Source: `config.CRITICAL_FORCE_LIMIT_N = 20.0`.
* What it is: a **configured simulation threshold** chosen by the team as a stand-in for the
  structural rating of a transceiver/port assembly. It is not measured from hardware and no
  datasheet value backs it. The dashboard labels it "limit 20 N (configured simulation threshold)".
* Note that peak forces in the failing case (300-900 N) are far above it because the gripper is
  driven kinematically into the cable; the number proves the collision, it is not a calibrated
  hardware force.

## Which condition rejects a procedure

`EpistemicRiskEngine.evaluate_policy_risk()` in `sim/risk_engine.py` runs the candidate in
P = 4 perturbed sandboxes (cable x offset +/-5 mm, lateral centre, friction) and computes:

1. `peak_force` (max over sandboxes), `p_violation` (fraction of sandboxes with peak >= 20 N),
   `omega` (normalised ensemble variance, Radic Webster et al.), `combined_risk = 0.5*min(1,omega) + 0.5*p_violation`.
2. `is_safe = combined_risk <= 0.30 and peak_force < 20 N`.
3. `task_ok` = in every sandbox the plug travelled >= 30 mm (`PLUG_DISENGAGE_THRESHOLD_X`),
   ended within 4 mm of seated, and the cable was never touched.
4. `approved = is_safe and task_ok`. `decision` is `APPROVED`, `INTERCEPTED` (not safe) or
   `REJECTED` (safe but task incomplete, e.g. a short pull or a grazing contact below 20 N).

In practice the condition that rejects the original 0.12 m procedure is the peak-force check
(hundreds of newtons in 4/4 sandboxes), which also drives `p_violation = 1` and a large omega; all
three fail together. A short pull (for example 0.02 m) is rejected by the task check alone: no
contact, forces zero, but the plug never disengaged. The guard reports the two cases with
different reasons.

## Verification of the two outcomes

`DataCenterRackEnv.outcome()` returns them separately:

* `transceiver_ok` = `disengaged` (max travel >= 30 mm) and `reseated` (final offset <= 4 mm),
  both read from the `plug_slide` joint position.
* `cable_ok` = no contact; `cable_within_force_limit` = peak < 20 N.
* `task_completed` = `transceiver_ok and cable_ok`.

Tests: `tests/test_sim.py::test_short_pull_fails_task_check_only` and
`tests/test_demo.py::test_incomplete_and_violating_pulls_rejected_for_different_reasons`.
