# Dashboard event contract (v2)

The dashboard (`demo/static/index.html`) and the backend (`demo/server.py`, `demo/comparison.py`,
`demo/nav_runner.py`) share this contract. Keep both sides in sync.

## HTTP endpoints (served by `python -m demo.server`, default `http://127.0.0.1:8000`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the dashboard page |
| GET | `/static/<file>` | static assets |
| GET | `/api/events` | Server-Sent Events stream (see below) |
| GET | `/api/state` | current snapshot as JSON (same shape as the `snapshot` event) |
| GET | `/api/missions` | list of missions: `[{"id","title","prompt","setup":[...]}]` (see Missions) |
| POST | `/api/run` | body `{"mission": "<id>", "task": "<text>", "setup": {...}}` -> `202 {"ok": true}` or `409 {"error": "busy"}` |
| POST | `/api/reset` | cancels a running comparison and clears state -> `200 {"ok": true}` |
| POST | `/api/replay` | replays the recorded run (`demo/recorded_run.json`) -> `202` or `404` if none |

## Missions

`GET /api/missions` returns:
```json
[
 {"id":"transceiver","title":"Transceiver reseat",
  "prompt":"Disengage the target optical transceiver in slot 1 without touching the neighbouring fibre cable, then reseat it.",
  "main_view":"camera",
  "setup":[{"key":"cable_shift_mm","label":"Move cable toward the gripper (mm)","type":"number","default":0,"min":0,"max":25,"step":1},
           {"key":"camera_occluded","label":"Occlude the wrist camera","type":"bool","default":false}]},
 {"id":"navigation","title":"Rack navigation",
  "prompt":"Go to Rack A and inspect it.",
  "main_view":"map",
  "setup":[{"key":"obstacle","label":"Move the maintenance cart into the aisle","type":"bool","default":false}]}
]
```
`main_view` says what the big view shows: `camera` = MuJoCo frames (`frame` view `main`), `map` = the live
2D map drawn from `map` + `nav` events. The setup values are sent back verbatim in `POST /api/run`.
The same setup is applied to BOTH the raw and the guarded path of a run.

## SSE stream

`GET /api/events` returns `text/event-stream`. Every message is `data: <json>\n\n`.
Every JSON object has a `type` field. On connect the server first sends one `snapshot`
with the full current state, then live events as they happen.

### `snapshot`
```json
{"type":"snapshot","mission":"transceiver","setup":{"cable_shift_mm":0,"camera_occluded":false},
 "task":"...","phase":"idle","status_message":"",
 "original_code":"","revised_code":"","diff_lines":[],"diff_summary":"","original_source":"","revised_source":"",
 "guard_initial":null,"guard_revised":null,"guard_message":null,
 "result_raw":null,"result_guarded":null,"fairness":null,"policy_ids":null,
 "skill_calls":[],"camera":null,"map":null,"nav":null,
 "timeline":[{"t":0.0,"level":"info","text":"..."}],
 "last_frame":{"main":null,"closeup":null},
 "recorded":false,"recorded_at":null,
 "llm":{"model":"qwen2.5-coder:7b","endpoint":"http://localhost:11434","streaming":true},
 "series":[{"t":1.2,"run":"raw","cable_force_n":0.0,"min_clearance_mm":45.0,"displacement_mm":3.0}],
 "done":null,"error":null}
```
`guard_*`, `guard_message`, `result_*`, `fairness`, `policy_ids`, `camera`, `map`, `nav`, `last_frame.*`
hold the payload of the latest event of that type (without the `type` field) or `null`.
`skill_calls` holds every `skill_call` payload of the run in order.

### `status`
`{"type":"status","phase":"<phase>","message":"<short human text>"}`

Phases, in the order they normally occur:
`idle`, `generating`, `raw_run`, `guard_check`, `patching`, `guard_recheck`, `guarded_run`,
`sandbox_replay`, `done`, `withheld`, `error`. `withheld` is also used when the guarded action is
HELD because the camera could not confirm the cable.

### `mission`
`{"type":"mission","id":"transceiver|navigation","title":"...","prompt":"...","setup":{...},"main_view":"camera|map"}`
sent at the start of every run.

### `task`
`{"type":"task","text":"..."}` the full task text the run is using (show it in full, never truncated).

### `timeline`
`{"type":"timeline","t":12.3,"level":"info|ok|warn|fail","text":"..."}`
`t` is seconds since the run started.

### `code_token` and `code`
`{"type":"code_token","target":"original|revised","token":"def "}` streamed tokens, append in order.
`{"type":"code","target":"original|revised","code":"<full text>","source":"ollama:qwen2.5-coder:7b"}`
final text; replace whatever was streamed. `source` may also be `"recorded (...)"` or `"stub"`.

### `diff`
`{"type":"diff","lines":[{"op":"+","text":"..."},{"op":"-","text":"..."},{"op":" ","text":"..."}],"summary":"extraction stroke 0.120 m -> 0.037 m"}`

### `policy_ids`
```json
{"type":"policy_ids","original_sha256":"<64 hex>","approved_sha256":"<64 hex>|null",
 "executed_sha256":"<64 hex>|null","executed_matches_approved":true|false|null}
```
Policy IDs are sha256 of the exact code text. `executed_matches_approved` is true when the code
that ran live is byte-identical to the code the guard approved. Show the short (10-char) ids.

### `skill_call`
```json
{"type":"skill_call","run":"raw|guarded|sandbox_replay","seq":3,"skill":"execute_primitive",
 "args":{"primitive":"pull_linear","distance":0.037},"step":412,"result":"displacement 37.0 mm",
 "policy_sha256":"<64 hex>","t":21.4}
```
One per restricted-skill call the simulator actually executed, in order. Navigation skills look like
`{"skill":"follow_route","args":{"route_id":"R1"},"result":"blocked at (1.1, 0.0) after 3.4 m"}` and
`{"skill":"inspect_rack","args":{"rack_id":"Rack A"},"result":"inspected (0.3 m from rack)"}`.

### `frame`
```json
{"type":"frame","run":"raw|guarded|sandbox_replay","view":"main|closeup",
 "image":"data:image/jpeg;base64,...","t":4.2,
 "hud":{"phase":"approach|grip|pull|dwell|push|drive|turn|inspect|idle","step":37,
        "displacement_mm":31.0,"cable_force_n":0.0,"peak_force_n":0.0,"min_clearance_mm":14.2},
 "labels":[{"name":"Target transceiver","u":0.52,"v":0.41,"color":"#38bdf8"}]}
```
For the navigation mission `view` is `main` only (a MuJoCo follow camera); `hud` then carries
`{"phase","step","speed_mps","lidar_min_m","distance_to_target_m"}`. `min_clearance_mm` here is the
SIMULATOR clearance (geometry); the camera estimate comes in `camera_frame`.

### `camera_frame` (transceiver mission)
```json
{"type":"camera_frame","run":"raw|guarded|sandbox_replay","t":4.3,
 "image":"data:image/jpeg;base64,...",
 "detected":true,"pixel_count":812,"bbox":[12,60,180,84],
 "camera_clearance_mm":44.8,"sim_clearance_mm":45.0,"depth_spread_mm":3.1,
 "confidence":"confident|uncertain|none","reason":"812 cable pixels, depth p10-p90 within 20 mm",
 "source":"wrist RGB-D camera (rendered), colour segmentation + aligned metric depth"}
```
The image is the wrist camera's RGB frame with the detection overlay already drawn (green tint on
cable pixels, bounding box). Display BOTH numbers with their labels: "Camera clearance (RGB-D)" and
"Simulator clearance (geometry)". When `confidence` is `uncertain` or `none`, show the word
"uncertain" in amber with `reason`; the backend holds the guarded motion in that case.

### `guard`
Same as v1: `{"type":"guard","stage":"initial|revised","decision":"APPROVED|INTERCEPTED|REJECTED","summary":"...",
"checks":[{"name","passed","measured","limit","unit","detail"}],"ensemble":[...],"omega","violation_probability","risk_score","threshold"}`.
For the navigation mission `ensemble` is a one-element list (single sandbox rollout) and omega/risk may be `null`.
Belongs in Technical details.

### `guard_message`
```json
{"type":"guard_message","stage":"initial|revised|hold","verdict":"APPROVED|INTERCEPTED|REJECTED|HELD",
 "text":"The guard blocked the original procedure: in all 4 sandboxes the gripper hit the neighbouring cable after about 44 mm of travel (peak 859 N, limit 20 N). Evidence sent back to the model."}
```
THE one plain-English guard message shown prominently. Replace on each new event.

### `result`
```json
{"type":"result","run":"raw|guarded","label":"Original procedure (sandbox)",
 "status":"COMPLETED|FAILED|WITHHELD|BLOCKED|HELD",
 "task_completed":false,"disengaged":true,"reseated":true,"cable_contact":true,"cable_within_force_limit":false,
 "peak_force_n":287.9,"min_clearance_mm":-7.5,"max_displacement_mm":120.0,"final_displacement_mm":0.1,
 "note":"Gripper swept into the fibre cable after 48 mm of travel"}
```
Navigation results carry instead: `arrived`, `inspected`, `blocked`, `route_id`, `distance_to_rack_m`,
`path_length_m`, `lidar_min_m`, plus `task_completed`, `status`, `note`.
`WITHHELD` = the guard refused to move the robot. `HELD` = motion was stopped/held during execution
(camera uncertain or LiDAR block). `BLOCKED` = never ran (static check).

### `fairness`
Same as v1 plus `setup` (the setup dict applied to both paths):
`{"type":"fairness","policy_sha256","initial_state_sha256","raw_state_sha256","guarded_state_sha256","identical","seed","cable_x_mm","cable_y_mm","setup":{...},"note"}`

### `series`
`{"type":"series","t":4.2,"run":"raw|guarded|sandbox_replay","cable_force_n":0.0,"min_clearance_mm":45.0,"displacement_mm":3.0}`
Navigation runs send `{"type":"series","t","run","lidar_min_m","speed_mps","distance_to_target_m"}`.

### `map` (navigation mission, sent once per run before any `nav`)
```json
{"type":"map","bounds":[-6,-5,6,5],
 "racks":[{"id":"Rack A","x":-2.5,"y":1.8,"w":1.2,"h":0.8,"face":"south"}],
 "routes":{"R1":{"label":"aisle (short)","waypoints":[[4.5,0],[0,0],[-2.5,0],[-2.5,0.9]],"length_m":7.9},
           "R2":{"label":"north corridor (long)","waypoints":[[4.5,0],[4.5,3.2],[-4.5,3.2],[-4.5,0],[-2.5,0],[-2.5,0.9]],"length_m":18.3}},
 "start":[4.5,0.0],"target":"Rack A","target_point":[-2.5,0.9],
 "obstacle":{"x":1.5,"y":0.0,"w":0.8,"h":0.6,"label":"maintenance cart"}|null,
 "pose_source":"MuJoCo simulator ground truth (no localisation estimate)"}
```
Coordinates are metres; draw the map with y up.

### `nav` (navigation mission, several per second while the robot moves)
```json
{"type":"nav","run":"raw|guarded|sandbox_replay","t":6.1,"predicted":false,
 "pose":[2.1,0.0,180.0],"pose_source":"simulator ground truth",
 "route_id":"R1","route":[[4.5,0],[0,0],[-2.5,0],[-2.5,0.9]],"waypoint_index":1,
 "lidar":[[1.3,0.0],[1.2,0.4]],"lidar_min_m":0.62,"lidar_rays":36,
 "blocked":false,"blocked_at":null,"arrived":false,"inspected":false,
 "distance_to_target_m":4.7,"speed_mps":1.2}
```
`lidar` holds the world-frame hit points of the scan (rays with no hit are omitted).
`predicted:true` marks sandbox-rollout data (guard); draw it differently (dashed/ghost) from live.
`blocked_at` is `[x,y]` when a LiDAR return inside the stop envelope halted the robot.

### `error`, `reset`, `recorded`, `done`
Unchanged from v1:
`{"type":"error","message","recoverable"}`, `{"type":"reset"}`,
`{"type":"recorded","active","recorded_at","note"}`,
`{"type":"done","outcome":"revised_passed|original_passed|withheld|held|error","summary"}`.
