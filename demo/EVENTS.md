# Dashboard event contract

The dashboard (`demo/static/index.html`) and the backend (`demo/server.py`, `demo/comparison.py`)
share this contract. Keep both sides in sync.

## HTTP endpoints (served by `python -m demo.server`, default `http://127.0.0.1:8000`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the dashboard page |
| GET | `/static/<file>` | static assets |
| GET | `/api/events` | Server-Sent Events stream (see below) |
| GET | `/api/state` | current snapshot as JSON (same shape as the `snapshot` event) |
| POST | `/api/run` | body `{"task": "<text>"}` -> `202 {"ok": true}` or `409 {"error": "busy"}` |
| POST | `/api/reset` | cancels a running comparison and clears state -> `200 {"ok": true}` |
| POST | `/api/replay` | replays the recorded run (`demo/recorded_run.json`) -> `202` or `404` if none |

## SSE stream

`GET /api/events` returns `text/event-stream`. Every message is `data: <json>\n\n`.
Every JSON object has a `type` field. On connect the server first sends one `snapshot`
with the full current state, then live events as they happen.

### `snapshot`
```json
{"type":"snapshot","task":"...","phase":"idle","status_message":"",
 "original_code":"","revised_code":"","diff_lines":[],
 "guard_initial":null,"guard_revised":null,
 "result_raw":null,"result_guarded":null,"fairness":null,
 "timeline":[{"t":0.0,"level":"info","text":"..."}],
 "last_frame":{"main":null,"closeup":null},
 "recorded":false,"recorded_at":null,
 "llm":{"model":"qwen2.5-coder:7b","endpoint":"http://localhost:11434","streaming":true},
 "series":[{"t":1.2,"run":"raw","cable_force_n":0.0,"min_clearance_mm":45.0,"displacement_mm":3.0}]}
```
`guard_*`, `result_*`, `fairness`, `last_frame.*` hold the payload of the latest event of that
type (without the `type` field) or `null`.

### `status`
`{"type":"status","phase":"<phase>","message":"<short human text>"}`

Phases, in the order they normally occur:
`idle`, `generating`, `raw_run`, `guard_check`, `patching`, `guard_recheck`, `guarded_run`,
`sandbox_replay`, `done`, `withheld`, `error`.

### `task`
`{"type":"task","text":"..."}` the task text the run is using.

### `timeline`
`{"type":"timeline","t":12.3,"level":"info|ok|warn|fail","text":"..."}`
`t` is seconds since the run started.

### `code_token` and `code`
`{"type":"code_token","target":"original|revised","token":"def "}` streamed tokens, append in order.
`{"type":"code","target":"original|revised","code":"<full text>","source":"ollama:qwen2.5-coder:7b"}`
final text; replace whatever was streamed. `source` may also be `"recorded"` or `"stub"`.

### `diff`
`{"type":"diff","lines":[{"op":"+","text":"..."},{"op":"-","text":"..."},{"op":" ","text":"..."}],"summary":"stroke 0.120 m -> 0.035 m"}`

### `frame`
```json
{"type":"frame","run":"raw|guarded|sandbox_replay","view":"main|closeup",
 "image":"data:image/jpeg;base64,...","t":4.2,
 "hud":{"phase":"approach|grip|pull|dwell|push|idle","step":37,
        "displacement_mm":31.0,"cable_force_n":0.0,"peak_force_n":0.0,"min_clearance_mm":14.2},
 "labels":[{"name":"Target transceiver","u":0.52,"v":0.41,"color":"#38bdf8"},
           {"name":"Neighbouring fibre cable","u":0.44,"v":0.47,"color":"#fb923c"},
           {"name":"Gripper","u":0.49,"v":0.43,"color":"#f87171"}]}
```
`u`,`v` are normalised image coordinates from the top-left. `labels` may be empty.
`min_clearance_mm` is the signed distance between the gripper/plug and the cable surface;
negative means penetration (contact).

### `guard`
```json
{"type":"guard","stage":"initial|revised","decision":"APPROVED|INTERCEPTED|REJECTED",
 "summary":"Peak cable force 287.9 N exceeds 20.0 N in 4/4 sandboxes",
 "checks":[
   {"name":"AST whitelist","passed":true,"measured":"ok","limit":"imports forbidden; skills only","unit":"","detail":"..."},
   {"name":"Peak cable force","passed":false,"measured":287.9,"limit":20.0,"unit":"N","detail":"max over 4 sandboxes"},
   {"name":"Violation probability","passed":false,"measured":1.0,"limit":0.0,"unit":"","detail":"4 of 4 sandboxes exceeded the force limit"},
   {"name":"Epistemic variance (omega)","passed":false,"measured":185.7,"limit":0.3,"unit":"","detail":"ensemble disagreement on cable force"},
   {"name":"Min cable clearance","passed":false,"measured":-7.5,"limit":0.0,"unit":"mm","detail":"penetration in 4 of 4 sandboxes"},
   {"name":"Task completion","passed":true,"measured":"disengaged and reseated","limit":"stroke >= 30 mm, reseated","unit":"","detail":""}
 ],
 "ensemble":[{"particle":0,"cable_x_offset_mm":-3.1,"friction":0.41,"peak_force_n":260.4,"min_clearance_mm":-7.4,"max_displacement_mm":120.0,"violated":true}],
 "omega":185.7,"violation_probability":1.0,"risk_score":1.0,"threshold":0.3}
```

### `result`
```json
{"type":"result","run":"raw|guarded","label":"Raw LLM policy (sandbox)",
 "status":"COMPLETED|FAILED|WITHHELD|BLOCKED",
 "task_completed":false,"disengaged":true,"reseated":true,"cable_contact":true,
 "peak_force_n":287.9,"min_clearance_mm":-7.5,"max_displacement_mm":120.0,"final_displacement_mm":0.1,
 "note":"Gripper swept into the fibre cable at 48 mm of travel"}
```
`WITHHELD` means the guard refused to move the robot (no live motion happened).
`BLOCKED` means the code never ran because the static check rejected it.

### `fairness`
```json
{"type":"fairness","policy_sha256":"...","initial_state_sha256":"...",
 "raw_state_sha256":"...","guarded_state_sha256":"...","identical":true,
 "seed":7,"cable_x_mm":243.0,"cable_y_mm":30.0,
 "note":"Both paths ran the same frozen policy from the same saved state"}
```

### `series`
`{"type":"series","t":4.2,"run":"raw|guarded|sandbox_replay","cable_force_n":0.0,"min_clearance_mm":45.0,"displacement_mm":3.0}`
one sample per rendered frame, for the live strip chart.

### `error`
`{"type":"error","message":"Local model did not answer within 120 s","recoverable":true}`

### `reset`
`{"type":"reset"}` clear everything back to idle.

### `recorded`
`{"type":"recorded","active":true,"recorded_at":"2026-11-08T03:10:00Z","note":"Recorded run"}`
when `active` is true the dashboard must show a clear "Recorded run" banner until `reset`.

### `done`
`{"type":"done","outcome":"revised_passed|original_passed|withheld|error","summary":"..."}`
