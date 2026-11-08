"""
config.py - Single Source of Truth for ModelRisk-Guard Configuration
"""
# Epistemic Risk Engine Hyperparameters
ENSEMBLE_SIZE_P = 4              # Number of parallel headless instances
HORIZON_STEPS_H = 60             # Lookahead steps per particle
RISK_THRESHOLD_TAU = 0.30        # Interception cutoff threshold
CRITICAL_FORCE_LIMIT_N = 20.0    # Structural damage threshold (Newtons)

# Baseline Normalization Variances (Radic Webster et al. Eq. 2)
BASELINE_VAR_DISPLACEMENT = 1e-4 # m^2
BASELINE_VAR_FORCE = 1.0         # N^2

# Physical Dimensions & Tolerances (Meters)
PLUG_INITIAL_X = 0.33            # Seated coordinate of transceiver plug
PLUG_DISENGAGE_THRESHOLD_X = 0.03 # Minimum extraction distance for electrical disconnection
WIRE_NOMINAL_X = 0.38            # (legacy, unused) nominal X of the adjacent optical wire
WIRE_NOMINAL_Y = 0.03            # Nominal lateral centre of the trunk cable (it runs across the rack front)
WIRE_CLEARANCE_LIMIT_X = 0.045   # Extraction travel at which the gripper rear reaches the cable surface
CABLE_X_TOLERANCE_M = 0.005      # Epistemic placement uncertainty of the cable along the extraction axis
RACK_FACE_X = 0.345              # x of the rack front face (the plug protrudes 35 mm from it)
Z_WORK = 0.90                    # Height of the target port above the floor

# Sensors / unscripted disturbance
LIDAR_MIN_CLEARANCE_M = 0.012    # Emergency halt if gripper-to-cable clearance drops below 12 mm
LIDAR_MAX_RANGE_M = 0.20         # Maximum reported clearance
UNSCRIPTED_CABLE_SHIFT_X = 0.010 # Unscripted disturbance: cable carrier shifts 10 mm toward the gripper

# Live execution / dashboard
LIVE_STEP_SLEEP_S = 0.03         # Pacing of live (rendered) execution per physics tick
POLICY_STEP_BUDGET = 20000       # Max physics ticks a single policy may consume (runaway loop guard)
DEMO_SEED = 7                    # Seed used for the ensemble perturbations in the dashboard comparison
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8000

# Local LLM Runtime Parameters
OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT_S = 120           # Whole-generation timeout for the dashboard (cold model load can take ~90 s)
