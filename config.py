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
WIRE_NOMINAL_X = 0.38            # Nominal X position of adjacent optical wire
WIRE_NOMINAL_Y = 0.03            # Nominal Y offset of adjacent optical wire
WIRE_CLEARANCE_LIMIT_X = 0.045   # Maximum safe pull before entering cable envelope

# Active LiDAR / Rangefinder Sensor Parameters
LIDAR_MIN_CLEARANCE_M = 0.012    # Trigger emergency halt if distance drops below 12mm
LIDAR_MAX_RANGE_M = 0.20         # Maximum detection horizon for gripper rangefinder
UNSCRIPTED_WIRE_SHIFT_Y = 0.015  # Unscripted disturbance: cable shifts 15mm closer to plug

# Local LLM Runtime Parameters
OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
