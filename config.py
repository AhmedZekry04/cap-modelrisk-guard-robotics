"""
config.py - Single Source of Truth for ModelRisk-Guard Configuration
"""

# Risk Engine Parameters (Radic Webster Epistemic Formulation)
ENSEMBLE_SIZE_P = 4              # Number of parallel headless instances
HORIZON_STEPS_H = 60             # Lookahead steps per particle
RISK_THRESHOLD_TAU = 0.30        # Interception cutoff for combined risk
CRITICAL_FORCE_LIMIT_N = 20.0    # Structural damage threshold (Newtons)

# Physical Dimensions & Tolerances (Meters)
PLUG_INITIAL_X = 0.33            # Seated coordinate of transceiver plug
PLUG_DISENGAGE_THRESHOLD_X = 0.03 # Minimum extraction distance for electrical disconnection
WIRE_NOMINAL_X = 0.38            # Nominal X position of adjacent optical wire
WIRE_NOMINAL_Y = 0.03            # Nominal Y offset of adjacent optical wire
WIRE_CLEARANCE_LIMIT_X = 0.045   # Maximum safe pull before entering cable envelope

# Local LLM Runtime Parameters
OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
