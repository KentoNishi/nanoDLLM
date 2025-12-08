"""Central definitions for supported BlockGPT model variants."""

MODEL_VARIANTS = {
    "base": {},
    "500m": {
        "model_dim": 1280,
        "num_heads": 20,
        "num_layers": 20,
        "head_dim": 64,
    },
}
