"""Central definitions for supported BlockGPT model variants."""

MODEL_VARIANTS = {
    "base": {},
    "domain60m": {
        "model_dim": 512,
        "num_heads": 8,
        "num_layers": 8,
        "head_dim": 64,
    },
    "500m": {
        "model_dim": 1280,
        "num_heads": 20,
        "num_layers": 20,
        "head_dim": 64,
    },
}
