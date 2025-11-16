# nanoDLLM

Lightweight Block Diffusion language model with a minimal training loop and evaluation utilities.

## Layout
- `model.py`: Block Diffusion model definition.
- `train.py`: training script for the diffusion model (FlexAttention + Block Diffusion objective).
- `requirements.txt`: core dependencies.

## Quickstart
```bash
pip install -r requirements.txt
# train
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) train.py
```

## Notes
Block Diffusion masks tokens per block, applies block-aware attention, and trains with a masked cross-entropy objective weighted by noise level. See `model.py` for details.
