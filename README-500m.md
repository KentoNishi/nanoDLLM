## Pre-train finewebedu50B
```
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) \
  train.py --dataset_mode=fineweb \
  --model_variant=500m \
  --run_id fineweb-base-500m
```

## Fine-tune without guidance
```
torchrun --standalone --nproc_per_node=1 \
  train.py --dataset_mode=combined \
  --model_variant=500m \
  --pretrained_checkpoint logs/fineweb-base-500m/state_latest.pt \
  --run_id combined-ft-500m
```

## Fine-tune with guidance
```
torchrun --standalone --nproc_per_node=1 \
  train.py --dataset_mode=combined_guidance \
  --model_variant=500m \
  --pretrained_checkpoint logs/fineweb-base-500m/state_latest.pt \
  --run_id combined-guidance-ft-500m
```
