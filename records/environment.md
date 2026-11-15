# Environment actually used (verified 2026-11-15)

| Item | Value |
|---|---|
| Machine | cloud sandbox container (ephemeral) |
| CPU | 4 cores, Intel Xeon @ 2.30 GHz |
| RAM | 15 GB |
| GPU | **none** (`nvidia-smi` absent, `torch.cuda.is_available()` = False) |
| Python / PyTorch | 3.13 / 2.15.0 (CPU execution) |
| Disk | about 30 GB free |

## Consequences for experiment size
- D3: recurrent SNN with 128 LIF units, 100 time steps, 30 epochs = about 5 min per run on one core;
  runs are executed 3-4 in parallel, one thread each.
- D1: **ResNet-8** (He-style CIFAR ResNet with one block per stage, 75,290 parameters), not ResNet-20.
  ResNet-20 costs about 2 min/epoch on one core here; with 5 seeds, six allocators and four budgets it
  does not fit a 1-2 hour round. Float training is 15 epochs, so float accuracy is well below the
  published ResNet-20 numbers. All D1 comparisons with the literature are therefore indirect.

## Data
- SHD from `https://zenkelab.org/datasets/shd_{train,test}.h5.gz` (official host). The sandbox network
  policy blocked this host at first; the owner approved a personal cloud environment ("HNN") that allows it.
- CIFAR-10 from the fast.ai S3 mirror `https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz`
  (same 50k/10k images as the original, PNG format); the Toronto host is blocked.
- `python prepare_data.py` converts both into `.npy` arrays under `data/` (not committed).
- snnTorch / SpikingJelly were not needed; the SNN is plain PyTorch.
