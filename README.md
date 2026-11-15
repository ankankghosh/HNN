# HNN: heterogeneous neural networks

Premise: a few neurons/channels are far more sensitive than the rest, so they should get more
resources (precision, richer dynamics) under a fixed budget. This repository holds small, honest
pilot experiments for the two open directions, run on a CPU-only cloud sandbox, with every record
kept here.

- **D3 (main)**: spiking networks. Use a sensitivity score to decide which neurons get which
  membrane time constant under an energy budget. Benchmark: SHD.
- **D1 (bounded side study)**: interaction-aware per-channel bit allocation at <= 2.5 average bits.
  Benchmark: CIFAR-10.

**Start with [`records/FINAL_REPORT.md`](records/FINAL_REPORT.md)** (verdict, "ours vs published"
table, evidence). If it does not exist yet, the study is still running; see `records/` for what is done.

## Layout

| Path | Content |
|---|---|
| `records/environment.md` | hardware actually used and what it forced us to scale down |
| `records/phase1_baselines.md` | baselines and the two diagnostics, before any new allocator |
| `records/round_N/` | one folder per loop round: `ideas.md` (pre-registered tests), `background.md` (related-work check and vetoes), `experiments.md` (run log), `analysis.md` (results, ours vs published, feedback) |
| `records/FINAL_REPORT.md` | final verdict for D3 and D1 |
| `results/` | raw JSON results and logs for every run |
| `d3/` | spiking-network harness (`snn.py`), baselines, diagnostic, allocators |
| `d1/` | quantization harness (`quant.py`), baselines, diagnostic, allocators |
| `tools/` | statistics helpers and table generators |
| `prepare_data.py` | converts the downloaded SHD and CIFAR-10 files to arrays under `data/` (not committed) |

## Reproduce

```bash
pip install -r requirements.txt
```

Download `shd_train.h5.gz` and `shd_test.h5.gz` from `https://zenkelab.org/datasets/` into `data/shd/`
and gunzip them; download and extract `https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz` into
`data/cifar/`. Then:

```bash
python prepare_data.py
```

```bash
cd d3 && python tune.py && python run_baselines.py 0.02 0.005 4 && python diagnostic.py 0.02
```

```bash
cd d1 && for s in 0 1 2 3 4; do python run_baselines.py $s 1 1 15 && python diagnostic.py $s 1 1; done
```

```bash
python tools/summarize_phase1.py
```

## Ground rules followed

- Validation splits are carved from the training sets; test sets are never used for model selection.
- At least 5 paired seeds per comparison; mean +/- standard error and win counts are reported.
- Success tests are written down before results exist (`records/round_N/ideas.md`).
- Ties and failures are reported as results. A win over our own baselines is not presented as a
  win over the literature; every analysis has an "ours vs published" table with a "comparable?" column.
