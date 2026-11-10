# Training and evaluation

**English** · [Italiano](../it/ADDESTRAMENTO.md)

This page explains how the released 1.1 and 1.5 models were trained and how to reproduce them or train a
variant. The [LightPFN 1.5 technical report](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf)
and the [report of 1.1](../../paper/LightPFN_report.pdf) give the reasons behind each choice.

- [Requirements](#requirements)
- [Synthetic priors](#synthetic-priors)
- [Generating a pool](#generating-a-pool)
- [Training a model](#training-a-model)
- [The 1.1 released recipe](#the-11-released-recipe)
- [The 1.5 released recipe](#the-15-released-recipe)
- [Evaluation](#evaluation)
- [Exporting weights for the package](#exporting-weights-for-the-package)

## Requirements

- Linux (the generator and the data stream use `fcntl` file locks).
- A CUDA or ROCm GPU with at least 16 GB for training; the 1.1 released run used two RTX 5090 and the 1.5 run
  used four RTX 5090.
- A checkout of this repository: the wheel on PyPI contains only the inference code.

```bash
git clone https://github.com/GioOtto/LightPFN.git && cd LightPFN
pip install -e ".[train,eval,dev]"
```

The `train` extra installs TabICL 2.2.0, whose BSD-3-Clause structural causal model code the graph prior reuses
(never its weights or predictions). Pools, checkpoints and evaluation data go to `data/` and `runs/`, which are
not tracked.

## Synthetic priors

| Prior | Version | Content |
|---|---|---|
| `graph` | 3 | structural causal models with random graphs and node functions (MLPs, trees, discretizations, Gaussian processes), as in TabICLv2, with an ExtraTrees filter that drops unlearnable tasks. 90% of the 1.1 released classification mix, and base graph of the 1.5 `r2` regression prior |
| `graph` | 4 | graph v3 with heavy tails, quantized high-cardinality categorical columns, more binary tasks and quantile-split multiclass targets. Used for evaluation (D1) and categorical experiments in 1.1, and for 90% of the 1.5 classification pretraining mix (at `p_binary=0.55`) |
| `rule` | 1 | graph v4 features labelled by an oracle over one to four drivers: XOR, parity, shallow trees, lookup tables, products, AND, OR. XOR and parity drivers are balanced so that no single feature predicts the label. 10% of the 1.1 and 1.5 classification mixes |
| `tree` | 4 | labels from fitted tree ensembles on graph features. Used for evaluation (D1) |
| `graph` (`r2`) | 3 | regression prior on graph v3 with `--reg-factors filter,noise,tails,discrete`: corrects learnability filtering, observation noise, heavy tails and discrete targets. Used to train the 1.5 regressor |

Task shapes come from presets: `s1b` 256 to 2,048 rows in groups of 8 (pretraining), `s2` 1,024 to 10,240 rows
in groups of 4, `s4` 4,096 to 60,000 rows in groups of 2 (long-context stage). `--geometry ratio` ties the
feature count to the rows; `--geometry cap` draws 2 to 100 features independently of the rows. Every task is
seeded by its coordinates, so a pool is byte-identical for any number of workers, and an interrupted generation
resumes where it stopped.

## Generating a pool

```bash
python -m lightpfn.prior.generate --preset s1b --prior graph --prior-version 3 --geometry ratio \
    --p-binary 0.55 --max-features 100 --seed 0 --n-tasks 256000 --n-jobs 16 --out data/prior/s1b_graph
python -m lightpfn.prior.generate --preset s1b --prior rule --prior-version 1 --geometry ratio \
    --p-binary 0.78 --max-features 100 --seed 20261011 --n-tasks 40000 --n-jobs 16 --out data/prior/s1b_rule_v1
```

These are the pools of the ablations in the 1.1 report. Generation time and storage depend on the geometry and
on the host; review any prior changes and run a small pilot before generating a large pool.

## Training a model

The architecture of the 1.1 released model (B4 in the 1.1 report, 4,603,088 parameters) is the default `Config`
plus four settings; the 1.5 classifier (4,609,472 parameters) adds `"cat_adapter":true`:

```bash
B4='{"row_refine":true,"row_refine_rounds":3,"row_mode":"summary","icl_drop_blocks":1}'
python -m lightpfn.train --run my_run --pools data/prior/s1b_graph:0.9 data/prior/s1b_rule_v1:0.1 \
    --steps 8000 --warmup 800 --max-cells 400000 --config "$B4" --no-lite
```

This is the 8,000-step ablation of the 1.1 report (C2). The trainer uses Muon for the block matrices and AdamW
elsewhere, a cosine schedule, bfloat16 autocast and an EMA of the weights, and writes
`runs/train/my_run/ema_stepXXXXXX.pt` (EMA weights) and `ckpt.pt` (everything needed to resume). Rerunning the
same command resumes from `ckpt.pt`. With several GPUs, launch it through
`python -m torch.distributed.run --nproc_per_node=N -m lightpfn.train ...`; each step then holds N times the
tasks.

## The 1.1 released recipe

<a id="the-released-recipe"></a>

**Pretraining** (`final` in the 1.1 report): 120,000 steps of 64 tasks on two GPUs, warmup 2,000, graph v3 90%
and rule v1 10% of fresh tasks streamed from CPU workers. The stream (`lightpfn.prior.stream`) generates chunks
of 7,500 steps ahead of the trainer and deletes each chunk once it has been read twice:

```bash
S=data/stream/final
python -m lightpfn.prior.stream init $PWD/$S --chunk-steps 7500 --chunks 16 --tasks-per-step 64 --passes 2 \
    --consumers final --prior v3=graph:3:0.55:0.9:3000000 --prior rule=rule:1:0.78:0.1:5000000
python -m lightpfn.prior.stream gen $PWD/$S --n-jobs 72 --ahead 3 &
python -m lightpfn.prior.stream train $PWD/$S --run final --gpu 0,1 --nproc 2 -- \
    --warmup 2000 --save-every 1000 --workers 4 --seed 0 --max-cells 800000 --config "$B4"
```

It took about 18 hours on two RTX 5090 with 72 CPU processes generating about 120 tasks per second.

**Long-context stage** (`final_long`, the 1.1 released weights): 39,250 more steps from the final checkpoint, on
tables of up to 60,000 rows, learning rate 1e-4 with warmup 200 and a cosine to 1e-5. Pools (the run used what
the generators had produced when each part of the stage started; the counts below are upper bounds):

```bash
G="python -m lightpfn.prior.generate --geometry cap --max-features 100"
$G --preset s2 --prior graph --prior-version 3 --p-binary 0.55 --seed 3000900 --n-tasks 345600 --out data/prior/s2cap_graph_v3
$G --preset s2 --prior rule --prior-version 1 --p-binary 0.78 --seed 5000900 --n-tasks 38400 --out data/prior/s2cap_rule_v1
$G --preset s4 --max-cells 800000 --prior graph --prior-version 3 --p-binary 0.55 --seed 3000902 --n-tasks 500000 --out data/prior/s4cap_graph_v3
$G --preset s4 --max-cells 800000 --prior rule --prior-version 1 --p-binary 0.78 --seed 5000902 --n-tasks 56000 --out data/prior/s4cap_rule_v1
```

Training starts from the final checkpoint (weights, EMA and optimizer state) with its step counter set to 0:

```bash
mkdir -p runs/train/final_long
python -c "import torch; c = torch.load('runs/train/final/ckpt.pt', weights_only=False); c['step'] = 0; torch.save(c, 'runs/train/final_long/ckpt.pt')"

python -m torch.distributed.run --standalone --nproc_per_node=2 -m lightpfn.train --run final_long \
    --pools data/prior/s1b_graph:0.135 data/prior/s1b_rule_v1:0.015 data/prior/s2cap_graph_v3:0.315 \
            data/prior/s2cap_rule_v1:0.035 data/prior/s4cap_graph_v3:0.45 data/prior/s4cap_rule_v1:0.05 \
    --sampling epoch --steps 39250 --warmup 200 --lr 1e-4 --groups-per-step 8 \
    --max-cells 1100000 --row-cost 8 --fit-rows --save-every 250 --workers 4 --seed 1 --config "$B4"
```

`--row-cost 8` counts every row as eight extra cells when micro-batches are formed (a measured memory model)
and `--fit-rows` subsamples the rows of a task that would exceed the budget. The 1.1 stage did not run in one
piece: a CUDA grid limit (fixed since) stopped long tables for the first 7,000 steps, and the stage was extended
twice. The 1.1 report (Section 9) describes the exact sequence.

## The 1.5 released recipe

Version 1.5 uses the 1.1 B4 trunk with the categorical adapter (`Config.cat_adapter=True`).
Classification (4,609,472 parameters) and regression (4,772,519 parameters) have separate weights, each below
five million parameters. No distillation or outputs from other tabular foundation models are used.

| Stage | Recipe |
|---|---|
| Main classification | 120,000 steps x 64 global tasks (7.68M task draws); `graph` v4 90%, `rule` v1 10%, both at `p_binary=0.55`; stream of 16 chunks x 7,500 steps, two passes; `--cat-mask-drop 0.25` |
| Main optimization | Muon + AdamW, bfloat16 autocast, EMA, warmup 2,000, peak LR 1e-3 with cosine decay to 1e-4, `--max-cells 800000` |
| Long-context classification | 15,000 steps from the main checkpoint, tables up to 60,000 rows; groups 15% `s1b` / 35% `s2cap` / 50% `s4cap`; warmup 200, LR 1e-4 with cosine decay to 1e-5, `--max-cells 1380000 --row-cost 10 --fit-rows` |
| Wide-table classification | 8,500 steps from the long-context checkpoint, same optimizer settings (warmup 200, LR 1e-4 with cosine decay to 1e-5); groups 8% `s1b` / 17% `s2cap` / 25% `s4cap` / 50% `s2w`, where `s2w` has 1,024 to 10,240 rows and 64 to 500 features (`--min-features 64 --max-features 500 --max-cells 1000000`) |
| Regression | Fine-tune the classification trunk on the `graph` v3 `r2` regression prior (`--reg-factors filter,noise,tails,discrete`, 2.5M streamed tasks), with a moment-retrieval decoder and a 999-level `quantiles` head (LR 1e-3, warmup 200); separate checkpoint |
| Long-context regression | 5,000 steps with a context stage up to 60,000 rows |

- **Equal class share (`p_binary=0.55`):** in 1.1, `rule` v1 used `p_binary=0.78`. Controlled 2,000-step
  comparisons showed that raising `p_binary` from `0.55` to `0.78` hurts multiclass accuracy, while `graph` v4
  at `p_binary=0.55` matches `graph` v3 on D1 and D3 and improves high-cardinality categorical tasks. Version
  1.5 therefore sets `p_binary=0.55` in both `graph` v4 and `rule` v1.
- **Categorical mask drop (`--cat-mask-drop 0.25`):** during pretraining, the categorical mask of 25% of tasks
  is dropped so that the model also handles unmasked integer-coded categories cleanly, while `cat="auto"`
  provides the mask at inference.
- **Regression prior (`r2`) and head:** auditing the plain `graph` v3 regression prior against real regression
  benchmarks showed 36% unlearnable tasks (versus 4% on real tasks) and too few heavy-tailed or discrete
  targets. The `r2` prior corrects learnability filtering, observation noise, heavy tails and discrete targets.
  The regression decoder retrieves context target moments in a single fused attention call and the `quantiles`
  head returns 999 sorted quantiles from `.001` to `.999`. Fine-tuning the pretrained classification trunk reaches
  lower CRPS than training the regressor from scratch at equal compute. The second-member `log(y)` / `-y` rule
  (`log_skew=2.0`) is an inference-time recipe and does not change training. Both released checkpoints use EMA
  weights.

## Evaluation

| Set | Content | Commands |
|---|---|---|
| D1 | 1,536 held-out synthetic tasks: graph v3, graph v4, tree and rule families, with a seed no training pool uses | `python -m lightpfn.eval.dev build`, then `python -m lightpfn.eval.dev evaluate --checkpoint CKPT --run NAME` |
| D2 | 25 probe configurations (XOR, parity, lookup tables, decoys...) with five seeds | run by `lightpfn.eval.dev evaluate` |
| D3 | 55 OpenML-CC18 classification datasets not in TabArena, at most 1,000 rows and 100 features, five-fold CV | `python -m lightpfn.eval.external build`, `run --checkpoint CKPT --name NAME`, `report` |
| scale | the 15 D3 datasets with at least 5,000 rows, training sets of 1,000 to 32,000 rows and all rows | `python -m lightpfn.eval.scale run --checkpoint CKPT --name NAME`, then `report --models NAME catboost` |
| D4 / Dcat | 13 OpenML datasets of 50,000 to 2.2 million rows outside TabArena and D3 (Dcat: the 6 tasks with high-cardinality categorical columns) | `python -m lightpfn.eval.large build`, `run --checkpoint CKPT --name NAME`, `report --models NAME catboost` |
| D3-reg / D3-reg-L | 23 OpenML-CTR23 regression tasks outside TabArena (up to 800 context rows, five-fold CV) and 15 larger CTR23 tasks (4,000 to 40,000 context rows, two splits) | `python -m lightpfn.eval.regression` |
| TabArena | the 38 classification and 13 regression tasks of TabArena v0.1, lite and full protocols (our harness, not the official pipeline) | `python -m lightpfn.eval.data`, then `python -m lightpfn.eval.harness --checkpoint CKPT --name NAME --n-estimators 4 --mode full` |

Model selection in both series used the development sets (D1, D2, D3, Dcat, D3-reg, D3-reg-L) with paired
bootstrap comparisons. TabArena is consulted only for finalists; local evaluations are distinct from the
official TabArena leaderboard. An earlier exploratory r2 evaluation in 1.1 is disclosed in the 1.1 report,
Section 7. Baselines (CatBoost, LightGBM, XGBoost, random forest) run through the same harness with default
settings: `python -m lightpfn.eval.external run --models catboost lightgbm xgboost rf`.

Use a different `--name` for each checkpoint and estimator configuration. The harness resumes successful splits
from the CSV named by `--name`; it does not check whether the checkpoint or configuration changed.

## Exporting weights for the package

```python
from lightpfn import load_model, save_model

# 1.5 classification and regression subfolders
cls_model = load_model("runs/train/cls_wide/ema_step008500.pt")  # EMA weights, weights_only loading
reg_model = load_model("runs/train/reg_long/ema_step005000.pt")
save_model(cls_model, "weights/classifier")                      # model.safetensors + config.json
save_model(reg_model, "weights/regressor")

# 1.1 checkpoint
v1_model = load_model("runs/train/final_long/ema_step039250.pt")
save_model(v1_model, "my_weights/")
```

`LightPFNClassifier(checkpoint="weights/classifier")` and `LightPFNRegressor(checkpoint="weights/regressor")`
(or `LightPFNClassifier(checkpoint="my_weights/")`) then load them directly.
