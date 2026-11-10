"""Pretrains LightPFN on the synthetic task pools.

Data: shards written by lightpfn.prior.generate, sampled from several pools with mixture weights
(e.g. graph SCM 0.73, fitted trees 0.27). Load-time augmentations per task: random feature order
and, for a fraction of tasks, missing values (MCAR, MAR or MNAR) since the pools contain none.
A step is `groups_per_step` groups; groups are split into micro-batches of at most `max_cells`
cells (tasks x rows x features) and gradients are accumulated.

Optimization: Muon on the 2D weights of the transformer blocks, AdamW on everything else, one
learning rate (Muon rescaled to match AdamW's update RMS), warmup + cosine, bf16 autocast,
gradient clipping, EMA of the weights. Every `eval_every` steps the EMA model runs the lite
TabArena harness and is ranked against the baselines in runs/eval/lite.

Usage (from the project root):
    python -m lightpfn.train --run r0 --pools data/prior/s1_graph:0.73 data/prior/s1_tree:0.27
"""

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from lightpfn.model.lightpfn import Config, LightPFN
from lightpfn.model.ccmm import ccmm_task_loss
from lightpfn.model.layers import Block
from lightpfn.model_v2.icl import SandwichBlock
from lightpfn.sklearn import stratified_subsample
from lightpfn.train_stability import make_optimizers, load_optimizer_states, optimizer_names, TrainingDiagnostic

ROOT = Path(__file__).resolve().parents[1]


class NanGuard:
    """Opt-in, read-only diagnostics. No hooks, anomaly mode, or per-micro GPU reads."""

    @classmethod
    def from_env(cls, *args, **kwargs):
        path = os.environ.get("LIGHTPFN_NAN_GUARD")
        return cls(path, *args, **kwargs) if path else None

    def __init__(self, path, model, ema, opts, args, cfg, rank, device):
        from collections import deque
        self.path, self.model, self.ema, self.opts = Path(path), model, ema, opts
        self.args, self.cfg, self.rank, self.device = args, cfg, rank, torch.device(device)
        self.trace = deque(maxlen=256)
        self.micro_index = 0

    def begin(self, step, fit_rng):
        self.step, self.fit_rng_state = step, copy.deepcopy(fit_rng.bit_generator.state)
        self.trace.clear()
        self.micro_index = 0

    def record(self, mb, group_index, task_offset):
        # Loader tensors are on CPU. Never retain X/y/codes or copy a GPU tensor here.
        self.trace.append(dict(micro_index=self.micro_index, group_index=group_index,
                               task_offset=task_offset, shape=tuple(mb["X"].shape), n_train=mb["n_train"],
                               d=mb["d"].clone(),
                               row_uid_edges=mb["row_uid"][:, [0, -1]].clone() if "row_uid" in mb else None,
                               cpu_rng=torch.get_rng_state(),
                               cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None))
        self.micro_index += 1

    def fail(self, reason, norm32=None, norm64=None):
        # Each bad local rank saves before exiting; after reduce only rank 0 saves.
        if reason == "before_allreduce" or self.rank == 0:
            try:
                self.capture(reason, norm32, norm64)
            except Exception as exc:
                raise FloatingPointError(f"non-finite gradients: capture failed at {self.path}: {exc}") from exc
        if reason == "after_allreduce" and torch.distributed.is_initialized():
            # Peers must not exit and let torchrun kill rank 0 during its atomic save.
            torch.distributed.barrier()
        raise FloatingPointError(f"non-finite gradients: {reason}, rank {self.rank}, step {self.step}; {self.path}")

    def capture(self, reason, norm32, norm64):
        grads = {n: p.grad.detach().cpu().clone() for n, p in self.model.named_parameters() if p.grad is not None}
        bad = [n for n, g in grads.items() if not bool(torch.isfinite(g).all())]
        self.path.mkdir(parents=True, exist_ok=True)
        target = self.path / f"failure_rank{self.rank}_step{self.step}.pt"
        tmp = target.with_suffix(".pt.tmp")
        torch.save(dict(step=self.step, rank=self.rank, reason=reason, nonfinite_params=bad,
                        norm32=norm32, norm64=norm64, grads=grads, model=self.model.state_dict(),
                        ema=self.ema.state_dict(), opts=[o.state_dict() for o in self.opts],
                        optimizer_param_names=optimizer_names(self.opts, self.model),
                        args=vars(self.args), config=asdict(self.cfg), micro_trace=list(self.trace),
                        micro_count=self.micro_index, fit_rng_state=self.fit_rng_state,
                        torch_version=str(torch.__version__), cuda=torch.version.cuda, hip=torch.version.hip,
                        device_name=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"), tmp)
        tmp.replace(target)

    def check_local(self, params):
        grads = [p.grad for p in params if p.grad is not None]
        # Foreach infinity norms fuse reads across tensors, without modifying gradients.
        # One scalar host read for the whole rank, not one per parameter/micro-batch.
        if grads and not bool(torch.isfinite(torch.stack(torch._foreach_norm(grads, float("inf")))).all()):
            self.fail("before_allreduce")

    def clip(self, params, max_norm):
        grads = [p.grad for p in params if p.grad is not None]
        norm = torch.nn.utils.get_total_norm(grads)
        if not bool(torch.isfinite(norm)):
            norm64 = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.double()) for g in grads]))
            self.fail("after_allreduce", float(norm), float(norm64))
        # Same two operations/defaults as clip_grad_norm_, with its norm reused.
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, norm)
        return norm


def inject_missing(rng, X, d, n_train=None):
    """In-place MCAR/MAR/MNAR, with at least two observed train cells per real column.

    MAR uses another column of the original unmasked table; for d=1 it falls back to MCAR.
    Mean rank is 1/2 even with ties, so the expected rate on selected columns is the sampled
    1%-50%. Restoring observations only affects tiny/very sparse training columns.
    """
    n, m = X.shape
    n_train = n if n_train is None else n_train
    if (not isinstance(d, (int, np.integer)) or not isinstance(n_train, (int, np.integer))
            or not 1 <= d <= m or not 1 <= n_train <= n or not np.issubdtype(X.dtype, np.floating)):
        raise ValueError("invalid missingness dimensions/dtype/train budget")
    original = X[:, :d].copy()
    if np.isinf(original).any() or np.any(np.isfinite(original[:n_train]).sum(0) == 0):
        raise ValueError("missingness requires an observed training value in every real column")
    cols = rng.choice(d, size=max(1, round(rng.uniform(0.1, 1.0) * d)), replace=False)
    mech = rng.choice(["mcar", "mar", "mnar"])
    for j in cols:
        rate = math.exp(rng.uniform(math.log(0.01), math.log(0.5)))
        if mech == "mcar" or (mech == "mar" and d == 1):
            probability = np.full(n, rate)
        else:
            # Mapping [0,d-1) around j excludes j without rejection sampling.
            k = j
            if mech == "mar":
                k = int(rng.integers(d - 1))
                k += k >= j
            src = original[:, k] * rng.choice([-1, 1])
            finite = np.isfinite(src)
            score = np.full(n, 0.5)
            if finite.sum() > 1:
                _, inverse, counts = np.unique(src[finite], return_inverse=True, return_counts=True)
                ranks = np.cumsum(counts) - (counts + 1) / 2
                score[finite] = ranks[inverse] / (finite.sum() - 1)
            probability = rate * 2 * score
        miss = rng.random(n) < probability
        observed = np.flatnonzero(np.isfinite(original[:n_train, j]))
        keep = min(2, len(observed))
        remaining = observed[~miss[observed]]
        if len(remaining) < keep:
            restore = rng.choice(observed[miss[observed]], keep - len(remaining), replace=False)
            miss[restore] = False
        X[miss, j] = np.nan


def _check_loaded_group(g):
    """Validate the training contract, including legacy pools, before augmentation.

    Training needs every class among the train rows. Test coverage of every class and
    non-constant columns are guaranteed by the generator (validate_group) for new pools,
    but not required here: v1 pools have rare classes missing from small test splits.
    """
    if "y_reg" in g:
        from lightpfn.prior.regression import validate_regression_group
        return validate_regression_group(g)
    X, y, d, nc = (g[k] for k in ("X", "y", "d", "n_classes"))
    if (not all(torch.is_tensor(v) for v in (X, y, d, nc)) or X.ndim != 3
            or not X.is_floating_point() or y.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64)
            or d.dtype not in (torch.int16, torch.int32, torch.int64)
            or nc.dtype not in (torch.int16, torch.int32, torch.int64)):
        raise ValueError("invalid group tensor types")
    B, n, m = X.shape
    ntr = g["n_train"]
    if (B < 1 or m < 1 or y.shape != (B, n) or d.shape != (B,) or nc.shape != (B,)
            or not isinstance(ntr, int) or not 1 <= ntr < n
            or not bool(torch.isfinite(X).all()) or float(X.abs().max()) > 1000):
        raise ValueError("invalid group shape/train budget/feature range")
    for i in range(B):
        C, di = int(nc[i]), int(d[i])
        if (not 1 <= di <= m or not 2 <= C <= 10 or ntr < C
                or not torch.equal(torch.unique(y[i]).long(), torch.arange(C))
                or not torch.equal(torch.unique(y[i, :ntr]).long(), torch.arange(C))
                or bool((X[i, :, di:] != 0).any())):
            raise ValueError("invalid class coverage/width/padding")


EPOCH_TAG = 0x45504F43  # "EPOC": keeps the shared epoch permutations apart from the worker streams


class PoolStream(IterableDataset):
    """Groups from shard pools mixed by weight. sampling="replace" draws each shard at random (the
    pool may grow while training reads it); "epoch" reads every shard once per epoch, from one
    permutation shared by the workers of every rank, each (rank, worker) taking its own slice
    (fixed shard list). With world > 1 (data parallel) the ranks share the seed and differ in rank."""

    def __init__(self, pools, p_missing=0.3, seed=0, augment="none", sampling="replace", rank=0, world=1,
                 typed=False, cat_mask_drop=0.0):
        super().__init__()
        if (not pools or not np.isfinite(p_missing) or not 0 <= p_missing <= 1
                or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= np.iinfo(np.uint32).max
                or any(not np.isfinite(w) or w < 0 for _, w in pools)
                or sum(w for _, w in pools) <= 0 or augment not in ("none", "v2")
                or sampling not in ("replace", "epoch")
                or not isinstance(world, int) or not isinstance(rank, int) or not 0 <= rank < world):
            raise ValueError("invalid pools/weights/missingness/seed/sampling/rank")
        self.pools = [(Path(p), float(w)) for p, w in pools if w > 0]
        self.p_missing = p_missing
        self.seed = seed
        self.augmentation = augment
        self.sampling = sampling
        self.rank, self.world = rank, world
        # typed: groups with a categorical mask also yield level IDs ("codes", -1 = missing; see typed_codes).
        self.typed = typed
        # cat_mask_drop (legacy view, v1 trunk): probability that a task's categorical mask is cleared, so the
        # model also learns tables whose categorical columns arrive undeclared (the "num view" of the lookup
        # probe, -1.7pp for V15 on 2026-11-04). 0 draws nothing: the stream is unchanged.
        if not (np.isfinite(cat_mask_drop) and 0 <= cat_mask_drop <= 1) or (cat_mask_drop and typed):
            raise ValueError("cat_mask_drop must be in [0, 1] and needs the legacy view")
        self.cat_mask_drop = cat_mask_drop

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, n_workers = (0, 1) if info is None else (info.id, info.num_workers)
        # No wall-clock seed: fixed pool, seed and worker count give a repeatable stream.
        rng = np.random.default_rng([self.seed, wid] if self.world == 1 else [self.seed, wid, self.rank])
        slot, n_slots = self.rank * n_workers + wid, self.world * n_workers
        weights = np.array([w for _, w in self.pools], dtype=float)
        weights /= weights.max()  # avoid overflow for otherwise valid weights near 1e308
        weights /= weights.sum()
        shards, last_scan = None, 0.0
        pending = [[] for _ in self.pools]
        epochs, queues = [0] * len(self.pools), [[] for _ in self.pools]
        while True:
            now = time.monotonic()
            if shards is None or (self.sampling == "replace" and now - last_scan > 300):
                shards = [sorted(p.glob("shard_*.pt")) for p, _ in self.pools]
                for (path, _), files in zip(self.pools, shards):
                    if not files:
                        raise FileNotFoundError(f"pool contains no committed shards: {path}")
                last_scan = now
            # Weights apply to groups even when the pools/shards contain different group counts.
            k = int(rng.choice(len(self.pools), p=weights))
            if not pending[k]:
                if self.sampling == "epoch":
                    if not queues[k]:
                        order = np.random.default_rng([self.seed, k, epochs[k], EPOCH_TAG]).permutation(len(shards[k]))
                        queues[k] = list(order[slot::n_slots]) if len(order) >= n_slots else [order[slot % len(order)]]
                        epochs[k] += 1
                    path = shards[k][queues[k].pop(0)]
                else:
                    path = shards[k][rng.integers(len(shards[k]))]
                try:
                    groups = torch.load(path, map_location="cpu", weights_only=True)
                    if not isinstance(groups, list) or not groups:
                        raise ValueError("empty/non-list shard")
                    if len({"y_reg" in g for g in groups}) != 1:
                        raise ValueError("mixed classification/regression shard")
                    for g in groups:
                        _check_loaded_group(g)
                except Exception as exc:
                    raise ValueError(f"cannot read valid shard {path}: {exc}") from exc
                pending[k] = [groups[i] for i in rng.permutation(len(groups))]
            out = self.augment(rng, pending[k].pop())
            if self.cat_mask_drop and "cat_mask" in out:
                drop = torch.from_numpy(rng.random(out["cat_mask"].shape[0]) < self.cat_mask_drop)
                out["cat_mask"] = out["cat_mask"] & ~drop[:, None]
            yield out

    def augment(self, rng, g):
        if "y_reg" in g:
            # Both views use the typed augmentation: the v1 trunk reads X and cat_mask (the categorical values
            # of the regression pools are level identities in X), v2 the level IDs too.
            if self.augmentation != "none":
                raise ValueError("regression requires augment=none")
            return self.augment_typed(rng, g)
        if self.typed and "cat_mask" in g:
            if self.augmentation != "none":
                raise ValueError("the typed view supports no augmentation")
            return self.augment_typed(rng, g)
        X = g["X"].float().numpy().copy()
        d = g["d"].numpy().copy()
        cat = g["cat_mask"].numpy().copy() if "cat_mask" in g else None
        if cat is None and self.augmentation == "v2":
            cat = np.zeros((X.shape[0], X.shape[2]), dtype=bool)
            for b, db in enumerate(d):
                cat[b, :int(db)] = [len(np.unique(X[b, :, j])) <= 10 for j in range(int(db))]
        for b in range(X.shape[0]):
            db = int(d[b])
            order = rng.permutation(db)
            X[b, :, :db] = X[b][:, order]
            if cat is not None:
                cat[b, :db] = cat[b, order]
            if self.augmentation == "v2":
                mask = cat[b] if cat is not None else np.r_[
                    [len(np.unique(X[b, :, j])) <= 10 for j in range(db)],
                    np.zeros(X.shape[2]-db, dtype=bool)]
                db = augment_v2(rng, X[b], db, int(g["n_train"]), mask)
                d[b] = db
            if rng.random() < self.p_missing:
                inject_missing(rng, X[b], db, int(g["n_train"]))
        out = dict(X=torch.from_numpy(X), y=g["y"].long(), d=torch.from_numpy(d).long(),
                   n_classes=g["n_classes"].long(), n_train=int(g["n_train"]))
        if cat is not None:
            out["cat_mask"] = torch.from_numpy(cat)
        return out

    def augment_typed(self, rng, g):
        """Typed view: the same column permutation for values, mask and level IDs; missing values (NaN, ID -1)
        injected as in the legacy view, with nominal columns ranked by a random score per level, not by ID."""
        X = g["X"].float().numpy().copy()
        cat = g["cat_mask"].numpy().copy()
        codes = typed_codes(g)
        d = g["d"].numpy()
        for b in range(X.shape[0]):
            db = int(d[b])
            order = rng.permutation(db)
            X[b, :, :db], cat[b, :db], codes[b, :, :db] = X[b][:, order], cat[b, order], codes[b][:, order]
            if rng.random() < self.p_missing:
                view = X[b].copy()
                for j in np.flatnonzero(cat[b, :db]):
                    _, inverse = np.unique(codes[b, :, j], return_inverse=True)
                    view[:, j] = rng.random(inverse.max() + 1)[inverse]
                inject_missing(rng, view, db, int(g["n_train"]))
                miss = np.isnan(view)
                X[b][miss] = np.nan
                codes[b][miss & cat[b]] = -1
        if "y_reg" in g:
            return dict(X=torch.from_numpy(X), y_reg=g["y_reg"], d=torch.from_numpy(d.copy()).long(),
                        n_train=int(g["n_train"]), row_uid=g["row_uid"],
                        cat_mask=torch.from_numpy(cat), codes=torch.from_numpy(codes))
        return dict(X=torch.from_numpy(X), y=g["y"].long(), d=torch.from_numpy(d.copy()).long(),
                    n_classes=g["n_classes"].long(), n_train=int(g["n_train"]),
                    cat_mask=torch.from_numpy(cat), codes=torch.from_numpy(codes))


def typed_codes(g):
    """(G, n, m) int32 level IDs of the categorical columns (0 elsewhere): the stored cat_codes, or, for pools
    whose categorical values are exact level identities in X (graph v4, rule), the rank of each distinct value."""
    G, n, m = g["X"].shape
    cat = g["cat_mask"].numpy()
    codes = np.zeros((G, n, m), dtype=np.int32)
    for b in range(G):
        cols = np.flatnonzero(cat[b])
        if "cat_codes" in g:
            codes[b][:, cols] = g["cat_codes"][b, :, :len(cols)].numpy()
        else:
            for j in cols:
                codes[b, :, j] = np.unique(g["X"][b, :, j].numpy(), return_inverse=True)[1]
    return codes


def augment_v2(rng, X, d, n_train, cat):
    """30% monotone tasks, 15% quantized tasks, 3% decoys (at most two).

    Parameters/cuts are fitted on train rows. Work in float64, cap the output
    to the pool's finite range, append only into padding. No source mutation.
    """
    if (X.ndim != 2 or not np.issubdtype(X.dtype, np.floating)
            or not isinstance(d, (int, np.integer)) or not isinstance(n_train, (int, np.integer))
            or not 1 <= d <= X.shape[1] or not 1 <= n_train < len(X)
            or cat.shape != (X.shape[1],) or cat.dtype != np.bool_
            or not np.isfinite(X).all() or np.any(np.abs(X) > 1000)):
        raise ValueError("invalid v2 input")
    numeric = np.flatnonzero(~cat[:d])
    if rng.random() < .30 and len(numeric):
        cols = rng.choice(numeric, max(1, round(.3*len(numeric))), replace=False)
        for j in cols:
            a = X[:, j].astype(np.float64)
            z = (a - np.median(a[:n_train])) / max(a[:n_train].std(), 1e-6)
            z = np.clip(z, -8, 8)
            kind = int(rng.integers(3))
            if kind == 0:
                v = np.sign(z) * np.log1p(np.abs(z))
            elif kind == 1:
                v = np.exp(z/3)
            else:
                v = np.sign(z) * np.abs(z)**rng.uniform(.5, 2.0)
            X[:, j] = np.clip((v-v[:n_train].mean())/max(v[:n_train].std(), 1e-6), -1000, 1000)
    if rng.random() < .15 and len(numeric) and n_train >= 2:
        cols = rng.choice(numeric, max(1, round(.1*len(numeric))), replace=False)
        for j in cols:
            k = int(rng.integers(3, min(64, max(3, n_train))+1))
            cuts = np.unique(np.quantile(X[:n_train, j], np.arange(1, k)/k))
            if not len(cuts) or X[:, j].min() == X[:, j].max():
                continue
            codes = rng.permutation(len(cuts)+1)
            if len(codes) > 2 and (np.all(np.diff(codes) > 0) or np.all(np.diff(codes) < 0)):
                codes[[0, 1]] = codes[[1, 0]]
            X[:, j] = codes[np.searchsorted(cuts, X[:, j], side="left")]
            cat[j] = True
    if rng.random() < .03 and d < X.shape[1]:
        count = min(X.shape[1]-d, int(rng.integers(1, 3)))
        sources = np.flatnonzero(X[:, :d].min(0) != X[:, :d].max(0))
        for _ in range(count):
            if not len(sources):
                break
            source = int(rng.choice(sources))
            a = X[:, source].astype(np.float64)
            scale = max(a[:n_train].std(), 1e-6)
            X[:, d] = np.clip(a + rng.normal(0, .05*scale, len(a)), -1000, 1000)
            cat[d] = False
            d += 1
    X[:, d:] = 0
    cat[d:] = False
    return d


def fit_rows(g, n_rows, rng):
    """The group with n_rows rows per task: the same train fraction, stratified train rows and random test
    rows, drawn per task."""
    if "y_reg" in g:
        return fit_rows_regression(g, n_rows, rng)
    X, y = g["X"], g["y"]
    G, n, _ = X.shape
    ntr = int(g["n_train"])
    ntr_new = min(max(1, round(ntr * n_rows / n)), ntr, n_rows - 1)
    nte_new = min(n_rows - ntr_new, n - ntr)
    rows = []
    for t in range(G):
        tr = np.arange(ntr) if ntr_new == ntr else stratified_subsample(rng, y[t, :ntr].numpy(), ntr_new)
        rows.append(torch.from_numpy(np.concatenate([tr, ntr + np.sort(rng.choice(n - ntr, nte_new, replace=False))])))
    out = dict(g, X=torch.stack([X[t, r] for t, r in enumerate(rows)]),
               y=torch.stack([y[t, r] for t, r in enumerate(rows)]), n_train=ntr_new)
    if "codes" in g:  # typed view: level IDs follow their rows
        out["codes"] = torch.stack([g["codes"][t, r] for t, r in enumerate(rows)])
    return out


def fit_rows_regression(g, n_rows, rng):
    """Subsample context and query separately, without inspecting target values."""
    B, n, _ = g["X"].shape
    ntr = g["n_train"]
    nt = min(max(1, round(ntr * n_rows / n)), ntr, n_rows - 1)
    nq = min(n_rows - nt, n - ntr)
    selected = [torch.from_numpy(np.r_[np.sort(rng.choice(ntr, nt, replace=False)),
                      ntr + np.sort(rng.choice(n - ntr, nq, replace=False))]) for _ in range(B)]
    out = dict(g, n_train=nt)
    for key in ("X", "y_reg", "codes", "row_uid"):
        if key in g:
            out[key] = torch.stack([g[key][i, r] for i, r in enumerate(selected)])
    return out


def micro_batches(g, max_cells, row_cost=0, rng=None):
    """Split only the task axis into micro-batches of at most max_cells cost, a task of n rows and m features
    costing n * (m + row_cost) (row_cost 0: its cells). An indivisible task may exceed the soft budget, unless
    rng is given: then the rows of the group's tasks are subsampled to fit (fit_rows)."""
    if (not isinstance(max_cells, (int, np.integer)) or max_cells < 1
            or g["X"].ndim != 3 or min(g["X"].shape) < 1):
        raise ValueError("max_cells and group dimensions must be positive")
    G, n, _ = g["X"].shape
    m = int(g["d"].max())  # real features: the padding columns are dropped below, so they cost nothing
    if rng is not None and n * (m + row_cost) > max_cells:
        n = max(2, int(max_cells // (m + row_cost)))
        g = fit_rows(g, n, rng)
    per = max(1, int(max_cells // (n * (m + row_cost))))
    for i in range(0, G, per):
        mb = {k: (v[i:i + per] if torch.is_tensor(v) and v.ndim else v) for k, v in g.items()}
        mb["X"] = mb["X"][:, :, :int(mb["d"].max())]
        if "cat_mask" in mb:
            mb["cat_mask"] = mb["cat_mask"][:, :mb["X"].shape[-1]]
        if "codes" in mb:
            mb["codes"] = mb["codes"][:, :, :mb["X"].shape[-1]]
        yield mb


def task_loss_regression(model, mb, device):
    """Per-task pinball loss of the v1 trunk with the quantiles head (Config.head="quantiles"), in the units
    of the targets normalized with the context rows (lightpfn/model/regression.py)."""
    from lightpfn.model.regression import pinball_loss

    if getattr(model, "reg", None) is None:
        raise ValueError("regression pools need --config with head=quantiles (or --arch v2)")
    m_max = int(mb["d"].max())
    has_padding = bool((mb["d"] < m_max).any())
    ntr = mb["n_train"]
    X = mb["X"][:, :, :m_max].to(device, non_blocking=True)
    y = mb["y_reg"].to(device, non_blocking=True)
    d = mb["d"].to(device, non_blocking=True)
    cat = None
    if "cat_mask" in mb and getattr(model.cfg, "cat_adapter", False) and bool(mb["cat_mask"].any()):
        cat = mb["cat_mask"][:, :m_max].to(device, non_blocking=True)
    ctx = model.encode(X[:, :ntr], y[:, :ntr], d=d, has_padding=has_padding, cat=cat, regression=True)
    q = model.predict_quantiles(ctx, X[:, ntr:], normalized=True)
    target = (y[:, ntr:].double() - ctx.extra["mean"][:, None]) / ctx.extra["scale"][:, None]
    return pinball_loss(q, target, reduction="task")


def task_loss(model, mb, device, label_slots, ccmm_weight=0.0):
    if "y_reg" in mb:
        if ccmm_weight:
            raise ValueError("CCMM loss does not support regression")
        return task_loss_regression(model, mb, device)
    if ccmm_weight:
        if getattr(model.cfg, "cat_adapter", False):
            raise ValueError("CCMM loss does not support the categorical adapter")
        return ccmm_task_loss(model, mb, device, label_slots, ccmm_weight)
    # PoolStream/micro_batches keep metadata on CPU. Read it before H2D: scalar
    # conversion after the copy would synchronize every microbatch with the GPU.
    C = int(mb["n_classes"].max())
    m_max = int(mb["d"].max())
    has_padding = bool((mb["d"] < m_max).any())
    X = mb["X"].to(device, non_blocking=True)
    y = mb["y"].to(device, non_blocking=True)
    d = mb["d"].to(device, non_blocking=True)
    nc = mb["n_classes"].to(device, non_blocking=True)
    ntr = mb["n_train"]
    B = X.shape[0]
    slots = torch.argsort(torch.rand(B, label_slots, device=device), dim=1)
    cat = None
    if "cat_mask" in mb and getattr(model.cfg, "cat_adapter", False) and bool(mb["cat_mask"].any()):
        cat = mb["cat_mask"][:, :m_max].to(device, non_blocking=True)
    logits = model(X[:, :, :m_max], y[:, :ntr], d=d, slots=slots,
                   n_classes=C, has_padding=has_padding, cat=cat).float()  # (B, M, C)
    logits = logits.masked_fill(torch.arange(C, device=device)[None, None] >= nc[:, None, None], -1e4)
    ce = F.cross_entropy(logits.transpose(1, 2), y[:, ntr:], reduction="none")  # (B, M)
    return ce.mean(1)  # per task


def task_loss_v2(model, mb, device, attn_zloss=0.0, stats=None):
    """Per-task CE or normalized pinball loss on a pool micro-batch: tasks with fewer features are
    zero-padded (Table.n_columns), as in task_loss. With the typed view (mb["codes"]) the categorical
    columns enter as level IDs; otherwise every column is numerical. attn_zloss > 0 adds to each
    classification task attn_zloss * mean over heads and query rows of logsumexp(retrieval attention
    logits)^2 and puts that detached per-task term in stats["attn_zloss"]; 0 runs the plain loss."""
    from lightpfn.model_v2 import Table

    # Metadata stays on CPU (see task_loss): read it before the copies, without host syncs.
    ntr, d = mb["n_train"], mb["d"]
    regression = "y_reg" in mb
    C = 2 if regression else max(int(mb["n_classes"].max()), 2)
    m = int(d.max())
    has_padding = bool((d < m).any())
    X = mb["X"][:, :, :m].to(device, non_blocking=True)
    y = mb["y_reg" if regression else "y"].to(device, non_blocking=True)
    b, n = X.shape[:2]
    missing = torch.isnan(X)
    rows = torch.arange(n, device=device)[None].expand(b, -1)
    if regression:
        rows = mb["row_uid"].to(device, non_blocking=True)
    cols = torch.arange(m, device=device)[None].expand(b, -1)
    if "codes" in mb:
        codes = mb["codes"][:, :, :m].to(device, non_blocking=True)
        cat = mb["cat_mask"][:, :m].to(device, non_blocking=True)
        missing = missing | (cat[:, None] & (codes < 0))
    else:
        codes = torch.zeros(X.shape, dtype=torch.int32, device=device)
        cat = torch.zeros((b, m), dtype=torch.bool, device=device)
    width = d.to(device, torch.int64, non_blocking=True) if has_padding else None
    table = Table(X, codes, cat, missing, rows, cols, width)
    if regression:
        from lightpfn.model_v2.heads import pinball_loss
        from lightpfn.model_v2.regression import normalize_targets
        if "y" in mb:
            raise ValueError("mixed task micro-batch")
        # Fit float64 moments on context only, before low precision arithmetic.
        z, mean, scale = normalize_targets(y, ntr)
        quantiles = model(table.rows(slice(0, ntr)), y[:, :ntr], table.rows(slice(ntr, n)), regression=True)
        normalized = (quantiles.double() - mean[:, None, None]) / scale[:, None, None]
        return pinball_loss(normalized, z[:, ntr:], reduction="task")
    if attn_zloss:
        model.retrieval.zloss = True
        try:
            logp = model(table.rows(slice(0, ntr)), y[:, :ntr], table.rows(slice(ntr, n)), n_classes=C)
            terms = attn_zloss * model.retrieval.zloss_terms
        finally:
            model.retrieval.zloss, model.retrieval.zloss_terms = False, None
        if stats is not None:
            stats["attn_zloss"] = terms.detach()
    else:
        logp = model(table.rows(slice(0, ntr)), y[:, :ntr], table.rows(slice(ntr, n)), n_classes=C)
    diagnostic = getattr(model.retrieval, "diagnostic", None)
    if diagnostic is not None:
        diagnostic.targets(y[:, ntr:])
    ce = F.nll_loss(logp.float().transpose(1, 2), y[:, ntr:], reduction="none").mean(1)
    return ce + terms if attn_zloss else ce


def param_groups(model):
    """Muon and AdamW parameters among the trainable ones (frozen parameters are in neither)."""
    muon, adamw = [], []
    muon_ids = {id(p) for blk in model.modules() if isinstance(blk, (Block, SandwichBlock))
                for name, p in blk.named_parameters() if p.ndim == 2 and "scaling" not in name}
    for p in model.parameters():
        if p.requires_grad:
            (muon if id(p) in muon_ids else adamw).append(p)
    return muon, adamw


def init_weights(model, path):
    """Load the weights of a plain checkpoint (safetensors directory/file or weights-only .pt) into model;
    a categorical adapter missing from it starts at the identity."""
    from lightpfn.checkpoint import load_model

    src = load_model(path, "cpu")
    missing, unexpected = model.load_state_dict(src.state_dict(), strict=False)
    # A classification checkpoint into a model with the quantiles head: the regression modules keep their
    # initialization (warm start).
    if unexpected or any(not k.startswith(("cells.cat_", "cat_stats.", "reg.")) for k in missing):
        raise ValueError(f"--init checkpoint does not match the model: missing {missing}, unexpected {unexpected}")
    if any(k.startswith(("cells.cat_", "cat_stats.")) for k in missing):
        model.init_cat_adapter()


def freeze_for(model, trainable):
    """trainable: "all", or "adapter" (only the categorical adapter learns; the rest stays fixed)."""
    if trainable == "all":
        return
    keep = {id(p) for p in model.adapter_parameters()}
    if not keep:
        raise ValueError("--trainable adapter needs a model with cat_adapter")
    for p in model.parameters():
        p.requires_grad_(id(p) in keep)


@torch.no_grad()
def allreduce_grads(params):
    """Data parallel without the DDP wrapper (the ranks run different numbers of micro-batches):
    sum every rank's gradients once per step. A parameter unused on every rank keeps grad None, as
    on one GPU (the optimizers skip it); one used on some rank gets zeros where it was unused."""
    import torch.distributed as dist
    used = torch.tensor([p.grad is not None for p in params], dtype=torch.uint8, device=params[0].device)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    live = [p for p, u in zip(params, used.tolist()) if u]
    if not live:
        return
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in live]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    for p, g in zip(live, torch._utils._unflatten_dense_tensors(flat, grads)):
        p.grad = g


@torch.no_grad()
def sync_replicas(modules):
    """Copy rank 0's parameters and buffers to every rank (start, resume, or after a drift check)."""
    import torch.distributed as dist
    for m in modules:
        for t in m.state_dict().values():
            dist.broadcast(t, 0)


@torch.no_grad()
def replicas_differ(model):
    import torch.distributed as dist
    flat = torch.cat([p.reshape(-1) for p in model.parameters()]).double()
    weights = (torch.arange(len(flat), device=flat.device) % 7 + 1).double()  # position-sensitive
    sig = torch.stack([flat.sum(), flat.abs().sum(), (flat * weights).sum()])
    hi, lo = sig.clone(), sig.clone()
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    return bool((hi != lo).any())


@torch.no_grad()
def ema_update(ema, model, decay):
    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)


def configure_acceleration(model, compile_region="none", tunableop_results=None, compile_workers=1):
    """Opt-in experiments; Module.compile preserves parameter/state_dict names."""
    if tunableop_results is not None:
        if not torch.version.hip:
            raise ValueError("TunableOp requires ROCm")
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        if not torch.cuda.tunable.read_file(str(tunableop_results)):
            raise ValueError(f"cannot load TunableOp results: {tunableop_results}")
    if compile_region != "none":
        import torch._inductor.config as inductor_config
        inductor_config.compile_threads = compile_workers
        for module in model.modules():
            if isinstance(module, Block):
                target = module.mlp if compile_region == "mlp" else module
                target.compile(dynamic=True)


def lite_eval(model, device, out_csv, arch="v1", categorical=False):
    from lightpfn.eval.harness import evaluate, make_jobs
    from lightpfn.eval.report import load_results, per_task, summarize, task_groups
    from lightpfn.sklearn import LightPFNClassifier

    if arch == "v2":
        from lightpfn.model_v2.estimator import V2Classifier
        factory = lambda n_threads, seed: V2Classifier(model, device=device, seed=seed,  # noqa: E731
                                                       categorical=categorical)
    else:
        factory = lambda n_threads, seed: LightPFNClassifier(model=model, device=device, seed=seed)  # noqa: E731
    df = evaluate(factory, make_jobs("lite"), n_threads=4, n_workers=1, verbose=False)
    df.to_csv(out_csv, index=False)
    base = load_results("lite", ["rf", "lightgbm", "xgboost", "catboost", "tabicl_gpu"])
    import pandas as pd

    t = per_task(pd.concat([base, df.assign(model="lightpfn")], ignore_index=True).fillna({"error": ""}))
    s, g_rank, _, _, n = summarize(t, task_groups())
    return s, g_rank


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--pools", nargs="+", required=True, help="dir:weight")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--stop-at", type=int, default=None,
                   help="stop at this step with a checkpoint; the schedule still runs to --steps (chunked streams)")
    p.add_argument("--sampling", choices=["replace", "epoch"], default="replace",
                   help="shards drawn with replacement, or each once per epoch (see PoolStream)")
    p.add_argument("--groups-per-step", type=int, default=4)
    p.add_argument("--max-cells", type=int, default=400_000, help="cells per micro-batch (soft budget)")
    p.add_argument("--row-cost", type=float, default=0.0,
                   help="cost of a row in cells, added to its features when splitting by --max-cells")
    p.add_argument("--fit-rows", action="store_true",
                   help="subsample the rows of a task over --max-cells instead of exceeding the budget")
    p.add_argument("--compile", choices=["none", "mlp", "block"], default="none")
    p.add_argument("--compile-workers", type=int, default=1)
    p.add_argument("--tunableop-results", type=Path, default=None, help="offline ROCm results; no online tuning")
    p.add_argument("--memory-fraction", type=float, default=0.8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=None, help="override runs/train/<run> for temporary runs")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--decoder-scaling-lr-factor", type=float, default=None,
                   help="opt-in decoder SoftmaxScaling AdamW group: LR multiplier (0,1), no decay; "
                        "1 preserves legacy groups; on resume omitted = saved setting or 1")
    p.add_argument("--decoder-attn-zloss", type=float, default=None,
                   help="v2 classification only: opt-in z-loss coefficient on the retrieval decoder's "
                        "attention logits, coef * mean logsumexp^2; 0 = plain loss; on resume omitted = "
                        "saved setting or 0")
    p.add_argument("--diagnostics-every", type=int, default=0,
                   help="v2 only: detached per-module gradients, scaling norms and attention maxima; "
                        "0 disables; otherwise a multiple of --log-every")
    p.add_argument("--p-missing", type=float, default=0.3)
    p.add_argument("--augment", choices=["none", "v2"], default="none")
    p.add_argument("--cat-mask-drop", type=float, default=None,
                   help="v1 legacy view: probability of clearing a task's categorical mask; "
                        "on resume omitted = saved setting, otherwise 0")
    p.add_argument("--view", choices=["legacy", "typed"], default=None,
                   help="typed: categorical columns of pools with a mask enter as level IDs (default for --arch v2; "
                        "legacy: every column as a number, the v1 view)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--arch", choices=["v1", "v2"], default="v1", help="v2: experimental C2v2 (lightpfn.model_v2)")
    p.add_argument("--config", type=str, default="{}", help="JSON overrides of model Config")
    p.add_argument("--ccmm-weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", type=Path, default=None,
                   help="start from these weights (fine-tuning; ignored when the run resumes from its ckpt.pt)")
    p.add_argument("--trainable", choices=["all", "adapter"], default="all",
                   help="adapter: train only the categorical adapter, every other weight frozen")
    p.add_argument("--no-lite", action="store_true", help="skip the lite TabArena eval (EMA checkpoints are still saved)")
    args = p.parse_args()
    if (args.diagnostics_every < 0 or (args.diagnostics_every and
            (args.arch != "v2" or args.diagnostics_every % args.log_every))):
        p.error("--diagnostics-every requires v2 and a positive multiple of --log-every")
    if args.view is None:
        args.view = "typed" if args.arch == "v2" else "legacy"
    if args.view == "typed" and args.arch != "v2":
        p.error("--view typed needs --arch v2")
    if args.cat_mask_drop is not None and (not math.isfinite(args.cat_mask_drop)
            or not 0 <= args.cat_mask_drop <= 1 or (args.cat_mask_drop and args.arch != "v1")):
        p.error("--cat-mask-drop needs --arch v1 and a probability in [0, 1]")
    if args.stop_at is not None and not 0 < args.stop_at <= args.steps:
        p.error("--stop-at must be in 1..--steps")
    end = args.steps if args.stop_at is None else args.stop_at
    # Data parallel when launched by torchrun: every rank draws --groups-per-step groups, so a step
    # holds world x groups-per-step groups. Rank 0 alone logs, saves and evaluates.
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    main_rank = rank == 0
    if world > 1:
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    run_dir = args.output_dir or ROOT / "runs" / "train" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt.pt"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False) if ckpt_path.exists() else None
    if args.cat_mask_drop is None:
        # args.json also covers checkpoints made before the setting was stored in ckpt.pt.
        saved_args = json.loads((run_dir / "args.json").read_text()) if ck and (run_dir / "args.json").exists() else {}
        args.cat_mask_drop = (ck or {}).get("cat_mask_drop", saved_args.get("cat_mask_drop", 0.0))
    if (not math.isfinite(args.cat_mask_drop) or not 0 <= args.cat_mask_drop <= 1
            or (args.cat_mask_drop and args.arch != "v1")):
        p.error("--cat-mask-drop needs --arch v1 and a probability in [0, 1]")
    if not math.isfinite(args.ccmm_weight) or args.ccmm_weight < 0:
        p.error("--ccmm-weight must be finite and nonnegative")
    if args.decoder_attn_zloss is None:
        args.decoder_attn_zloss = float((ck or {}).get("decoder_attn_zloss", 0.0))
    if not math.isfinite(args.decoder_attn_zloss) or args.decoder_attn_zloss < 0:
        p.error("--decoder-attn-zloss must be finite and nonnegative")
    if args.decoder_attn_zloss and args.arch != "v2":
        p.error("--decoder-attn-zloss needs --arch v2")
    if args.arch == "v2":
        from lightpfn.model_v2 import Config as ConfigV2, LightPFNV2
        if args.ccmm_weight or args.trainable != "all" or args.init is not None or args.compile != "none":
            p.error("--arch v2 supports neither CCMM, adapters, --init nor --compile")
        overrides = json.loads(args.config)
        if ck is not None:
            for key in ("decoder_scale_cap", "icl_scale_cap", "retrieval_smoothing"):
                overrides.setdefault(key, ck["config"].get(key, 0.0))
            overrides.setdefault("decoder_scale_mode", ck["config"].get("decoder_scale_mode", "legacy"))
            for key, default in (("icl_scale_mode", "legacy"), ("icl_log_reference", 1024),
                                 ("frontend_qk_norm", False), ("decoder_head_bound", 8.0),
                                 ("retrieval_qk_affine", True)):
                overrides.setdefault(key, ck["config"].get(key, default))
        cfg = ConfigV2(**overrides)
        model = LightPFNV2(cfg)
    else:
        overrides = json.loads(args.config)
        if ck is not None:
            overrides.setdefault("grad_checkpoint", ck["config"].get("grad_checkpoint", False))
        cfg = Config(**overrides)
        if args.ccmm_weight and not cfg.ccmm:
            p.error("--ccmm-weight requires --config containing ccmm=true")
        if args.ccmm_weight and cfg.cat_adapter:
            p.error("CCMM loss does not support the categorical adapter")
        model = LightPFN(cfg)
    if args.init is not None and not (run_dir / "ckpt.pt").exists():
        init_weights(model, args.init)
    freeze_for(model, args.trainable)
    model = model.to(device)
    ema = copy.deepcopy(model).eval()
    muon_p, adamw_p = param_groups(model)
    step = 0
    if args.decoder_scaling_lr_factor is None:
        args.decoder_scaling_lr_factor = (ck or {}).get("decoder_scaling_lr_factor", 1.0)
    opts = make_optimizers(model, muon_p, adamw_p, args.lr, args.weight_decay, args.decoder_scaling_lr_factor)
    if ckpt_path.exists():
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        load_optimizer_states(opts, ck, model, (muon_p, adamw_p))
        step = ck["step"]
        if main_rank:
            print(f"resumed from step {step}", flush=True)
    configure_acceleration(model, args.compile, args.tunableop_results, args.compile_workers)
    if world > 1:
        sync_replicas([model, ema])
        torch.manual_seed(args.seed + 7919 * rank)  # label-slot draws differ across ranks
    if main_rank:
        (run_dir / "args.json").write_text(json.dumps(dict(vars(args), config_full=asdict(cfg), world=world),
                                                      indent=2, default=str))
        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"model {n_params / 1e6:.2f}M params ({n_train:,} trainable), muon {sum(p.numel() for p in muon_p) / 1e6:.2f}M, "
              f"{world} rank(s)", flush=True)

    if step >= end:
        if main_rank:
            print(f"step {step} already at --stop-at/--steps {end}: nothing to do", flush=True)
        if world > 1:
            dist.destroy_process_group()
        return
    pools = [(s.rsplit(":", 1)[0], float(s.rsplit(":", 1)[1])) for s in args.pools]
    # the stream is seeded without the clock: a resumed run needs a different seed so that it
    # does not replay the groups it saw from step 0
    stream_seed = (args.seed + 1_000_003 * step) % 2**32
    fit_rng = np.random.default_rng([stream_seed, rank, 17])
    loader = DataLoader(PoolStream(pools, args.p_missing, stream_seed, args.augment, args.sampling, rank, world,
                                   typed=args.view == "typed", cat_mask_drop=args.cat_mask_drop),
                        batch_size=None, num_workers=args.workers,
                        pin_memory=True, prefetch_factor=4 if args.workers else None,
                        persistent_workers=bool(args.workers))
    it = iter(loader)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        t = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    log = open(run_dir / "log.csv", "a") if main_rank else None
    t0, tasks_seen, n_acc = time.time(), 0, 0
    loss_acc = torch.zeros((), device=device, dtype=torch.float64)
    zloss_acc = torch.zeros((), device=device, dtype=torch.float64)
    params = [p for p in model.parameters() if p.requires_grad]
    diagnostic = TrainingDiagnostic(model) if args.diagnostics_every else None
    diagnostic_norms = []
    nan_guard = NanGuard.from_env(model, ema, opts, args, cfg, rank, device)
    model.train()
    while step < end:
        if nan_guard is not None:
            nan_guard.begin(step + 1, fit_rng)
        sample_diagnostic = diagnostic is not None and (step + 1) % args.diagnostics_every == 0
        if sample_diagnostic:
            diagnostic.begin()
        lr = lr_at(step)
        for o in opts:
            for gr in o.param_groups:
                gr["lr"] = lr * gr["lr_scale"] if "lr_scale" in gr else lr
        groups = [next(it) for _ in range(args.groups_per_step)]
        n_local = n_tasks = sum(g["X"].shape[0] for g in groups)
        if world > 1:  # the mean over every rank's tasks; a GPU tensor, so no host sync
            n_tasks = torch.tensor(float(n_local), device=device)
            dist.all_reduce(n_tasks)
        step_loss = torch.zeros((), device=device, dtype=torch.float64)
        micro_batch_count = 0
        zloss_stats = {} if args.decoder_attn_zloss else None
        for group_index, g in enumerate(groups):
            task_offset = 0
            for mb in micro_batches(g, args.max_cells, args.row_cost, fit_rng if args.fit_rows else None):
                if sample_diagnostic:
                    micro_batch_count += 1
                if args.trainable == "adapter":
                    cat = mb.get("cat_mask")
                    if cat is None or not bool((cat & (torch.arange(cat.shape[1])[None] < mb["d"][:, None])).any()):
                        continue  # the frozen model's loss on tables without categorical columns has no gradient
                if nan_guard is not None:
                    nan_guard.record(mb, group_index, task_offset)
                    task_offset += mb["X"].shape[0]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.decoder_attn_zloss:
                        losses = task_loss_v2(model, mb, device, args.decoder_attn_zloss, zloss_stats)
                    else:
                        losses = (task_loss_v2(model, mb, device) if args.arch == "v2"
                                  else task_loss(model, mb, device, cfg.label_slots, args.ccmm_weight))
                loss = losses.sum() / n_tasks
                loss.backward()
                # Match Python's double accumulation, without a host read per microbatch.
                step_loss.add_(loss.detach().double())
                if zloss_stats:  # the z-loss share of loss (regression micro-batches add nothing)
                    zloss_acc.add_(zloss_stats.pop("attn_zloss").sum().double() / n_tasks)
        if nan_guard is not None:
            nan_guard.check_local(params)
        if world > 1:
            allreduce_grads(params)
        details = diagnostic.finish(opts) if sample_diagnostic else None
        if details is not None:
            details["rank_batch"] = dict(rank=rank, tasks=n_local, micro_batches=micro_batch_count,
                                       n_train_min=min(g["n_train"] for g in groups),
                                       n_train_max=max(g["n_train"] for g in groups),
                                       features_max=max(int(g["d"].max()) for g in groups))
        gnorm = (torch.nn.utils.clip_grad_norm_(params, args.clip) if nan_guard is None
                 else nan_guard.clip(params, args.clip))
        if diagnostic is not None:
            diagnostic_norms.append(gnorm.detach())
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        ema_update(ema, model, args.ema if step > args.warmup else 0.99)
        step += 1
        tasks_seen += n_local * world
        loss_acc.add_(step_loss)
        n_acc += 1

        if step % args.log_every == 0:
            if world > 1:
                dist.all_reduce(loss_acc)  # each rank holds its share of the global mean
                if args.decoder_attn_zloss:
                    dist.all_reduce(zloss_acc)
            logged_loss = float(loss_acc) / n_acc
            dt = time.time() - t0
            line = dict(step=step, loss=logged_loss, lr=lr, gnorm=float(gnorm), tasks_per_s=tasks_seen / dt,
                        s_per_step=dt / n_acc)
            if args.decoder_attn_zloss:  # included in loss; loss - attn_zloss is the task loss
                line["attn_zloss"] = float(zloss_acc) / n_acc
                zloss_acc.zero_()
            if details is not None:
                norms = torch.stack(diagnostic_norms)
                details["window"] = dict(steps=len(norms), gnorm_median=float(norms.median()),
                                         gnorm_max=float(norms.max()),
                                         clip_fraction=float((norms > args.clip).float().mean()),
                                         nonfinite_norms=int((~torch.isfinite(norms)).sum()))
                # Keep every norm to reconstruct exact medians over longer windows.
                details["window"]["gnorm_values"] = norms.cpu().tolist()
                details["optimizer_groups"] = [
                    [dict(lr=g["lr"], weight_decay=g["weight_decay"], parameters=len(g["params"]))
                     for g in o.param_groups] for o in opts]
                line["diagnostics"] = details
                diagnostic_norms.clear()
            if main_rank:
                print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in line.items()}), flush=True)
                log.write(json.dumps(line) + "\n")
                log.flush()
            t0, tasks_seen, n_acc = time.time(), 0, 0
            loss_acc.zero_()
        if step % args.save_every == 0 or step == end:
            # Same gradients and optimizer math keep the replicas equal; check it before saving.
            if world > 1 and replicas_differ(model):
                if main_rank:
                    print(f"step {step}: replicas differ, copying rank 0 to the others", flush=True)
                sync_replicas([model, ema])
            if main_rank:
                ck = dict(model=model.state_dict(), ema=ema.state_dict(), opts=[o.state_dict() for o in opts],
                          step=step, config=asdict(cfg), cat_mask_drop=args.cat_mask_drop)
                if args.decoder_scaling_lr_factor != 1:
                    ck.update(optimizer_param_names=optimizer_names(opts, model),
                              decoder_scaling_lr_factor=args.decoder_scaling_lr_factor)
                if args.decoder_attn_zloss:
                    ck["decoder_attn_zloss"] = args.decoder_attn_zloss
                torch.save(ck, run_dir / "ckpt.pt.tmp")
                (run_dir / "ckpt.pt.tmp").replace(ckpt_path)
        if main_rank and (step % args.eval_every == 0 or step == args.steps):
            # The EMA checkpoint first, published by rename (readers never see a partial file); the lite
            # eval after it is informative only and may fail without losing the checkpoint.
            ema_path = run_dir / f"ema_step{step:06d}.pt"
            torch.save(dict(ema=ema.state_dict(), step=step, config=asdict(cfg)), run_dir / "ema.pt.tmp")
            (run_dir / "ema.pt.tmp").replace(ema_path)
            if args.no_lite:
                continue
            te = time.time()
            s, g_rank = lite_eval(ema, device, run_dir / f"lite_step{step:06d}.csv", args.arch, args.view == "typed")
            print(f"=== lite eval step {step} ({time.time() - te:.0f}s)\n{s.round(4).to_string()}\n{g_rank.round(2).to_string()}",
                  flush=True)
            model.train()
    if diagnostic is not None:
        diagnostic.close()
    if world > 1:
        dist.barrier()  # rank 0 may still be saving or evaluating the last step
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
