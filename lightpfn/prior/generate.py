"""Deterministic, resumable pools of graph-SCM and fitted-tree classification tasks.

A group shares its sampled row count, requested width and train fraction. Resampling selects
exactly that many row indices without replacement; repeated values are allowed. Shards contain float16 features,
uint8 contiguous labels, true widths and generation diagnostics. The priors are described in the technical report.
"""

import argparse
import collections
from contextlib import contextmanager
import fcntl
import importlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import re
import shutil
import time

import numpy as np
import torch

PRESETS = {
    "s1": dict(n_min=64, n_max=2048, train_min=0.3, train_max=0.9, group_size=8),
    "s1b": dict(n_min=256, n_max=2048, train_min=0.3, train_max=0.9, group_size=8),
    "s2": dict(n_min=1024, n_max=10240, train_min=0.5, train_max=0.9, group_size=4),
    "s3": dict(n_min=4096, n_max=50000, train_min=0.6, train_max=0.9, group_size=2),
    "s4": dict(n_min=4096, n_max=60000, train_min=0.6, train_max=0.9, group_size=2),
}
P_BINARY = 0.4
P_RESAMPLE = 0.5
MAX_CLASSES = 10
LABEL_SLOTS = 16
MAX_ABS_X = 1000.0
ROWS_PER_FEATURE = 8
P_UNCAPPED = 0.1
RATIO_BANDS = [(0.7, 0.01, 0.06), (0.2, 0.06, 0.15), (0.1, 0.15, 0.5)]
PRIORS = ["graph", "tree", "rule"]  # append: legacy seed coordinates must not change
# Remove constant features, retain duplicates, and rank-match resampling quotas.
PRIOR_VERSIONS = {"graph": 4, "tree": 4, "rule": 1}
SUPPORTED_VERSIONS = {"graph": (3, 4), "tree": (4,), "rule": (1,)}
DEFAULT_VERSIONS = {"graph": 3, "tree": 4, "rule": 1}
FILTER_TRIES = 63  # initial draft plus retries; every draft is checked, never force-accepted
MAX_TASK_DRAFTS = 128  # also bounds structural failures (constant columns, missing classes, ...)
MAX_GRAPH_DRAWS = 128
SEED_SCHEME = "seedsequence-coordinate-v1"


def _prior():
    from tabicl.prior.graph_lib._config import PriorConfig

    return PriorConfig(
        add_gaussian_noise=False, allow_act_warping=False,
        filter_unpredictable_graphs=True, filter_unpredictable_datasets=True,
        min_n_nodes=2, max_n_nodes=32, cauchy_dag_offset=0.0,
    )


_PRIOR = None
_THREAD_LIMIT = None


def _init_worker(prior_name):
    global _PRIOR, _THREAD_LIMIT
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    _THREAD_LIMIT = threadpool_limits(limits=1)
    _PRIOR = _prior() if prior_name in ("graph", "rule") else None


def coordinate_seed(seed, prior, shard, group, task=0, domain=0):
    """Unambiguous coordinates, unlike the old arithmetic seed formula (gi>=1000 collided)."""
    values = (seed, shard, group, task, domain)
    if any(not isinstance(x, (int, np.integer)) for x in values):
        raise ValueError("seed coordinates must be integers")
    coords = [int(seed), PRIORS.index(prior), int(shard), int(group), int(task), int(domain)]
    if any(not 0 <= x <= np.iinfo(np.uint32).max for x in coords):
        # SeedSequence expands large integers into several words: bound each coordinate
        # to one word so [2**32, 0, ...] cannot alias [0, 1, 0, ...].
        raise ValueError("seed coordinates must fit uint32")
    return coords


def _seed_task(seed):
    ss = np.random.SeedSequence(seed)
    # NumPy's legacy API accepts a full seed vector: no truncation to 32 bits.
    np.random.seed(ss.generate_state(4))
    torch.manual_seed(int(ss.generate_state(1, dtype=np.uint64)[0]))
    random.seed(int(ss.generate_state(1, dtype=np.uint64)[0]))
    return np.random.default_rng(ss)


def _graph_dataset(spec, n_gen):
    """One TabICLv2 GraphSCM draft, with canonical feature-group iteration.

    TabICL 2.2.0 GraphPrior.generate_dataset returns (padded X, y, effective d), removes
    constant columns and retries indefinitely; it can return fewer classes than requested.
    We use the same graph sampler/converters/preprocessing, but own the retry/shape policy.
    RandomDataset.sample iterates list(set(groups)); sorting x/y fixes PYTHONHASHSEED drift.
    """
    from tabicl.prior.graph_lib._base import Context, Dataset, DatasetProperties
    from tabicl.prior.graph_lib._dataset import RandomDataset, check_x_y_ancestors_overlap
    from tabicl.prior.graph_lib._graph import RandomDAG
    from tabicl.prior.graph_lib._graph_function import RandomGraphFunction
    from tabicl.prior.graph_lib._properties import sample_categorical_sizes
    from tabicl.prior._reg2cls import outlier_removing, standard_scaling

    cfg = _PRIOR if _PRIOR is not None else _prior()
    modern = spec.get("prior_version", DEFAULT_VERSIONS[spec["prior"]]) != 3
    if modern:
        from lightpfn.prior.v4 import CONFIG
        if np.random.random() < CONFIG["softmax_only_probability"]:
            from dataclasses import replace
            cfg = replace(cfg, cat_modes="softmax")
    context = Context(config=cfg, device="cpu")
    d = spec["n_features"]
    properties = DatasetProperties(
        n_train=n_gen, n_test=0,
        cat_sizes={"x": sample_categorical_sizes(d, context, max_cat_size=200),
                   "y": [0 if spec.get("balanced_target", False) else spec["n_classes"]]},
    )
    sampler = RandomDataset(context).sampler
    for _ in range(MAX_GRAPH_DRAWS):
        n_nodes = sampler.randint("n_nodes", cfg.min_n_nodes, cfg.max_n_nodes + 1, use_log=True)
        graph = RandomDAG(context).sample(n_nodes)
        node_specs = [dict() for _ in range(n_nodes)]
        for group in sorted({s.group for s in properties.feature_specs.values()}):
            features = {k: v for k, v in properties.feature_specs.items() if v.group == group}
            if cfg.subsample_feature_nodes:
                n_feature_nodes = sampler.randint("n_feature_nodes", 1, n_nodes + 1)
                nodes = np.random.permutation(n_nodes)[:n_feature_nodes]
            else:
                nodes = np.arange(n_nodes)
            assignments = np.random.choice(nodes, replace=True, size=len(features))
            if modern and group == "y" and np.random.random() < CONFIG["sink_target_probability"]:
                # A sink target prevents its categorical value from being copied into
                # observed descendants. The ancestral-overlap check still applies.
                assignments[:] = n_nodes - 1
            for node, (key, value) in zip(assignments, features.items()):
                node_specs[node][key] = value
        if not cfg.filter_unpredictable_graphs or check_x_y_ancestors_overlap(graph, node_specs):
            break
    else:
        raise RuntimeError("TabICLv2: graph connectivity retries exhausted")
    fn = RandomGraphFunction(context, dag=graph, node_feature_specs=node_specs)
    if cfg.ensure_iid:
        fn(n_gen)
    data = Dataset(tensors=fn(n_gen), feature_specs=properties.feature_specs,
                   graph=graph, n_train=n_gen).get_concat_tensors()
    X = torch.cat([data[k] for k in ("x_cat", "x_num") if k in data], dim=-1).float()
    if modern:
        from lightpfn.prior.v4 import preprocess_graph
        X, info = preprocess_graph(X, data)
    else:
        X = standard_scaling(outlier_removing(X, threshold=4))
    order = torch.randperm(d)
    X = X[:, order]
    if spec.get("balanced_target", False):
        signal = data["y_num"].view(-1).double().numpy()
        cuts = np.quantile(signal, np.arange(1, spec["n_classes"])/spec["n_classes"])
        y = torch.from_numpy(np.searchsorted(cuts, signal)).long()
    else:
        y = data["y_cat"].view(-1).long()
    if bool((y >= 0).all()) and bool((y < spec["n_classes"]).all()):
        y = torch.randperm(spec["n_classes"])[y]
    if modern:
        info["cat_mask"] = info["cat_mask"][order.numpy()]
        info["config"] = cfg
        return X.numpy(), y.numpy(), info
    return X.numpy(), y.numpy()


def _target_proportions(rng, n_classes):
    if n_classes == 2:
        minority = math.exp(rng.uniform(math.log(0.01), math.log(0.5)))
        return np.array([minority, 1 - minority])
    alpha = math.exp(rng.uniform(math.log(0.3), math.log(3.0)))
    return rng.dirichlet(np.full(n_classes, alpha))


def _unpredictable(X, y, config=None, *, return_metrics=False):
    """Deterministic TabICLv2 OOB Brier/MSE test (ET seed 1, bootstrap seed 0)."""
    from tabicl.prior._dataset import should_filter
    from tabicl.prior.graph_lib._config import PriorConfig

    cfg = config if config is not None else PriorConfig(filter_unpredictable_datasets=True)
    if return_metrics:
        from lightpfn.prior.v4 import filter_metrics
        return filter_metrics(X, y, cfg)
    return should_filter(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y), cfg, is_classif=True)


def _base_dataset(rng, spec, n_gen):
    if spec["prior"] == "tree":
        from lightpfn.prior.trees import generate_tree_task

        return generate_tree_task(rng, n_gen, spec["n_features"], spec["n_classes"])
    return _graph_dataset(spec, n_gen)


def _clean_candidate(X, y, spec):
    """Validate before casts; drop float16 constants and retain repeated rows/columns."""
    X, y = np.asarray(X), np.asarray(y)
    C, d = spec["n_classes"], spec["n_features"]
    if (X.ndim != 2 or len(X) == 0 or X.shape[1] != d or y.shape != (len(X),)
            or not np.isfinite(X).all() or not np.isfinite(y).all()
            or not np.equal(y, np.floor(y)).all() or not np.array_equal(np.unique(y), np.arange(C))):
        return None
    # Both priors are standardized already; reject overflows rather than silently clipping.
    if np.any(X < -MAX_ABS_X) or np.any(X > MAX_ABS_X):
        return None
    X = X.astype(np.float16)
    X = X[:, X.min(0) != X.max(0)]
    if X.shape[1] == 0:
        return None
    return X, y.astype(np.int64)


def _quotas(proportions, n, minimum=2):
    """Exact largest-remainder allocation with a per-class minimum (never overshoots n)."""
    p = np.asarray(proportions, dtype=float)
    if (p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p < 0)
            or p.sum() <= 0 or minimum < 1 or n < minimum * len(p)):
        raise ValueError("infeasible class proportions/row budget")
    ideal = p / p.sum() * n
    take = np.maximum(np.floor(ideal).astype(int), minimum)
    while take.sum() > n:
        eligible = take > minimum
        take[np.argmax(np.where(eligible, take - ideal, -np.inf))] -= 1
    while take.sum() < n:
        take[np.argmax(ideal - take)] += 1
    return take


def _select_rows(rng, y, take):
    return rng.permutation(np.concatenate([
        rng.choice(np.flatnonzero(y == c), size=int(k), replace=False) for c, k in enumerate(take)
    ]))


def generate_task(spec):
    """A fully seed-determined task; exhausted retries raise instead of accepting bad data.

    Filter the quantized source. Resampled tasks retain this source-level signal guarantee:
    the final OOB Brier test is an audit, not a second gate penalizing rare classes (see review).
    """
    version = spec.get("prior_version", DEFAULT_VERSIONS.get(spec["prior"]))
    if version not in SUPPORTED_VERSIONS.get(spec["prior"], ()):
        raise ValueError("unsupported prior version")
    if spec["prior"] == "rule" or (spec["prior"] == "graph" and version == 4):
        from lightpfn.prior.v4 import generate_modern_task
        return generate_modern_task(spec)
    n, d, C = (spec[k] for k in ("n_rows", "n_features", "n_classes"))
    if (any(not isinstance(v, (int, np.integer)) for v in (n, d, C))
            or spec["prior"] not in PRIORS or not 1 <= d <= np.iinfo(np.int16).max
            or not 2 <= C <= min(MAX_CLASSES, LABEL_SLOTS, np.iinfo(np.uint8).max + 1)
            or n < 2 * C):
        raise ValueError("invalid task specification (requires n_rows >= 2*n_classes)")
    rng = _seed_task(spec["seed"])
    requested = rng.random() < P_RESAMPLE
    n_gen = 2 * n if requested else n
    rejected = invalid = checks = 0
    for draft in range(1, MAX_TASK_DRAFTS + 1):
        X, y = _base_dataset(rng, spec, n_gen)
        # Check raw shapes before indexing: a malformed source must use the bounded
        # structural retry path rather than raise IndexError during its shuffle.
        clean = _clean_candidate(X, y, spec)
        if clean is None:
            invalid += 1
            continue
        X, y = clean
        counts = np.bincount(y, minlength=C)
        if len(y) < n or np.any(counts < 2):
            invalid += 1
            continue
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]
        checks += 1
        if _unpredictable(X, y):
            rejected += 1
            if rejected > FILTER_TRIES:
                raise RuntimeError(f"predictability retries exhausted: {spec}, rejected={rejected}")
            continue
        take = _quotas(_target_proportions(rng, C), n) if requested else counts.copy()
        # Labels are permuted below, so rank matching preserves the target distribution
        # while maximizing feasibility. Stable sorting makes ties reproducible.
        if requested:
            take[np.argsort(counts, kind="stable")] = np.sort(take)
        resampled = bool(requested and np.all(take <= counts))
        if not resampled:
            # Capacity-aware sampling with all classes preserved. Start with two per class,
            # then uniformly draw from the remaining rows; no ad hoc group truncation.
            reserved = _select_rows(rng, y, np.full(C, 2))
            available = np.ones(len(y), dtype=bool)
            available[reserved] = False
            idx = rng.permutation(np.r_[reserved, rng.choice(np.flatnonzero(available), n - 2 * C, replace=False)])
        else:
            idx = _select_rows(rng, y, take)
        X, y = X[idx], y[idx]
        # A selected subset can lose rare categories. Remove its constant columns too.
        X = X[:, X.min(0) != X.max(0)]
        if X.shape[1] == 0:
            invalid += 1
            continue
        y = rng.permutation(C)[y]
        return dict(X=X, y=y.astype(np.uint8), n_features=d, resampled=resampled,
                    resample_requested=bool(requested), rejected=rejected, invalid=invalid,
                    drafts=draft, filter_checks=checks)
    raise RuntimeError(f"structural retries exhausted: {spec}, invalid={invalid}, rejected={rejected}")


def stratified_split(rng, y, n_rows, n_train):
    """Unique row indices, every class in train and every nonsingleton class in test.

    Infeasible budgets raise; callers must explicitly allocate enough test rows.
    """
    y = np.asarray(y)
    if y.ndim != 1 or not 1 <= n_train < n_rows <= len(y):
        raise ValueError("invalid row/train budget")
    classes, counts = np.unique(y, return_counts=True)
    if n_train < len(classes) or n_rows - n_train < np.sum(counts >= 2):
        raise ValueError("split cannot reserve classes in train/test")
    train, test, rest = [], [], []
    for c in classes:
        rows = rng.permutation(np.flatnonzero(y == c))
        train.append(rows[0])
        if len(rows) >= 2:
            test.append(rows[1])
        rest.extend(rows[2:] if len(rows) >= 2 else rows[1:])
    rest = rng.permutation(rest).astype(np.int64)
    n_fill = n_train - len(train)
    train = np.r_[train, rest[:n_fill]].astype(np.int64)
    test = np.r_[test, rest[n_fill:n_fill + n_rows - n_train - len(test)]].astype(np.int64)
    return np.r_[rng.permutation(train), rng.permutation(test)]


def assemble_group(rng, tasks, train_frac):
    if not tasks or not 0 < train_frac < 1:
        raise ValueError("empty group or invalid train fraction")
    n_rows = len(tasks[0]["y"])
    if any(len(t["y"]) != n_rows for t in tasks):
        raise ValueError("a group must retain its requested row count")
    Cmax = max(len(np.unique(t["y"])) for t in tasks)
    test_min = max(sum(np.unique(t["y"], return_counts=True)[1] >= 2) for t in tasks)
    if Cmax > n_rows - max(1, test_min):
        raise ValueError("group cannot fit class reservations")
    n_train = min(max(round(train_frac * n_rows), Cmax), n_rows - max(1, test_min))
    G = len(tasks)
    d_max = max(t.get("n_features", t["X"].shape[1]) for t in tasks)
    if any(not 1 <= t["X"].shape[1] <= t.get("n_features", d_max) for t in tasks):
        raise ValueError("effective width exceeds the requested width")
    X = np.zeros((G, n_rows, d_max), dtype=np.float16)
    Y = np.zeros((G, n_rows), dtype=np.uint8)
    d, n_classes = np.zeros(G, dtype=np.int16), np.zeros(G, dtype=np.int16)
    for g, t in enumerate(tasks):
        sel = stratified_split(rng, t["y"], n_rows, n_train)
        if "rule_meta" in t:
            for attempt in range(MAX_TASK_DRAFTS):
                oracle = t["rule_oracle"][sel[n_train:]]
                if len(np.unique(oracle)) >= 2 and np.mean(oracle == t["y"][sel[n_train:]]) >= .70:
                    break
                sel = stratified_split(rng, t["y"], n_rows, n_train)
            else:
                raise ValueError("rule signal split retries exhausted")
        X[g, :, :t["X"].shape[1]] = t["X"][sel]
        _, Y[g] = np.unique(t["y"][sel], return_inverse=True)
        d[g], n_classes[g] = t["X"].shape[1], len(np.unique(Y[g]))
    group = dict(X=torch.from_numpy(X), y=torch.from_numpy(Y), d=torch.from_numpy(d),
                 n_classes=torch.from_numpy(n_classes), n_train=int(n_train),
                 resampled=torch.tensor([t["resampled"] for t in tasks]))
    if all("cat_mask" in t for t in tasks):
        mask = np.zeros((G, d_max), dtype=bool)
        for i, t in enumerate(tasks):
            mask[i, :int(d[i])] = t["cat_mask"]
        group["cat_mask"] = torch.from_numpy(mask)
        group["unclipped_mode"] = [t["unclipped_mode"] for t in tasks]
        for key in ("trivial_rejected", "balance_rejected", "oob_auc"):
            group[key] = torch.tensor([t[key] for t in tasks])
    if all("rule_meta" in t for t in tasks):
        group["rule_meta"] = [t["rule_meta"] for t in tasks]
        from lightpfn.prior.rules import evaluate_rule
        group["rule_oracle"] = torch.from_numpy(np.stack([
            evaluate_rule(X[i, :, :int(d[i])], t["rule_meta"]) for i, t in enumerate(tasks)]))
    for key in ("resample_requested", "rejected", "invalid", "drafts", "filter_checks"):
        if all(key in t for t in tasks):
            group[key] = torch.tensor([t[key] for t in tasks])
    validate_group(group)
    return group


def validate_group(g, *, preset=None, max_features=None, group_size=None):
    """Stored-shard contract. Also usable to audit existing pools without modifying them."""
    X, y, d, nc = (g[k] for k in ("X", "y", "d", "n_classes"))
    if (not all(torch.is_tensor(v) for v in (X, y, d, nc)) or X.ndim != 3
            or y.shape != X.shape[:2] or X.dtype != torch.float16 or y.dtype != torch.uint8):
        raise ValueError("invalid X/y shape or storage dtype")
    B, n, m = X.shape
    ntr = g["n_train"]
    if (B < 1 or m < 1 or d.shape != (B,) or nc.shape != (B,)
            or d.dtype != torch.int16 or nc.dtype != torch.int16
            or not isinstance(ntr, int) or not 1 <= ntr < n
            or not bool(torch.isfinite(X).all()) or float(X.abs().max()) > MAX_ABS_X):
        raise ValueError("invalid group dimensions/train budget/finite range")
    if group_size is not None and B != group_size:
        raise ValueError("invalid group size")
    if preset and not preset["n_min"] <= n <= preset["n_max"]:
        raise ValueError("row count outside preset")
    if max_features is not None and m > max_features:
        raise ValueError("too many features")
    for i in range(B):
        di, C = int(d[i]), int(nc[i])
        if not 1 <= di <= m or not 2 <= C <= min(MAX_CLASSES, LABEL_SLOTS) or ntr < C:
            raise ValueError("invalid d/n_classes")
        if (not torch.equal(torch.unique(y[i]), torch.arange(C, dtype=y.dtype))
                or not torch.equal(torch.unique(y[i, :ntr]), torch.arange(C, dtype=y.dtype))
                or bool((X[i, :, di:] != 0).any())):
            raise ValueError("labels/classes/train coverage/padding disagree")
        Xi, yi = X[i, :, :di].numpy(), y[i].numpy()
        counts = np.bincount(yi, minlength=C)
        if np.any(counts < 2) or not np.array_equal(np.unique(yi[ntr:]), np.arange(C)):
            raise ValueError("class has fewer than two rows or is absent from test")
        if np.any(Xi.min(0) == Xi.max(0)):
            raise ValueError("constant real column")
    if "resampled" not in g or g["resampled"].shape != (B,) or g["resampled"].dtype != torch.bool:
        raise ValueError("invalid resampling flags")
    for key in ("resample_requested", "rejected", "invalid", "drafts", "filter_checks"):
        if key in g and g[key].shape != (B,):
            raise ValueError(f"invalid {key} shape")
    if "resample_requested" in g:
        if g["resample_requested"].dtype != torch.bool or bool((g["resampled"] & ~g["resample_requested"]).any()):
            raise ValueError("inconsistent resampling diagnostics")
    for key in ("rejected", "invalid", "drafts", "filter_checks"):
        if key in g and (g[key].dtype != torch.int64 or bool((g[key] < 0).any())):
            raise ValueError(f"invalid {key} diagnostics")
    if all(k in g for k in ("rejected", "invalid", "drafts", "filter_checks")):
        minimum_checks = g["rejected"] + (0 if "rule_meta" in g else 1)
        if (not torch.equal(g["drafts"], g["rejected"] + g["invalid"] + 1)
                or bool((g["filter_checks"] < minimum_checks).any())
                or bool((g["filter_checks"] > g["drafts"]).any())
                or bool((g["drafts"] > MAX_TASK_DRAFTS).any())
                or bool((g["rejected"] > FILTER_TRIES).any())):
            raise ValueError("inconsistent draft/filter diagnostics")
    if "cat_mask" in g:
        mask = g["cat_mask"]
        if not torch.is_tensor(mask) or mask.shape != (B, m) or mask.dtype != torch.bool:
            raise ValueError("invalid categorical mask")
        for i, di in enumerate(d):
            if bool(mask[i, int(di):].any()):
                raise ValueError("categorical padding is not zero")
    if "rule_meta" in g:
        from lightpfn.prior.rules import evaluate_rule
        if len(g["rule_meta"]) != B or g["rule_oracle"].shape != (B, n) or g["rule_oracle"].dtype != torch.uint8:
            raise ValueError("invalid rule metadata/oracle")
        if bool((g["filter_checks"] != 0).any()):
            raise ValueError("rule tasks must not use ExtraTrees")
        if bool(g["resampled"].any()) or bool(g["resample_requested"].any()):
            raise ValueError("rule tasks must not use class resampling")
        for i, meta in enumerate(g["rule_meta"]):
            drivers = meta["drivers"]
            C = int(nc[i])
            if (meta["n_classes"] != C or sorted(meta["class_map"]) != list(range(C))
                    or meta["family"] not in ("xor", "parity", "tree", "lookup", "product", "and", "or")):
                raise ValueError("invalid rule classes/family")
            if (not drivers or len(set(drivers)) != len(drivers)
                    or any(not isinstance(j, int) or not 0 <= j < int(d[i]) for j in drivers)
                    or (meta["family"] == "lookup" and len(drivers) != 1)
                    or (meta["family"] != "lookup" and not 2 <= len(drivers) <= 4)):
                raise ValueError("invalid rule driver indices")
            oracle = evaluate_rule(X[i, :, :int(d[i])].numpy(), meta)
            if not np.array_equal(oracle, g["rule_oracle"][i].numpy()):
                raise ValueError("rule oracle disagrees with observed features")
            if (len(np.unique(oracle)) != int(nc[i]) or len(np.unique(oracle[ntr:])) < 2
                    or np.bincount(oracle, minlength=C).min() < 2
                    or np.mean(oracle[ntr:] == y[i, ntr:].numpy()) < .70):
                raise ValueError("rule has insufficient test signal")


def sample_n_features(rng, n_rows, max_features, geometry):
    if n_rows < 1 or not 1 <= max_features <= np.iinfo(np.int16).max:
        raise ValueError("invalid feature/row limits")
    low = min(2, max_features)
    if geometry == "cap":
        high = max_features if rng.random() < P_UNCAPPED else min(max_features, max(4, n_rows // ROWS_PER_FEATURE))
        return int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))
    if geometry != "ratio":
        raise ValueError("unknown geometry")
    probs, lows, highs = zip(*RATIO_BANDS)
    k = rng.choice(len(RATIO_BANDS), p=probs)
    ratio = math.exp(rng.uniform(math.log(lows[k]), math.log(highs[k])))
    return int(min(max_features, max(low, round(ratio * n_rows))))


def group_specs(rng, preset, max_features, base_seed, prior, geometry="cap", p_binary=P_BINARY, max_cells=None):
    """max_cells caps rows x features: long tables draw from a narrower feature range. None leaves the
    draws exactly as before (existing pools resume and validate unchanged)."""
    if (not np.isfinite(p_binary) or not 0 <= p_binary <= 1 or prior not in PRIORS
            or not 2 * MAX_CLASSES <= preset["n_min"] <= preset["n_max"]
            or preset["group_size"] < 1
            or not 0 < preset["train_min"] <= preset["train_max"] < 1):
        raise ValueError("invalid preset/probability/prior")
    n_rows = round(math.exp(rng.uniform(math.log(preset["n_min"]), math.log(preset["n_max"]))))
    if max_cells is not None:
        max_features = min(max_features, max(2, max_cells // n_rows))
    d = sample_n_features(rng, n_rows, max_features, geometry)
    frac = rng.uniform(preset["train_min"], preset["train_max"])
    specs = []
    for g in range(preset["group_size"]):
        C = 2 if rng.random() < p_binary else int(rng.integers(3, MAX_CLASSES + 1))
        seed = [*base_seed, g] if isinstance(base_seed, (list, tuple)) else [int(base_seed), g]
        specs.append(dict(seed=seed, n_rows=n_rows, n_features=d, n_classes=C, prior=prior))
    return specs, frac


def shard_specs(args, shard, n_groups):
    preset, specs, fractions = PRESETS[args.preset], [], []
    for gi in range(n_groups):
        coords = coordinate_seed(args.seed, args.prior, shard, gi, domain=1)
        s, frac = group_specs(np.random.default_rng(coords), preset, args.max_features,
                              coordinate_seed(args.seed, args.prior, shard, gi, domain=2),
                              args.prior, args.geometry, args.p_binary, getattr(args, "max_cells", None))
        specs.extend(s)
        fractions.append(frac)
    for s in specs:
        s["prior_version"] = getattr(args, "prior_version", DEFAULT_VERSIONS[args.prior])
    return specs, fractions


def assemble_shard(args, shard, results, fractions):
    G = PRESETS[args.preset]["group_size"]
    return [assemble_group(np.random.default_rng(coordinate_seed(args.seed, args.prior, shard, gi, domain=3)),
                           results[gi * G:(gi + 1) * G], fractions[gi]) for gi in range(len(fractions))]


def make_shard(pool, args, shard, n_groups):
    specs, fractions = shard_specs(args, shard, n_groups)
    return assemble_shard(args, shard, pool.map(generate_task, specs, chunksize=1), fractions)


def generator_meta(args):
    versions = {}
    for package in ("numpy", "torch", "scipy", "scikit-learn", "tabicl", "xgboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            # Some conda builds (notably xgboost) omit wheel distribution metadata.
            module = importlib.import_module("sklearn" if package == "scikit-learn" else package)
            versions[package] = module.__version__
    version = getattr(args, "prior_version", DEFAULT_VERSIONS[args.prior])
    extra = {}
    if args.prior == "rule" or (args.prior == "graph" and version == 4):
        from dataclasses import asdict
        from lightpfn.prior.v4 import CONFIG
        extra = dict(prior_config=asdict(_prior()), modern_config=CONFIG)
    return json.loads(json.dumps(dict(vars(args), preset_cfg=PRESETS[args.preset],
        p_resample=P_RESAMPLE, max_classes=MAX_CLASSES, label_slots=LABEL_SLOTS, max_abs_x=MAX_ABS_X,
        rows_per_feature=ROWS_PER_FEATURE, p_uncapped=P_UNCAPPED, ratio_bands=RATIO_BANDS,
        prior_version=version, seed_scheme=SEED_SCHEME,
        filter_tries=FILTER_TRIES, max_task_drafts=MAX_TASK_DRAFTS, max_graph_draws=MAX_GRAPH_DRAWS,
        versions=versions, python=platform.python_version(), **extra)))


def check_resume(out, meta):
    """Validate metadata even with zero shards; deserialize all committed shards before reuse."""
    files = list(out.glob("shard_*.pt"))
    by_id = {}
    for f in files:
        match = re.fullmatch(r"shard_(\d{6,})\.pt", f.name)
        if match is None:
            raise ValueError(f"malformed shard name: {f}")
        index = int(match[1])
        if f.name != f"shard_{index:06d}.pt":
            raise ValueError(f"noncanonical shard name: {f}")
        by_id[index] = f
    ids = sorted(by_id)
    files = [by_id[i] for i in ids]
    if ids != list(range(len(ids))):
        raise ValueError("shards must be contiguous from zero")
    path = out / "meta.json"
    if files and not path.exists():
        raise ValueError("shards without meta.json")
    if path.exists():
        old = json.loads(path.read_text())
        run_only = {"out", "n_tasks", "n_jobs", "min_free_gb"}
        diff = {k: (old.get(k), meta.get(k)) for k in set(old) | set(meta)
                if k not in run_only and old.get(k) != meta.get(k)}
        if diff:
            raise ValueError(f"incompatible pool settings (old, new): {diff}")
    lengths = []
    for i, f in enumerate(files):
        try:
            groups = torch.load(f, map_location="cpu", weights_only=True)
            if not isinstance(groups, list) or not 1 <= len(groups) <= meta["groups_per_shard"]:
                raise ValueError("invalid shard group count")
            if i < len(files) - 1 and len(groups) != meta["groups_per_shard"]:
                raise ValueError("only the last shard may be partial")
            for gi, g in enumerate(groups):
                validate_group(g, preset=meta["preset_cfg"], max_features=meta["max_features"],
                               group_size=meta["preset_cfg"]["group_size"])
                # Detect a valid shard accidentally copied/renamed from other coordinates.
                specs, frac = group_specs(
                    np.random.default_rng(coordinate_seed(meta["seed"], meta["prior"], i, gi, domain=1)),
                    meta["preset_cfg"], meta["max_features"],
                    coordinate_seed(meta["seed"], meta["prior"], i, gi, domain=2),
                    meta["prior"], meta["geometry"], meta["p_binary"], meta.get("max_cells"))
                n, d = specs[0]["n_rows"], specs[0]["n_features"]
                classes = torch.tensor([s["n_classes"] for s in specs], dtype=torch.int16)
                Cmax = int(classes.max())
                if (g["X"].shape[1:] != (n, d) or not bool((g["d"] <= d).all())
                        or not torch.equal(g["n_classes"], classes)
                        or g["n_train"] != min(max(round(frac * n), Cmax), n - Cmax)):
                    raise ValueError("shard geometry/classes disagree with its seeded specification")
            lengths.append(len(groups))
        except Exception as exc:
            raise ValueError(f"invalid committed shard {f}: {exc}") from exc
    if sum(lengths) * meta["preset_cfg"]["group_size"] > meta["n_tasks"]:
        raise ValueError("requested task count is smaller than the committed pool")
    return lengths


def atomic_write(path, writer):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def pool_lock(out):
    with (out / ".generate.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another generator holds the pool lock") from exc
        yield


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=list(PRESETS), default="s1")
    p.add_argument("--prior", choices=PRIORS, default="graph")
    p.add_argument("--prior-version", type=int, default=None,
                   help="graph: 3 (legacy default) or 4; tree: 4; rule: 1")
    p.add_argument("--geometry", choices=["cap", "ratio"], default="cap")
    p.add_argument("--p-binary", type=float, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--n-tasks", type=int, default=256_000)
    p.add_argument("--groups-per-shard", type=int, default=32)
    p.add_argument("--max-features", type=int, default=100)
    p.add_argument("--max-cells", type=int, default=None,
                   help="cap on rows x features per task (long tables get fewer features); default: none")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-free-gb", type=float, default=200.0)
    p.add_argument("--lookahead", type=int, default=4,
                   help="shards in flight: workers start the next shards while one waits for its slowest task")
    args = p.parse_args()
    # Scheduling only: the shards are identical for every value, so it stays out of meta.json.
    lookahead = args.lookahead
    del args.lookahead
    if args.prior_version is None:
        args.prior_version = DEFAULT_VERSIONS[args.prior]
    if args.prior_version not in SUPPORTED_VERSIONS[args.prior]:
        p.error("unsupported prior version")
    if args.p_binary is None:
        args.p_binary = 0.78 if args.prior == "rule" or (args.prior == "graph" and args.prior_version == 4) else P_BINARY
    G = PRESETS[args.preset]["group_size"]
    if (args.n_tasks <= 0 or args.n_tasks % G or args.groups_per_shard < 1 or args.n_jobs < 1
            or not 0 <= args.seed <= np.iinfo(np.uint32).max
            or not 1 <= args.max_features <= np.iinfo(np.int16).max
            or (args.max_cells is not None and args.max_cells < 2 * PRESETS[args.preset]["n_max"])
            or not np.isfinite(args.p_binary) or not 0 <= args.p_binary <= 1
            or not np.isfinite(args.min_free_gb) or args.min_free_gb < 0 or lookahead < 1):
        p.error("invalid limits; n-tasks must be a positive multiple of group_size")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = generator_meta(args)
    # One thread from the start: the resume check of a large pool with every core's threads, on a busy box at
    # nice 19, read 8 of 47 GB in an hour (0.5 s per shard with one thread)
    torch.set_num_threads(1)
    with pool_lock(out):
        lengths = check_resume(out, meta)
        atomic_write(out / "meta.json", lambda f: f.write(json.dumps(meta, indent=2).encode()))
        ng = args.n_tasks // G
        start = len(lengths)
        if lengths and lengths[-1] < args.groups_per_shard and sum(lengths) < ng:
            start -= 1  # deterministic extension of the last partial shard
        n_shards = math.ceil(ng / args.groups_per_shard)
        print(f"pool {out}: {sum(lengths)*G} tasks present; shards {start}..{n_shards-1}", flush=True)
        if start >= n_shards:
            return
        with mp.get_context("spawn").Pool(args.n_jobs, initializer=_init_worker, initargs=(args.prior,)) as pool:
            pending, nxt, t0 = collections.deque(), start, time.monotonic()
            while nxt < n_shards or pending:
                while nxt < n_shards and len(pending) < lookahead:
                    if shutil.disk_usage(out).free / 1e9 < args.min_free_gb:
                        print("stopping: free disk space below min-free-gb", flush=True)
                        n_shards = nxt
                        break
                    specs, fractions = shard_specs(args, nxt, min(args.groups_per_shard, ng-nxt*args.groups_per_shard))
                    pending.append((nxt, pool.map_async(generate_task, specs, chunksize=1), fractions))
                    nxt += 1
                if not pending:
                    break
                shard, result, fractions = pending.popleft()
                groups = assemble_shard(args, shard, result.get(), fractions)
                # Shards are written in order, so a resume still sees a contiguous prefix.
                atomic_write(out / f"shard_{shard:06d}.pt", lambda f: torch.save(groups, f))
                rejected = sum(int(g["rejected"].sum()) for g in groups)
                invalid = sum(int(g["invalid"].sum()) for g in groups)
                drafts = sum(int(g["drafts"].sum()) for g in groups)
                checked = sum(int(g["filter_checks"].sum()) for g in groups)
                dt, t0 = time.monotonic()-t0, time.monotonic()
                print(f"shard {shard:06d}: {len(groups)*G/dt:.2f} tasks/s; "
                      f"filter rejected {rejected}/{checked} checks; structural rejects {invalid}/{drafts}", flush=True)


if __name__ == "__main__":
    main()
