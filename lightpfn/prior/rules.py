"""Sparse known mechanisms on graph-v4 features, with no learned signal gate."""
import math

import numpy as np

from lightpfn.prior.v4 import CONFIG, quantize_column


def evaluate_rule(X, meta):
    """Exact oracle from stored observed columns and serializable rule parameters."""
    D = X[:, meta["drivers"]].astype(np.float64)
    family = meta["family"]
    if family == "lookup":
        levels = np.asarray(meta["levels"])
        index = np.searchsorted(levels, D[:, 0])
        if np.any(index >= len(levels)) or np.any(levels[np.minimum(index, len(levels)-1)] != D[:, 0]):
            raise ValueError("lookup encountered an unknown oracle level")
        y = np.asarray(meta["table"])[index]
    elif family == "tree":
        node = np.zeros(len(X), dtype=int)
        for _ in range(meta["depth"]):
            drivers = np.asarray(meta["node_drivers"])[node]
            cuts = np.asarray(meta["node_cuts"])[node]
            side = D[np.arange(len(D)), drivers] > cuts
            node = 2 * node + 1 + side
        y = np.asarray(meta["table"])[node - (2**meta["depth"] - 1)]
    elif family == "product":
        score = np.prod(D[:, :2] - np.asarray(meta["centers"]), axis=1)
        y = np.asarray(meta["table"])[np.searchsorted(meta["product_cuts"], score)]
    elif family in ("xor", "parity") and meta["n_classes"] > 2:
        digits = np.column_stack([np.searchsorted(cuts, D[:, j], side="left")
                                  for j, cuts in enumerate(meta["digit_thresholds"])])
        y = digits.sum(1) % meta["n_classes"]
    else:
        bits = D > np.asarray(meta["thresholds"])
        if meta["n_classes"] == 2 and family in ("xor", "parity"):
            y = bits.sum(1) % 2
        elif meta["n_classes"] == 2 and family == "and":
            y = bits.all(1).astype(int)
        elif meta["n_classes"] == 2 and family == "or":
            y = bits.any(1).astype(int)
        else:
            cells = bits @ (2**np.arange(D.shape[1]))
            y = np.asarray(meta["table"])[cells]
    return np.asarray(meta["class_map"])[y].astype(np.uint8)


def _cell_table(rng, size, C):
    # Every class represented; random mapping rather than ordinal cell numbering.
    return rng.permutation(np.arange(size) % C).tolist()


def _make_rule(rng, X, cat, C, family):
    d = X.shape[1]
    minimum = max(2, math.ceil(math.log2(C))) if family in ("and", "or") else 2
    if d < minimum or (family == "tree" and C > 8):
        family = "lookup"
    k = min(d, int(rng.integers(minimum, max(minimum, 4) + 1)))
    if family == "xor" and C == 2:
        k = min(d, 2)
    if family == "product":
        k = min(d, 2)
    if family == "lookup":
        eligible = [j for j in range(d) if cat[j] and len(np.unique(X[:, j])) >= C]
        if not eligible:
            eligible = [j for j in range(d) if len(np.unique(X[:, j])) >= C]
            if not eligible:
                return None
            j = int(rng.choice(eligible))
            X[:, j] = quantize_column(rng, X[:, j], min(256, max(16, C*4)), 0.8)
            cat[j] = True
            eligible = [j]
        drivers = [int(rng.choice(eligible))]
    else:
        drivers = rng.choice(d, k, replace=False).tolist()
    if family in ("xor", "parity") and C > 2:
        eligible = [j for j in range(d) if len(np.unique(X[:, j])) >= C]
        if len(eligible) < 2:
            return None
        drivers = rng.choice(eligible, min(k, len(eligible)), replace=False).tolist()
    meta = dict(family=family, drivers=drivers, n_classes=C, class_map=list(range(C)))
    D = X[:, drivers].astype(float)
    if family == "lookup":
        levels = np.unique(D[:, 0])
        if len(levels) < C:
            return None
        meta.update(levels=levels.tolist(), table=_cell_table(rng, len(levels), C))
    elif family == "tree":
        depth = int(rng.integers(max(2, math.ceil(math.log2(C))), 4))
        node_drivers, cuts = [], []
        regions = [np.arange(len(X))]
        for rows in regions:
            if len(node_drivers) == 2**depth-1:
                break
            candidates = []
            for j in range(len(drivers)):
                cut = float(np.quantile(D[rows, j], rng.uniform(.3, .7))) if len(rows) else 0
                if len(rows) and D[rows, j].min() <= cut < D[rows, j].max():
                    candidates.append((j, cut))
            if not candidates:
                return None
            j, cut = candidates[int(rng.integers(len(candidates)))]
            node_drivers.append(j); cuts.append(cut)
            side = D[rows, j] > cut
            regions.extend([rows[~side], rows[side]])
        meta.update(depth=depth, node_drivers=node_drivers, node_cuts=cuts,
                    table=_cell_table(rng, 2**depth, C))
    elif family == "product":
        centers = np.median(D, axis=0)
        score = np.prod(D-centers, axis=1)
        cuts = [0.0] if C == 2 else np.unique(np.quantile(score, np.arange(1, C)/C)).tolist()
        if len(cuts) != C-1:
            return None
        meta.update(centers=centers.tolist(), product_cuts=cuts, table=rng.permutation(C).tolist())
    elif family in ("xor", "parity") and C > 2:
        # C-ary parity (a Latin table): independent uniform driver symbols,
        # class = sum(symbols) mod C. Arbitrary binary-cell -> C maps would leak
        # marginal signal, and cannot be pure for C=10 with only four bits.
        thresholds = []
        for j, driver in enumerate(drivers):
            levels = np.unique(D[:, j])
            positions = np.linspace(0, len(levels)-1, C+1).astype(int)[1:-1]
            cuts = [(float(levels[p])+float(levels[p+1]))/2 for p in positions]
            thresholds.append(cuts)
            symbols = rng.integers(C, size=len(X))
            original_symbols = np.searchsorted(cuts, D[:, j], side="left")
            for symbol in range(C):
                pool = D[:, j][original_symbols == symbol]
                rows = np.flatnonzero(symbols == symbol)
                if not len(pool):
                    return None
                X[rows, driver] = rng.choice(pool, len(rows), replace=True)
        meta.update(digit_thresholds=thresholds, interaction_radix=C)
    else:
        q = rng.uniform(.3, .7, len(drivers))
        thresholds = np.quantile(D, q, axis=0).diagonal()
        meta.update(thresholds=thresholds.tolist(), source_quantiles=q.tolist())
        if family in ("xor", "parity"):
            # Source quantiles are 30-70%, but each final bit cell receives equal
            # mass. Conditional draws preserve observed marginal values and ties;
            # independence removes graph-induced marginal leakage into parity.
            size = 2**len(drivers)
            cells = rng.permutation(np.arange(len(X)) % size)
            for j, driver in enumerate(drivers):
                for bit in (0, 1):
                    pool = D[:, j][(D[:, j] > thresholds[j]) == bit]
                    rows = np.flatnonzero(((cells >> j) & 1) == bit)
                    if not len(pool):
                        return None
                    X[rows, driver] = rng.choice(pool, len(rows), replace=True)
        meta["table"] = _cell_table(rng, 2**len(drivers), C)
    return meta


def generate_rule_task(spec):
    from lightpfn.prior import generate as g
    rng = g._seed_task(spec["seed"])
    n, d, C = (spec[k] for k in ("n_rows", "n_features", "n_classes"))
    family = str(rng.choice(CONFIG["rule_families"], p=CONFIG["rule_probabilities"]))
    noise = float(rng.uniform(*CONFIG["rule_noise"]))
    a = float(rng.uniform(*CONFIG["rule_weight"]))
    # Pure-interaction slice has no graph admixture by construction.
    b = 0.0 if family in ("xor", "parity") or rng.random() >= CONFIG["rule_mix_probability"] else float(rng.uniform(0, .5))
    for draft in range(1, g.MAX_TASK_DRAFTS + 1):
        X, graph_y, info = g._graph_dataset(spec, n)
        # The graph target is discarded. A shortage of its classes is irrelevant.
        if (X.shape != (n, d) or not np.isfinite(X).all() or np.max(np.abs(X)) > g.MAX_ABS_X):
            continue
        X = X.astype(np.float16)
        keep = X.min(0) != X.max(0)
        X, cat = X[:, keep], info["cat_mask"][keep]
        if not X.shape[1]:
            continue
        meta = _make_rule(rng, X, cat, C, family)
        if meta is None:
            continue
        oracle = evaluate_rule(X, meta).astype(int)
        if len(np.unique(oracle)) != C or np.bincount(oracle, minlength=C).min() < 2:
            continue
        # The graph API exposes a sampled category, not its latent logits. The
        # centered one-hot category is an explicit logit proxy, recorded in meta.
        scores = a * np.eye(C)[oracle]
        if b:
            if (graph_y.shape != (n,) or not np.isfinite(graph_y).all()
                    or not np.equal(graph_y, np.floor(graph_y)).all()
                    or np.any(graph_y < 0) or np.any(graph_y >= C)):
                continue
            scores += b * (np.eye(C)[graph_y.astype(int)] - 1/C)
        if b:
            logits = scores / CONFIG["rule_mix_temperature"]
            probabilities = np.exp(logits-logits.max(1, keepdims=True))
            probabilities /= probabilities.sum(1, keepdims=True)
            y = (rng.random(n)[:, None] > np.cumsum(probabilities, axis=1)).sum(1)
            y = np.minimum(y, C-1)
        else:
            y = oracle.copy()
        flip = rng.random(n) < noise
        y[flip] = (y[flip] + rng.integers(1, C, flip.sum())) % C
        if np.any(np.bincount(y, minlength=C) < 2) or np.mean(y == oracle) < .70:
            continue
        mapping = rng.permutation(C)
        meta.update(class_map=mapping.tolist(), noise=noise, rule_weight=a, graph_weight=b, mix_temperature=CONFIG["rule_mix_temperature"],
                    pure_interaction=meta["family"] in ("xor", "parity"),
                    driver_policy=CONFIG["interaction_driver_policy"] if meta["family"] in ("xor", "parity") else "observed")
        return dict(X=X, y=mapping[y].astype(np.uint8), n_features=d,
                    resampled=False, resample_requested=False, rejected=0, invalid=draft-1,
                    drafts=draft, filter_checks=0, cat_mask=cat, rule_meta=meta,
                    rule_oracle=mapping[oracle].astype(np.uint8), unclipped_mode=info["unclipped_mode"],
                    trivial_rejected=0, balance_rejected=0, oob_auc=float("nan"))
    raise RuntimeError(f"rule mechanism retries exhausted: {spec}")
