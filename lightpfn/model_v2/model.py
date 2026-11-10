"""Experimental C2v2 model. No v1 wrapper, registry or public API changes."""

from dataclasses import dataclass, field

import torch
from torch import nn

from lightpfn.model.layers import OrthogonalEmbedding, MLP

from .config import Config
from .frontend import Frontend
from .heads import QuantileHead, Retrieval, TransportHead
from .ich import target_view
from .icl import WorkspaceICL
from .types import Table, canonical_labels


@dataclass
class Context:
    frontend: object
    icl: object
    keys: torch.Tensor
    y: torch.Tensor
    n_classes: int
    regression: bool
    mean: torch.Tensor | None
    scale: torch.Tensor | None
    # Bank targets sorted once per context for the transport head, not per query
    # chunk. Derived in __post_init__, so dataclasses.replace(y=...) refreshes it.
    sorted_y: tuple | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.sorted_y = tuple(self.y.sort(-1)) if self.regression else None


class LightPFNV2(nn.Module):
    """One resident network with classification and at most ONE regression head.

    Regression configurations also support classification for later controlled
    multitask experiments. The choice of training mix/checkpoints remains open.
    """

    limit = 5_000_000

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg = cfg or Config()
        self.frontend = Frontend(cfg)
        self.label = OrthogonalEmbedding(10, cfg.width)
        self.reg_label = nn.Linear(1, cfg.width) if cfg.head != "classification" else None
        self.task = nn.Parameter(torch.zeros(1 if cfg.head == "classification" else 2, cfg.width))
        self.icl = WorkspaceICL(cfg, hidden=1)
        self.retrieval = Retrieval(cfg.width, cfg.decoder_scale_cap, cfg.retrieval_smoothing,
                                   cfg.decoder_scale_mode, cfg.retrieval_qk_affine, cfg.decoder_head_bound)
        if cfg.head == "classification":
            self.regression_head = None  # no random draw here: classification init is unchanged
        else:
            # The head draws from its own stream (seeded by one draw inside fork_rng), and the global CPU
            # stream is restored afterwards: the FF re-initialisation below, and so every common tensor, is
            # identical for quantiles and transport at the same seed (2026-11-04; before, the two heads
            # consumed different amounts of randomness and shifted the FF init).
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(torch.randint(0, 2**62, (1,)).item()))
                self.regression_head = (QuantileHead(cfg.width) if cfg.head == "quantiles"
                                        else TransportHead(cfg.width))
        # Spend the remaining budget on active FF units uniformly across unique
        # blocks. No unused padding parameter or second, dormant regression head.
        count = self.parameter_count()
        step = 2 * cfg.width * cfg.unique_blocks
        maximum = 1 + (self.limit - count) // step
        hidden = maximum if cfg.ff_hidden is None else cfg.ff_hidden
        if hidden < 1 or hidden > maximum:
            raise ValueError(f"FF hidden {hidden} exceeds 5M budget (maximum {maximum})")
        self.ff_hidden = hidden
        for block in (self.icl.prelude, *self.icl.core, self.icl.coda):
            block.mlp = MLP(cfg.width, hidden)
            nn.init.normal_(block.mlp.fc2.weight, std=.005)
        if self.parameter_count() > self.limit:
            raise ValueError("resident parameter limit exceeded")

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self):
        # Recursive leaves partition every parameter, including direct parameters.
        counts = {}
        for name, parameter in self.named_parameters():
            key = name.rsplit(".", 1)[0] if "." in name else name
            counts[key] = counts.get(key, 0) + parameter.numel()
        return counts

    def encode(self, context: Table, y, *, regression=False, n_classes=2):
        context = context.canonical()
        if y.shape != context.row_uid.shape or y.device != context.numerical.device:
            raise ValueError("y must match context rows and device")
        if regression:
            if self.regression_head is None or not y.is_floating_point() or not torch.isfinite(y).all():
                raise ValueError("regression needs a regression configuration and finite floating targets")
            full = [target_view(yy, torch.ones_like(yy, dtype=torch.bool), True) for yy in y]
            mean, scale = torch.stack([v.mean for v in full]), torch.stack([v.scale for v in full])
            z = torch.stack([v.values[:, 0] for v in full])
            slots, mapping = None, None
            bank_label = self.reg_label(z[..., None])
        else:
            slots, mapping = canonical_labels(y, context.row_uid, n_classes)
            z, mean, scale = None, None, None
            bank_label = self.label(slots)
        rows, frontend = self.frontend.encode(context, y, slots, mapping, z, regression)
        task = self.task[int(regression)]
        icl = self.icl.encode(rows + bank_label + task)
        keys = self.retrieval.keys(icl.bank)
        return Context(frontend, icl, keys, z if regression else y, n_classes, regression, mean, scale)

    def predict(self, query: Table, context: Context):
        query = query.canonical()
        rows = self.frontend.query(query, context.frontend) + self.task[int(context.regression)]
        out = []
        # Query rows are independent. Only the regression heads need bounded chunks
        # (O(chunk*N) retrieval weights); classification runs them all at once.
        step = self.cfg.query_chunk if context.regression else rows.shape[1]
        for i in range(0, rows.shape[1], step):
            h = self.icl.query(rows[:, i:i + step], context.icl)
            if context.regression:
                weights = self.retrieval.weights(h, context.keys)
                q = (self.regression_head(h, weights, context.y, context.sorted_y)
                     if isinstance(self.regression_head, TransportHead)
                     else self.regression_head(h, weights, context.y))
                q = context.mean[:, None, None] + context.scale[:, None, None] * q
                out.append(q)
            else:
                out.append(self.retrieval.classification(h, context.keys, context.y, context.n_classes))
        return torch.cat(out, 1)

    def forward(self, context: Table, y_context, query: Table, *, regression=False, n_classes=2):
        return self.predict(query, self.encode(context, y_context, regression=regression, n_classes=n_classes))
