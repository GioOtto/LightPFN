"""Explicit S0 candidates; auto budgeting spends only on active ICL feed-forwards."""

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class Config:
    width: int = 320
    unique_blocks: int = 4
    repeats: int = 3
    workspace: bool = True
    slots: int = 512
    head: str = "classification"  # classification, quantiles, transport
    ich: bool = True
    bypass: bool = True
    chunk_cells: int = 8192
    # Training (autograd on) bounds cell stages by train_chunk_cells instead: one piece per
    # micro-batch of the usual budget. Every piece costs a full set of kernel launches.
    train_chunk_cells: int = 1 << 19
    query_chunk: int = 32
    checkpoint: bool = True
    ff_hidden: int | None = None
    decoder_scale_cap: float = 0.0  # abs(base(log n)); modulation adds a factor in [0, 2]
    decoder_scale_mode: str = "legacy"  # atan: smooth base in [-4, 4]; head: one bounded scalar per head
    decoder_head_bound: float = 8.0  # head mode only: scale in (0, bound), no query modulation
    icl_scale_cap: float = 0.0
    icl_scale_mode: str = "legacy"  # log: one coefficient per head, no query gate
    icl_log_reference: int = 1024  # initial log scale is one at this key length
    frontend_qk_norm: bool = False  # non-affine head normalization before attention
    retrieval_smoothing: float = 0.0  # additive class probability mass before log, renormalized
    retrieval_qk_affine: bool = True  # False: no learned q/k norm gains, so atan bounds the decoder scale

    def __post_init__(self):
        if self.icl_scale_mode not in ("legacy", "log"):
            raise ValueError("icl_scale_mode must be legacy or log")
        if self.icl_scale_mode == "log" and self.icl_scale_cap:
            raise ValueError("icl_scale_cap is only supported with legacy icl_scale_mode")
        if not isinstance(self.icl_log_reference, int) or self.icl_log_reference < 2:
            raise ValueError("icl_log_reference must be an integer >= 2")
        if not isinstance(self.retrieval_qk_affine, bool):
            raise ValueError("retrieval_qk_affine must be boolean")
        if not isinstance(self.frontend_qk_norm, bool):
            raise ValueError("frontend_qk_norm must be boolean")
        if self.decoder_scale_mode not in ("legacy", "atan", "head"):
            raise ValueError("decoder_scale_mode must be legacy, atan or head")
        if not 1 < self.decoder_head_bound <= 64:  # range checked numerically (backward included)
            raise ValueError("decoder_head_bound must be in (1, 64]")
        if self.decoder_scale_mode != "head" and self.decoder_head_bound != 8.0:
            raise ValueError("decoder_head_bound only applies to decoder_scale_mode head")
        if self.decoder_scale_mode != "legacy" and self.decoder_scale_cap:
            raise ValueError("smooth decoder_scale_mode excludes decoder_scale_cap")
        for value in (self.decoder_scale_cap, self.icl_scale_cap, self.retrieval_smoothing):
            if not math.isfinite(value) or value < 0:
                raise ValueError("stability settings must be finite and nonnegative")
        if self.width not in (256, 320) or self.unique_blocks < 3 or self.repeats < 1:
            raise ValueError("unsupported ICL geometry")
        if self.head not in ("classification", "quantiles", "transport"):
            raise ValueError("choose one regression head, or classification only")
        if self.slots < 1 or self.chunk_cells < 1 or self.train_chunk_cells < 1 or self.query_chunk < 1:
            raise ValueError("chunk sizes and slots must be positive")
        if self.bypass and not self.ich:
            raise ValueError("bypass requires ICH")

    def to_dict(self):
        return asdict(self)


def ablation_configs():
    """All budgeted arms. A5 depths are implementation proposals, not a winner."""
    result = {}
    for name, fields in {
        "A4_320x4": {},
        "A3_320x8_distinct": dict(unique_blocks=8, repeats=1),
        "A5_320x6_proposed": dict(unique_blocks=6, repeats=1),
        "A5_256x8_proposed": dict(width=256, unique_blocks=8, repeats=1),
    }.items():
        for head in ("classification", "quantiles", "transport"):
            for workspace in (True, False):
                for ich, bypass in ((True, True), (True, False), (False, False)):
                    key = f"{name}/{head}/{'workspace' if workspace else 'dense'}/ich{int(ich)}_bypass{int(bypass)}"
                    result[key] = Config(**fields, head=head, workspace=workspace, ich=ich, bypass=bypass)
    # Fixed-trunk controls for causal ICH/bypass/workspace and head comparisons.
    # The 999-quantile workspace arm has the smallest FF budget of these heads.
    # Instantiate the real module instead of maintaining a second count formula.
    from .model import LightPFNV2
    reference_hidden = LightPFNV2(Config(head="quantiles")).ff_hidden
    for head in ("classification", "quantiles", "transport"):
        for workspace, ich, bypass in ((True, True, True), (True, True, False),
                                       (True, False, False), (False, True, True)):
            key = f"A4_fixed_trunk/{head}/{'workspace' if workspace else 'dense'}/ich{int(ich)}_bypass{int(bypass)}"
            result[key] = Config(head=head, workspace=workspace, ich=ich, bypass=bypass, ff_hidden=reference_hidden)
    # Causal sharing control (OPUS S0 review M-2): A4 with A3's FF width per head, so A3 vs A4 changes
    # only sharing at equal applications and FF FLOPs. A3 vs budget A4 is an A5-type question.
    for head in ("classification", "quantiles", "transport"):
        a3 = LightPFNV2(Config(unique_blocks=8, repeats=1, head=head)).ff_hidden
        result[f"A4_ffA3/{head}/workspace/ich1_bypass1"] = Config(head=head, ff_hidden=a3)
    return result
