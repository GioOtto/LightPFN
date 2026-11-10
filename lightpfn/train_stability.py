"""Opt-in optimizer experiments and sampled, detached training diagnostics."""

import math

import torch

from lightpfn.model.layers import SoftmaxScaling
from lightpfn.model_v2.scaling import BoundedHeadScaling, LogScaling


def make_optimizers(model, muon, adamw, lr, weight_decay, decoder_scaling_lr_factor=1.0):
    if not math.isfinite(decoder_scaling_lr_factor) or not 0 < decoder_scaling_lr_factor <= 1:
        raise ValueError("decoder scaling LR factor must be in (0, 1]")
    opts = []
    if muon:
        opts.append(torch.optim.Muon(muon, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw"))
    if adamw:
        groups = adamw
        if decoder_scaling_lr_factor != 1:
            decoder = getattr(model, "retrieval", getattr(model, "decoder", None))
            scaling = {id(p) for p in decoder.scaling.parameters()} if decoder is not None else set()
            special = [p for p in adamw if id(p) in scaling]
            if not special:
                raise ValueError("no trainable decoder scaling tensors")
            groups = [dict(params=[p for p in adamw if id(p) not in scaling]),
                      dict(params=special, lr=lr * decoder_scaling_lr_factor,
                           lr_scale=decoder_scaling_lr_factor, weight_decay=0.0)]
        opts.append(torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98)))
    return opts


def optimizer_names(opts, model):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[[names[id(p)] for p in group["params"]] for group in opt.param_groups] for opt in opts]


def load_optimizer_states(opts, checkpoint, model, legacy_groups):
    """Regroup by name, keeping moments and step counters from legacy or experimental checkpoints."""
    old = checkpoint["opts"]
    names = checkpoint.get("optimizer_param_names")
    current = optimizer_names(opts, model)
    if names is None:
        lookup = {id(p): n for n, p in model.named_parameters()}
        names = [[[lookup[id(p)] for p in group]] for group in legacy_groups if group]
    if len(old) != len(opts) or len(names) != len(opts):
        raise ValueError("checkpoint optimizer layout does not match the model")
    for opt, saved, before, after in zip(opts, old, names, current):
        if len(before) != len(saved["param_groups"]) or any(
                len(nn) != len(g["params"]) for nn, g in zip(before, saved["param_groups"])):
            raise ValueError("checkpoint parameter names do not match optimizer state")
        if before == after:
            # Preserve the exact legacy load path; explicit experimental group settings win.
            settings = [dict(lr_scale=g.get("lr_scale", 1.0), weight_decay=g["weight_decay"])
                        for g in opt.param_groups]
            opt.load_state_dict(saved)
            for group, setting in zip(opt.param_groups, settings):
                if "lr_scale" in group or setting["lr_scale"] != 1:
                    group.update(setting)
            continue
        states = {n: saved["state"].get(i) for nn, g in zip(before, saved["param_groups"])
                  for n, i in zip(nn, g["params"])}
        if set(states) != {n for group in after for n in group}:
            raise ValueError("checkpoint optimizer parameters changed")
        target = opt.state_dict()
        target["state"] = {i: states[n] for nn, g in zip(after, target["param_groups"])
                           for n, i in zip(nn, g["params"]) if states[n] is not None}
        opt.load_state_dict(target)


class TrainingDiagnostic:
    """Maxima over every micro-batch of ONE sampled step; gradients after all-reduce, before clip.

    Attention logits are recomputed in fp32 after the SDPA autocast input conversion, tiled to
    avoid an NxN allocation. This is not the internal accumulation of flash attention.
    """

    def __init__(self, model):
        self.model = model
        self.names = {id(m): name for name, m in model.named_modules()}
        self.active = False
        self.values = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, (LogScaling, BoundedHeadScaling)):
                self.handles.append(module.register_forward_hook(self.scaling))
            if isinstance(module, SoftmaxScaling):
                self.handles.append(module.register_forward_hook(self.scaling))
                self.handles.append(module.base.register_forward_hook(self.base))
            if name.endswith(("q_norm", "k_norm", "post_attn", "post_ff")):
                self.handles.append(module.register_forward_pre_hook(self.normalization))
            if hasattr(module, "q_norm") and (hasattr(module, "classification") or hasattr(module, "attend")):
                module.diagnostic = self

    def begin(self):
        self.values = {}
        self.raw_loss_sum = None
        self.raw_tasks = 0
        self.active = True

    @torch.no_grad()
    def maximum(self, name, value):
        value = value.detach().float()
        self.values[name] = value if name not in self.values else torch.maximum(self.values[name], value)

    @torch.no_grad()
    def scaling(self, module, args, out):
        if self.active:
            name = self.names[id(module)]
            self.maximum(name + ".q_before_abs_max", args[0].detach().abs().amax())
            self.maximum(name + ".q_after_abs_max", out.detach().abs().amax())
            if isinstance(module, LogScaling):
                self.maximum(name + ".coefficient_abs_max", module.head_scale.detach().abs().amax())
                self.maximum(name + ".effective_abs_max",
                             module.head_scale.detach().abs().amax() * math.log(max(args[1], 2)))
            if isinstance(module, BoundedHeadScaling):
                scale = module.scale(args[1]).detach()
                self.maximum(name + ".effective_abs_max", scale.amax())
                # Running maxima only: the smallest head scale is stored negated.
                self.maximum(name + ".negated_effective_min", (-scale).amax())
                self.maximum(name + ".near_bound_fraction_max", (scale >= .95 * module.bound).float().mean())

    @torch.no_grad()
    def base(self, module, args, out):
        if self.active:
            name = self.names[id(module)]
            value = out.detach().float()
            self.maximum(name + ".output_abs_max", value.abs().amax())
            self.maximum(name + ".output_rms_max", value.square().mean().sqrt())
            scaling = self.model.get_submodule(name.rsplit(".", 1)[0])
            cap = scaling.base_cap
            effective = scaling.effective_base(out.detach()).float()
            self.maximum(name + ".effective_abs_max", effective.abs().amax())
            if scaling.base_mode == "atan":
                self.maximum(name + ".near_bound_fraction_max", (effective.abs() >= 3.8).float().mean())
            if cap:
                self.maximum(name + ".cap_fraction_max", (value.abs() > cap).float().mean())

    @torch.no_grad()
    def normalization(self, module, args):
        if self.active:
            with torch.autocast(args[0].device.type, enabled=False):
                rms2 = args[0].detach().float().square().mean(-1)
                self.maximum(self.names[id(module)] + ".inverse_rms_max", (rms2 + module.eps).rsqrt().amax())

    @torch.no_grad()
    def attention(self, module, q, k):
        if not self.active:
            return
        name = self.names[id(module)]
        if torch.is_autocast_enabled(q.device.type):
            dtype = torch.get_autocast_dtype(q.device.type)
            q, k = q.to(dtype), k.to(dtype)
        self.maximum(name + ".q_abs_max", q.detach().abs().amax())
        with torch.autocast(q.device.type, enabled=False):
            q, k = q.detach().float(), k.detach().float()
            for i in range(0, q.shape[-2], 64):
                row_hi, row_lo = None, None
                for j in range(0, k.shape[-2], 512):
                    logits = q[..., i:i + 64, :] @ k[..., j:j + 512, :].transpose(-1, -2)
                    logits = logits / math.sqrt(q.shape[-1])
                    self.maximum(name + ".logit_max", logits.amax())
                    self.maximum(name + ".logit_abs_max", logits.abs().amax())
                    hi, lo = logits.amax(-1), logits.amin(-1)
                    row_hi = hi if row_hi is None else torch.maximum(row_hi, hi)
                    row_lo = lo if row_lo is None else torch.minimum(row_lo, lo)
                self.maximum(name + ".logit_range_max", (row_hi - row_lo).amax())

    @torch.no_grad()
    def probabilities(self, p):
        if self.active:
            self.raw_logp = p.detach().clamp_min(1e-8).log()
            self.raw_probability = p.detach()
            self.maximum("retrieval.probability_floor_fraction_max", (p < 1e-8).float().mean())
            # Before smoothing: comparable between control and smoothing experiments.
            self.maximum("retrieval.raw_nll_max", -p.clamp_min(1e-8).log().amin())

    @torch.no_grad()
    def targets(self, y):
        if self.active:
            target = self.raw_probability.gather(-1, y[..., None])
            self.maximum("retrieval.target_floor_fraction_max", (target < 1e-8).float().mean())
            self.maximum("retrieval.target_nll_max", -target.clamp_min(1e-8).log().amin())
            loss = (-self.raw_logp.gather(-1, y[..., None]).squeeze(-1)).mean(-1).double().sum()
            self.raw_loss_sum = loss if self.raw_loss_sum is None else self.raw_loss_sum + loss
            self.raw_tasks += y.shape[0]
            del self.raw_logp
            del self.raw_probability

    @torch.no_grad()
    def finish(self, opts):
        self.active = False
        squared, unused = {}, []
        for name, p in self.model.named_parameters():
            owner = name.rsplit(".", 1)[0]
            if p.grad is None:
                unused.append(name)
            else:
                norm2 = p.grad.detach().float().norm().square()
                squared[owner] = squared.get(owner, 0) + norm2
            if ".scaling." in name:
                self.values["weight_norm." + name] = p.detach().float().norm()
        self.values.update({"gradient_norm." + n: v.sqrt() for n, v in squared.items()})
        for i, opt in enumerate(opts):
            for j, group in enumerate(opt.param_groups):
                state = [opt.state[p] for p in group["params"] if p in opt.state]
                for key in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                    norms = [s[key].detach().float().norm().square() for s in state if key in s]
                    if norms:
                        self.values[f"optimizer.{i}.{j}.{key}_norm"] = torch.stack(norms).sum().sqrt()
        # Reduce diagnostics over ranks too: micro-batches/data differ, even with identical weights.
        keys = sorted(self.values)
        tensor = torch.stack([self.values[k] for k in keys])
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        details = dict(max_over_ranks=dict(zip(keys, tensor.cpu().tolist())), unused_parameters=unused)
        if self.raw_loss_sum is not None:
            raw = torch.stack([self.raw_loss_sum, self.raw_loss_sum.new_tensor(self.raw_tasks)])
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(raw)
            details["unsmoothed_sample_loss"] = float(raw[0] / raw[1])
        return details

    def close(self):
        for handle in self.handles:
            handle.remove()
        for module in self.model.modules():
            if getattr(module, "diagnostic", None) is self:
                del module.diagnostic
