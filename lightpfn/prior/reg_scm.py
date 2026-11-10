"""Regression-only operator policy for TabICL's synthetic graph, never weights.

Instance-local fit hooks are installed by a dedicated Context. No package class,
global function or classification context is patched. Operator parameters are
bounded before transforming any rows, including the independent fit sample.
"""

from types import MethodType

import torch

POLICY = dict(version=3, root_sources="persistent family, transform and covariance; fit on independent draw only",
              normalizer_max_gain=100., exp_log_bound=8.,
              power_magnitude_max=2., integer_power=2,
              structural_output_bound=32., structural_map="32*tanh(value/32), before observation",
              bounded_outputs=["Standardize", "L2Normalize", "RandomLinearFunction",
                               "RandomQuadraticFunction", "RandomProductFunction",
                               "RandomAggregationMultiFunction", "RandomNodeFunction numerical outputs"],
              aggregation_product_accumulator="float64",
              exclusions=[], channel_weight_renormalization=1.,
              observed_target_clipping=False, observation_retries=0)


def bounded_exp(x):
    return torch.exp(POLICY["exp_log_bound"] * torch.tanh(x / POLICY["exp_log_bound"]))


def structural_bound(x):
    if not torch.isfinite(x).all():
        raise FloatingPointError("nonfinite SCM operator, before observation; no silent repair")
    return POLICY["structural_output_bound"] * torch.tanh(x / POLICY["structural_output_bound"])


def _standardize_fit(self, x):
    # Float64 moments prevent overflow/cancellation inside the float32 SCM.
    v = x.double()
    self.mean_ = v.mean(0, keepdim=True).to(x.dtype)
    sd = v.std(0, keepdim=True, correction=0)
    self.inv_std_ = sd.clamp_min(1 / POLICY["normalizer_max_gain"]).reciprocal().to(x.dtype)


def _l2_fit(self, x):
    rms = x.double().square().sum(-1).mean().sqrt()
    self.factor_ = rms.clamp_min(1 / POLICY["normalizer_max_gain"]).reciprocal().to(x.dtype)


def _standardize_transform(self, x):
    return structural_bound((x.double() - self.mean_.double()) * self.inv_std_.double()).to(x.dtype)


def _l2_transform(self, x):
    return structural_bound(x.double() * self.factor_.double()).to(x.dtype)


def _bounded_transform(self, *args, **kwargs):
    return structural_bound(self._reg_original_transform(*args, **kwargs))


def _bounded_node_transform(self, *args, **kwargs):
    latent, features = self._reg_original_transform(*args, **kwargs)
    features = {k: structural_bound(v) if self.feature_specs[k].cat_size == 0 else v
                for k, v in features.items()}
    return structural_bound(latent), features


def _aggregation_transform(self, x):
    if self.op_type_ != "product":
        return structural_bound(self._reg_original_transform(x))
    # Every component is structurally bounded; <=32 factors of magnitude <=32
    # fit float64 even when their float32 product would overflow.
    components = torch.stack([f(t) for f, t in zip(self.funcs_, x)], 0)
    return structural_bound(components.double().prod(0)).to(components.dtype)


def _root_sample(self, n_batch, n):
    from tabicl.prior.graph_lib._function import RandomFunction
    from tabicl.prior.graph_lib._points import (RandomUniformPoints, RandomCirclePoints,
                                               RandomGaussianPoints, RandomCovariancePoints)
    if not hasattr(self, "_reg_root_transform"):
        source = self.sampler.choice("random_base_points", [RandomUniformPoints, RandomCirclePoints,
                                                            RandomGaussianPoints, RandomCovariancePoints])
        self._reg_root_transform = RandomFunction(self.context, n, n)
        self._reg_root_source = source(self.context)
        self._reg_root_width = n
    if n != self._reg_root_width:
        raise ValueError("persistent root source width changed")
    return self._reg_root_transform(self._reg_root_source.sample(n_batch, n))


def _covariance_sample(self, n_batch, n):
    from tabicl.prior.graph_lib._points import RandomUniformPoints, RandomGaussianPoints
    from tabicl.prior.graph_lib._weights import RandomWeights
    if not hasattr(self, "_reg_covariance"):
        self._reg_covariance = torch.randn(n, n) * RandomWeights(self.context).sample(1, n)
        source = self.sampler.choice("random_covariance_base_points", [RandomUniformPoints, RandomGaussianPoints])
        self._reg_covariance_source = source(self.context)
    return self._reg_covariance_source.sample(n_batch, n) @ self._reg_covariance.t()


def context(config, device="cpu"):
    from tabicl.prior.graph_lib._base import Context, GlobalSampler
    from tabicl.prior.graph_lib._activation import Standardize, L2Normalize, RandomTorchActivation
    from tabicl.prior.graph_lib._function import RandomLinearFunction, RandomQuadraticFunction, RandomProductFunction
    from tabicl.prior.graph_lib._node_function import RandomNodeFunction
    from tabicl.prior.graph_lib._multi_function import RandomAggregationMultiFunction
    from tabicl.prior.graph_lib._points import RandomPoints, RandomCovariancePoints

    class Sampler(GlobalSampler):
        def numerical(self, name, low, high, *args, **kwargs):
            if name in ("random_power_exponent", "relu_random_power_exponent", "neg_random_power_exponent"):
                high = POLICY["power_magnitude_max"]
            elif name == "rand_int_power_exponent":
                low, high = POLICY["integer_power"], POLICY["integer_power"] + 1
            return super().numerical(name, low, high, *args, **kwargs)

        def choice(self, name, categories, *args, **kwargs):
            # Preserve number/order/probability of operator choices. exp becomes
            # a smooth bounded operator, not a truncation of observed targets.
            if name == "acts_torch":
                categories = [bounded_exp if f is torch.exp else f for f in categories]
            return super().choice(name, categories, *args, **kwargs)

    class RegressionContext(Context):
        def get_recursion_context(self, obj):
            # Patch only these instances. The second draw samples innovations,
            # never a new root function, family or covariance parameter.
            if type(obj) is RandomPoints:
                obj.sample = MethodType(_root_sample, obj)
            elif type(obj) is RandomCovariancePoints:
                obj.sample = MethodType(_covariance_sample, obj)
            if isinstance(obj, Standardize):
                obj._fit = MethodType(_standardize_fit, obj)
                obj._transform = MethodType(_standardize_transform, obj)
            elif isinstance(obj, L2Normalize):
                obj._fit = MethodType(_l2_fit, obj)
                obj._transform = MethodType(_l2_transform, obj)
            elif isinstance(obj, (RandomLinearFunction, RandomQuadraticFunction, RandomProductFunction,
                                  RandomNodeFunction, RandomAggregationMultiFunction)):
                obj._reg_original_transform = obj._transform
                transform = (_bounded_node_transform if isinstance(obj, RandomNodeFunction) else
                             _aggregation_transform if isinstance(obj, RandomAggregationMultiFunction) else
                             _bounded_transform)
                obj._transform = MethodType(transform, obj)
            # Shared counters describe every constructed operator, including
            # dormant branches; fitted counts are not inferred from y values.
            name = type(obj).__name__
            if isinstance(obj, (Standardize, L2Normalize, RandomTorchActivation, RandomNodeFunction,
                                RandomLinearFunction, RandomQuadraticFunction, RandomProductFunction,
                                RandomAggregationMultiFunction)):
                counts[name] = counts.get(name, 0) + 1
            return RegressionContext(device=self.device, sampler=self.sampler, config=self.config,
                                     observed_classes=self.observed_classes + [type(obj)])

    counts = {}
    return RegressionContext(device=device, config=config, sampler=Sampler(seed=None, config=config)), counts
