"""
LARS: Layer-wise Adaptive Rate Scaling
Converted from TensorFlow to PyTorch
https://github.com/google-research/simclr/blob/master/lars_optimizer.py
"""

import torch
from torch.optim.optimizer import Optimizer, required
import re

EETA_DEFAULT = 0.001

class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
            self,
            params,
            lr=required,
            momentum=0.9,
            use_nesterov=False,
            weight_decay=0.0,
            exclude_from_weight_decay=None,
            exclude_from_layer_adaptation=None,
            classic_momentum=True,
            eeta=EETA_DEFAULT,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.ge(0),
                        torch.where(
                            g_norm.ge(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    # next_v.mul_(momentum).add_(grad, scaled_lr)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True

#
# from contextlib import contextmanager
# import torch
# from torch.optim.optimizer import Optimizer
# from torchlars._adaptive_lr import compute_adaptive_lr
#
#
# class LARS(Optimizer):
#     """Implements 'LARS (Layer-wise Adaptive Rate Scaling)'__ as Optimizer a
#     :class:`~torch.optim.Optimizer` wrapper.
#     __ : https://arxiv.org/abs/1708.03888
#     Wraps an arbitrary optimizer like :class:`torch.optim.SGD` to use LARS. If
#     you want to the same performance obtained with small-batch training when
#     you use large-batch training, LARS will be helpful::
#     Args:
#         optimizer (Optimizer):
#             optimizer to wrap
#         eps (float, optional):
#             epsilon to help with numerical stability while calculating the
#             adaptive learning rate
#         trust_coef (float, optional):
#             trust coefficient for calculating the adaptive learning rate
#     Example::
#         base_optimizer = optim.SGD(model.parameters(), lr=0.1)
#         optimizer = LARS(optimizer=base_optimizer)
#         output = model(input)
#         loss = loss_fn(output, target)
#         loss.backward()
#         optimizer.step()
#     """
#
#     def __init__(self, optimizer, eps=1e-8, trust_coef=0.001):
#         if eps < 0.0:
#             raise ValueError('invalid epsilon value: , %f' % eps)
#         if trust_coef < 0.0:
#             raise ValueError("invalid trust coefficient: %f" % trust_coef)
#
#         self.optim = optimizer
#         self.eps = eps
#         self.trust_coef = trust_coef
#         self.adaptive_lr = torch.ones([])
#
#     def __getstate__(self):
#         lars_dict = {}
#         lars_dict['eps'] = self.eps
#         lars_dict['trust_coef'] = self.trust_coef
#         lars_dict['adaptive_lr'] = self.adaptive_lr
#         return (self.optim, lars_dict)
#
#     def __setstate__(self, state):
#         self.optim, lars_dict = state
#
#         self.eps = lars_dict['eps']
#         self.trust_coef = lars_dict['trust_coef']
#         self.adaptive_lr = lars_dict['adaptive_lr']
#
#     def __repr__(self):
#         return '%s(%r)' % (self.__class__.__name__, self.optim)
#
#     @property
#     def param_groups(self):
#         return self.optim.param_groups
#
#     def state_dict(self):
#         return self.optim.state_dict()
#
#     def load_state_dict(self, state_dict):
#         self.optim.load_state_dict(state_dict)
#
#     def zero_grad(self):
#         self.optim.zero_grad()
#
#     def add_param_group(self, param_group):
#         self.optim.add_param_group(param_group)
#
#     @contextmanager
#     def hide_weight_decays(self):
#         weight_decays = []
#
#         for group in self.optim.param_groups:
#             if 'weight_decay' in group:
#                 weight_decays.append(group['weight_decay'])
#                 group['weight_decay'] = 0
#             else:
#                 weight_decays.append(None)
#
#         try:
#             yield weight_decays
#         finally:
#             for group, weight_decay in zip(self.optim.param_groups, weight_decays):
#                 if weight_decay is None:
#                     continue
#                 group['weight_decay'] = weight_decay
#
#     def apply_adaptive_lrs(self, weight_decays):
#         with torch.no_grad():
#             for group, weight_decay in zip(self.optim.param_groups, weight_decays):
#                 if weight_decay is None:
#                     weight_decay = 0.0
#                 for p in group['params']:
#                     if p.grad is None:
#                         continue
#
#                     param_norm = p.norm()
#                     grad_norm = p.grad.norm()
#
#                     # The optimizer class has no method to change `dtype` of
#                     # its inner tensors (like `adaptive_lr`) and to select to
#                     # use CPU or GPU with Tensor. LARS's interface follows the
#                     # optimizer class's interface, so LARS cannot change
#                     # `dtype` of inner tensors explicitly also. In that
#                     # context, we have constructed LARS can modify its member
#                     # variable's spec implicitly by comparing with given spec
#                     # by the original optimizer's element.
#                     param_norm_spec = (param_norm.is_cuda, param_norm.type())
#                     adaptive_lr_spec = (self.adaptive_lr.is_cuda, self.adaptive_lr.type())
#
#                     if param_norm_spec != adaptive_lr_spec:
#                         self.adaptive_lr = torch.ones_like(param_norm)
#
#                     # calculate adaptive lr & weight decay
#                     adaptive_lr = compute_adaptive_lr(
#                         param_norm,
#                         grad_norm,
#                         weight_decay,
#                         self.eps,
#                         self.trust_coef,
#                         self.adaptive_lr)
#
#                     p.grad.add_(weight_decay, p.data)
#                     p.grad.mul_(adaptive_lr)
#
#     def step(self, *args, **kwargs):
#         with self.hide_weight_decays() as weight_decays:
#             self.apply_adaptive_lrs(weight_decays)
#             return self.optim.step(*args, **kwargs)


