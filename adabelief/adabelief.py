import math

import torch
from torch.optim.optimizer import Optimizer


class AdaBelief(Optimizer):
    r"""Implements AdaBelief algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdaBelief variant was proposed in `AdaBelief Optimizer, adapting stepsizes by the belief in
    observed gradients`_.
    It builds on:
        - the AdamW variant proposed in `Decoupled Weight Decay Regularization`_.
        - the AMSGrad variant proposed in `On the Convergence of Adam and Beyond`_.
        - the RAdam variant proposed in `On the Variance of the Adaptive Learning Rate and
        Beyond`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its variance (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this algorithm,
            incompatible with the RAdam variant  (default: False)
        rectify (boolean, optional): whether to use the RAdam variant of this algorithm,
            incompatible with the AMSGrad variant  (default: False)
        weight_decouple (boolean, optional): whether to use the AdamW variant of this algorithm,
            (default: True)


    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients:
        https://arxiv.org/pdf/2010.07468.pdf
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://arxiv.org/pdf/1904.09237.pdf
    .. _On the Variance of the Adaptive Learning Rate and Beyond:
        https://arxiv.org/abs/1908.03265
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        rectify=False,
        weight_decouple=True,
    ):

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if amsgrad and rectify:
            raise ValueError("AMSGrad and RAdam variants are not compatible, pick only one")

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if "betas" in param and (
                    param["betas"][0] != betas[0] or param["betas"][1] != betas[1]
                ):
                    param["buffer"] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(AdaBelief, self).__init__(params, defaults)

        self.weight_decouple = weight_decouple
        self.rectify = rectify

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")

                # Perform stepweigth decay
                if self.weight_decouple:
                    # Like AdamW
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                elif group["weight_decay"] != 0:
                    # Like regular Adam
                    grad.add_(p, alpha=group["weight_decay"])

                # Perform optimization step
                amsgrad = group["amsgrad"]
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of variance values
                    state["exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of variance values
                        state["max_exp_avg_var"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg_var = state["exp_avg"], state["exp_avg_var"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Decay the variance running average coefficient
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if not self.rectify:
                    # AdamW update
                    if amsgrad:
                        #  AMSGrad variant
                        max_exp_avg_var = state["max_exp_avg_var"]
                        # Maintains the maximum of all variance running avg. until now
                        torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    else:
                        denom = (exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = group["lr"] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    # RAdam update
                    buffered = group["buffer"][int(state["step"] % 10)]
                    if state["step"] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state["step"]
                        beta2_t = beta2 ** state["step"]
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma
                        step_size = 1 / bias_correction1

                        # 5 instead of 4 as a precaution
                        if N_sma >= 5:
                            step_size *= math.sqrt(
                                ((1 - beta2_t) * (N_sma - 4) * (N_sma - 2) * N_sma_max)
                                / ((N_sma_max - 4) * (N_sma_max - 2) * N_sma)
                            )
                        buffered[2] = step_size

                    if N_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(eps)
                        p.addcdiv_(exp_avg, denom, value=-step_size * group["lr"])
                    else:
                        # Un-adapted update if the variance is untractable
                        p.add_(exp_avg, alpha=-step_size * group["lr"])

        return loss
