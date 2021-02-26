# AdaBelief Slim

This repository contains the code for the `adabelief-slim` Python package, from which you can use a Pytorch implementation of the AdaBelief optimizer.

## Installation

Using Python 3.6 or higher:

```bash
pip install adabelief-slim
```

## Usage

```python
from adabelief import AdaBelief

model = ...
kwargs = ...

optimizer = AdaBelief(model.parameters(), **kwargs)
```

The following hyperparameters can be passed as keyword arguments:

- `lr`: learning rate (default: `1e-3`)
- `betas`: 2-tuple of coefficients used for computing the running averages of the gradient and its "variance" (see paper) (default: `(0.9, 0.999)`)
- `eps`: term added to the denominator to improve numerical stability (default: `1e-8`)
- `weight_decay`: weight decay coefficient (default: `1e-2`)
- `amsgrad`: whether to use the AMSGrad variant of the algorithm (default: `False`)
- `rectify`: whether to use the RAdam variant of the algorithm (default: `False`)
- `weight_decouple`: whether to use the AdamW variant of this algorithm (default: `True`)

Be aware that the AMSGrad and RAdam variants **can't** be used simultaneously.

## Motivation

As you're probably aware, one of the paper's main authors ([Juntang Zhuang](https://juntang-zhuang.github.io/)) released his code in this [repository](https://github.com/juntang-zhuang/Adabelief-Optimizer), which is used to maintain the `adabelief_pytorch` package. Thus, you may be wondering why this repository exists, and how it differs with his. The reason is actually pretty simple: the author made some decisions regarding his code which made it an unsuitable option for me. While it wasn't the only thing that bugged me, my main issue was with adding unnecessary packages as dependencies.

Regarding differences, the main ones are:

- I removed the `fixed_decay` option, as the author's experiments showed it wasn't great
- I removed the `degenerate_to_sgd` option, as the author copied the RAdam codebase, but it seems recommended to always use it
- I removed all logging related features, along with the `print_change_log` option
- I removed all code specific to older version of Pytorch (I think all versions above `1.4` should work), as I don't care for them
- I changed the flow of the code to be closer to the official implementation of AdamW
- I removed all usage of the `.data` property as it isn't recommended, and can be avoided with the `torch.no_grad` decorator
- I moved the code specific to AMSGrad so that it isn't executed if the RAdam variant is selected
- I added an exception if both RAdam and AMSGrad are selected, as they can't both be used (in the official repository RAdam is used if both RAdam and AMSGrad are selected)
- I removed half-precision support, as I don't care for it

## References

### Codebases

- [Official AdaBelief implementation](https://github.com/juntang-zhuang/Adabelief-Optimizer)
- [Official RAdam implementation](https://github.com/LiyuanLucasLiu/RAdam)
- [Official AdamW implementation](https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW)
- [Pytorch Optimizers](https://github.com/jettify/pytorch-optimizer)

### Papers

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980): proposed Adam
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101): proposed AdamW
- [On the Convergence of Adam and Beyond](https://arxiv.org/pdf/1904.09237.pdf): proposed AMSGrad
- [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265): proposed RAdam
- [AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients](https://arxiv.org/abs/1908.03265): proposed AdaBelief

## License

[MIT](LICENSE)
