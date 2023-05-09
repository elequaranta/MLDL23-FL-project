import torch
import torch.nn as nn


class HardNegativeMining(nn.Module):

    def __init__(self, perc=0.25):
        super().__init__()
        self.perc = perc

    def forward(self, loss, _):
        b = loss.shape[0]
        loss = loss.reshape(b, -1)
        p = loss.shape[1]
        tk = loss.topk(dim=1, k=int(self.perc * p))
        loss = tk[0].mean()
        return loss


class MeanReduction:
    def __call__(self, x, target):
        x = x[target != 255]
        return x.mean()


def get_scheduler(opts, optimizer, max_iter=None):
    if opts.lr_policy == 'poly':
        assert max_iter is not None, "max_iter necessary for poly LR scheduler"
        return torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                 lr_lambda=lambda cur_iter: (1 - cur_iter / max_iter) ** opts.lr_power)
    if opts.lr_policy == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)

    return None