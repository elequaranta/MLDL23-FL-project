import copy
from typing import Any, OrderedDict
import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabV3

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
    
class DistillationLoss:

    def __init__(self, model: DeepLabV3, alpha: float = 0.5, beta: float = 0.5, tau: int = 1) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    def __call__(self, x: torch.Tensor, targets: torch.Tensor, imgs) -> Any:
        teacher_logits = self.model(imgs)["out"]
        distillation_loss = self.beta * self.cross_entropy(self.softmax(x/self.tau), self.softmax(teacher_logits/self.tau))
        student_loss = self.alpha * self.cross_entropy(self.softmax(x), targets)
        return distillation_loss + student_loss
    
    def update_model(self, params_dict: OrderedDict):
        self.model.load_state_dict(params_dict)
        self.model.to(self.device)