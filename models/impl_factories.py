from overrides import override
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, StepLR
from torch.optim import SGD, Adam, Optimizer
from torchvision.models.segmentation.deeplabv3 import _SimpleSegmentationModel

from models.abs_factories import ModelFactory, OptimizerFactory, SchedulerFactory
from config.enums import DatasetOptions
from models.deeplabv3 import deeplabv3_mobilenetv2

class DeepLabV3MobileNetV2Factory(ModelFactory):

    def __init__(self, dataset_type: DatasetOptions) -> None:
        super().__init__(dataset_type)
    
    def construct(self) -> _SimpleSegmentationModel:
        return deeplabv3_mobilenetv2(num_classes=self.dataset_class_number)
    
class SGDFactory(OptimizerFactory):

    def __init__(self, lr: float, weight_decay: float, momentum:float, model_params_iter) -> None:
        super().__init__(lr=lr, 
                         weight_decay=weight_decay, 
                         model_params=model_params_iter)
        self.momentum = momentum

    @override
    def construct(self) -> Optimizer:
        return SGD(self.params, 
                   lr=self.lr, 
                   momentum=self.momentum,
                   weight_decay=self.weight_decay,
                   nesterov=True)

class AdamFactory(OptimizerFactory):
    def __init__(self, lr: float, weight_decay: float, model_params_iter) -> None:
        super().__init__(lr=lr, 
                         weight_decay=weight_decay, 
                         model_params=model_params_iter)
    
    @override
    def construct(self) -> Optimizer:
        return Adam(self.params, lr=self.lr, weight_decay=self.weight_decay)
    
class LambdaSchedulerFactory(SchedulerFactory):

    def __init__(self, lr_power: float, optimizer, max_iter: int) -> None:
        super().__init__(optimizer)
        self.max_iter = max_iter
        self.lr_power = lr_power

    @override
    def construct(self) -> _LRScheduler:
        assert self.max_iter is not None, "max_iter necessary for poly LR scheduler"
        return LambdaLR(self.optimizer, lr_lambda=lambda cur_iter: (1 - cur_iter / self.max_iter) ** self.lr_power)
    
class StepLRSchedulerFactory(SchedulerFactory):

    def __init__(self, lr_decay_step: int, lr_decay_factor: float , optimizer) -> None:
        super().__init__(optimizer)
        self.lr_decay_step = lr_decay_step
        self.lr_decay_factor = lr_decay_factor

    @override
    def construct(self) -> _LRScheduler:
        return StepLR(self.optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay_factor)