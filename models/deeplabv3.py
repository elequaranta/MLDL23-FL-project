from collections import OrderedDict
import copy
from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from .mobilenetv2 import MobileNetV2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead
from torchvision._internally_replaced_utils import load_state_dict_from_url


def _deeplabv3_mobilenetv2(
        backbone: MobileNetV2,
        num_classes: int,
) -> DeepLabV3:
    backbone = backbone.features

    out_pos = len(backbone) - 1
    out_inplanes = backbone[out_pos][0].out_channels
    return_layers = {str(out_pos): "out"}

    backbone = create_feature_extractor(backbone, return_layers)
    classifier = DeepLabHead(out_inplanes, num_classes)

    return DeepLabV3(backbone, classifier)


def deeplabv3_mobilenetv2(
        num_classes: int = 21,
        in_channels: int = 3
) -> DeepLabV3:
    width_mult = 1
    backbone = MobileNetV2(width_mult=width_mult, in_channels=in_channels)
    model_urls = {
        0.5: 'https://github.com/d-li14/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_0.5-eaa6f9ad.pth',
        1.: 'https://github.com/d-li14/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_1.0-0c6065bc.pth'}
    if torch.cuda.is_available():
        state_dict = load_state_dict_from_url(model_urls[width_mult], progress=True)
    else:
        state_dict = load_state_dict_from_url(model_urls[width_mult], progress=True, map_location=torch.device('cpu'))
    state_dict_updated = state_dict.copy()
    for k, v in state_dict.items():
        if 'features' not in k and 'classifier' not in k:
            state_dict_updated[k.replace('conv', 'features.18')] = v
            del state_dict_updated[k]

    if in_channels == 4:
        aux = torch.zeros((32, 4, 3, 3))
        aux[:, 0, :, :] = copy.deepcopy(state_dict_updated['features.0.0.weight'][:, 0, :, :])
        aux[:, 1, :, :] = copy.deepcopy(state_dict_updated['features.0.0.weight'][:, 1, :, :])
        aux[:, 2, :, :] = copy.deepcopy(state_dict_updated['features.0.0.weight'][:, 2, :, :])
        aux[:, 3, :, :] = copy.deepcopy(state_dict_updated['features.0.0.weight'][:, 2, :, :])
        state_dict_updated['features.0.0.weight'] = aux
    backbone.load_state_dict(state_dict_updated, strict=False)

    model = _deeplabv3_mobilenetv2(backbone, num_classes)
    model.task = 'segmentation'

    return model


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(OrderedDict([
            ("aspp", ASPP(in_channels, [12, 24, 36])),
            ("5conv", nn.Conv2d(256, 256, 3, padding=1, bias=False)),
            ("5bn", nn.BatchNorm2d(256)),
            ("5relu",nn.ReLU()),
            ("6conv", nn.Conv2d(256, num_classes, 1))
            ])
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = OrderedDict([
            ("2conv", nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)),
            ("2bn", nn.BatchNorm2d(out_channels)),
            ("2relu",nn.ReLU()),
        ])
        super().__init__(modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(OrderedDict([
            ("1pool", nn.AdaptiveAvgPool2d(1)),
            ("3conv", nn.Conv2d(in_channels, out_channels, 1, bias=False)),
            ("3bn", nn.BatchNorm2d(out_channels)),
            ("3relu", nn.ReLU())])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(OrderedDict([
                ("1conv", nn.Conv2d(in_channels, out_channels, 1, bias=False)), 
                ("1bn", nn.BatchNorm2d(out_channels)),
                ("1relu", nn.ReLU())
                ]))
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(OrderedDict([
            ("4conv", nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False)),
            ("4bn",nn.BatchNorm2d(out_channels)),
            ("4relu",nn.ReLU()),
            ("1drop",nn.Dropout(0.5))])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)