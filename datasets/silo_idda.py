from typing import List
import torch
from numpy.typing import ArrayLike

from datasets.sl_idda import IDDADatasetSelfLearning
from datasets.ss_transforms import FDA, ToTensor
import datasets.ss_transforms as tr


class SiloIddaDataset(IDDADatasetSelfLearning):

    def __init__(self, 
                 root: str, 
                 list_samples: List[str], 
                 transform: tr.Compose = None, 
                 test_mode: bool = False, 
                 client_name: str = None):
        super().__init__(root, 
                         list_samples, 
                         transform, 
                         test_mode, 
                         client_name)
        self.style: ArrayLike = self.get_style().numpy()

    @property
    def style(self) -> ArrayLike:
        return self.style

    def get_style(self):
        to_tensor = ToTensor()
        amp_trg_sum = torch.zeros(size=(3, 1080, 961))
        for sample in self.list_samples:
            sample = sample.strip().split(".")[0]
            img = self.open_image(sample)
            img = to_tensor(img)
            fft_img = torch.fft.rfft2(img.clone(), dim=(-2, -1))
            amp_trg, pha_trg = FDA._extract_ampl_phase(fft_img.clone())
            amp_trg_sum = amp_trg_sum.add(amp_trg)
        return amp_trg_sum / len(len(self.list_samples))