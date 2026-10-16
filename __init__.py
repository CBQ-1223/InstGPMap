from .models import *
from .datasets import *
import os
import torch
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr
#删除tmp_gts_nusc_60x30.pkl
# os.system('rm -rf tmp_gts_nusc_60x30.pkl')