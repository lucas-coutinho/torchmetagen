import torch
from torch import nn
from .utils import *
from typing import Any

__all__ = ['DeepVirFinder', 'deepvirfinder']

model_urls = {
    'deepvirfinder': 'https://mosquitopretrainedmodels.s3-sa-east-1.amazonaws.com/small_random_w_dvf_0.001_truncated_500.pth'
    }

class DeepVirFinder(nn.Module):
  def __init__(self, M = 10, K = 8, N = 100, p=0.1):
    """
    DeepVirFinder constructor

    Args:
      M : number of feature maps in the convolutional layer
      K : dimensionality of the kernel 
      N : number of neurons in the Linear layer
      p : dropout's probability
    """
    super().__init__()

    self.K = K
    self.M = M

    self.conv1d_1 = nn.Conv1d(in_channels = 4, out_channels = M, kernel_size =  K)
    self.maxpool = GlobalMaxPooling1D('channesl_first')
    self.dropout = nn.Dropout(p = p)
        
  
    self.dense_1 = nn.Linear(M, N)

    self.dense_2 = nn.Linear(N, 1)

    
    self.relu1 =nn.ReLU()
    self.relu2 =nn.ReLU()

    self.sigmoid = nn.Sigmoid() 
        
  
  def forward(self, x) -> torch.Tensor:

    x = self.conv1d_1(x)
    x = self.relu1(x)

    x = self.maxpool(x)

    x = self.dropout(x)

    x = self.dense_1(x)
    x = self.dropout(x)
    x = self.relu2(x)
    x = self.dense_2(x)
    x = self.sigmoid(x)

    return x

class GlobalMaxPooling1D(nn.Module):
  def __init__(self, data_format='channels_last'):
    
    super(GlobalMaxPooling1D, self).__init__()
    self.data_format = data_format
    self.step_axis = 1 if self.data_format == 'channels_last' else 2

  def forward(self, input) -> torch.Tensor:

      return torch.max(input, axis=self.step_axis).values




def deepvirfinder(pretrained: bool = False, progress: bool = False, **kwargs: Any) -> DeepVirFinder:
    r""" Jie Ren's et al DeepVirFinder architecture from the
    `"Identifying viruses from metagenomic data by deep learning" <https://arxiv.org/abs/1806.07810>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on metavirome dataset
        progress (bool): If True, displays a progress bar of the download to stderr

    """
    model = DeepVirFinder(**kwargs)


    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['deepvirfinder'],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model