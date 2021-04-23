import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from typing import Any

__all__ = ['ViraMINER', 'viraminer']

model_urls = {
    'viraminer': 'https://mosquitopretrainedmodels.s3-sa-east-1.amazonaws.com/small_random_w_viraminer_0.001_padding_1000.pth'
    }

class ViraMINER(nn.Module):
  def __init__(self, M = 10, K = 8, N = 100, p=0.3):
    """
    ViraMINER constructor

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
    self.maxpool = GlobalMaxPooling1D('channels_first')
    self.dropout = nn.Dropout(p = p)
    self.avgpool = GlobalAvgPooling1D('channels_first')
  
    self.dense_1 = nn.Linear(M, N)

    self.dense_2 = nn.Linear(N, 1)

    
    self.relu1 = [nn.ReLU(), nn.ReLU()]
    self.relu2 = [nn.ReLU(), nn.ReLU()]
    self.sigmoid = nn.Sigmoid()

  def forward(self, x) -> torch.Tensor:
    res = []
    for i in range(2):
      y = self.conv1d_1(x)
      y = self.relu1[i](y)
    
      y = self.maxpool(y) if i == 1 else self.avgpool(y).squeeze(2)

      y = self.dropout(y)
      y = self.dense_1(y)
      y = self.dropout(y)
      y = self.relu2[i](y)
      res.append(y)
  
      

    x = self.dense_2(res[0] + res[1])

    x = self.sigmoid(x)

    return x

class GlobalMaxPooling1D(nn.Module):
  def __init__(self, data_format='channels_last'):
      super(GlobalMaxPooling1D, self).__init__()
      self.data_format = data_format
      self.step_axis = 1 if self.data_format == 'channels_last' else 2

  def forward(self, input) -> torch.Tensor:

      return torch.max(input, axis=self.step_axis)[0]

class GlobalAvgPooling1D(nn.Module):
  def __init__(self, data_format='channels_last'):
      super(GlobalAvgPooling1D, self).__init__()
      self.data_format = data_format
      self.step_axis = 1 if self.data_format == 'channels_last' else 2

  def forward(self, input)-> torch.Tensor:

      return torch.mean(input, dim=self.step_axis, keepdim=True)


def viraminer(pretrained: bool = False, progress: bool = False, **kwargs:Any) -> ViraMINER:
    r""" ViramMiner architecture from the
    `"ViraMiner: Deep Learning on Raw DNA Sequences for Identifying Viral Genomes in Human Samples" <https://www.biorxiv.org/content/10.1101/602656v2>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on metavirome dataset
        progress (bool): If True, displays a progress bar of the download to stderr

    """
    model = ViraMINER(**kwargs)


    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['viraminer'],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model
