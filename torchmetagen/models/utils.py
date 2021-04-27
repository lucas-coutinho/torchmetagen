import keras
from keras.layers.convolutional import Conv1D
import torch
import numpy as np

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url



def keras_to_pyt(km, pm=None):
	r"""
	Convert pretrained keras models to torch

	Args:
		km                     : pretrained keras model
		[Optional] pm          : torch model 

	Returns:
		Union[nn.Module, Dict] : transloaded torch model OR state dict
	"""
	weight_dict = {}
	for layer in km.layers:
		if (type(layer) is Conv1D):
			weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], ( 2, 1, 0 ))
			weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
		elif type(layer) is keras.layers.Dense:
			weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
			weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
	if pm:
		pyt_state_dict = pm.state_dict()
		for key in pyt_state_dict.keys():
			pyt_state_dict[key] = torch.from_numpy(weight_dict[key])
		pm.load_state_dict(pyt_state_dict)
		return pm
	return weight_dict
