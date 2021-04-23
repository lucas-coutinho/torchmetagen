from torch.utils.data import Dataset
from .utils import FastaHandler
from pandas import DataFrame
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union, Iterable

__all__=['MetagenomicSequenceData']

"""
.. note::
    the structure of the dataset is designed as:
    |
    --train
    |      |
    |       --viral.fasta
    |       --nonvral.fasta
    |       ...
    --validation
    |      |
    |       --viral.fasta
    |       --nonvral.fasta
    |       ...      
           

"""
class MetagenomicSequenceData(Dataset):
    def __init__(self,
                data: DataFrame,
                transform: Optional[Callable]=None,
                target_transform: Optional[Callable]=None,
                labels: Optional[List[str]]=None 
                ) -> None:
        
        self._data=data
        self._labels=labels
        self.transform=transform
        self.target_transform=target_transform


        self.classes, self.classes_to_idx = labels, {cls_name:i for i, cls_name in enumerate(labels)}

    def __getitem__(self, idx):
        
        X=self._data.iloc[idx]['data']
        y=self.classes_to_idx[self._data.iloc[idx]['class']]

        if self.transform is not None:
            X=self.transform(X)
        
        if self.target_transform is not None:
            y=self.target_transform(y)


        return X, y

    def __len__(self):
        return len(self._data)
            
    def __iter__(self):
        return iter([self[i] for i in range(len(self))])

        


