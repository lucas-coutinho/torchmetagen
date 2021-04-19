import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import torch 

class KmerBase(object):
  def __k_mer_iter(self, contig, k, stride=1):
    for idx in range(0, len(contig) -k + 1,stride): yield contig[idx:idx + k ] 

class ToKmer(KmerBase):
  def __call__(self, sample, **kwargs):
    return self.__k_mer_encoding(sample, **kwargs)

  def __k_mer_encoding(self, contig, k=3):
    from itertools import product
    kmer_table = np.zeros((4, 4**(k-1)), dtype=np.float32)
    kmer_list =[''.join(x) for x in  product('ACTG', repeat= k-1 )]
    for mer in self.__k_mer_iter(contig, k):
      try:
        kmer_table['ACTG'.index(mer[0])][kmer_list.index(mer[1:])] += 1
      except:
        print(mer)

    return [np.transpose(kmer_table/len(contig))]   

class ToW2V(KmerBase):
  def __init__(self, model: str):
    import gensim
    self.model=gensim.models.FastText.load(model)

  def __call__(self, sample: str, window_size: int, stride: int):
    return np.array((stride/(len(sample) - window_size) + 1)*sum([self.model.wv[mer]*(1/(1 + (self.model.wv.vocab[mer].count/len(self.model.wv.vocab) if mer in self.model.wv.vocab.keys() else 0) )) for mer in __k_mer_iter(sample, window_size, stride)]))

        

class ToTensor(object):
  def __init__(self, method: Optional[str]=None):
    self._method=method

  @property
  def method(self) -> str:
      return self._method

  def __call__(self, sample: Union[Any, Tuple[Any, Any]]) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    if self.method=='one-hot':
      return tuple(map(lambda x: torch.from_numpy(x).transpose(1,0), sample)) if isinstance(sample, tuple) else torch.from_numpy(sample).transpose(1,0)

    else:
      return tuple(map(torch.from_numpy, sample)) if isinstance(sample, tuple) else torch.from_numpy(sample)



class ToOneHot(object):
  def __init__(self, alphabet: List[str]):
    super().__init__()
    self.alphabet=alphabet

  def __toOneHot(self, seq):
    import numpy as np
    oneHot = [] 
    oneHot_map = dict(zip(self.alphabet, np.eye(len(self.alphabet))))
    for nt in seq:
        try:
            oneHot.append(oneHot_map[nt])
        except KeyError:
            if nt == 'N':
                oneHot.append([.25, .25, .25, .25])
            else:
                oneHot.append([0., 0., 0., 0.])

    return np.array(oneHot, dtype=np.float32)

  def __call__(self, seq: Union[Any, Tuple[Any, Any]]) -> Union[Any, Tuple[Any, Any]]:
    if isinstance(seq, tuple):
      return tuple(map(self.__toOneHot, seq))
    else:
      return self.__toOneHot(seq)


class Chunking(object):
  def __init__(self, chunk_size: int, seed: Optional[int]=None) -> None:
    self.chunk_size=chunk_size
    self.seed=seed

  def __call__(self, sample: Any) -> Any:
    np.random.seed(self.seed) if self.seed is not None else None
    init_chr=np.clip(np.random.randint(0, len(sample)) , 0, len(sample) - self.chunk_size)

    return sample[init_chr:(init_chr+self.chunk_size)]


class ReverseComplement(object):
  def __call__(self,seq: str) -> Tuple[str, str]:
    complement_ = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return seq, "".join(complement_.get(base, base) for base in reversed(seq))
