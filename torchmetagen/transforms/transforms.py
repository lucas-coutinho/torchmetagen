import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union, Iterable
import torch 

__all__=['ToKmer', 'ToW2V', 'ToTensor', 'ToOneHot', 'Chunking', 'ReverseComplement']

class KmerBase(object):
  """Base class to kmer-based representations.
  """
  def _k_mer_iter(self, contig, k, stride) -> Iterable:
    for idx in range(0, len(contig) -k + 1,stride): yield contig[idx:idx + k ] 
  
  def __repr__(self):
    return self.__class__.__name__ + '()'

class ToKmer(KmerBase):
  """Convert a genomic sequence from raw representation, ``str``, to ``numpy.ndarray`` kmer representation.
  Converts a str to kmer frequency representation. 

  Args:
      k       (int) :  length of the sliding window.
      stride  (int) :  the sliding window's step length.

  """
  def __init__(self, k: int = 3, stride: int = 1):
    self.k=k
    self.stride=stride

  def __call__(self, sample):
    return self._k_mer_encoding(sample)

  def _k_mer_encoding(self, contig):
    from itertools import product
    kmer_table = np.zeros((4, 4**(self.k-1)), dtype=np.float32)
    kmer_list =[''.join(x) for x in  product('ACTG', repeat= self.k-1 )]
    for mer in self._k_mer_iter(contig, self.k, self.stride):
      try:
        kmer_table['ACTG'.index(mer[0])][kmer_list.index(mer[1:])] += 1
      except:
        print(mer)

    return kmer_table/len(contig)

  def __repr__(self):
    return self.__class__.__name__ + '()'

class ToW2V(KmerBase):
  """Convert a genomic sequence from raw representation, ``str``, to ``numpy.ndarray`` word2vec representation.

  Converts a str to word2vec numeric representation. 

  Args:
      model       (str) : path where the gensim w2v model is saved.
      window_size (int) : length of the sliding window.
      stride      (int) : the sliding window's step length.
  """
  def __init__(self, model: str, window_size: int = 3, stride: int = 1):
    import gensim
    self.model=gensim.models.FastText.load(model)
    self.window_size=window_size
    self.stride=stride

  def __call__(self, sample: str):
    return np.array((self.stride/(len(sample) - self.window_size) + 1)*sum([self.model.wv[mer]*(1/(1 + (self.model.wv.vocab[mer].count/len(self.model.wv.vocab) if mer in self.model.wv.vocab.keys() else 0) )) for mer in self._k_mer_iter(sample, self.window_size, self.stride)]))

  def __repr__(self):
    return self.__class__.__name__ + '()'
        

class ToTensor(object):
  """Convert a ``numpy.ndarray`` to tensor.

  Converts, if is in one hot representation, a numpy.ndarray (W x C) in the range of [0, 1] to a torch.FloatTensor of shape (C x W). 
  Otherwise converts from numpy.ndarray to torch.FloatTensor regardeless the shape.

  Args:
      method (str) : Optional method used to encode the data. Options [``one-hot`` | ``None``]
  """
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

  def __repr__(self):
    return self.__class__.__name__ + '()'


class ToOneHot(object):
  """ Convert a genomic sequence from raw representation, ``str``, to ``numpy.ndarray`` One Hot representation.

  Args:
    alphabet (list of characters) : list of charactes representing the alphabet.

  Exemple:
    Given an alphabet, the representation is made as:
    >>>oneHot=ToOneHot("ACTG")
    >>>oneHot("AACCT")
    >>>array([[1., 0., 0., 0.],
              [1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 1., 0.]], dtype=float32)

  """
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

  def __repr__(self):
    return self.__class__.__name__ + '()'


class Chunking(object):
  """Randomly select a chunk from the original sequence.

  Selects a random chunk of the original sequence by sampling over a uniform distribution.
  If the chunk > ``len(sequence)`` then the sample's shape will be in the sequence's, otherwise the chunk's.
  """
  def __init__(self, chunk_size: int, seed: Optional[int]=None) -> None:
    """Chunking constructor

    Args:
        chunk_size (int) : length of the sampled chunk
        sedd       (int) : Optional seed for reproducibility 
    """
    self.chunk_size=chunk_size
    self.seed=seed

  def __call__(self, sample: Any) -> Any:
    np.random.seed(self.seed) if self.seed is not None else None
    init_chr=np.clip(np.random.randint(0, len(sample)) , 0, len(sample) - self.chunk_size)

    return sample[init_chr:(init_chr+self.chunk_size)]

  def __repr__(self):
    return self.__class__.__name__ + '()'

class ReverseComplement(object):
  """Reverse Complement of a genomic sequence.

  Complements a sequence of [ACTG]+ following a pre-defined dictionary
  and returns a tuple with forward and backward sequence.

  """
  def __call__(self,seq: str) -> Tuple[str, str]:
    complement_ = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return seq, "".join(complement_.get(base, base) for base in reversed(seq))
  def __repr__(self):
    return self.__class__.__name__ + '()'
