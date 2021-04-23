
from Bio import SeqIO
import os 
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union, Iterable

class FastaHandler:
    """Fasta Handler for formating fasta-like files into lists.

    Formats fasta-like files into list of sequences. 

    Args:
        root_dir            (str): path where the fastas are saved
        file_name           (str): name of the fasta-like file.
        with_name (optional bool): adds the header content to each sequence.
            Every read or contig in fasta-like files contains meta data which are indexed by `>` symbol.
    Raises:
        IOError: Expect any of .fasta, .fa, .faa, .fna or .fsa file extension.

    
    """
    def __init__(self, 
                 root_dir: str,
                 file_name: str, 
                 with_name: bool=False,
                 ):

        self.root_dir = root_dir = os.path.expanduser(root_dir)
        self._file_name=file_name
        self.with_name=with_name

        if file_name.endswith(('.fasta', '.fa', '.faa', '.fna', '.fsa')):
            parse = file_name.split(sep='.')
            self._class_name = parse[0]
            self._extension = parse[-1]

        else:
            raise IOError("Expecting fasta-like files, got {}".format(file_name.split(sep='.')[-1]))

        self.data= self.load()

    @property  
    def class_name(self) -> str:
        return self._class_name
    
    @property  
    def file_name(self) -> str:
        return self._file_name
    
    @property  
    def extension(self) -> str:
        return self._extension

    def __add__(self, other):
        return  self.data + other.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str]:

        return self.data[idx]

    def __iter__(self) -> Iterable[Tuple[str,str]]:
        return next(self.data)

    def load(self) -> List[Tuple[str, str]]:
        data=[]
        with open(os.path.join(self.root_dir, self.file_name), 'r') as handle:
            for record in SeqIO.parse(handle, self.extension):
                if self.with_name:
                    data.append((record.id, str(record.seq)))
                else:
                    data.append(str(record.seq))
        return data
    

class DatasetSplit(object):
    """Split the dataset into train and validation sets.

    Args:
        split (dict of floats indexed by str): split proportion of each set.

    Raises:
        AssertionError: Expectes to train's and val's proportions sum up to 1.
    """
    def __init__(self, split: Dict[str, float]= {'train': 0.7, 'val': 0.3}):
        
        check=sum([value for key, value in split.items()])

        assert check == 1., "Expected to train's and val's proportions sum up to 1, got {}".format(check)

        self._split_factor=split

    @property
    def split(self) -> Dict[str, float]:
        return self._split_factor


    def __call__(self, fasta_data: FastaHandler) -> List:
        from sklearn.model_selection import train_test_split
        import numpy as np
        X_train, X_val= train_test_split([i for i in range(len(fasta_data))],
                                                train_size= self.split['train'], 
                                                test_size=self.split['val']
                                               )

        return np.array(fasta_data.data)[X_train], np.array(fasta_data.data)[X_val]


class InflateDataset(object):
    """Inflate the dataset into regularized contigs sizes.

    Inflates the raw, non-normalized length dataset points into nomarlized length points. The length nomarlization
    helps the composition of baches higher than 1, improving training time and gradient estimation. 

    Args:
        chunk_size (int): windows size to crop.
        method     (str): method used to crop. [`truncated` | `sliding` | `padding` ]
    
    Raises:
        AssertionError: expects one of the methods [`truncated` | `sliding` | `padding` ], got another.

    """
    def __init__(self, chunk_size: int, method: str, tol: float):
        
        assert method in ['truncated', 'sliding', 'padding'], "got {}, expect truncated, sliding or padding.".format(method)

        self.chunk_size=chunk_size
        self.method=method
        self.tol=tol

    def __call__(self, dataset: List) -> List:
        import numpy as np
        inflated_dataset=[]

        for contig in dataset:

            n_times = int(np.floor(len(contig)/self.chunk_size))
            trunc_error = len(contig) - n_times*self.chunk_size

            if self.method == 'truncated':
            
                inflated_dataset+= [contig[i*self.chunk_size:self.chunk_size*(i + 1)] for i in range(n_times)] 

            if self.method == 'sliding':
                
                partial_contigs = [contig[i*self.chunk_size:self.chunk_size*(i + 1)] for i in range(n_times)]
                partial_contigs.append(contig[self.chunk_size*(n_times - 1) + trunc_error:])
                
                inflated_dataset+= partial_contigs

            if self.method == 'padding':

                partial_contigs = [contig[i*self.chunk_size:self.chunk_size*(i + 1)] for i in range(n_times)]
                if trunc_error/self.chunk_size >= self.tol :partial_contigs.append(contig[n_times*self.chunk_size:] + 'S'*(self.chunk_size - trunc_error)) 

                inflated_dataset+=partial_contigs
        
        return inflated_dataset


