
from Bio import SeqIO
import os 
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union, Iterable

class FastaHandler:

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
    def __init__(self, split: Dict[str, float]= {'train': 0.7, 'val': 0.3}):
        
        check=sum([value for key, value in split.items()])
        if check != 1.:
            raise "Expected to train's and val's proportions sum up to 1, got {}".format(check)

        self._split_factor=split

    @property
    def split(self) -> Dict[str, float]:
        return self._split_factor


    def __call__(self, fasta_data: FastaHandler) -> List:
        from sklearn.model_selection import train_test_split

        X_train, X_val, _, _= train_test_split([i for i in range(len(fasta_data))],
                                                train_size= self.split['train'], 
                                                test_size=self.split['test']
                                               )

        return fasta_data.data[X_train], fasta_data.data[X_val]