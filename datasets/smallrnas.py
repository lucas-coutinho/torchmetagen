from torch.utils.data import Dataset
from .utils import *


class MetagenomicSequenceqData(Dataset):
  def __init__(self,
               root_dir:str, 
               format:str,
               complement:bool=False,
               name:bool=False,
               faster:bool=False,
               evenFaster:bool=False, 
               transform: Optional[Callable]=None,
               target_transform: Optional[Callable]=None,
               labels: Optional[List[str]]=None 
              ) -> None:
    super(Dataset, self).__init__()

    self.transform=transform
    self.target_transform=target_transform
    self.root_dir=root_dir
    self.format=format
    self.faster=faster

    labels.sort() if labels is not None else None
    classes, classes_to_idx = self.__find_classes(root_dir) if labels is None else (labels, {cls_name:i for i, cls_name in enumerate(labels)})

    self.lens={}
    self.classes=classes
    self.classes_to_idx=classes_to_idx
    self.name=name
    self.dict_offset={}
    self.evenFaster=evenFaster
    


    count=0
    for class_ in self.classes:
      with open(os.path.join(self.root_dir, class_ + self.format ), 'rb') as origin_file:
          self.lens[class_] = 0
          for line in origin_file:
            if b'>' in line: 
              self.lens[class_]+=1
              self.dict_offset[count] = origin_file.tell()
              count+=1

    self.len = sum([values for keys, values in self.lens.items()])



    if evenFaster:
      self.data_dict={}
      count=-1
      for class_ in self.classes:
        with open(os.path.join(self.root_dir, class_ + self.format )) as origin_file:
            for line in origin_file:
              if '>' in line: 
                count+=1
                self.data_dict[count] = ""
              else:
                self.data_dict[count]+=line

          
    elif faster:
      self.file = {}
      for label in labels:
        try:
          self.file[label] =  open(os.path.join(self.root_dir, label + self.format ))
        except:
          raise RuntimeError("Unable to open this file")
            
  def __find_classes(self, dir: str) -> Tuple[Any, Any]:
    classes = [d.name.replace(self.format, '') for d in os.scandir(dir) if d.is_file()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
  
  def __del__(self) -> None:
    if hasattr(self, 'file'):
      for label in self.classes:
        self.file[label].close()
    else:
      pass

  def __len__(self) -> int:
    return self.len

  def __iter__(self):
    return iter([self[i] for i in range(len(self))])

  def __getitem__(self, idx) -> Union[Tuple[Any, Any], Tuple[Any, Any, Any]]:
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    if idx < 0 : idx = len(self) + idx 
  
    assert idx < len(self), f"{idx} index exceeds bounds"

    acc_len=0
    for label in self.classes:
      acc_len = acc_len + self.lens[label]
      if idx < acc_len:
        sel_label = label
        break
  
    counter=0    
    found=False
    data_point=""
    
    if self.evenFaster:
      data_point=self.data_dict[idx]
    elif self.faster:
      origin_file = self.file[sel_label]
      origin_file.seek(self.dict_offset[idx])
      for line in origin_file:
        if '>' in line: break
        data_point+=line
    else:
      with open(os.path.join(self.root_dir, sel_label + self.format)) as origin_file:
        origin_file.seek(self.dict_offset[idx])
        for line in origin_file:
          if '>' in line: break
          data_point+=line


    # with open(os.path.join(self.root_dir, sel_label + self.format)) as origin_file:
    #   for line in origin_file:
    #     if '>' in line:
    #       counter+=1        
    #       if found: break  

    #     if found:
    #       data_point+=line
      
    #     if counter == (idx % self.lens[sel_label]) + 1 and not found:
    #         name = line
    #         found=True
          


    data_point = data_point.replace('\n', '')
    if self.transform:
      data_point = self.transform(data_point)

    target=self.classes_to_idx[sel_label]
    if self.target_transform:
      target=self.target_transform(target)
    if self.name:
      return  (name.replace('\n', ''), 
               data_point,
               target
               )
    else:
      return data_point,self.classes_to_idx[sel_label]
    
         
          
  