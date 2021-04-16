import time
import random
import copy
from sklearn import metrics as mtr
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 
from torch import nn
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union


def train_model(model: nn.Module, criterion: nn.Module, optimizer: nn.Module, scheduler: nn.Module, num_epochs: int = 25) -> Tuple[List, List]:
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  loss_per_batch = {'train':[],'val':[]}
  loss_per_epoch = {'train':[],'val':[]}
  for epoch in range(num_epochs):
      print('Epoch {}/{}'.format(epoch, num_epochs - 1))
      print('-' * 10)

      # Each epoch has a training and validation phase
      for phase in ['train', 'val']:
          if phase == 'train':
              model.train()  # Set model to training mode
          else:
              model.eval()   # Set model to evaluate mode

          running_loss = 0.0
          running_corrects = 0

          # Iterate over data.
          for inputs, labels in dataloaders[phase]:
              #forwards, backwards = inputs
              forwards = inputs
              forwards  = forwards.to(device)
              #backwards = backwards.to(device)

              labels    = labels.to(device)

              # zero the parameter gradients
              optimizer.zero_grad()

              # forward
              # track history if only in train
              with torch.set_grad_enabled(phase == 'train'):
                  outputs_fw = model(forwards)
                  #outputs_bk = model(backwards)
                  # print(outputs_fw.shape, outputs_bk.shape, labels.unsqueeze(1).shape)
                  #outmean = (outputs_fw + outputs_bk)/2.
                  outmean = outputs_fw
                  # _, preds = torch.max(outmean, 1)
                  preds = torch.round(outmean)
                  loss = criterion(outmean, labels.unsqueeze(1).to(torch.float32))
              
                  # backward + optimize only if in training phase
                  if phase == 'train':
                      loss.backward()
                      optimizer.step()



              # statistics
              running_loss += loss.item() * forwards.size(0)

              loss_per_batch[phase].append(loss.item() * forwards.size(0))
              running_corrects += torch.sum(preds == labels.unsqueeze(1).data)
          # if phase == 'train':
          #     scheduler.step()

          epoch_loss = running_loss / dataset_sizes[phase]
  
          epoch_acc = running_corrects.double() / dataset_sizes[phase]
          loss_per_epoch[phase].append(epoch_acc.item())
          print('{} Loss: {:.4f} Acc: {:.4f}'.format(
              phase, epoch_loss, epoch_acc))

          # deep copy the model
          if phase == 'val' and epoch_acc > best_acc:
              best_acc = epoch_acc
              best_model_wts = copy.deepcopy(model.state_dict())

      print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return loss_per_epoch, loss_per_batch

#import pandas_ml

def evaluate(model: nn.Module, data_loader: DataLoader, report: bool = True) -> Union[Tuple[List,List], Dict[str, float]]:
  acc = 0.0

  model.eval() 

  y_hat = []
  y_true = []
  score = []
  with torch.no_grad():
    for inputs, labels in data_loader:
        #forwards, backwards = inputs
        forwards = inputs
        
        labels = labels.to(device)
        forwards  = forwards.to(device)
       # backwards = backwards.to(device)


        outputs_fw = model(forwards)
       # outputs_bw = model(backwards)
        # print(outputs_fw)
        # print(outputs_bw)
        mean = outputs_fw
        #mean = (outputs_fw + outputs_bw)/2.

        score.append(mean.squeeze(1))
      # _, preds = torch.max((outputs_fw + outputs_bw)/2 , 1)
        preds = torch.round(mean)
        y_hat.append(preds.squeeze(1))
        y_true.append(labels.squeeze(0))
        acc += torch.sum(preds == labels.unsqueeze(1).data)
      
  

  
  print('Acc test_set: {:.4f}'.format(acc.double()/len(data_loader.dataset)))
  Y_true = [label for joint in y_true for label in joint.cpu().numpy()] 
  #print(Y_true)
  Y_hat =  [label for joint in (y_hat if report else score) for label in joint.cpu().numpy()]
  # print(Y_hat[1], Y_true[1])
  if report:
    return  mtr.classification_report(Y_true, Y_hat, output_dict=True)
  else:
    return Y_hat, Y_true


def genDataLoader(dataset_metagenomic: Dict[str, torch.utils.data.Dataset], batch_size: Dict[str, int]) -> DataLoader:
  class_count = [dataset_metagenomic['train'].lens['nonviral'], dataset_metagenomic['train'].lens['viral']]

  weights = [len(dataset_metagenomic['train'])/class_count[int(label)] for _, label in dataset_metagenomic['train']]
  weights = torch.Tensor(weights)

  sampler = {}
  sampler['train'] = torch.utils.data.sampler.WeightedRandomSampler(weights=weights.double(), num_samples=len(dataset_metagenomic['train']))
  sampler['val']  = None
  dataloaders = {x : DataLoader(dataset_metagenomic[x], batch_size=batch_size[x],
                                              sampler=sampler[x] ) for x in ['train', 'val']}
  return dataloaders