import argparse
import os
import logging
import numpy as np 
import pandas as pd 
import torch
from torchvision import transforms as tf

from datasets import metagenomicdataset as meta
from datasets.utils import FastaHandler, DatasetSplit, InflateDataset
from models import DeepVirFinder, deepvirfinder
from transforms import *
from utils import *


def main():
    # %% [markdown]
    # ## Check for GPU devices
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                        level=logging.INFO,
                        filename=args.log)
    # %%
    if args.gpu:
        device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            logging.warning('CUDA device not found, migrating to CPU')
    else:
        device='cpu'

    logging.info('Device name: %s' % device)

    # %%
    path_to_file = args.dataset

    viral = FastaHandler(path_to_file, 'viral.fasta',)
    nonviral= FastaHandler(path_to_file, 'nonviral.fasta',)


    # %%
    splitter = DatasetSplit({'train':args.train_split,
                             'val'  :args.test_split 
                             })

    viral_train, viral_test= splitter(viral)
    nonviral_train, nonviral_test= splitter(nonviral)


    # %%

    inflate=InflateDataset(method='truncated', 
                           tol=0.5, 
                           chunk_size=500)

    viral_train_inflated = inflate(viral_train) if args.inflate_train else viral_train
    viral_test_inflated = inflate(viral_test)   if args.inflate_test  else viral_test


    nonviral_train_inflated = inflate(nonviral_train) if args.inflate_train else nonviral_train
    nonviral_test_inflated  = inflate(nonviral_test)  if args.inflate_test  else nonviral_test


    # %%
    transforms_train=tf.Compose([
        ReverseComplement(),
        ToOneHot(['G','T', 'C', 'A']),
        ToTensor('one-hot')
    ])

    transforms_test=tf.Compose([
        ReverseComplement(),
        ToOneHot(['G','T', 'C', 'A']),
        ToTensor('one-hot')
    ])


    dataset_train= meta.MetagenomicSequenceData(pd.DataFrame({"data":np.concatenate((nonviral_train_inflated, viral_train_inflated)),
                                                            "class":np.concatenate((np.repeat("nonviral",len(nonviral_train_inflated)),
                                                                                    np.repeat("viral",len(viral_train_inflated))))}),
                                                        labels=['nonviral', 'viral'], transform=transforms_train)

    dataset_test= meta.MetagenomicSequenceData(pd.DataFrame({"data":np.concatenate((nonviral_test_inflated, viral_test_inflated)),
                                                            "class":np.concatenate((np.repeat("nonviral",len(nonviral_test_inflated)),
                                                                                    np.repeat("viral",len(viral_test_inflated))))}),
                                                        labels=['nonviral', 'viral'], transform=transforms_test)

    dataset={'train': dataset_train, 
             'val': dataset_test
             }
    dataset_sizes = {'train':len(dataset_train), 
                     'val':len(dataset_test)
                     }

    # %%
    dataloaders = genDataLoader(dataset, {'train':args.batch_size_train, 
                                          'val':args.batch_size_val
                                          })


    # %%
    model_torch = deepvirfinder(pretrained=args.pretrained, 
                                progress=True,
                                M = 1000, 
                                K = 10, 
                                N = 1000)


    # %%

    #device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model_torch.parameters(), lr = args.lr)
    criterion = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,4)
    per_epoch, per_batch = train_model(model_torch.to(device),
                                       criterion, 
                                       optimizer, 
                                       scheduler, 
                                       dataloaders, 
                                       device, 
                                       dataset_sizes, 
                                       num_epochs=args.epochs)


    # %%


    logging.info(evaluate(model_torch.to(device),
                          dataloaders['val'],
                          device))


    if args.checkpoint is not None:
        torch.save(model_torch.state_dict(), args.checkpoint)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size_train', type=int, default=250)
    parser.add_argument('--batch_size_val', type=int, default=250)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', nargs='?', const=True, help='GPU enable')
    parser.add_argument('--pretrained', nargs='?', const=True, help='Load model checkpoint')
    parser.add_argument('--inflate_train', nargs='?', const=True, help='Inflate the dataset by a predefined method')
    parser.add_argument('--inflate_test', nargs='?', const=True, help='Inflate the dataset by a predefined method')
    parser.add_argument('--train_split', type=float, default= 0.7, help='training set proportion')
    parser.add_argument('--test_split', type=float, default= 0.3, help='testing set proportion')
    parser.add_argument('--dataset', type=str, default='.dataset', help="path to dataset")    
    parser.add_argument('--log', type=str, default='stat.log', help='name of the logging file')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help="model's checkpoint file name")    
    args = parser.parse_args()

    main()