# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Loading data and traing a viral classifier on metagenomic sequences of mosquitoes
# 
# ## Import dependencies 

# %%
import pandas as pd 
import numpy as np 
from datasets.utils import FastaHandler, DatasetSplit, InflateDataset
from datasets import metagenomicdataset as meta
from transforms import *
from torchvision import transforms as tf
import torch

from models import DeepVirFinder, deepvirfinder
from utils import *

# %% [markdown]
# ## Check for GPU devices

# %%
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%
path_to_file = '~/Desktop/ViralClassificationAWS/datasets/Dataset_v1_2'

viral = FastaHandler(path_to_file, 'viral.fasta',)
nonviral= FastaHandler(path_to_file, 'nonviral.fasta',)


# %%
splitter = DatasetSplit({'train':0.7,'val':0.3 })

viral_train, viral_test= splitter(viral)
nonviral_train, nonviral_test= splitter(nonviral)


# %%

inflate=InflateDataset(method='truncated', tol=0.5, chunk_size=500)

viral_train_inflated = inflate(viral_train)
viral_test_inflated = inflate(viral_test) #viral_test #


nonviral_train_inflated =  inflate(nonviral_train)
nonviral_test_inflated = inflate(nonviral_test) #nonviral_test#


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

dataset={'train': dataset_train, 'val': dataset_test}
dataset_sizes = {'train':len(dataset_train), 'val':len(dataset_test)}

# %%
dataloaders = genDataLoader(dataset, {'train':250, 'val':250})


# %%
model_torch = deepvirfinder(pretrained=True, progress=True, M = 1000, K = 10, N = 1000)


# %%

#device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model_torch.parameters(), lr = 1e-4)
criterion = torch.nn.BCELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,4)
per_epoch, per_batch = train_model(model_torch.to(device), criterion, optimizer, 
                      scheduler, dataloaders, device, dataset_sizes, num_epochs=1)


# %%


print(pd.DataFrame(evaluate(model_torch.to(device), dataloaders['val'], device)))


# %%
# torch.save(checkpoint_3.state_dict(), './first_checkpoint.pth')

"""
# %%
cp aliagned_t18.fasta  aliagned_t18_teste.fasta
grep '>' aliagned_t18_teste.fasta  > temp_headers
while read p
do 
 h_query=`echo $p| cut -f1 -d " "| cut -f2 -d ">"`
 h_subject=`grep $h_query -A5 diamondx_t12_f0.out| grep ">"|tr ">" "|"`
 sed -i "s/$h_query/$h_query $h_subject/g" aliagned_t18_teste.fasta
done < temp_headers

fasta_formatter -i aliagned_t18_teste.fasta  -o aliagned_t18_teste_linear.fasta
cp aliagned_t18.fasta  aliagned_t18_teste.fasta
grep '>' aliagned_t18_teste.fasta  > temp_headers
while read p
do 
 h_query=`echo $p| cut -f1 -d " "| cut -f2 -d ">"`
 h_subject=`grep $h_query -A5 diamondx_t12_f0.out| grep ">"|tr ">" "|"`
 sed -i "s/$h_query/$h_query $h_subject/g" aliagned_t18_teste.fasta
done < temp_headers

fasta_formatter -i aliagned_t18_teste.fasta  -o aliagned_t18_teste_linear.fasta

grep -A1 -i "virus" aliagned_t18_teste_linear.fasta > viral_blastx_hits.fasta

grep '>' alia
grep -A1 -i "virus" aliagned_t18_teste_linear.fasta > viral_blastx_hits.fasta

grep '>' alia

"""
