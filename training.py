import models
import utils
from torchvision import transforms 
import transforms as metatr

def main():
    
    data_transforms={}
        
    root_dir='./Dataset_v1_2'
    data_transforms['train']=transforms.Compose([                                 
                                        metatr.Chunking(500),
                                        metatr.ToOneHot(),
                                        metatr.ToTensor('one-hot')

    ])
    data_transforms['val']=transforms.Compose([
                                        metatr.Chunking(500),
                                        metatr.ToOneHot(),
                                        metatr.ToTensor('one-hot')
    ])



    dataset_metagenomic = {x: MetagenomicSequenceqData(root_dir=root_dir, 
                                                format='.fasta',
                                                labels=['viral', 'nonviral'], 
                                                faster=True, 
                                                transform=data_transforms[x],
                                                target_transform=np.float,
                                                                                                    for x in ['train', 'val']}

    dataset_sizes = {x : len(dataset_metagenomic[x]) for x in ['train', 'val']}


if __name__ == '__main__':

    main()
