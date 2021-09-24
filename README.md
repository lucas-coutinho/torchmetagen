# Torchmetagen Package
Amazon SageMaker project for viral classification in metagenomic sequence data of mosquitoes. 

# torchmetagen

The torchmetagen package consists of dataset formatter, model architectures and common sequence transformations for metagenomic sequences classification.

# Installation

To fulfill properly the installation one need to install dependent packages:

```  
pytorch>=1.5
torchvision>=0.7
pandas>=1.22
biopython==1.78
```
Can install via pip install:
```shell
pip install torchmetagen
```



# Dataset format

Torchmetagen supports the following fasta-like files: 
-   .fasta (default)
-   .fa 
-   .faa 
-   .fna
-   .fsa
