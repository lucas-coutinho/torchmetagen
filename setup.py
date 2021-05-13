from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='torchmetagen',
    version='0.1.8',
    license='MIT',

    description='Torchmetagen is a python package for metagenomic sequence data processing and inference.',

    author='lucas-coutinho',
    url='http://github.com/lucas-coutinho/ViralClassificationAWS',

    keywords=['metagenomic', 'deep learning', 'viral classification'],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],

    packages=find_packages(),
    install_requires=[
                'numpy',
                'torch',
                'torchvision',
                'pandas',
                'biopython',
    ],

)