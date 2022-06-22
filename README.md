# CDNet

This repo is the official implementation of ["Deep-learning-enabled crack detection and analysis in commercial lithium-ion battery cathodes"](https://onlinelibrary.wiley.com/doi/10.1002/adfm.202203070). It currently includes code for the following tasks:

## Abstract

In Li-ion batteries, the mechanical degradation initiated by micro cracks is one of the bottlenecks for enhancing the performance. Quantifying the crack formation and evolution in complex composite electrodes can provide important insights into electrochemical behaviors under prolonged and/or aggressive cycling. However, observation and interpretation of the complicated crack patterns in battery electrodes through imaging experiments are often time-consuming, labor intensive, and subjective. Herein, we develop a deep learning-based approach to extract the crack patterns from nanoscale hard X-ray holo-tomography data of a commercial 18650-type battery cathodes. We demonstrate efficient and effective quantification of the damage heterogeneity with automation and statistical significance. We further associate the crack characteristics with the active particles’ packing densities and discuss a potentially viable architectural design for suppressing the structural degradation in an industry-relevant battery configuration.

<img src="https://github.com/YijinLiu-Lab/CDNet/blob/main/Figure/fig3.png" width="900px">


## Workflow 
The workflow is shown in the figure below:

<img src="https://github.com/YijinLiu-Lab/CDNet/blob/main/Figure/fig2.png" width="900px">


## Network structure 
The network structure is shown in the figure below:

<img src="https://github.com/YijinLiu-Lab/CDNet/blob/main/Figure/fig4.png" width="900px">



## Getting Started

![train.py](https://github.com/YijinLiu-Lab/CDNet/blob/main/train.py) Shows how to train network on your own dataset. 

![model.py](https://github.com/YijinLiu-Lab/CDNet/blob/main/model.py) These files contain the main network implementation.

![utils.py](https://github.com/YijinLiu-Lab/CDNet/blob/main/utils.py) Different pre-processing steps to prepare the input data.

![evaluation.py](https://github.com/YijinLiu-Lab/CDNet/blob/main/evaluation.py) Shows how to evaluation network on your own dataset. 


## Installation
1.Clone this repository via git clone https://github.com/YijinLiu-Lab/CDNet.git
2.Install dependencies and current repo
```
pip install -r requirements.txt
```


## Training on your own dataset

Train a new model starting from your own dataset:
```
python3 train.py train --train_dataroot=/path/to/data/train/ --target_dataroot=/path/to/data/target/ --trained_model_path=/path/to/your/model/
```
evaluat a new model starting from your own dataset:
```
python3 evaluation.py train --evaluation_dataroot=/path/to/data/evaluation/ --model_dataroot=/path/to/your/model/ --out_dataroot=/path/to/your/out/
```

## Citation 
Use this bibtex to cite this repository:
```
@article{jiang_lib_segmentation2020,
  title={Deep-learning-enabled crack detection and analysis in commercial lithium-ion battery cathodes},
  author={Tianyu Fu, Federico Monaco, Jizhou Li, Kai Zhang, Qingxi Yuan, Peter Cloetens, Piero Pianetta, and Yijin Liu},
  journal={Advanced Functional Materials Advanced Functional Materials},
  year={2022},
  doi={10.1002/adfm.202203638},
}
```

## Contributing
Contributions to this repository are always welcome. Examples of things you can contribute:
