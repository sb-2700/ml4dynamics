# Mitigating distribution shift in machine-learning augmented hybrid simulation


Codebase for mitigating distribution shift in MLHS using tangent-space reegularized algorithm.

Based on the paper [Mitigating distribution shift in machine-learning augmented hybrid simulation]().


## Installing & Getting Started

1. Clone the repository.


We should attempt to increase the safety of the code, namely, only several modules or code can change the network model parameters and others only freeze them and do inference  


##  data generation
`python generate_RDdata.py`

`python generate_NSdata.py`

need to add arguments to these two functions to specifies the dimension of the data

currently, n=256, Re=100, 400, 500 the data is unavailable

### Questions
Where should I define the data variable `u` first time? Either in the script where we use to data to training, or the utils.py file where we define the read_data function


## training
`python training.py`


## plots of the distribution shift phenomena
exp1.py
this script plots the Fig. 1 and Fig. 2 which illustrate the distribution shift 
phenomena for RD and NS

## plots of the comparison of distribution shift
ssh jiaxi@10.246.112.152

### finished file (means it can handle both RD & NS)
utils.py
simulator.py
model.py
generate_RDdata.py 
generate_NSdata.py 
generate_data.sh
training.py
training.sh
exp1.py


### environment setup
conda create -n TR python=3.11.3

conda install pytorch::pytorch torchvision torchaudio -c pytorch

pip install torch

### comment
just come up with an idea that we can put all the exp.py file in our local repo and download the data to print locally

`conda env create -f environment.yml`

Traceback (most recent call last):
  File "training.py", line 144, in <module> 
    loss_tr = loss_tr + lambda_*criterion(sum_, torch.tensor(0.0))
  File "/home/jiaxi/anaconda3/envs/Applymath/lib/python3.6/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/jiaxi/anaconda3/envs/Applymath/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 520, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/home/jiaxi/anaconda3/envs/Applymath/lib/python3.6/site-packages/torch/nn/functional.py", line 3112, in mse_loss
    return torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
RuntimeError: iter.device(arg).is_cuda()INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/native/cuda/Loops.cuh":98, please report a bug to PyTorch. argument 2: expected a C
UDA device but found cpu