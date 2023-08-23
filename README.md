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