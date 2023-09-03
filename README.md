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
Where should I define the data variable `u` first time? Either in the script where we use to data to training, or the utils.py file where we define the read_data function.

Why the test loss is always smaller than the training loss?

Try to plant a module which send u a msg when there is some issue encounter in training, either loss become NaN or some CUDA issues

One of my recently really hard to detect bug is that I forget to load my learned model parameters in the test file, which results in wrong checking result in check.py. How can I avoid this?

Currently, the data structure for RD and NS equations are not consistent, NS is nx * ny, while RD is n * 1


## training
`python training.py`


## plots of the distribution shift phenomena
exp1.py
this script plots the Fig. 1 and Fig. 2 which illustrate the distribution shift 
phenomena for RD and NS

how to archive all the learned models?

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


### Unsolved problems in the numerical experiment
1. During training of the OLS model, the training loss is 7e-3 and test error 7e-3. However, when we download the model to local and test it over test trajectory, the error is 1e-1 which is significantly large than the error report.
2. 


### General tips on code writing for MLHS
> Procedure overview of MLHS
> > 1. Generating the data using some classical numerical methods (In the project, we simulating the Reaction-diffusion equation using Crank-Nicolson scheme and NS equation using projection method).
> > * First step is to check if the simulated data is physical. Sometimes the data blow up as NS equation and sometimes we simply forget to save the data into the file (they remain $0$ as initialization).
> > * It is advantageous to pack all the parameters such as grid size, time step, etc into the data .npz file. Moreover, try to make all the dataset file of the same structure.
> > * It is not necessary to fix the size of the storing data at first. In numerical simulation we may introduce grid cell, but this should not be included when performing inference as it is non-physical. In this project, this happens to our NS experiment, as we will need some ghost cell to force BC while they are not suitable to be included in unresolved model to predict the pressure.
> > * It is recommended to use a warm-up time, i.e. discarding the first several iterations of the simulated configuration. Since most initial condition is un-physical, e.g. we use random Gaussian field for RD initialization and set all the velocity & pressure interior to $0$ for NS equation, this will deteriorate the performance of data-driven model.
> > 2. Using the generated data, we will run optimization on unresolved model to learn the parameters.
> > * We need to test the accuracy carefully to avoid trivial optimization. In this project, at first we forgot to store the pressure label, which is initialized to $0$. This also result in our gradient descent keeps decreasing the loss. However, usually this loss (around 1e-7~1e-10) is much lower than the loss in typical MLHS (around 1e-2~1e-4)
> > 3. Using the learned model, we can do trajectories inference or other validation task and plot the figures, which is usually suitable to be done at local.
> > 4. General tips: it is convenient to save some *.bash file either locally on PC or the remote server, e.g. we can save the following commends on PC, remote server respectively
> > + `scp file_name usr@address:pwd`
> > + `tar -zxcf file_name pwd`
> > + `git clone repo_address`
> > 5. Be careful when you want to recheck the numerical results in remote server on local PC. Not only should the models and all the hyperparameters be the same, but also the data should be the same. 
> > 6. It is a good habit to save the models and maybe prediction several times during the training, e.g. if the total training epoch is 50000, it is recommended to save the model per 5000 epochs.ß