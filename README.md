# Mitigating distribution shift in machine-learning augmented hybrid simulation


Codebase for mitigating distribution shift in MLHS using tangent-space reegularized algorithm.

Based on the paper [Mitigating distribution shift in machine-learning augmented hybrid simulation]().


## Installing & Getting Started
1. Environment setup
`conda create -n TR python=3.9.18`
`python -m pip install -r requirements.txt`
2. Clone the repository.
`git clone https://github.com/jiaxi98/TR.git`

## Data generation
`bash generate_data.sh`

## Training
`bash train.sh`

## Plots
1. plots of the distribution shift phenomena
`python exp1.py`
this script plots the Fig. 1 and Fig. 2 which illustrate the distribution shift 
phenomena for RD and NS
2. plots of the linear dynamics experiments
`python exp2.py`
3. plots of the comparison of distribution shift
`python exp3.py`