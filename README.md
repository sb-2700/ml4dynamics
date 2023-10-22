# Mitigating distribution shift in machine learning-augmented hybrid simulation

Codebase for mitigating distribution shift in MLHS using tangent-space reegularized algorithm based on the paper [[1]]() by Jiaxi Zhao and Qianxiao Li.


## Installing & Getting Started
1. Environment setup: if the device is not prepared for GPU programming, please comment out the package related to cuda.

```bash
conda create -n TR python=3.9.18
conda activate TR
python -m pip install -r requirements.txt
```

2. Clone the repository.
   
```bash
git clone https://github.com/jiaxi98/TR.git
cd TR
```

## Reproduce the results
In the following we will provide the detailed procedure to reproduce the full experiments in the paper. All the estimated execution times are based on 4 GPUs (NVIDIA GeForce RTX 3090). All the checkpoints of network models are provided and can be used to directly generate the plots in `demo.ipynb` notebook without training the models.
### Data generation
Estimated execution time: 1 hour
```bash
mkdir ../data/NS
mkdir ../data/RD
bash generate_data.sh
```

### Training
Estimated execution time: 24 hour
```bash
mkdir ../models/NS
mkdir ../models/RD
bash train.sh
```

### Plots
For a quick view of all the plots, we high recommand to run the `demo.ipynb` notebook.

Estimated execution time: 5 minutes
1. plots of the distribution shift phenomena, this script plots the Fig. 1 and Fig. 2 which illustrate the distribution shift 
phenomena for RD and NS
```bash
cd exp
python exp1.py
```
2. plots of the linear dynamics experiments
```bash
python exp2.py
```
3. plots of the comparison of distribution shift with different simulating parameters
```bash
python exp3.py
```
4. plots of the comparison of TR, OLS, and the ground truth
```bash
python exp4.py
```
5. generate the table
```bash
python exp5.py
```

## Reference
[1] [Mitigating distribution shift in machine learning-augmented hybrid simulation]()