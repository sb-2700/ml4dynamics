# Mitigating distribution shift in machine learning-augmented hybrid simulation

Codebase for mitigating distribution shift in MLHS using tangent-space reegularized algorithm based on the paper [[1]]() by Jiaxi Zhao and Qianxiao Li.


## Installing & Getting Started
Install the package and set the environment

```bash
git clone git@github.com:jiaxi98/ml4dynamics.git
mkdir venv
mkdir venv/ml4dynamics
python3 -m venv venv/ml4dynamics
source venv/ml4dynamics/bin/activate
cd ml4dynamics
pip install --upgrade pip
python -m pip install -r requirements.txt
pip install -e .
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

## Citation
If you find this codebase useful for your research, please consider citing:
```bibtex
@article{zhao2025mitigating,
  title={Mitigating Distribution Shift in Machine Learning--Augmented Hybrid Simulation},
  author={Zhao, Jiaxi and Li, Qianxiao},
  journal={SIAM Journal on Scientific Computing},
  volume={47},
  number={2},
  pages={C475--C500},
  year={2025},
  publisher={SIAM}
}
```