# This script plot \cref{linear}
import numpy.random as r
from utils import *
from simulator import *
from model import *


n = 5
A = r.rand(n, n)
eps = .001


for i in range(100):
    B = r.rand(n, n)
    C = r.rand(n, n)
    C_OLS = A
    C_TR = A


    for j in range(10):
        