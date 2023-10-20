# This script plots \cref{RD-table}
# This script plots \cref{NS-table}
from utils import *
from simulator import *
from models import *
from test import test_simulator

model_type = ['OLS', 'mOLS', 'aOLS', 'TR']
ds_parameter = [1, 2, 3, 4, 5]
test_number = 1
rt = np.zeros([4, 5])

for i in range(len(model_type)):
    for j in range(len(ds_parameter)):
        for k in range(test_number):
            simulator, _ = test_simulator(n=128,
                                          model_type=model_type[i], 
                                          ds_parameter=ds_parameter[j], 
                                          test_index=k)
            rt[i,j] = rt[i,j] + simulator.error_hist[-1]
        rt[i, j] = rt[i, j]/test_number

with open("../../fig/exp5/TableRD.tbl", "w") as f:
    for i in range(len(model_type)):
        f.write("{} & {:.2e} & {:.2e} & {:.2e} & {:.2e} & {:.2e} \\\\\n".format(model_type[i], rt[i,0], rt[i,1], rt[i,2], rt[i,3], rt[i,4]))
        f.write("\\hline\n")