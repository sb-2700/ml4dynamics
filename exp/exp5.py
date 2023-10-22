# This script plots \cref{RD-table}
# This script plots \cref{NS-table}
from utils import *
from simulator import *
from models import *
from test import test_simulator

def main():
    model_type = ['OLS', 'mOLS', 'aOLS', 'TR']
    ds_parameter = [1, 2, 3, 4, 5]
    test_number = 1
    rt = np.zeros([4, 5])

    for i in range(len(model_type)):
        for j in range(len(ds_parameter)):
            for k in range(test_number):
                simulator, _ = test_simulator(n=64,
                                            model_type=model_type[i], 
                                            ds_parameter=ds_parameter[j], 
                                            test_index=k)
                rt[i,j] = rt[i,j] + simulator.error_hist[500]
            rt[i, j] = rt[i, j]/test_number

    with open("../../fig/exp5/TableRD.tbl", "w") as f:
        for i in range(len(model_type)):
            f.write("{} & {:.2e} & {:.2e} & {:.2e} & {:.2e} & {:.2e} \\\\\n".format(model_type[i], rt[i,0], rt[i,1], rt[i,2], rt[i,3], rt[i,4]))
        f.write("Diff & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% \\\\\n".format(100-rt[3,0]*100/rt[0,0], 
                                                                                    100-rt[3,1]*100/rt[0,1], 
                                                                                    100-rt[3,2]*100/rt[0,2], 
                                                                                    100-rt[3,3]*100/rt[0,3], 
                                                                                    100-rt[3,4]*100/rt[0,4]))
        f.write("\\hline\n")

if __name__ == "__main__":
    main()