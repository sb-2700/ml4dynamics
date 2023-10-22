# This script plot \cref{linear}
from utils import *
from simulator import *
from models import *

def main():
    dim = 2
    int_dim = 1         # intrinsic dimension
    eps = .001
    step_num = 50
    SAMPLE_NUM = 5
    MAX_SIZE = 2
    size_arr = np.linspace(.5, MAX_SIZE, SAMPLE_NUM)
    lambda_array = [2,4,8,16,32,64,128,256,512,1024,2048]
    r.seed(0)
    rt = [[], [], []]
    var = [[], [], []]

    i = 0
    F = np.eye(dim) * 1.2
    F[:int_dim, :int_dim] = np.eye(int_dim) * 0.95
    B = r.rand(dim, dim)
    Q, R = nalg.qr(B)
    B = Q
    B = np.eye(dim)

    U = np.zeros([dim, step_num])
    U_OLS = np.zeros([dim, step_num])
    U_mOLS = np.zeros([dim, step_num])
    U_TR = np.zeros([dim, step_num])
    err_OLS = np.zeros(step_num)
    err_mOLS = np.zeros(step_num)
    err_TR = np.zeros(step_num)
    U[:int_dim, 0] = r.rand(int_dim)
    U_OLS[:, 0] = copy.deepcopy(U[:, 0])
    U_mOLS[:, 0] = copy.deepcopy(U[:, 0])
    U_TR[:, 0] = copy.deepcopy(U[:, 0])
    for j in range(1, step_num):
        U[:, j] = F @ U[:, j-1]

    for lambda_ in lambda_array:
        tmp1 = []
        tmp2 = []
        tmp3 = []
        for i in range(SAMPLE_NUM):
            # generate parameters
            #C = r.rand(dim, dim) * size_arr[i]
            #C = B.T @ F
            C = np.zeros([dim, dim])
            #C[:int_dim, :int_dim] = np.eye(int_dim) * size_arr[i]
            C[:int_dim, :int_dim] = r.rand(int_dim, int_dim) * size_arr[i]
            A = F - B @ C
            # it is a little bit weird that noise are all the same in different scale
            #eps = 0
            noise = r.rand(dim, step_num) * eps
            
            U_pinv = nalg.pinv(U)
            P_V = U @ U_pinv
            P_V_ = np.eye(dim) - P_V
            P0 = np.zeros([dim, dim])
            P0[:int_dim, :int_dim] = np.eye(int_dim)
            #if nalg.norm(P_V-P0) > 1e-5:
                #print(nalg.norm(P_V-P0))
            #    continue
            
            P_V = P0
            P_V_ = np.eye(dim) - P0
            C_OLS = C @ P_V + noise @ U_pinv
            C_TR = nalg.inv(np.eye(dim) + lambda_ * B.T @ P_V_ @ B) \
                        @ (C_OLS - lambda_ * B.T @ P_V_ @ A @ P_V)
            C_mOLS = C_OLS @ nalg.inv((lambda_ * np.eye(dim) + P_V))
            #pdb.set_trace()
            #print(C_OLS)
            #print(C_TR)
            F_OLS = A + B @ C_OLS
            F_mOLS = A + B @ C_mOLS
            F_TR = A + B @ C_TR
            #print(lambda_)
            #print(F_OLS)
            #print(F_TR)

            for j in range(1, step_num):
                U_OLS[:, j] = F_OLS @ U_OLS[:, j-1]
                U_mOLS[:, j] = F_mOLS @ U_mOLS[:, j-1]
                U_TR[:, j] = F_TR @ U_TR[:, j-1]
                err_OLS[j] = nalg.norm(U_OLS[:, j] - U[:, j])
                err_mOLS[j] = nalg.norm(U_mOLS[:, j] - U[:, j])
                err_TR[j] = nalg.norm(U_TR[:, j] - U[:, j])

            tmp1.append(err_OLS[-1])
            tmp2.append(err_mOLS[-1])
            tmp3.append(err_TR[-1])
        tmp1 = np.array(tmp1)
        tmp2 = np.array(tmp2)
        tmp3 = np.array(tmp3)
        rt[0].append(np.mean(tmp1))
        rt[1].append(np.mean(tmp2))
        rt[2].append(np.mean(tmp3))
        var[0].append(np.std(tmp1))
        var[1].append(np.std(tmp2))
        var[2].append(np.std(tmp3))
    rt = np.array(rt)
    var = np.array(var)

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi']  = 300 
    labelsize = 10
    fontsize = 10
    eps = 1e-7
    plt.plot(lambda_array, np.log(rt[0]), label='OLS')
    plt.fill_between(lambda_array, np.log(np.maximum(rt[0]-var[0], eps)), np.log(rt[0]+var[0]), color="gray", alpha=0.5)
    plt.plot(lambda_array, np.log(rt[1]), label='mOLS')
    plt.fill_between(lambda_array, np.log(np.maximum(rt[1]-var[1], eps)), np.log(rt[1]+var[1]), color="gray", alpha=0.5)
    plt.plot(lambda_array, np.log(rt[2]), label='TR')
    plt.fill_between(lambda_array, np.log(np.maximum(rt[2]-var[2], eps)), np.log(rt[2]+var[2]), color="gray", alpha=0.5)
    #plt.xlabel(r'$B\widehat C$')
    plt.xlabel(r'$\lambda$', fontsize=fontsize)
    plt.ylabel(r'$\log\| u(T) - \widehat u(T) \|_{2}$', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.7))
    plt.savefig('../../fig/exp2/linear.pdf')
    plt.show()
    '''plt.scatter(U_OLS[0, :], U_OLS[1, :], label='OLS')
    plt.scatter(U_TR[0, :], U_TR[1, :], label='TR')
    plt.show()'''

if __name__ == '__main__':
    main()