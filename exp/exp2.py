# This script plot \cref{linear}
from utils import *
from simulator import *
from models import *

dim = 5
int_dim = 3         # intrinsic dimension
eps = .1
step_num = 50
SAMPLE_NUM = 100
MAX_SIZE = 2
size_arr = np.linspace(1, MAX_SIZE, SAMPLE_NUM) * -1
lambda_ = 10000
r.seed(0)
rt = [[], [], []]

i = 0
F = np.eye(dim) * 1.2
F[:int_dim, :int_dim] = np.eye(int_dim) * 0.98
B = r.rand(dim, dim)
Q, R = nalg.qr(B)
B = Q
B = np.eye(dim)

U = np.zeros([dim, step_num])
U_OLS = np.zeros([dim, step_num])
U_TR = np.zeros([dim, step_num])
err_OLS = np.zeros(step_num)
err_TR = np.zeros(step_num)
U[:int_dim, 0] = r.rand(int_dim)
U_OLS[:, 0] = copy.deepcopy(U[:, 0])
U_TR[:, 0] = copy.deepcopy(U[:, 0])
for j in range(1, step_num):
    U[:, j] = F @ U[:, j-1]

while i < SAMPLE_NUM:
    # generate parameters
    #C = r.rand(dim, dim) * size_arr[i]
    #C = B.T @ F
    C = np.zeros([dim, dim])
    C[:int_dim, :int_dim] = np.eye(int_dim) * size_arr[i]
    A = F - B @ C
    # it is a little bit weird that noise are all the same in different scale
    #eps = 0
    noise = r.rand(dim, step_num) * eps
    rt[0].append(nalg.norm(B @ C))
    
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
    #pdb.set_trace()
    F_OLS = A + B @ C_OLS
    F_TR = A + B @ C_TR

    for j in range(1, step_num):
        U_OLS[:, j] = F_OLS @ U_OLS[:, j-1]
        U_TR[:, j] = F_TR @ U_TR[:, j-1]
        err_OLS[j] = nalg.norm(U_OLS[:, j] - U[:, j])
        err_TR[j] = nalg.norm(U_TR[:, j] - U[:, j])

    #rt[1].append((np.log(err_OLS[-1]/err_OLS[step_num//2])
    #              - np.log(err_TR[-1]/err_TR[step_num//2]))/step_num*2)
    rt[2].append(np.log(err_OLS[-1]/err_TR[-1])/step_num)
    print(nalg.norm(A+B@C_OLS@P_V, ord=2))
    rt[1].append(nalg.norm(B@C_OLS, ord=2)**2 
                 + nalg.norm(A+B@C_OLS@P_V, ord=2)**2 
                 - nalg.norm(A+B@C_TR@P_V, ord=2)**2)
    i = i+1

#print(rt[0])
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi']  = 300 
labelsize = 10
fontsize = 10
plt.scatter(rt[1], rt[2])
#plt.xlabel(r'$B\widehat C$')
plt.xlabel(r'$\left\| A + B\widehat{C}_{OLS}P_V \right\|^2 + \left\| B\widehat{C}_{OLS} \right\|^2 - \left\| A + B\widehat{C}_{TR}P_V \right\|^2$', fontsize=fontsize)
plt.ylabel(r'$\frac{1}{T}\log\frac{e_{OLS}}{e_{TR}}$', fontsize=fontsize)
plt.tick_params(labelsize=labelsize)
plt.savefig('../../fig/exp2/linear.pdf')
plt.show()
'''plt.scatter(U_OLS[0, :], U_OLS[1, :], label='OLS')
plt.scatter(U_TR[0, :], U_TR[1, :], label='TR')
plt.show()'''