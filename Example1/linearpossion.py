# %% 引入模块
import csv
import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pandas import DataFrame
from tqdm.notebook import tqdm
import seaborn as sns


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)


# %% 配置方程
class linearpossion(object):
    def __init__(self, dimension):
        self.dim = dimension
        self.low = -1
        self.high = 1
        self.sigma = np.sqrt(2)

    def bound_cond(self, x):
        # base = np.sum(x, axis=1, keepdims=True) / self.dim
        # return base ** 2 + np.sin(base)
        return self.u_exact(x)


    def f(self, x):
        # base = self.u_exact(x)
        # temp= (4000000*(x[:,0]**2+x[:,1]**2-x[:,0]-x[:,1])+998000)
        # f = np.reshape(temp, (x.shape[0],1))*np.reshape(base,(x.shape[0],1))
        # return f
        base = np.sum(x, axis=1, keepdims=True) / self.dim
        return (np.sin(base) - 2) / self.dim
        # return f
    def u_exact(self, x):
        # u = np.exp(-1000 * ((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2))
        base = np.sum(x, axis=1, keepdims=True) / self.dim
        return base ** 2 + np.sin(base)
        # return u
    def unif_sample_domain(self, num_sample):
        return np.random.uniform(low=self.low, high=self.high, size=(num_sample, self.dim))

    def unif_sample_bound(self, num_sample):
        x = np.random.uniform(low=self.low, high=self.high, size=(num_sample, self.dim))
        for i in range(num_sample):
            idx_num1 = random.randint(0, self.dim - 1)
            if idx_num1 == 0:
                idx_num2 = random.randint(1, self.dim - 1)
            else:
                idx_num2 = random.randint(0, self.dim - 1)

            idx1 = random.sample(list(range(self.dim)), idx_num1)
            x[i, idx1] = 1

            idx2 = random.sample(list(range(self.dim)), idx_num2)
            x[i, idx2] = -1
        return x

    def unif_sample_bound2(self, num_sample):
        x_boundary = []
        for i in range(self.dim):
            x = np.random.uniform(low=self.low, high=self.high, size=(num_sample // (2 * self.dim), self.dim))
            x[:, i] = self.high
            x_boundary.append(x)
            x = np.random.uniform(low=self.low, high=self.high, size=(num_sample // (2 * self.dim), self.dim))
            x[:, i] = self.low
            x_boundary.append(x)
        x_boundary = np.concatenate(x_boundary, axis=0)
        return x_boundary

    def is_in_domain(self, x):
        sup_flag = np.all(x < self.high, axis=1, keepdims=True)
        sub_flag = np.all(x > self.low, axis=1, keepdims=True)
        flag = sup_flag * sub_flag
        return flag

    def exit_estimate(self, x0, x1):
        sup_bound = np.ones_like(x0)
        sub_bound = np.ones_like(x0) * -1
        delta_x = x1 - x0
        sup_alpha = (sup_bound - x0) / delta_x
        sub_alpha = (sub_bound - x0) / delta_x
        alpha = np.concatenate((sup_alpha, sub_alpha), axis=1)
        alpha = min(alpha[alpha > 0])
        x_e = x0 + alpha * delta_x
        return x_e

    def transit(self, x0, delta_t):
        M = x0.shape[0]
        D = x0.shape[1]
        delta_W = np.sqrt(delta_t) * np.random.normal(size=(M, D))
        x1 = x0 + self.sigma * delta_W
        return x1

    def D(self, x0, delta_t):
        return np.ones((x0.shape[0], 1))

    def R(self, x0, x1, delta_t):
        return (self.f(x0) + self.f(x1)) * delta_t * 0.5
        # return

    def spread(self, x0, J, delta_t):
        M = x0.shape[0]
        D = x0.shape[1]
        delta_W = np.sqrt(delta_t) * np.random.normal(size=(M * J, D))
        # delta_W = np.reshape(delta_W, (M, m, D))
        x0 = np.expand_dims(x0, axis=1)
        x0 = np.broadcast_to(x0, (M, J, D))
        x0 = np.reshape(x0, (M * J, D))
        x1 = x0 + self.sigma * delta_W
        return x1

class Draw():
    def __init__(self):
        pass

    def plot_line(self, X0, X1, func):
        line_x = [x for x in np.linspace(min(X0[0], X1[0]), max(X0[0], X1[0]), 100)]
        line_y = [func(x) for x in line_x]
        plt.plot(line_x, line_y, 'k')

    # def plot_background(self, vertexes):
    #     self.plot_line(vertexes[0,:], vertexes[1,:], lambda x: 0 * x)
    #     self.plot_line(vertexes[0,:], vertexes[2,:], lambda x: 1/np.sqrt(3) * x)
    #     self.plot_line(vertexes[1,:], vertexes[2,:], lambda x: np.sqrt(1-x**2))

    def plot_background(self):
        plt.vlines(x=1, ymin=-1, ymax=1, color='k')
        plt.vlines(x=-1, ymin=-1, ymax=1, color='k')
        plt.hlines(y=1, xmin=-1, xmax=1, color='k')
        plt.hlines(y=-1, xmin=-1, xmax=1, color='k')

        # plt.vlines(x=0.5, ymin=-0.5, ymax=0.5,color = 'k')
        # plt.vlines(x=-0.5, ymin=-0.5, ymax=0.5,color = 'k')
        # plt.hlines(y=0.5, xmin=-0.5, xmax=0.5,color = 'k')
        # plt.hlines(y=-0.5, xmin=-0.5, xmax=0.5,color = 'k')
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))

    def plot_points(self, X0, style):
        plt.plot(X0[:, 0], X0[:, 1], style)

    def plot_transit(self, X0, X1, style='k-'):
        axis_x = np.vstack((X0[:,0],X1[:,0]))
        axis_y = np.vstack((X0[:,1],X1[:,1]))
        plt.plot(axis_x, axis_y, style)

    def plot_abs_error(self, x, y, z, cmp):
        plt.contourf(x, y, z, 20, cmap=plt.cm.get_cmap(cmp))
        plt.colorbar()


# class Resnet(nn.Module):
#     def __init__(self,dim):
#         super(Resnet, self).__init__()
#         self.input_layer = nn.Linear(dim, 60)
#         self.linear = nn.ModuleList()
#         params = torch.FloatTensor(np.random.rand(1,4)).to(device)
#         self.a = nn.Parameter(params)
#         for _ in range(3):
#             self.linear.append(nn.Linear(60, 60))
#         self.output_layer = nn.Linear(60, 1)
#         self.activation = self.swish
#
#
#     def ac(self, x):
#         return torch.nn.LeakyReLU(x)
#
#     def swish(self, x, beta=1):
#         return x * torch.sigmoid(beta * x)
#         # return torch.sin(x)
#
#     def forward(self, x):
#         out = self.a[:,0] * self.activation(self.input_layer(x))
#         # out = self.activation(self.input_layer(x))
#         # out = nn.LeakyReLU(0.01, True)(self.input_layer(x))
#         # out = torch.mul(out, self.a[0])
#         # out = x
#         for i in range(3):
#             x_temp = self.a[:,i+1] * self.activation(self.linear[i](out))
#             # x_temp = self.activation(self.linear[i](out))
#             # x_temp = self.activation(self.linear[i](x_temp))
#             # x_temp = torch.mul(x_temp, self.a)
#             out = x_temp + out
#         out = self.output_layer(out)
#         return out

class Resnet(nn.Module):
    def __init__(self, dim):
        super(Resnet, self).__init__()
        self.input_layer = nn.Linear(dim, 60)
        self.hidden_layer = nn.Linear(60, 60)
        self.output_layer = nn.Linear(60, 1)
        self.num_layers = 3
        self.activation = self.swish
        # params = torch.FloatTensor(np.random.rand(1, 4)).to(device)
        # self.a = nn.Parameter(params)

    def swish(self, x, beta=1):
        return x * torch.sigmoid(beta * x)

    def forward(self, x):
        out = self.input_layer(x)
        # out = self.a[:,0]*self.activation(out)
        out = self.activation(out)

        for i in range(self.num_layers):
            shortcut = out
            out = self.hidden_layer(out)
            # out = self.a[:,i+1]*self.activation(out)
            out = self.activation(out)
            out = out + shortcut

        # out = self.a[:,3]*self.activation(out)
        out = self.activation(out)
        out = out + shortcut
        out = self.output_layer(out)

        return out
# %% 设置参数
class FC(nn.Module):
    def __init__(self, dim):
        super(FC, self).__init__()
        self.fc_i = nn.Linear(dim, 20)
        self.fc = nn.Linear(20, 20)
        self.fc_o = nn.Linear(20, 1)
        self.activation = self.swish
        # self.a = nn.Parameter(torch.ones(1).to(device).requires_grad_())

    def swish(self, x, beta=1):
            return torch.tanh(x)

    def forward(self, x):
        x = self.activation(self.fc_i(x))
        for i in range(7):
            x = self.activation(self.fc(x))
            i+=1
        x = self.fc_o(x)
        return x



NUM_TRAIN_SAMPLE = 500
NUM_TEST_SAMPLE  = 3000
NUM_BOUND_SAMPLE = 200
DIMENSION        = 40
LEARNING_RATE    = 0.01
NUM_ITERATION    = 20000
Delta_t          = 0.0005
m = 20
# %% 额外函数
def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    # if type(m) == nn.Parameter:
    #     torch.nn.init.xavier_uniform_(m.params)
# %% 配置训练

Eq = linearpossion(DIMENSION)
# U = FC(DIMENSION).to(device)
U = Resnet(DIMENSION).to(device)
U.apply(weights_init)
optU = optim.Adam(U.parameters(), lr=LEARNING_RATE)
# U.load_state_dict(torch.load('/home/GPU3/xingyu/Ad-DFLM/PINN/20D-092319-PINN'))


# %% 开始训练
start_time = time.time()
training_history = []

Loss=0
X0 = Eq.unif_sample_domain(NUM_TRAIN_SAMPLE)
Lossb = 0
for step in range(NUM_ITERATION + 1):
    torch.cuda.empty_cache()
    if step % 200 == 0:
        for p in optU.param_groups:
            p['lr'] *= 0.95
            print(p['lr'])
    # if (step/5000) % 2 == 0:
    #     for p in optU.param_groups:
    #         p['lr'] *= 0.5
    # #         print(p['lr'])
    # if step!=0 and (step / 2000) % 2 == 1:
    #     for p in optU.param_groups:
    #         p['lr'] *= 0.5
    #         print(p['lr'])
    # if step!=0 and (step / 5000) % 2 == 1:
    #     for p in optU.param_groups:
    #         p['lr'] /= 5
    #         print(p['lr'])


    if step % 5 == 0:
        np.random.seed(60)
        elapsed_time = time.time() - start_time
        X0t = Eq.unif_sample_domain(NUM_TEST_SAMPLE)
        Y0t = U(torch.FloatTensor(X0t).to(device))
        Y0t = Y0t.cpu().detach().numpy()
        U0t = Eq.u_exact(X0t)
        L2 = np.sqrt(np.mean((Y0t - U0t) ** 2) / np.mean(U0t ** 2))
        L1 = np.abs(U0t-Y0t)

        if step % 200 == 0:
            print(f'\nStep: {step:>5}, '\
                f'Loss: {Loss:>10.5f}, '\
                f'L2: {L2:.6f},\n'\
                # f'L1: {L1:.6f},\n'\
                f'Time: {elapsed_time:.2f}')
        training_history.append([step, Loss, L2, elapsed_time])

    # if step %5000 == 0:
    #     x0 = np.linspace(-1,1,50)
    #     y0 = np.linspace(-1,1,50)
    #     x, y = np.meshgrid(x0, y0)
    #     # points = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)),axis=1)
    #     points = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), np.zeros((x.shape[0] * x.shape[1], Eq.dim - 2))), axis=1)
    #     Yp = U(torch.FloatTensor(points).to(device))
    #     Yp = Yp.cpu().detach().numpy()
    #     Up = Eq.u_exact(points)
    #     error = np.abs(Yp-Up)
    #
    #     plt.contourf(x,y,error.reshape(x0.shape[0], -1), 20, cmap=plt.cm.get_cmap('jet'))
    #     plt.colorbar()
    #     plt.show()



##############################################################################################################################################################

    Loss = 0

##MTD###
    # Y0 = U(torch.FloatTensor(X0).to(device))
    # X1 = Eq.transit(X0, Delta_t)
    # Y1 = U(torch.FloatTensor(X1).to(device))
    # flag = Eq.is_in_domain(X1)
    # #
    # if np.any(flag == False):
    #     # if Eq == LinearPoisson:
    #     #     X0m_out = X0m[flag == False, :]
    #     #     Xm_out = Xm[flag == False, :]
    #     #     fgs = Eq.exit_estimate1(X0m_out, Xm_out)
    #     #     for i in range(len(fgs)):
    #     #         if fgs[i].any():
    #     #             Xm_out = Eq.estimate_exit(X0m_out, Xm_out, fgs[i], i + 1)
    #     # else:
    #         X0_out = X0[flag.squeeze(-1) == False, :]
    #         X1_out = X1[flag.squeeze(-1) == False, :]
    #         X1_new = Eq.exit_estimate(X0_out, X1_out)
    #         Y1[flag.squeeze(-1) == False, :] = torch.FloatTensor(Eq.bound_cond(X1_new)).to(device)
    # #
    # #
    # D_t = Eq.D(X0, Delta_t)
    # R_t = Eq.R(X0, Delta_t)
    # Target = - torch.FloatTensor(R_t).to(device) + torch.FloatTensor(D_t).to(device) * Y1
    # Loss = Loss + torch.sum(torch.square(Y0 - Target))
    #
    # Xb = Eq.unif_sample_bound2(NUM_BOUND_SAMPLE)
    # Yb = U(torch.FloatTensor(Xb).to(device))
    # Gb = torch.FloatTensor(Eq.bound_cond(Xb)).to(device)
    # Loss = Loss + torch.sum(torch.square(Yb - Gb))
    #
    # optU.zero_grad()
    # Loss.backward()
    # optU.step()
    #
    # if np.any(flag == False):
    #     X1[flag.squeeze(-1)==False, :] = Eq.unif_sample_domain(len(X1[flag.squeeze(-1)==False, :]))
    # X0 = X1

    # X0 = Eq.unif_sample_domain(NUM_TRAIN_SAMPLE)
##########################Adaptive#########################################################
    Y0 = U(torch.FloatTensor(X0).to(device))
    Xm = Eq.spread(X0,m, Delta_t)
    X0m = np.expand_dims(X0, axis=1)
    X0m = np.broadcast_to(X0m, (NUM_TRAIN_SAMPLE, m, DIMENSION))
    X0m = np.reshape(X0m, (NUM_TRAIN_SAMPLE * m, DIMENSION))
    Y0m = U(torch.FloatTensor(Xm).to(device))


    flag = Eq.is_in_domain(Xm)

    if np.any(flag == False):
        X0m_out = X0m[flag.squeeze(-1) == False, :]
        Xm_out = Xm[flag.squeeze(-1) == False, :]
        Xm_new = Eq.exit_estimate(X0m_out, Xm_out)
        Y0m[flag.squeeze(-1) == False, :] = torch.FloatTensor(Eq.bound_cond(Xm_new)).to(device)
        Xm[flag.squeeze(-1) == False, :] = Xm_new
        # if step % 500 == 0:
        #     draw.plot_background()
        #     draw.plot_points(X0m, 'b.')
        #     draw.plot_transit(X0m[flag.squeeze(-1) == False, :], Xm[flag.squeeze(-1) == False, :], 'b-')
        #     draw.plot_points(Xm_new, 'rx')
        #     plt.show()

    D_t = Eq.D(X0m, Delta_t)
    R_t = Eq.R(X0m, Xm, Delta_t)
    R_t = np.reshape(R_t, (D_t.shape[0], 1))
    Target = - torch.FloatTensor(R_t).to(device)*torch.FloatTensor(D_t).to(device) + torch.FloatTensor(D_t).to(device) *Y0m
# #
# # ###zsy
    def Ind(x):
        # x0m = X0m
        # Delta_t = 0.00001
        J2 = 2
        xm = Eq.spread(x, J2, Delta_t)
        x0m = np.expand_dims(x, axis=1)
        x0m = np.broadcast_to(x0m, (NUM_TRAIN_SAMPLE * m, J2, DIMENSION))
        x0m = np.reshape(x0m, (NUM_TRAIN_SAMPLE * m * J2, DIMENSION))
        y0m = U(torch.FloatTensor(xm).to(device))

        flag2 = Eq.is_in_domain(xm)

        if np.any(flag2 == False):
            x0m_out = x0m[flag2.squeeze(-1) == False, :]
            xm_out = xm[flag2.squeeze(-1) == False, :]
            xm_new = Eq.exit_estimate(x0m_out, xm_out)
            y0m[flag2.squeeze(-1) == False, :] = torch.FloatTensor(Eq.bound_cond(xm_new)).to(device)


        D_t = Eq.D(x0m, Delta_t)
        R_t = Eq.R(x0m, xm, Delta_t)
        Target1 = (- torch.FloatTensor(R_t).to(device) + y0m) * torch.FloatTensor(D_t).to(device)
        Target1 = torch.reshape(Target1, (NUM_TRAIN_SAMPLE * m, J2, 1))
        Target1 = (1 / J2) * torch.sum(Target1, dim=1)

        Ind = torch.square(Target1 - U(torch.FloatTensor(x).to(device)))
        method = 'global'
        if method == 'global':
            # NEW = []
            index = torch.topk(Ind, k=NUM_TRAIN_SAMPLE, dim=0)[1].cpu().detach().numpy()
            NEW = x[index].squeeze(1)

        if method == 'local':
            Ind = torch.reshape(Ind, (NUM_TRAIN_SAMPLE, m))
            # x = np.reshape(x, (NUM_TRAIN_SAMPLE, m, DIMENSION))
            _, index = torch.max(Ind, dim=1)
            index = index.unsqueeze(1).cpu().detach().numpy()
            NEW = x[index].squeeze(1)

        return NEW




    td = torch.abs(Target - U(torch.FloatTensor(X0m).to(device)))
    Target = torch.reshape(Target, (NUM_TRAIN_SAMPLE, m, 1))
    Y0_pred = 1 / m * torch.sum(Target, dim=1)


    # if step != 0 and (step / 1000) % 2 == 0 or (step / 1000) % 2 == 1:
    # index = torch.topk(td, k=NUM_TRAIN_SAMPLE, dim=0)[1].cpu().detach().numpy()
    # NEW = Xm[index].squeeze(1)
    # X1 = NEW
    #     # X1 = Ind(Xm)
    #     print('Adaptive sample is down')
    # else:
    #     X1 = Eq.transit(X0, Delta_t)
    # X1 = Ind(Xm)
    X1=Eq.transit(X0,Delta_t)
    # x0 = Eq.transit(X0,Delta_t)[0:400]
    # X1=np.vstack((x1, x0))




    Loss = Loss + 1 / NUM_TRAIN_SAMPLE * torch.sum(torch.square(Y0_pred - Y0))
#
#
    Xb = Eq.unif_sample_bound2(NUM_BOUND_SAMPLE)
    Yb = U(torch.FloatTensor(Xb).to(device))
    Gb = torch.FloatTensor(Eq.bound_cond(Xb)).to(device)
    Loss = Loss + 1/NUM_BOUND_SAMPLE * torch.sum(torch.square(Yb - Gb))

    optU.zero_grad()
    Loss.backward()
    optU.step()

    flag3 = Eq.is_in_domain(X1)
    if np.any(flag3 == False):
        X1[flag3.squeeze(-1) == False, :] = Eq.unif_sample_domain(len(X1[flag3.squeeze(-1) == False, :]))
    X0 = X1

training_history =np.array(torch.FloatTensor(training_history).cpu().detach())
print('Min L2:', np.min(training_history[:,2]))
print('Mean L2:', np.mean(training_history[:,2]))
# Iteration = np.arange(0, 20001, 5)
# Iteration = Iteration[::50]
# plt.figure()
# plt.axes(yscale='log')
# plt.grid(axis="y")
# plt.xlabel('Iteration')
# plt.ylabel('Relative Error')
# glo = pd.read_csv('/home/GPU3/xingyu/Ad-DFLM/high dimension possion/30D-20000-1e-05dt-0.01-global(4000)-[051544].csv').values[:,2]
# # glo = pd.read_csv('/home/GPU3/xingyu/Ad-DFLM/high dimension possion/30D-20000-0.0005dt-0.01-TD-[081601].csv').values[:,2]
# # td = pd.read_csv('/home/GPU3/xingyu/Ad-DFLM/high dimension possion/30D-20000-1e-05dt-0.01-TD-[211215].csv').values[:,2]
# td = pd.read_csv('/home/GPU3/xingyu/Ad-DFLM/high dimension possion/30D-20000-0.0005dt-0.01-TD-[081601].csv').values[:,2]
# DFLM = pd.read_csv('/home/GPU3/xingyu/Ad-DFLM/high dimension possion/30D-20000-0.0005dt-0.01-DFLM-[081827].csv').values[:,2]
# PINN = np.load('/home/GPU3/xingyu/Ad-DFLM/old states/30D-PINN-L2curve.npy')
# PINN2 = pd.read_csv('/home/GPU3/xingyu/Ad-DFLM/PINN/30D-PINN-global-[051138].csv').values[:,2]
# # DRM_L2 = np.load('/home/GPU3/xingyu/Ad-DFLM/old states/10D-DRM-L2curve.npy')
# # MTD_L2 = pd.read_csv('/home/GPU3/xingyu/Ad-DFLM/high dimension possion/10D-20000-1e-05dt-0.01-TD-lp-[071615].csv').values[:,2]
# local = pd.read_csv('/home/GPU3/xingyu/Ad-DFLM/high dimension possion/30D-20000-1e-05dt-0.01-local-[211211].csv').values[:,2]
#
# line1, = plt.plot(Iteration, training_history[:,2][::50], color='#E5086A', linestyle='solid')
# # line2, = plt.plot(Iteration, local[::50], color='#FF9200', linestyle='--')
# line3, = plt.plot(Iteration, td[::50], color='#0094FF', linestyle='-', marker='+')
# line4, = plt.plot(Iteration, DFLM[::50], color='#008D00', linestyle='dashdot')
# line5, = plt.plot(Iteration, PINN[::50], color='#80766E', linestyle='dotted')
# line6, = plt.plot(Iteration, PINN2[::50], color='#FF9200', linestyle='--')
# plt.legend([line1, line3, line4, line5, line6],
#            ["ADFLM-global", 'ADFLM-TD', "DFLM", 'PINN', 'APINN'],
#            loc='upper right')
# plt.show()


# training_history =np.array(training_history.cpu().detach())
record_time = time.localtime()
np.savetxt('/home/GPU3/xingyu/Ad-DFLM/high dimension possion/{}D-{}-{}dt-{}-global-[{:0>2d}{:0>2d}{:0>2d}].csv'.format(
    # Eq.__class__.__name__,
    DIMENSION,
    NUM_ITERATION,
    Delta_t,
    LEARNING_RATE,
    record_time.tm_mday,
    record_time.tm_hour,
    record_time.tm_min,),
            training_history,
            delimiter=",",
            header="step, loss, L2, Time",
            comments='')
print('Training History Saved!')

torch.save(U.state_dict(), f'{DIMENSION}-{record_time.tm_mday:0>2d}{record_time.tm_hour:0>2d}{record_time.tm_min:0>2d}-global')
print('Model Saved!')

#测试

U.load_state_dict(torch.load('2D-20000-0.0005dt-0.01-[011140]-adaptive-u2(500-20-3000)'))
print('load from ckpt!')
