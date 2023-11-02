# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import *
import os
from GenerateData import *
import random
import pandas as pd
import matplotlib.pyplot as plt
from optparse import OptionParser

# Omega 空间域
a = [-1, -1]
b = [ 1,  1]
DIMENSION = 10      # Dimension
# Netword
DIM_INPUT  = DIMENSION     # 输入维数
NUM_UNIT   = 20             # 单层神经元个数
DIM_OUTPUT = 1              # 输出维数
NUM_LAYERS = 6              # 模型层数
# Optimizer
IS_DECAY           = 1
LEARN_RATE         = 0.01      # 学习率
LEARN_FREQUENCY    = 400         # 学习率变化间隔
LEARN_LOWWER_BOUND = 1e-5       # 学习率下限
LEARN_DECAY_RATE   = 0.95      # 学习率衰减乘子
LOSS_FN            = nn.MSELoss()
# Training
CUDA_ORDER = "2"
NUM_TRAIN_SMAPLE   = 500  # 内部训练集大小
NUM_BOUND          = 400    # 每个边界的训练集大小
NUM_ITERATION      = 20000  # 单份样本训练次数
NUM_TRAIN_TIMES    = 1      # 训练样本份数
# Re-sampling
IS_RESAMPLE = 0             # 是否重采样
SAMPLE_FREQUENCY   = 5000   # 重采样间隔
# Testing
NUM_TEST_SAMPLE    = 3000   # 测试集大小
TEST_FREQUENCY     = 10     # 输出间隔
# Loss weight
BETA = 1000                  # 边界损失函数权重
# Save model
IS_SAVE_MODEL = 1           # 是否保存模型
Delta_t          = 0.00001

class EquationBase(object):
    def __init__(self, dimension, device):
        self.D = dimension
        self.device = device

    def f(self, t, X, Y, Z):
        raise NotImplementedError

    def g(self, X):
        raise NotImplementedError

    def terminal_grad(self, X):
        g = self.terminal_cond(X)
        Dg = torch.autograd.grad(outputs=[g],
                                 inputs=[X],
                                 grad_outputs=torch.ones_like(g),
                                 allow_unused=True,
                                 retain_graph=True,
                                 create_graph=True)[0]
        return Dg

    def mu(self, t, X):
        raise NotImplementedError

    def sigma(self, t, X):
        raise NotImplementedError

    def u_exact(self, t, X):
        raise NotImplementedError

    def transit(self, X_init, t, delta_t):
        M = X_init.shape[0]
        delta_W = np.sqrt(delta_t) * np.random.normal(size=(M, self.D))
        delta_W = torch.FloatTensor(delta_W).to(self.device)
        X_next = X_init + torch.squeeze(torch.matmul(delta_W.unsqueeze(1), self.sigma(t, X_init)), 1)
        return delta_W, X_next

class PossionQuation(object):
    def __init__(self, dimension, device):
        # super(PossionQuation, self).__init__(dimension, device)
        self.D = dimension
        self.device=device
        self.sigma = np.sqrt(2)


    def f(self, X):
        base = self.u_exact(X)
        # temp= (4000000*(X[:,0]**2+X[:,1]**2-X[:,0]-X[:,1])+998000)
        # f = torch.reshape(temp, (base.shape[0],1))*base
        x = torch.mean(X, dim=1)
        f = (torch.sin(x) - 2)/self.D
        # f.reshape(-1, 1).detach().requires_grad_(True)
        return f.requires_grad_(True)

    def g(self, X):
        # return self.u_exact(X)
        x = torch.mean(X, axis=1)
        u = x.pow(2) + torch.sin(x)
        return u.reshape(-1, 1).detach().requires_grad_(True)

    def u_exact(self, X):
        # u = torch.exp(-1000*((X[:,0]-0.5)**2 + (X[:,1]-0.5)**2))
        x = torch.mean(X, dim=1)
        u = x.pow(2) + torch.sin(x)
        return u.reshape(-1, 1).detach().requires_grad_(True)

    # 区域内部的采样
    def interior(self, N=NUM_TRAIN_SMAPLE):
        X =  np.random.uniform(low=-1, high=1, size=(N, self.D))
        # eps = np.spacing(1)
        # l_bounds = [l+eps for l in a]
        # u_bounds = [u-eps for u in b]
        X = torch.FloatTensor(X)
        # # X = torch.FloatTensor( sampleCubeQMC(self.D, l_bounds, u_bounds, N))
        return X.requires_grad_(True).to(self.device)

    # 边界采样
    def boundary(self, n=NUM_BOUND):
        x = np.random.uniform(low=-1, high=1, size=(n, self.D))
        for i in range(n):
            idx_num1 = random.randint(0, self.D - 1)
            if idx_num1 == 0:
                idx_num2 = random.randint(1, self.D - 1)
            else:
                idx_num2 = random.randint(0, self.D - 1)

            idx1 = random.sample(list(range(self.D)), idx_num1)
            x[i, idx1] = 1

            idx2 = random.sample(list(range(self.D)), idx_num2)
            x[i, idx2] = -1
        x_boundary = torch.FloatTensor(x).requires_grad_(True).to(self.device)
        return x_boundary

    def is_in_domain(self, x):
        x = x.cpu().detach().numpy()
        sup_flag = np.all(x < 1, axis=1, keepdims=True)
        sub_flag = np.all(x > -1, axis=1, keepdims=True)
        flag = sup_flag * sub_flag
        return flag

    def exit_estimate(self, x0, x1):
        x0 = x0.cpu().detach().numpy()
        x1 = x1.cpu().detach().numpy()
        sup_bound = np.ones_like(x0)
        sub_bound = np.ones_like(x0) * -1
        delta_x = x1 - x0
        sup_alpha = (sup_bound - x0) / delta_x
        sub_alpha = (sub_bound - x0) / delta_x
        alpha = np.concatenate((sup_alpha, sub_alpha), axis=1)
        alpha = min(alpha[alpha > 0])
        x_e = x0 + alpha * delta_x
        x_e = torch.FloatTensor(x_e).to(DEVICE)
        return x_e

    def D1(self, x0, delta_t):
        return torch.ones((x0.shape[0], 1))

    def R(self, x0, x1, delta_t):
        return (self.f(x0) + self.f(x1)) * delta_t * 0.5

class Singularity(object):
    def __init__(self, dimension):
        self.device = DEVICE
        self.r = 1
        self.theta = np.pi / 6
        self.sigma = np.sqrt(2)
        self.vertexes = np.array([[0.0, 0.0], [1.0, 0.0], [np.sqrt(3) / 2, 1 / 2]])

    # h(x)
    def g(self, x):
        x = x.cpu().detach().numpy()
        a = x[:, 0]
        a = a[:, np.newaxis]
        b = x[:, 1]
        b = b[:, np.newaxis]
        u1 = a ** 2 - b ** 2 - (1/4) * a * b
        r = np.sqrt(a ** 2 + b ** 2)
        theta = np.arctan2(b, a)
        u2 = pow(r, 2/3) * np.sin((2/3) * theta)
        u1 = torch.FloatTensor(u1).to(DEVICE)
        u2 = torch.FloatTensor(u2).to(DEVICE)
        return u2

    # f(x)
    def f(self, x):
        f = np.zeros(x.shape[0])
        f = torch.FloatTensor(f).to(DEVICE).requires_grad_(True)
        return f

    # u(x)
    def u_exact(self, x):
        x = x.cpu().detach().numpy()
        a = x[:,0]
        a = a[:,np.newaxis]
        b = x[:,1]
        b = b[:,np.newaxis]
        u1 = a**2-b**2-1/4*a*b
        r = np.sqrt(a ** 2 + b ** 2)
        theta = np.arctan2(b , a)
        u2 = pow(r, 2/3) * np.sin((2/3) * theta)
        u1 = torch.FloatTensor(u1).to(DEVICE).requires_grad_(True)
        u2 = torch.FloatTensor(u2).to(DEVICE).requires_grad_(True)
        return u2

    # 内部随机采样
    def interior(self, num_sample):
        r = self.r * np.random.uniform(low=0.0, high=1.0, size=(num_sample))
        theta = self.theta * np.random.uniform(low=0.0, high=1.0, size=(num_sample))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        samples = np.transpose(np.vstack((x, y)))
        return torch.FloatTensor(samples).to(DEVICE).requires_grad_(True)


    def boundary(self, num_sample):
        line1_x = np.random.uniform(size=(num_sample, 1))
        line1_y = np.zeros_like(line1_x)
        line2_x = np.random.uniform(size=(num_sample, 1)) * np.sqrt(3) / 2
        line2_y = np.sqrt(3) / 3 * line2_x
        line3_x = np.random.uniform(low=np.sqrt(3) / 2, high=1, size=(num_sample, 1))
        line3_y = np.sqrt(1 - np.power(line3_x, 2))
        samples = np.concatenate((np.concatenate((line1_x, line1_y), axis=1), np.concatenate((line2_x, line2_y), axis=1),
        np.concatenate((line3_x, line3_y), axis=1)), axis=0)

        return torch.FloatTensor(samples).to(DEVICE).requires_grad_(True)

    # 判断是否超出边界
    def is_in_domain(self, x):
        x = x.cpu().detach().numpy()
        r2 = x[:, 0] ** 2 + x[:, 1] ** 2
        k = x[:, 1] / x[:, 0]
        is_in_domain = (r2 <= 1.0) * (k >= 0) * (k <= np.sqrt(3) / 3) * (x[:, 0] >= 0)
        return is_in_domain


    def exit_estimate1(self, X0, X1):
        flag_area = []
        for vertex in self.vertexes:
            line_func = lambda x, y: (x - X0[:, 0]) * (vertex[1] - X0[:, 1]) - (y - X0[:, 1]) * (vertex[0] - X0[:, 0])
            flag_area.append(line_func(X1[:, 0], X1[:, 1]) > 0)
        spec_line_func = lambda x, y: x + (2 - np.sqrt(3)) * y - 1
        # spec_line_func = lambda x, y: (-2 - np.sqrt(3))*x + y + (2 + np.sqrt(3))
        flag_area.append(spec_line_func(X0[:, 0], X0[:, 1]) <= 0)
        flag_area = np.transpose(np.vstack(flag_area))
        flag_area1 = (flag_area[:, 0] == False) * (flag_area[:, 1] == True) * (flag_area[:, 3] == True) + (
                    flag_area[:, 0] == False) * (flag_area[:, 1] == True) * (flag_area[:, 2] == False) * (
                                 flag_area[:, 3] == False)
        flag_area2 = (flag_area[:, 0] == True) * (flag_area[:, 2] == False) * (flag_area[:, 3] == True) + (
                    flag_area[:, 0] == True) * (flag_area[:, 1] == True) * (flag_area[:, 2] == False) * (
                                 flag_area[:, 3] == False)
        flag_area3 = (flag_area[:, 1] == False) * (flag_area[:, 2] == True) * (flag_area[:, 3] == True) + (
                    1 - (flag_area[:, 1] == True) * (flag_area[:, 2] == False)) * (flag_area[:, 3] == False)
        return [flag_area1, flag_area2, flag_area3]

    def estimate_exit(self, X_init, X_next, flag_area, case):
        X0 = X_init[flag_area == True, :]
        X1 = X_next[flag_area == True, :]
        # 先检查是否有垂直的情况
        flag_equal = (X0[:, 0] - X1[:, 0]) == 0
        X0_k = X0[flag_equal == False, :]
        X1_k = X1[flag_equal == False, :]

        k = (X1_k[:, 1] - X0_k[:, 1]) / (X1_k[:, 0] - X0_k[:, 0])
        if case == 1:
            Xb_k = (k * X0_k[:, 0] - X0_k[:, 1]) / k
        elif case == 2:
            Xb_k = (-k * X0_k[:, 0] + X0[:, 1]) / (np.sqrt(3) / 3 - k)
        elif case == 3:
            a = np.power(k, 2) + 1
            b = 2 * k * (-k * X0_k[:, 0] + X0_k[:, 1])
            c = np.power(-k * X0_k[:, 0] + X0_k[:, 1], 2) - 1
            d = np.sqrt(np.power(b, 2) - 4 * a * c)
            Xb_k = (-b + d) / (2 * a)

        X1[flag_equal == False, 0] = Xb_k
        if case == 1:
            X1[:, 1] = np.zeros_like(X1[:, 1])
        elif case == 2:
            X1[:, 1] = np.sqrt(3) / 3 * X1[:, 0]
        elif case == 3:
            X1[:, 1] = np.sqrt(1.0 - X1[:, 0] ** 2 + 1e-6)

        X_next[flag_area == True, :] = X1

        return X_next

    def D(self, x0, delta_t):
        return torch.ones((x0.shape[0], 1))

    def R(self, x0, delta_t):
        return torch.zeros((x0.shape[0], 1))



# 内部损失函数
def loss_interior(Eq, model, x, f_test):
    # 计算拉普拉斯算子
    # u = model(x)
    # du = torch.autograd.grad(u, x,
    #                         grad_outputs = torch.ones_like(u),
    #                         create_graph = True,
    #                         retain_graph = True)[0]
    # # 计算各维二阶偏导
    # laplace = torch.zeros_like(u)
    # for i in range(Eq.D):
    #     d2u = torch.autograd.grad(du[:,i], x,
    #                             grad_outputs = torch.ones_like(du[:,i]),
    #                             create_graph = True,
    #                             retain_graph = True)[0][:,i]
    #     laplace += d2u.reshape(-1, 1)
    # Ind = torch.abs(-laplace.detach()-f_test)

    u = model(x)
    du1 = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    # fTerm = torch.DoubleTensor(Eq.f(x)).to(DEVICE)
    # loss_int = torch.mean(0.5*(torch.sum(du1*du1, 1).unsqueeze(1)) - f_test * u)
    Ind = torch.abs(0.5*(torch.sum(torch.abs(du1*du1), 1).unsqueeze(1)) - f_test * u)
    loss_int = torch.mean(Ind)
    # du = torch.autograd.grad(u, x,
    #                         grad_outputs = torch.ones_like(u),
    #                         create_graph = True,
    #                         retain_graph = True)[0]
    # # 计算各维二阶偏导
    # laplace = torch.zeros_like(u)
    # for i in range(2):
    #     d2u = torch.autograd.grad(du[:,i], x,
    #                             grad_outputs = torch.ones_like(du[:,i]),
    #                             create_graph = True,
    #                             retain_graph = True)[0][:,i]
    #     laplace += d2u.reshape(-1, 1)
    # Ind = torch.abs(-laplace.detach()-f_test)
    return loss_int, Ind

    # return LOSS_FN(-laplace.detach(), f_test), Ind

# 边界损失函数
def loss_boundary(Eq, model, x_boundary):
    # x_boundary = Eq.boundary(100)
    u_theta    = model(x_boundary).reshape(-1,1)
    u_bd       = Eq.g(x_boundary).reshape(-1,1)
    loss_bd    = LOSS_FN(u_theta, u_bd) 
    return loss_bd

# Test function
def TEST(Eq, model, NUM_TEST_SAMPLE, x_test):
    with torch.no_grad():
        # x_test = torch.Tensor(NUM_TESTING, Eq.D).uniform_(-1, 1).requires_grad_(True).to(Eq.device)

        u_real  = Eq.u_exact(x_test)
        u_pred  = model(x_test)
        Error   =  u_real - u_pred
        L2error = torch.sqrt(torch.mean(Error*Error)) / torch.sqrt(torch.mean(u_real*u_real))
    return L2error.cpu().detach().numpy()

def construct_test_set(Eq, NUM_TEST_SAMPLE):
    np.random.seed(512)
    x_test = Eq.interior(NUM_TEST_SAMPLE).requires_grad_(True).to(Eq.device)
    return x_test



#@title
# 残差网络
class Resnet(nn.Module):
    def __init__(self, dim):
        super(Resnet, self).__init__()
        self.input_layer = nn.Linear(dim, 60)
        self.hidden_layer = nn.Linear(60, 60)
        self.output_layer = nn.Linear(60, 1)
        self.num_layers = 3
        self.activation = self.swish
        # params = torch.FloatTensor(np.random.rand(1, 4)).to(DEVICE)
        # self.a = nn.Parameter(params)

    def swish(self, x, beta=1):
        # return x * torch.sigmoid(beta * x)
        return torch.sin(x)

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

# class Resnet(nn.Module):
#     def __init__(self, dim):
#         super(Resnet, self).__init__()
#         self.layer1 = nn.Linear(dim, 60)
#         self.layer2 = nn.Linear(60, 60)
#         self.layer3 = nn.Linear(60, 1)
#         self.epsilon = 0.01
#         # self.weights_init()
#
#     def swish(self, x, beta=1):
#         # return torch.sin(x)
#         return x * torch.nn.Sigmoid()(x * beta)
#
#     def forward(self, x):
#         u = x
#
#         out = self.layer1(x)
#         out = self.swish(out)
#         # out = out * torch.sigmoid(out)
#         # out = F.relu(out)
#         # out = F.tanh(out)
#
#         for i in range(3):
#             shortcut = out
#             out = self.layer2(out)
#             out = self.swish(out)
#             # out = F.relu(out)
#             # out = F.tanh(out)
#             out = out + shortcut
#
#         # out = out * torch.sigmoid(out)
#         out = self.swish(out)
#         # out = F.relu(out)
#         # out = F.tanh(out)
#         out = out + shortcut
#
#         out = self.layer3(out)
#         return out


# class FC(nn.Module):
#     def __init__(self, dim):
#         super(FC, self).__init__()
#         self.fc_i = nn.Linear(dim, 20)
#         self.fc = nn.Linear(20, 20)
#         self.fc_o = nn.Linear(20, 1)
#         self.activation = self.swish
#         # self.a = nn.Parameter(torch.ones(1).to(device).requires_grad_())
#
#     def swish(self, x, beta=1):
#         # return x * torch.sigmoid(beta * x)
#         return torch.tanh(x)
#     def forward(self, x):
#         x = self.activation(self.fc_i(x))
#         # x = torch.mul(self.a,self.activation(self.fc_i(x)))
#         for i in range(7):
#             x = self.activation(self.fc(x))
#             # x = torch.mul(self.a, self.activation(self.fc(x)))
#             i+=1
#         x = self.fc_o(x)
#         return x


def transit(x,J):
    delta_t = Delta_t
    x = x.cpu().detach().numpy()
    M = x.shape[0]
    D = x.shape[1]
    delta_W = np.sqrt(delta_t) * np.random.normal(size=(M * J, D))
    # delta_W = np.reshape(delta_W, (M, m, D))
    x0 = np.expand_dims(x, axis=1)
    x0 = np.broadcast_to(x0, (M, J, D))
    x0 = np.reshape(x0, (M * J, D))
    x1 = x0 + delta_W
    x1=torch.FloatTensor(x1)
    return x1.requires_grad_(True).to(DEVICE)

def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


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
        # self.plot_line(vertexes[0,:], vertexes[1,:], lambda x: 0 * x)
        # self.plot_line(vertexes[0,:], vertexes[2,:], lambda x: 1/np.sqrt(3) * x)
        # self.plot_line(vertexes[1,:], vertexes[2,:], lambda x: np.sqrt(1-x**2))

        plt.vlines(x=1, ymin=-1, ymax=1, color='k')
        plt.vlines(x=-1, ymin=-1, ymax=1, color='k')
        plt.hlines(y=1, xmin=-1, xmax=1, color='k')
        plt.hlines(y=-1, xmin=-1, xmax=1, color='k')

        # plt.vlines(x=0.5, ymin=-0.5, ymax=0.5,color = 'k')
        # plt.vlines(x=-0.5, ymin=-0.5, ymax=0.5,color = 'k')
        # plt.hlines(y=0.5, xmin=-0.5, xmax=0.5,color = 'k')
        # plt.hlines(y=-0.5, xmin=-0.5, xmax=0.5,color = 'k')

        # ax = plt.gca()
        # ax.spines['right'].set_color('none')
        # ax.spines['top'].set_color('none')
        # ax.spines['left'].set_position(('data', 0))
        # ax.spines['bottom'].set_position(('data', 0))

    def plot_points(self, X0, style):
        plt.plot(X0[:, 0], X0[:, 1], style)

    def plot_transit(self, X0, X1, style='k-'):
        axis_x = np.vstack((X0[:,0],X1[:,0]))
        axis_y = np.vstack((X0[:,1],X1[:,1]))
        plt.plot(axis_x, axis_y, style)

    def plot_abs_error(self, x, y, z, cmp):
        # Yp = Yp.cpu().detach().numpy()
        # Up = Eq.u_exact(points)
        # error = np.abs(Yp - Up)
        plt.contourf(x, y, z, 20, cmap=plt.cm.get_cmap(cmp))
        plt.colorbar()

draw = Draw()

# 训练
Equations = {'LP': PossionQuation, 'SG': Singularity}

parser = OptionParser(usage="Select Parameters")
parser.add_option('-e', dest='equations', type='string', default='LP')
parser.add_option('-g', dest='gpu', type='string', default='0')
options, args = parser.parse_args([])
EQ_NAME = options.equations
CUDA_ORDER = options.gpu

DEVICE = torch.device(f"cuda:{CUDA_ORDER}" if torch.cuda.is_available() else "cpu")
# Eq = Equations[EQ_NAME](DIMENSION, DEVICE)
Eq=PossionQuation(DIMENSION, DEVICE)
# Eq = Singularity(DIMENSION)
print(f'Current Equations: {EQ_NAME}')
print(f'Current Device {DEVICE}')
x_test  = construct_test_set(Eq, NUM_TEST_SAMPLE)
x0 = Eq.interior(NUM_TRAIN_SMAPLE)
def train_pipeline():

    # global x
    torch.cuda.empty_cache()
    model = Resnet(DIMENSION).to(DEVICE)
    # .double()
    # model = FC(DIMENSION).to(DEVICwvvE)
    model.apply(weights_init)
    optA = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    model.load_state_dict(torch.load('/home/GPU3/xingyu/Ad-DFLM/PINN/10D-021347-ADRM'))



    # 网络迭代
    elapsed_time     = 0    # 计时
    training_history = []    # 记录数据
    x0 = Eq.interior(NUM_TRAIN_SMAPLE)
    for step in tqdm(range(NUM_ITERATION+1)):
        if step and step % 200 == 0:
            for p in optA.param_groups:
                p['lr'] *= 0.95
                print(p['lr'])

        if step % 5000 == 0:
            x = np.linspace(-1, 1, 50)
            y = np.linspace(-1, 1, 50)
            x, y = np.meshgrid(x, y)
            # points = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)),axis=1)
            points = np.concatenate(
                (x.reshape(-1, 1), y.reshape(-1, 1), np.zeros((x.shape[0] * x.shape[1], Eq.D - 2))), axis=1)
            Yp = model(torch.FloatTensor(points).to(DEVICE))
            Yp = Yp.cpu().detach().numpy()
            Up = Eq.u_exact(torch.FloatTensor(points).to(DEVICE))
            Up = Up.cpu().detach().numpy()
            error = np.abs(Yp - Up)

            plt.contourf(x, y, error.reshape(x.shape[0], -1), 20, cmap=plt.cm.get_cmap('jet'))
            plt.colorbar()
            plt.show()



        x_boundary = Eq.boundary(NUM_BOUND)
        x0 = Eq.interior(NUM_TRAIN_SMAPLE)
        f_test = Eq.f(x0).detach().reshape(-1, 1)
        #
        # if step % 5000 == 0:
        #     x0 = Eq.interior(NUM_TRAIN_SMAPLE)
        #     f_test  = Eq.f(x0).detach().reshape(-1, 1)

        # if step % 200 == 0:
        #     draw.plot_background()
        #
        #     draw.plot_points(x0.detach().cpu().numpy(), 'b.')
        #     # draw.plot_transit(X0m[flag == False, :], Xm[flag == False, :], 'b-')
        #     # draw.plot_points(Xm_out, 'rx')
        #     plt.show()

        start_time = time.time()
        loss_int, _ = loss_interior(Eq, model, x0, f_test)
        loss_bd  = loss_boundary(Eq, model, x_boundary)
        loss     = loss_int + BETA*loss_bd

        optA.zero_grad()
        loss.backward()
        optA.step()

        def adptive(x0,stype):
            J=10
            def spread(x, a, b, c):
                x = x.unsqueeze(1)
                # x = x.cpu().detach().numpy()
                # x = np.expand_dims(x, axis=1)
                x = torch.broadcast_to(x, (a, b, c))
                x = torch.reshape(x, (a * b, c))

                return x.to(Eq.device)

            sx = transit(x0,J)
            flag = Eq.is_in_domain(sx)

            if np.any(flag==False):
                sx[flag.squeeze(-1) == False, :] = Eq.interior(sx[flag.squeeze(-1) == False, :].shape[0])
            # if np.any(flag==False):
            #     sx[flag == False, :] = Eq.interior(sx[flag == False, :].shape[0])


            x0 = x0.cpu().detach().numpy()
            x0m = np.expand_dims(x0, axis=1)
            x0m = np.broadcast_to(x0m, (NUM_TRAIN_SMAPLE, J, DIMENSION))
            x0m = np.reshape(x0m, (NUM_TRAIN_SMAPLE * J, DIMENSION))
            x0m= torch.FloatTensor(x0m).to(DEVICE)
            D_t = Eq.D1(x0m, Delta_t).to(DEVICE)
            R_t = Eq.R(x0m, sx, Delta_t)
            # D_t = Eq.D(x0m, Delta_t).to(DEVICE)
            # R_t = Eq.R(x0m, Delta_t).to(DEVICE)


            # if step!=0 and (step / 200) % 2 == 0 or (step / 200) % 2 == 1:
            # _,Ind = loss_interior(Eq, model, sx, spread(f_test,NUM_TRAIN_SMAPLE,2,1))
            Yp = model(sx)

            if stype=='indp':
                # Yp = Yp.cpu().detach().numpy()
                # Up = Eq.u_exact(sx)
                # Up = Up.cpu().detach().numpy()
                # Ind = np.abs(Yp - Up)
                _, Ind = loss_interior(Eq, model, sx, spread(f_test, NUM_TRAIN_SMAPLE, J, 1))
                # Ind = torch.FloatTensor(Ind).to(DEVICE)
                # Ind = np.abs()
                index = torch.topk(Ind, k=NUM_TRAIN_SMAPLE, dim=0)[1].cpu().detach().numpy()
                NEW = sx[index].squeeze(1)

            if stype=='indm':
                Target = (- R_t + Yp) * D_t
                td = torch.abs(Target - model(x0m))
                index = torch.topk(td, k=NUM_TRAIN_SMAPLE, dim=0)[1].cpu().detach().numpy()
                NEW = sx[index].squeeze(1)

            # flag3 = Eq.is_in_domain(NEW)
            # if np.any(flag3 == False):
            #     NEW[flag3 == False, :] = Eq.interior(len(NEW[flag3 == False, :]))

            return NEW

        # if step%1000 == 0 and step!=0:
        # x1 = adptive(x0, 'indp')
        # x0=x1
        # else:
        #     x0 = Eq.interior(NUM_TRAIN_SMAPLE)

        elapsed_time = elapsed_time + time.time() - start_time
        if step % 5 == 0:
            loss_int     = loss_int.cpu().detach().numpy()
            loss_bd      = loss_bd.cpu().detach().numpy()
            loss         = loss.cpu().detach().numpy()
            L2error      = TEST(Eq, model, NUM_TEST_SAMPLE, x_test)
            if step % 200 == 0:
                print(f'\nStep: {step:>5}, '\
                      f'Loss: {loss:>10.5f}, '\
                      f'L2: {L2error:.6f},\n'\
                          # f'L1: {L1:.6f},\n'\
                      f'Time: {elapsed_time:.2f}')
            training_history.append([step, loss, L2error, elapsed_time])

    training_history = np.array(training_history)
    print(np.min(training_history[:,2]))



    record_time = time.localtime()
    np.savetxt(
        '/home/GPU3/xingyu/Ad-DFLM/PINN/20D-PINN-[{:0>2d}{:0>2d}{:0>2d}].csv'.format(
            # Eq.__class__.__name__,
            record_time.tm_mday,
            record_time.tm_hour,
            record_time.tm_min, ),
        training_history,
        delimiter=",",
        header="step, loss, L2error, elapsed_time",
        comments='')
    record_time = time.localtime()
    torch.save(model.state_dict(),
   f'/home/GPU3/xingyu/Ad-DFLM/PINN/20D-{record_time.tm_mday:0>2d}{record_time.tm_hour:0>2d}{record_time.tm_min:0>2d}-PINN')

    save_time = f'[{save_time.tm_mday:0>2d}{save_time.tm_hour:0>2d}{save_time.tm_min:0>2d}]'
    dir_path  = os.getcwd() + f'/PossionEQ/{DIMENSION}D/'
    file_name = f'PINN-{NUM_ITERATION}itr-{SAMPLE_FREQUENCY}N-decay{IS_DECAY}.csv'
    file_path = dir_path + save_time + file_name

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 保存训练数据
    np.savetxt(file_path, training_history,
                delimiter =",",
                header    ="step, L2error, loss, loss_int, loss_bd, elapsed_time",
                comments  ='')
    print('Training History Saved!')
    # 保存训练模型
    if IS_SAVE_MODEL:
        torch.save(model.state_dict(), dir_path + save_time + '-U_net')
        print('PINN Network Saved!')


if __name__ == "__main__":
    train_pipeline()
