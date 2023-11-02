import torch
import torch.nn as nn
import os
import numpy as np
import sys 
import shutil 
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('FI_PINNs')

from utils.freeze_weights import freeze_by_idxs


device = torch.device("cuda:0")
Dimension = 80

class DNN(nn.Module):
    """This class carrys out DNN"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_hiddens):
        super().__init__()
        self.num_layers   = num_hiddens
        self.input_layer  = nn.Linear(input_dim, hidden_dim)
        self.linear = nn.ModuleList()
        for _ in range(num_hiddens):
            self.linear.append( nn.Linear(hidden_dim, hidden_dim) )
            self.linear.append( nn.Linear(hidden_dim, hidden_dim) )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = torch.sin

    def forward(self, x):
        out = self.activation( self.input_layer(x) )
        for i in range( self.num_layers ):
            res = out
            res = self.activation( self.linear[2*i](res) )
            res = self.activation( self.linear[2*i+1](res) )
            out = out + res
        out = self.output_layer(out)
        return out
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, 0.1)
            m.bias.data.fill_(0.001)


class PinnOnePeak:
    """This script carrys out unbounded pinn pdes"""
    def __init__(self, X_b_train, u_b, img_save_path) -> None:
        self.img_save_path = img_save_path
        self.loss_func = nn.MSELoss()
        self.iter = 0
        self.net = DNN(Dimension, 256, 1, 2).to(device)
        # self.net.apply(self.net.init_weights)

        self.u_b = torch.tensor(u_b.reshape(-1, 1), dtype = torch.float32).to(device)
        self.x_b = torch.tensor(X_b_train, dtype=torch.float32, requires_grad=True).to(device)

        # u_true = lambda x: np.mean(x, axis=1)
        self.points = np.random.uniform(-1,1,(10000, Dimension))
        self.true_u = self.u_true(self.points).reshape(100, 100)
        self.X, self.Y = self.points[:, 0].reshape(100, 100), self.points[:, 1].reshape(100, 100)
        self.optim_adam = torch.optim.Adam(self.net.parameters(), lr = 1e-4)

    def u_true(self, x):
        X = np.mean(x, axis=1).reshape(-1, 1)
        return np.power(X, 2) + np.sin(X)

    def update_bound(self, X_b_train, u_b):
        self.u_b = torch.tensor(u_b, dtype = torch.float32, requires_grad=True).to(device)
        self.x_b = torch.tensor(X_b_train, dtype=torch.float32, requires_grad=True).to(device)       

    def net_u(self, x):
        u = self.net( x )
        return u

    def net_f(self, x, FLAG=True):
        u = self.net_u(x)
        u_x = torch.autograd.grad(u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        laplace_u = torch.zeros_like(u)
        for i in range(Dimension): 
            u_xx = torch.autograd.grad(u_x[:,i], x, 
                grad_outputs=torch.ones_like(u_x[:,i]),
                retain_graph=True,
                create_graph=FLAG)[0][:, i:i+1]
            laplace_u += u_xx.reshape(-1, 1)

        f = - laplace_u - self.source_function(x)
        return f
    
    def source_function(self, x):
        X = torch.mean(x, dim=1).reshape(-1, 1)
        f = (torch.sin(X) - 2) / Dimension
        return f.detach().requires_grad_(True)


    def closure(self):
        self.optimizer.zero_grad()
        
        # u & f predictions:
        u_b_prediction = self.net_u(self.x_b, self.y_b)
        f_prediction = self.net_f(self.x_f, self.y_f)

        # losses:
        u_b_loss = self.loss_func(u_b_prediction, self.u_b)
        f_loss = self.loss_func(f_prediction, torch.zeros_like(f_prediction).to(device))
        ls = f_loss + u_b_loss

        # derivative with respect to net's weights:
        ls.backward()
        self.error.append(self.calculate_error())

        # increase iteration count:
        self.iter += 1

        # print report:
        if not self.iter % 10:
            print('Epoch: {0:}, Loss: {1:.4f}'.format(self.iter, ls))
            print('             Error: {1:.4f}'.format(self.error[-1]))

        return ls

    def train(self, X_f_train, adam_iters, i = 0):
        self.update(X_f_train)
        self.net.train()
        self.error = []
        batch_sz  = self.x_f.shape[0]
        n_batches = self.x_f.shape[0] // batch_sz
        # if i >= 1:
        #     freeze_by_idxs(self.net, [0, 1, 2])
        # self.optimizer.step(self.closure)
        for i in range(adam_iters):
            for j in range(n_batches):
                x_f_batch = self.x_f[j*batch_sz:(j*batch_sz + batch_sz),]
                self.optim_adam.zero_grad()
                u_b_prediction = self.net_u(self.x_b)
                f_prediction = self.net_f(x_f_batch)

                # losses:
                u_b_loss = self.loss_func(u_b_prediction, self.u_b)
                f_loss = self.loss_func(f_prediction, torch.zeros_like(f_prediction).to(device))
                ls = f_loss + u_b_loss # 
                ls.backward()
                
                self.optim_adam.step()
            # scheduler.step(ls)
            if not i % 100:
                self.error.append(self.calculate_error())
        #         print('current epoch: %d, loss: %.7f, error: %.7f'%(i, ls.item(), self.error[-1]))
        print('Min L2 relative error: %.7f'%(np.min(self.error)))
        print('---------------------------------------------------')
        # self.optimizer.step(self.closure)
        return np.min(self.error)
    

    def update(self, X_f_train):
        self.x_f = torch.tensor(X_f_train,
                                dtype=torch.float32,
                                requires_grad=True).to(device)

    def predict(self, points, FLAG=True):
        x = torch.tensor(points, requires_grad = True).float().to(device)

        self.net.eval()
        u = self.net_u(x)
        f = self.net_f(x,FLAG)
        u = u.to('cpu').detach().numpy()
        f = f.to('cpu').detach().numpy()
        return u, f

    def plot_error(self, add_points = None, prefix = None):
        """ plot the solution on new data """
        u_predict, f_predict = self.predict(self.points, FLAG=False)
    
        u_predict = u_predict.reshape(self.true_u.shape)
        f_predict = f_predict.reshape(self.true_u.shape)

        fig = plt.figure(figsize=(15,10))
        fig.suptitle('Initial points:' + str(len(self.x_f)))
        ax1 = fig.add_subplot(221)
        im1 = ax1.contourf(self.X, self.Y, abs(f_predict), cmap = "winter")
        if add_points is not None:
            fig.suptitle('Initial points: ' + str(len(self.x_f)) +  ' ' + 'add points: ' + str(len(add_points)))
            ax1.scatter(add_points[:,0], add_points[:,1], marker = 'o', edgecolors = 'red', facecolors = 'white')
        ax1.set_title("Equation error")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='2%', pad=0.08)
        fig.colorbar(im1, cax=cax, orientation='vertical')
    
        ax2 = fig.add_subplot(222)
        im2 = ax2.contourf(self.X, self.Y, abs(u_predict - self.true_u), cmap = "winter")
        ax2.set_title("Solution error")
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='2%', pad=0.08)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        
        ax3 = fig.add_subplot(223)
        ax3.plot(self.error, label = "l_2 error")
        ax3.set_xlabel('Epoches * 100')
        ax3.legend()
        plt.savefig(os.path.join(self.img_save_path, prefix + '.png'))
        plt.close()
        # plt.show()
    
    def calculate_error(self):
        u_predict, _ = self.predict(self.points, FLAG=False)
        error = np.linalg.norm(u_predict.squeeze() - self.true_u.flatten())/np.linalg.norm(self.true_u.flatten())
        return error
    
    def absolute_error(self):
        u_predict, _ = self.predict(self.points, FLAG=False)
        error = np.linalg.norm(u_predict.squeeze() - self.true_u.flatten())
        return error

