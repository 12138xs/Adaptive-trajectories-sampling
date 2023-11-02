import numpy as np
import torch


class ATS():
    def __init__(self, high, low, J1, J2, device, type) -> None:
        self.DIMENSION = 2
        self.high = high
        self.low = low
        self.sigma = np.sqrt(2)
        self.delta_t = 0.0001

        self.J1 = J1
        self.J2 = J2
        self.device = device
        self.type = type

    def bound_cond(self, x):
        return np.exp( -1000 * np.sum((x-0.5)**2, axis=1) )

    def source_function(self, x):
        temp = -1000 * np.sum((x - 0.5)**2, axis=1)
        return 4000 * (temp + 1) * np.exp(temp)

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

    def transit(self, x0):
        M = x0.shape[0]
        D = x0.shape[1]
        delta_W = np.sqrt(self.delta_t) * np.random.normal(size=(M, D))
        x1 = x0 + self.sigma * delta_W
        return x1

    def D(self, x0):
        return np.ones((x0.shape[0], 1))

    def R(self, x0, x1):
        return (self.source_function(x0) + self.source_function(x1)) * self.delta_t * 0.5
        # return

    def spread(self, x0, J):
        M = x0.shape[0]
        D = x0.shape[1]
        delta_W = np.sqrt(self.delta_t) * np.random.normal(size=(M * J, D))
        # delta_W = np.reshape(delta_W, (M, m, D))
        x0 = np.expand_dims(x0, axis=1)
        x0 = np.broadcast_to(x0, (M, J, D))
        x0 = np.reshape(x0, (M * J, D))
        x1 = x0 + self.sigma * delta_W
        return x1
    
    def Ind(self, x, model, J2=1, tol=0.1):
        I = x.shape[0]
        xm = self.spread(x, J2)
        x0m = np.expand_dims(x, axis=1)
        x0m = np.broadcast_to(x0m, (I, J2, self.DIMENSION))
        x0m = np.reshape(x0m, (I * J2, self.DIMENSION))
        y0m = model.predict(xm)[0]

        flag = self.is_in_domain(xm)
        if np.any(flag == False):
            x0m_out = x0m[flag.squeeze(-1) == False, :]
            xm_out  = xm[flag.squeeze(-1) == False, :]
            xm_new  = self.exit_estimate(x0m_out, xm_out)
            y0m[flag.squeeze(-1) == False, :] = self.bound_cond(xm_new).reshape(-1, 1)

        D_t = self.D(x0m).reshape(-1, 1)
        R_t = self.R(x0m, xm).reshape(-1, 1)
        Target1 = np.reshape((- R_t + y0m) * D_t, (I, J2, 1))
        Target1 = np.mean(Target1, axis=1)
        Ind = -np.abs(Target1.squeeze() - model.predict(x)[0].squeeze()) + tol
        return Ind


    def resample_ats(self, X_train, model, func):
        def Ind(x, I, J, J2, method='global'):
            Ind = -func(x)
            Ind = torch.tensor(Ind, dtype=torch.float32).to(self.device)
            
            if method == 'global':
                # NEW = []
                index = torch.topk(Ind, k=500, dim=0)[1].cpu().detach().numpy()
                NEW = x[index] # .squeeze(1)

            if method == 'local':
                Ind = torch.reshape(Ind, (I, J))
                # x = np.reshape(x, (I, m, self.DIMENSION))
                _, index = torch.max(Ind, dim=1)
                index = index.unsqueeze(1).cpu().detach().numpy()
                NEW = x[index].squeeze(1)

            return NEW

        # X0, x_train = np.split(X_train, [300])
        X0 = X_train
        # print(X0)
        # print(x_train)
        X1 = self.spread(X0, self.J1)
        flag = self.is_in_domain(X1)
        if np.any(flag == False):
            X1[flag.squeeze(-1) == False, :] = np.random.uniform(self.low, self.high, (len(X1[flag.squeeze(-1) == False, :]), 2))
        X1 = Ind(X1, X0.shape[0], self.J1, self.J2, self.type)
        return X1