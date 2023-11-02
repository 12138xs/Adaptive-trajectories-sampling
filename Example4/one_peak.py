import sys
sys.path.append('FI_PINNs')

import torch 
import numpy as np
from functools import partial
import os
import copy 
import pickle 
from model.pinn_one_peak_torch import PinnOnePeak
from utils.ISAG import SAIS
from utils.density import Uniform
from utils.gene_data import generate_peak1_samples
from utils.ats_data import ATS
# torch.manual_seed(1)
import matplotlib.pyplot as plt
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('root', type=str, default='./results')
args = parser.parse_args()

model_save_path = os.path.join(args.root, 'models') 
data_save_path  = os.path.join(args.root, 'data') 
img_save_path   = os.path.join(args.root, 'figures') 

if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

Error = []
P_failure = []
Dimension = 10
device = torch.device("cuda:0")

def power_f(x, model, tol = 0.1):
    f = np.zeros(x.shape[0])
    f = -abs(model.predict(x, False)[1].squeeze()) + tol
    return f

def generate_rar_samples(power_f, num_samples, num, x_interval, y_interval):
    samples = np.zeros((num, Dimension))
    for i in range(Dimension):
        samples[:,i] = np.random.uniform(x_interval[i], y_interval[i], num)
    nacp = power_f(samples)
    samples = samples[np.argsort(nacp.squeeze())]
    return samples[:num_samples]

def shuffle_data(X_f_train, len=None):
    if len is None:
        len = X_f_train.shape[0]
    index  = np.arange(0, X_f_train.shape[0])
    np.random.shuffle(index)
    return X_f_train[index[:len], :]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

### Repeat training
for j in range(1):

    ###Parameters for SAIS
    lb = np.array([-1 for i in range(Dimension)])
    ub = np.array([ 1 for i in range(Dimension)])
    mu = np.array([ 0 for i in range(Dimension)])
    p_failure = []
    samples = []
    tol_r  = 0.01
    tol_p = 0.1
    N1 = 500 
    N2 = 2000
    p0 = 0.1
    sais = SAIS(N1, N2, p0, 500)

    J1 = 10
    J2 = 1
    ats  = ATS(Dimension, J1, J2, device, type='global')

    error = {'PINN_model':[],
            'SAIS_model':[],
            'RAR_model':[], 
            'ATS_model':[],
            'ATS_m_model':[]}
    sample_times = {'PINN_model':[],
            'SAIS_model':[],
            'RAR_model':[], 
            'ATS_model':[],
            'ATS_m_model':[]}
    times = {'PINN_model':[],
            'SAIS_model':[],
            'RAR_model':[], 
            'ATS_model':[],
            'ATS_m_model':[]}
    sizes = {'PINN_model':[],
            'SAIS_model':[],
            'RAR_model':[], 
            'ATS_model':[],
            'ATS_m_model':[]}
    epoches = 2000
    Iters = 9
    
    ###Generate data
    N_b = 1000
    N_f = 2000
    X_f_train, X_b_train, u_b = generate_peak1_samples(N_b, N_f, lb, ub, Dimension)
    # print(X_b_train)
    
    # ###Initial training
    pinn = PinnOnePeak(X_b_train, u_b, img_save_path)
    print('Start initial training----------------------------------------')
    begin = time.time()
    L2E = pinn.train(X_f_train, epoches, 0)
    end = time.time()
    times['PINN_model'].append(end - begin)
    times['SAIS_model'].append(end - begin)
    times['RAR_model'].append(end - begin)
    times['ATS_model'].append(end - begin)
    times['ATS_m_model'].append(end - begin)
    torch.save(pinn, os.path.join(model_save_path, 'initial_model'))
    sais_pinn  = torch.load(os.path.join(model_save_path, 'initial_model'))
    rar_pinn   = torch.load(os.path.join(model_save_path, 'initial_model'))
    ats_pinn   = torch.load(os.path.join(model_save_path, 'initial_model'))
    ats_m_pinn = torch.load(os.path.join(model_save_path, 'initial_model'))

    # sais_X_f_train = copy.deepcopy(X_f_train)
    # rar_X_f_train = copy.deepcopy(X_f_train)
    ats_X_f_train = copy.deepcopy(X_f_train)
    ats_m_X_f_train = copy.deepcopy(X_f_train)

    error['PINN_model'].append(L2E)
    error['SAIS_model'].append(L2E)
    error['RAR_model'].append(L2E)
    error['ATS_model'].append(L2E)
    error['ATS_m_model'].append(L2E)

    for i in range(Iters):
        pinn.plot_error(prefix = "PINN_model" + str(i))
        # sais_pinn.plot_error(prefix = "SAIS_model" + str(i))
        # rar_pinn.plot_error(prefix = "rar_model" + str(i))
        ats_pinn.plot_error(prefix = "ats_model" + str(i))
        ats_m_pinn.plot_error(prefix = "ats_m_model" + str(i))

        power_function = partial(power_f, model = pinn, tol = tol_r)
        rar_power_function = partial(power_f, model = rar_pinn, tol = 0)
        ats_power_function = partial(power_f, model = ats_pinn, tol = 0)
        ats_indecater = partial(ats.Ind, model = ats_m_pinn, J2 = 1, tol = 0)

        ### Generate new samples
        # begin = time.time()
        # sais_samples, p = sais.sample_uniform(power_function, Uniform, lb, ub)
        # sample_times['SAIS_model'].append(time.time() - begin)
        # begin = time.time()
        # rar_samples = generate_rar_samples(rar_power_function, 500, 2000, lb, ub)
        # sample_times['RAR_model'].append(time.time() - begin)
        begin = time.time()
        ats_samples = ats.resample_ats(ats_X_f_train, ats_pinn, ats_power_function)
        sample_times['ATS_model'].append(time.time() - begin)
        begin = time.time()
        ats_m_samples = ats.resample_ats(ats_m_X_f_train, ats_m_pinn, ats_indecater)
        sample_times['ATS_m_model'].append(time.time() - begin)


        ### concat new samples
        # sais_X_f_train = np.vstack([sais_X_f_train, sais_samples])
        # rar_X_f_train = np.vstack([rar_X_f_train, rar_samples])
        ats_X_f_train = np.vstack([ats_X_f_train, ats_samples]) # ats_samples # 
        ats_m_X_f_train = np.vstack([ats_m_X_f_train, ats_m_samples]) # ats_m_samples # 

        # sais_X_f_train = shuffle_data(sais_X_f_train)
        # rar_X_f_train = shuffle_data(rar_X_f_train)
        ats_X_f_train = shuffle_data(ats_X_f_train, N_f)
        ats_m_X_f_train = shuffle_data(ats_m_X_f_train, N_f)

        sizes['PINN_model'].append(X_f_train.shape[0])
        # sizes['SAIS_model'].append(sais_X_f_train.shape[0])
        # sizes['RAR_model'].append(rar_X_f_train.shape[0])
        sizes['ATS_model'].append(ats_X_f_train.shape[0])
        sizes['ATS_m_model'].append(ats_m_X_f_train.shape[0])

        _, X_b_train, u_b = generate_peak1_samples(N_b, N_f, lb, ub, Dimension)

        # with open(data_save_path + '/sais_samples' + str(i), 'wb') as f:
        #     pickle.dump(sais_samples, f)
        # with open(data_save_path + '/rar_samples' + str(i), 'wb') as f:
        #     pickle.dump(rar_samples, f)
        with open(data_save_path + '/ats_samples' + str(i), 'wb') as f:
            pickle.dump(ats_samples, f)
        with open(data_save_path + '/ats_m_samples' + str(i), 'wb') as f:
            pickle.dump(ats_m_samples, f)

        # sais_pinn.plot_error(add_points = sais_samples, prefix = "SAIS_model_add_points" + str(i))
        # rar_pinn.plot_error(add_points = rar_samples, prefix = "rar_model_add_points" + str(i))
        ats_pinn.plot_error(add_points = ats_samples, prefix = "ats_model_add_points" + str(i))
        ats_m_pinn.plot_error(add_points = ats_m_samples, prefix = "ats_m_model_add_points" + str(i))

        # print('Start pinn retraining----------------------------------------')
        # begin = time.time()
        # pL2E = pinn.train(X_f_train, epoches, i + 1)
        # times['PINN_model'].append(time.time() - begin)
        # pinn.plot_error(prefix="PINN_model_after training" + str(i))
        # print('Start sais retraining----------------------------------------')
        # begin = time.time()
        # sais_pinn.update_bound(X_b_train, u_b)
        # fL2E = sais_pinn.train(sais_X_f_train, epoches, i + 1)
        # times['SAIS_model'].append(time.time() - begin)
        # pinn.plot_error(prefix="SAIS_model_after training" + str(i))
        # print('Start rar retraining----------------------------------------')
        # begin = time.time()
        # rar_pinn.update_bound(X_b_train, u_b)
        # rL2E = rar_pinn.train(rar_X_f_train, epoches, i+1)
        # times['RAR_model'].append(time.time() - begin)
        # rar_pinn.plot_error(prefix="rar_model_after training" + str(i))
        print('Start ats retraining----------------------------------------')
        begin = time.time()
        ats_pinn.update_bound(X_b_train, u_b)
        a_L2E = ats_pinn.train(ats_X_f_train, epoches, i+1)
        times['ATS_model'].append(time.time() - begin)
        ats_pinn.plot_error(prefix="ats_model_after training" + str(i))
        print('Start ats_m retraining----------------------------------------')
        begin = time.time()
        ats_m_pinn.update_bound(X_b_train, u_b)
        mL2E = ats_m_pinn.train(ats_m_X_f_train, epoches, i+1)
        times['ATS_m_model'].append(time.time() - begin)
        ats_m_pinn.plot_error(prefix="ats_m_model_after training" + str(i))

        # torch.save(pinn, os.path.join(model_save_path, 'PINN_model' + str(i)))
        # torch.save(sais_pinn, os.path.join(model_save_path, 'sais_model' + str(i)))
        # torch.save(rar_pinn, os.path.join(model_save_path, 'rar_model'+ str(i)))
        torch.save(ats_pinn, os.path.join(model_save_path, 'ats_model'+ str(i)))
        torch.save(ats_m_pinn, os.path.join(model_save_path, 'ats_m_model'+ str(i)))
        # error['PINN_model'].append(pL2E)
        # error['SAIS_model'].append(fL2E)
        # error['RAR_model'].append(rL2E)
        error['ATS_model'].append(a_L2E)
        error['ATS_m_model'].append(mL2E)

    # power_function = partial(power_f, model = pinn, tol = tol_r)
    # samples, p = sais.sample_uniform(power_function, Uniform, lb, ub)
    # p_failure.append(p)

    Error.append(error)
    # P_failure.append(p_failure)
    
    
with open(os.path.join(data_save_path, 'one_peak_error'), 'wb') as f:
    pickle.dump(Error, f)
# with open(os.path.join(data_save_path, 'one_peak_failure'), 'wb') as f:
#     pickle.dump(P_failure, f)
with open(os.path.join(data_save_path, 'one_peak_samples'), 'wb') as f:
    pickle.dump(sample_times, f)
with open(os.path.join(data_save_path, 'one_peak_times'), 'wb') as f:
    pickle.dump(times, f)
with open(os.path.join(data_save_path, 'one_peak_sizes'), 'wb') as f:
    pickle.dump(sizes, f)



