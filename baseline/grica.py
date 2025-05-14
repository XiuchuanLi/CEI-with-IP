import torch
import numpy as np

from src.methods import graphical_rica
from src.generate_data import generate_data, generate_complex_data
from src.utils import performance

# hyper-parameters of RICA
lr = 0.1
w_init = 'cov_guess'
lmbda_GRICA = 0
momentum = 0
epochs = 400
samples_list = [500,1000,2000,5000,10000,15000,20000,25000]


latent, observed = 1, 3
for distribution in ['laplace', 'uniform']:
    for graph in ['a', 'b', 'c']:
        print(distribution, 'Fig. 3(', graph, ')')
        results = [[], []] # mean, std
        for n_samples in samples_list:
            weight_GRICA_pred, weight_GRICA_true = [], []
            weight_CM_pred, weight_CM_true = [], []
            for seed in range(100):
                data, g, weights, w_id = generate_data(graph, n_samples=n_samples, distribution=distribution, latent=latent, observed=observed, seed=seed)
                #GRICA
                d_cov = (data.t()).cov()
                U, S, V = d_cov.svd()
                S_2=torch.inverse(torch.diag(S.sqrt()))
                W_w = S_2.matmul(V.t())
                data_whitened = W_w.matmul(data.t()).t()
                w_true = weights
                loss_data, w_loss_data, weight_pred, weight_true = graphical_rica(latent, observed, g, data, data_whitened, epochs, lr, W_w, w_init, w_true, momentum, lmbda_GRICA)
                tmp_pred, tmp_true = weight_pred[w_id].detach().item(), weight_true[w_id].detach().item()
                weight_GRICA_pred.append(tmp_pred)
                weight_GRICA_true.append(tmp_true)
                
            weight_GRICA_true = np.array(weight_GRICA_true)
            weight_GRICA_pred = np.array(weight_GRICA_pred)
            error_GRICA = np.abs(weight_GRICA_true - weight_GRICA_pred) / np.maximum(np.abs(weight_GRICA_true), np.abs(weight_GRICA_pred))
            mean, std = performance(error_GRICA)
            results[0].append(mean)
            results[1].append(std)
            print(f'{n_samples}: {mean:.2f}±{std:.2f}')
        # np.save(f'data/grica_{distribution}_{graph}.npy', results)


latent, observed = 2, 4
for distribution in ['laplace', 'uniform']:
    for graph in ['a', 'b', 'c']:
        print(distribution, 'Fig. 6(', graph, ')')
        results = [[], []] # mean, std
        for n_samples in samples_list:
            weight_GRICA_pred, weight_GRICA_true = [], []
            weight_CM_pred, weight_CM_true = [], []
            for seed in range(100):
                data, g, weights, w_id = generate_complex_data(graph, n_samples=n_samples, distribution=distribution, latent=latent, observed=observed, seed=seed)
                #GRICA
                d_cov = (data.t()).cov()
                U, S, V = d_cov.svd()
                S_2=torch.inverse(torch.diag(S.sqrt()))
                W_w = S_2.matmul(V.t())
                data_whitened = W_w.matmul(data.t()).t()
                w_true = weights
                loss_data, w_loss_data, weight_pred, weight_true = graphical_rica(latent, observed, g, data, data_whitened, epochs, lr, W_w, w_init, w_true, momentum, lmbda_GRICA)
                tmp_pred, tmp_true = weight_pred[w_id].detach().item(), weight_true[w_id].detach().item()
                weight_GRICA_pred.append(tmp_pred)
                weight_GRICA_true.append(tmp_true)
            
            weight_GRICA_true = np.array(weight_GRICA_true)
            weight_GRICA_pred = np.array(weight_GRICA_pred)
            error_GRICA = np.abs(weight_GRICA_true - weight_GRICA_pred) / np.maximum(np.abs(weight_GRICA_true), np.abs(weight_GRICA_pred))
            mean, std = performance(error_GRICA)
            results[0].append(mean)
            results[1].append(std)
            print(f'{n_samples}: {mean:.2f}±{std:.2f}')
        # np.save(f'data/grica_{distribution}_{graph}_complex.npy', results)
