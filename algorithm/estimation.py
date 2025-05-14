from src.generate_data import generate_data, generate_complex_data
from src.subcase import CalCase, CalComplexCase
from src.utils import performance
import numpy as np

# configuration of data
samples_list = [500,1000,2000,5000,10000,15000,20000,25000]

latent, observed = 1, 3
for distribution in ['laplace', 'uniform']:
    for graph in ['a', 'b', 'c']:
        print(distribution, 'Fig. 3(', graph, ')')
        results = [[], []] # mean, std
        for n_samples in samples_list:
            weight_pred, weight_true = [], []
            for seed in range(100):
                data, weights, w_id = generate_data(graph, n_samples=n_samples, distribution=distribution, latent=latent, observed=observed, seed=seed)
                Z, T, O = data[:, 0], data[:, 1], data[:, 2]
                weight_pred.append(CalCase(T, O, Z, graph))
                weight_true.append(weights[w_id])
            weight_true = np.array(weight_true)
            weight_pred = np.array(weight_pred)
            error = np.abs((weight_true - weight_pred)) / np.maximum(np.abs(weight_true), np.abs(weight_pred))
            mean, std = performance(error)
            results[0].append(mean)
            results[1].append(std)
            print(f'{n_samples}: {mean:.2f}±{std:.2f}')
        print('\n')
        # np.save(f'data/ours_{distribution}_{graph}.npy', results)

latent, observed = 2, 4
for distribution in ['laplace', 'uniform']:
    for graph in ['a', 'b', 'c']:
        print(distribution, 'Fig. 6(', graph, ')')
        results = [[], []] # mean, std
        for n_samples in samples_list:
            weight_pred, weight_true = [], []
            for seed in range(100):
                data, weights, w_id = generate_complex_data(graph, n_samples=n_samples, distribution=distribution, latent=latent, observed=observed, seed=seed)
                Z1, Z2, T, O = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
                weight_pred.append(CalComplexCase(T, O, Z1, Z2, graph))
                weight_true.append(weights[w_id])
            weight_true = np.array(weight_true)
            weight_pred = np.array(weight_pred)
            error = np.abs((weight_true - weight_pred)) / np.maximum(np.abs(weight_true), np.abs(weight_pred))
            mean, std = performance(error)
            results[0].append(mean)
            results[1].append(std)
            print(f'{n_samples}: {mean:.2f}±{std:.2f}')
        # np.save(f'data/ours_{distribution}_{graph}_complex.npy', results)