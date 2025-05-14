import torch
import numpy as np
import networkx as nx

def generate_data(id, n_samples=1000, distribution='laplace', latent=1, observed=3, seed=0):
    if id == 'a':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        w_id = 3
    elif id == 'b':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        w_id = 4
    elif id == 'c':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
        w_id = 4
    elif id == 'd':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
        w_id = 3
    elif id == 'e':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
        w_id = 5
    elif id == 'f':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0]])
        w_id = 4
    elif id == 'g':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 0]])
        w_id = 5
    elif id == 'h':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 0]])
        w_id = 4
    else:
        raise ValueError
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    g = nx.DiGraph(iv_adj)
    n_weights = len(g.edges())
    while True:
        weights = torch.Tensor(n_weights).uniform_(-0.5, 0.5)
        for i in range(n_weights):
            if weights[i]>0:
                weights[i] += 0.5
            else:
                weights[i] -= 0.5
        
        adj = torch.zeros([latent+observed, latent+observed])
        for i in range(n_weights):
            adj[list(g.edges)[i][1], list(g.edges)[i][0]]=weights[i]
        mix = (torch.inverse(torch.eye(latent+observed) - adj))
        if np.all(np.abs(mix.numpy()[np.abs(mix.numpy()) > 1e-6]) > 0.2): # faithfulness
            break
    epsilon = np.load(f'../{distribution}_noises/noise_{n_samples}.npy')[np.random.choice(np.arange(10), latent+observed, replace=False)]
    epsilon = epsilon - np.mean(epsilon, axis=1,keepdims=True)
    epsilon = torch.Tensor(epsilon)            
    data = mix.matmul(epsilon).t()
    data = data[:,range(latent, observed+latent)]
    
    return data.numpy(), weights.numpy(), w_id


def generate_complex_data(id, n_samples=1000, distribution='laplace', latent=2, observed=4, seed=0):
    if id == 'a':
        iv_adj = np.array([[0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]])
        w_id = 6
    elif id == 'b':
        iv_adj = np.array([[0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0]])
        w_id = 8
    elif id == 'c':
        iv_adj = np.array([[0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0]])
        w_id = 9
    else:
        return 0
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    g = nx.DiGraph(iv_adj)
    n_weights = len(g.edges())
    while True:
        weights = torch.Tensor(n_weights).uniform_(-0.5, 0.5)
        for i in range(n_weights):
            if weights[i]>0:
                weights[i] += 0.5
            else:
                weights[i] -= 0.5
        
        adj = torch.zeros([latent+observed, latent+observed])
        for i in range(n_weights):
            adj[list(g.edges)[i][1], list(g.edges)[i][0]]=weights[i]
        mix = (torch.inverse(torch.eye(latent+observed) - adj))
        if np.all(np.abs(mix.numpy()[np.abs(mix.numpy()) > 1e-6]) > 0.2): # faithfulness
            break
    epsilon = np.load(f'../{distribution}_noises/noise_{n_samples}.npy')[np.random.choice(np.arange(10), latent+observed, replace=False)]
    epsilon = epsilon - np.mean(epsilon, axis=1,keepdims=True)
    epsilon = torch.Tensor(epsilon)            
    data = mix.matmul(epsilon).t()
    data = data[:,range(latent, observed+latent)]
    
    return data.numpy(), weights.numpy(), w_id

# data, g, adj, weights = generate_data('a', n_samples=1000, latent=1, observed=3)