import torch
import numpy as np
import networkx as nx

from .utils import intersection, count_lists
from torch.nn import Parameter


def get_ratio_(Z, D, deg=2):
    """
    Computes ratio between alpha_d/alpha_z

    Parameters:
        - Z (np.array): proxy variable observations
        - D (np.array): treatment observations
        - deg (int): moment of non-guassianity (equal to the (n-1) from the original paper)
    """
    var_u = np.mean(Z*D)
    sign = np.sign(var_u)
    
    diff_normal_D = np.mean(D**(deg)*Z) - deg*var_u*np.mean(D**(deg-1))
    diff_normal_Z = np.mean(Z**(deg)*D) - deg*var_u*np.mean(Z**(deg-1))
    
    alpha_sq = ((diff_normal_D) / (diff_normal_Z))
    if alpha_sq < 0:
        alpha_sq = -(abs(alpha_sq)**(1/(deg-1)))
    else:
        alpha_sq = alpha_sq**(1/(deg-1))
    alpha_sq = abs(alpha_sq)*sign
    
    return alpha_sq


def cross_moment(Z, D, Y, deg=2):
    denominator = 0
    while denominator==0:
        alpha_sq = get_ratio_(Z, D, deg)
        numerator = np.mean(D*Y) - alpha_sq*np.mean(Y*Z)
        denominator = np.mean(D*D) - alpha_sq*np.mean(D*Z)
        deg += 1
    return numerator / denominator


def double_cross_moment(Z1, Z2, D, Y, deg=2):
    denominator = 0
    while denominator==0:
        alpha_sq1 = get_ratio_(Z1, D, deg)
        alpha_sq2 = get_ratio_(Z2, D, deg)
        numerator = np.mean(D*Y) - alpha_sq1*np.mean(Y*Z1) - alpha_sq2*np.mean(Y*Z2)
        denominator = np.mean(D*D) - alpha_sq1*np.mean(D*Z1) - alpha_sq2*np.mean(D*Z2)
        deg += 1
    return numerator / denominator


# Remark: data - before withening
def init_w_guess_(data, g, latent, observed):
    up_data = data.t()
    w = torch.zeros(len(g.edges()))
    mask = torch.zeros(len(g.edges()))

    for i, e in enumerate(g.edges()):
        if e[0] < latent:
            w[i] = torch.Tensor(1).normal_().item()
            mask[i] = 1
        else:
            G_cov = up_data.cov()
            w[i] = G_cov[e[0]-latent,e[1]-latent]/G_cov[e[0]-latent,e[0]-latent]
            up_data[:, e[1]-latent] = up_data[:, e[1]-latent]-w[i]*up_data[:,e[0]-latent]

            an_s = sorted(nx.ancestors(g, e[0]))
            i_s = intersection(an_s,list(range(latent)))

            if len(i_s) > 0 :
                an_t = sorted(nx.ancestors(g, e[1]))
                i_t = intersection(an_t,list(range(latent)))
                if len(i_t)>0:
                    ints = intersection(i_s,i_t)
                    if len(ints)>0:
                        mask[i] = 1
    return w, mask


def graphical_rica(latent, observed, g, data, data_whitening, epochs, lr, W_w, w_init, w_true, momentum=0, lmbda=0):
    
    """
        Graphical adaptation of RICA
        
        Parameters:
            - latent (int): Number of hidden variables.
            - observed (int): Number of observed variables.
            - g (nx.DiGraph): The DAG as a NetworkX DiGraph object.
            - data (torch.Tensor): Input data.
            - lr(double): Learning rate of the optimizer.
            - epochs (int): Number of optimization epochs.
            - W_w (torch.Tensor): Whitening matrix.
            - w_init (str): Weight initialization strategy ('random', 'true', 'cov_guess').
            - w_true (torch.Tensor): True weights of the DAG edges.
        
        Returns:
            - loss_data (torch.Tensor): Loss data during optimization.
            - w_loss_data (torch.Tensor): Squared distance of the difference between the true and the estimated parameters during optimization.
            
        """
    
    loss_data = torch.zeros(epochs)
    w_loss_data = torch.zeros(len(w_true), epochs)
                    
    mask = None
    if w_init=='cov_guess':
        w, mask = init_w_guess_(data, g, latent, observed)
        weight_true = w_true[mask==1]
        weight = Parameter(w[mask==1])
        fix_weight = w[mask == 0]        
        c_list = count_lists(mask)
    elif w_init=="true":
        weight_true = w_true
        weight = Parameter(torch.clone(w_true).detach().requires_grad_(True))
    else:
        weight_true = w_true
        weight = Parameter(torch.Tensor(len(g.edges())).normal_(0,1))

    optimizer = torch.optim.RMSprop([weight], lr, momentum=momentum)

    min_loss = None
    for epoch in range(epochs):
        adj = torch.eye(len(g.nodes()))
        if w_init == 'cov_guess':
            for ii, e in enumerate(g.edges()):
                if mask[ii] == 1:
                    adj[e]=-weight[int(c_list[ii])]
                else:
                    adj[e]=-fix_weight[int(c_list[ii])] 
        else:
            for e in range(len(g.edges())):
                adj[list(g.edges)[e]]=-weight[e]

        B = (torch.inverse(adj)).t()
        B = B[latent:latent+observed,:]
        B = W_w.matmul(B)

        latents = data_whitening.matmul(B)
        output = latents.matmul(B.t())
    
        diff = output - data_whitening
        loss_recon = 0
        if lmbda!=0:
            loss_recon = (diff * diff).mean()
        loss_latent = latents.abs().mean()
        loss = lmbda * loss_recon + loss_latent
        
        loss_data[epoch] = (loss.data).item()
        if min_loss is None or min_loss>loss_data[epoch]:
            min_loss = loss_data[epoch]
            weight_pred = weight.detach().clone()
        
        if  w_init == 'cov_guess':
            w_loss_data[mask==1, epoch] = (weight-w_true[mask==1].detach()).abs()
            w_loss_data[mask==0, epoch] = (fix_weight-w_true[mask==0].detach()).abs()
        else:
            w_loss_data[:, epoch] = (weight-w_true.detach().item).abs()       
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_data, w_loss_data, weight_pred, weight_true