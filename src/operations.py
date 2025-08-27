import torch
import torch.nn as nn


def scale(net, prefactor):
    return lambda u: prefactor * net(u)

def add(net1, net2):
    return lambda u: net1(u) + net2(u)

def multi(net1, net2):
    return lambda u: net1(net2(u))

def linear(nets: list[nn.Module], W, scale=scale, add=add, multi=multi):
    """ 
    Linear combination of multiple networks.

    Args:
        nets: list of networks
        W: list of weights; maps N inputs to M outputs
        scale: scaling function
        add: addition function
        multi: multiplication function
    """
    N, M = W.shape[0], W.shape[-1]
    new_nets = [None] * M
    
    for m in range(M):
        for n in range(N):
            mask = torch.zeros_like(W)
            mask[n, m] = 1
            mask *= W
            mask = mask.sum()
            
            if new_nets[m] is None:
                new_nets[m] = scale(net=nets[n], prefactor=mask)
            else:    
                new_nets[m] = add(new_nets[m], scale(net=nets[n], prefactor=mask))
    return new_nets

def act(nets: list[nn.Module], scale=scale, add=add, multi=multi): 
    
    new_nets = [multi(net, net) for net in nets]
    return new_nets
