# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: MutuInfo.py
@Time: 2021/4/1 9:46
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Mine(nn.Module):
    """
    Neural network implementation of Mutual Information Neural Estimator (MINE).
    Used to estimate mutual information between transmitted (X) and received (Y) signals.
    The network learns to score whether a pair of signals (X,Y) came from the same transmission.
    """

    def __init__(self, in_dim=2, hidden_size=10):
        """
        Args:
            in_dim: Input dimension (2 for concatenated pairs of X,Y signals)
            hidden_size: Number of neurons in hidden layers
        """
        super(Mine, self).__init__()
        self.dense1 = linear(in_dim, hidden_size)
        self.dense2 = linear(hidden_size, hidden_size)
        self.dense3 = linear(hidden_size, 1) # Outputs a single score for each X, Y pair

    def forward(self, inputs):
        """Forward pass through the network"""
        x = self.dense1(inputs)
        x = F.relu(x)  # ReLU activation after first layer
        x = self.dense2(x)
        x = F.relu(x)  # ReLU activation after second layer
        output = self.dense3(x)  # Final score output

        return output
    
def linear(in_dim, out_dim, bias=True):
    """
    Creates a linear layer with custom initialization.
    Uses normal distribution for weights and zeros for bias.
    """
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    # Initialize weights with normal distribution
    lin.weight = torch.nn.Parameter(
        torch.normal(0.0, 0.02, size=lin.weight.shape))
    # xavier_uniform_(lin.weight)  # Alternative initialization (commented out)
    if bias:
        lin.bias.data.zero_()  # Initialize bias to zero

    return lin

def mutual_information(joint, marginal, mine_net):
    # T(x,y) : T là minet
    t = mine_net(joint)
    # T(x, y')
    marginal_scores = mine_net(marginal).clamp(min=-50, max=50)
    # log E(e^T) = max_score + logE(e^(T-max_score))
    et = torch.exp(marginal_scores)
    max_score = torch.max(marginal_scores)
    log_mean_et = max_score + torch.log(
        torch.mean(torch.exp(marginal_scores - max_score)))
    # MI = E_joint[T] - log_marginal[e^T]
    mi_lb = torch.mean(t) - log_mean_et
    mi_bits = mi_lb / torch.log(torch.tensor(2.0))  # Compute but don’t return
    return mi_lb, t, et  # Return only 3 values

def learn_mine(batch, mine_net, ma_et, ma_rate=0.01):
    joint, marginal = batch
    joint = torch.FloatTensor(joint)
    marginal = torch.FloatTensor(marginal)
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)

    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
    loss = -(torch.mean(t) - (1 / torch.mean(ma_et)) * torch.mean(et))

    return loss, ma_et, mi_lb

def sample_batch(rec, noise):
    """
    Creates paired samples for mutual information estimation.
    IMPORTANT: From the function parameters:
    - rec should be X (transmitted signal) based on variable name convention in MINE papers
    - noise should be Y (received signal) based on variable name convention in MINE papers
    However, the actual usage in train_mi() shows:
    - rec = Tx_sig (transmitted signal X)
    - noise = Rx_sig (received signal Y, NOT noise)
    The parameter names are misleading!

    Args:
        rec: Transmitted signal (X) [despite misleading parameter name]
        noise: Received signal (Y) [despite misleading parameter name]
    """
    # Reshape both signals into column vectors (-1 means infer dimension)
    # Shape becomes [batch_size, 1]
    rec = torch.reshape(rec, shape=(-1, 1))  # Reshape X
    noise = torch.reshape(noise, shape=(-1, 1))  # Reshape Y

    # Split each signal into two halves
    # If batch_size=128, each sample will be 64
    rec_sample1, rec_sample2 = torch.split(rec, int(rec.shape[0] / 2),
                                           dim=0)  # Split X into X1, X2
    noise_sample1, noise_sample2 = torch.split(noise, int(noise.shape[0] / 2),
                                               dim=0)  # Split Y into Y1, Y2

    # Create joint distribution samples (X1,Y1)
    # These are pairs where X and Y are from the same transmission
    # Shape: [batch_size/2, 2] where each row is [x,y] from same transmission
    joint = torch.cat((rec_sample1, noise_sample1), 1)

    # Create marginal distribution samples (X1,Y2)
    # These are pairs where X and Y are from different transmissions
    # Shape: [batch_size/2, 2] where each row is [x,y] from different transmissions
    marg = torch.cat((rec_sample1, noise_sample2), 1)

    return joint, marg
