import torch
import numpy as np

class QfunPARAFAC(torch.nn.Module):
    def __init__(self, dims, k, scale=1.0):
        super().__init__()

        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale*torch.randn(dim, k, dtype=torch.double, requires_grad=True)
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

    def forward(self, indices):
        bsz = indices.shape[0]
        if indices.shape[0] == 4:
            print(indices.shape[1])
        prod = torch.ones(bsz, self.k, dtype=torch.double)
        for i in range(indices.shape[1]):
            idx = indices[:, i]
            factor = self.factors[i]
            prod *= factor[idx, :]
        if indices.shape[1] < len(self.factors):
            return torch.matmul(prod, self.factors[-1].T)
        return torch.sum(prod, dim=-1)


class QfunNN(torch.nn.Module):

    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(QfunNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for h in num_hiddens:
            self.layers.append(torch.nn.Linear(num_inputs, h))
            self.layers.append(torch.nn.ReLU())
            num_inputs = h
        action_layer = torch.nn.Linear(num_inputs, num_outputs)
        action_layer.weight.data.mul_(0.1)
        action_layer.bias.data.mul_(0.0)
        self.layers.append(action_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class PolicyPARAFAC(torch.nn.Module):
    def __init__(self, dims, k, scale=1.0, model='gaussian'):
        super().__init__()

        self.k = k
        self.n_factors = len(dims)
        self.dims = dims

        factors = []
        for dim in dims:
            factor = scale*torch.randn(dim, k, dtype=torch.double, requires_grad=True)
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

        self.model = model
        if model == 'gaussian':
            self.log_sigma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, indices):
        indices=indices.long()
        if len(indices.shape) == 1:
            indices = indices.view(1, -1)
        bsz = indices.shape[0]
        prod = torch.ones(bsz, self.k, dtype=torch.double)
        for i in range(indices.shape[1]):
            idx = indices[:, i]
            factor = self.factors[i]
            prod *= factor[idx, :]
        if indices.shape[1] < len(self.factors):
            res = []
            nA = len(self.factors)-indices.shape[1]
            for cols in zip(
                *[self.factors[-(a + 1)].t() for a in reversed(range(nA))]
            ):
                kr = cols[0]
                for j in range(1, nA):
                    kr = torch.kron(kr, cols[j])
                res.append(kr)
            factors_action = torch.stack(res, dim=1)
            return torch.matmul(prod, factors_action.T).view(-1, *self.dims[-nA:])
        else:
            res = torch.sum(prod, dim=-1)
        if self.model == 'gaussian':
            return res, torch.clamp(self.log_sigma, min=-2.5, max=0.0)
        return res


class ValuePARAFAC(torch.nn.Module):
    def __init__(self, dims, k, scale=1.0):
        super().__init__()

        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale*torch.randn(dim, k, dtype=torch.double, requires_grad=True)
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

    def forward(self, indices):
        indices=indices.long()
        if len(indices.shape) == 1:
            indices=indices.view(1, -1)
        bsz = indices.shape[0]
        prod = torch.ones(bsz, self.k, dtype=torch.double)
        for i in range(indices.shape[1]):
            idx = indices[:, i]
            factor = self.factors[i]
            prod *= factor[idx, :]
        if indices.shape[1] < len(self.factors):
            return torch.matmul(prod, self.factors[-1].T)
        return torch.sum(prod, dim=-1)

class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, model='gaussian'):
        super(PolicyNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for h in num_hiddens:
            self.layers.append(torch.nn.Linear(num_inputs, h))
            self.layers.append(torch.nn.Tanh())
            num_inputs = h
        if len(num_outputs) > 0:
            self.num_outputs = num_outputs
            product = 1
            for out in num_outputs:
                product *= out
        else:
            product = num_outputs[0]
        action_layer = torch.nn.Linear(num_inputs, product)
        action_layer.weight.data.mul_(0.1)
        action_layer.bias.data.mul_(0.0)
        self.layers.append(action_layer)

        self.model = model
        if model == 'gaussian':
            self.log_sigma = torch.nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.model == 'gaussian':
            return x, torch.clamp(self.log_sigma, min=-2.0, max=0.0)
        return x.view(-1, *self.num_outputs)


class ValueNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(ValueNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for h in num_hiddens:
            self.layers.append(torch.nn.Linear(num_inputs, h))
            self.layers.append(torch.nn.Tanh())
            num_inputs = h
        action_layer = torch.nn.Linear(num_inputs, num_outputs)
        action_layer.weight.data.mul_(0.1)
        action_layer.bias.data.mul_(0.0)
        self.layers.append(action_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x