import torch
from torch import nn
from math import sqrt
import numpy as np



class TwoCompartmentLIFLayer(nn.Module):
    def __init__(self, n_in, n_out, t_mem, t_ref, rho, bias=False):
        super().__init__()
        
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.alpha_mem = 1./t_mem
        self.alpha_ref = 1./t_ref
        self.rho = rho

        if bias:
            self.bias = nn.Parameter(torch.empty(n_out), requires_grad=True)
            torch.nn.init.uniform_(self.bias, -.5, .5)
        else:
            self.bias = False
        
        self.reset()
        
    def threshold(self, u):
        return (u >= 1.).float()
        
    def forward(self, x):
        self.cached_input = x
        input_ = self.linear(x)
        
        if self.R is None:
            self.R = torch.zeros_like(input_, requires_grad=False)
            self.V = torch.zeros_like(input_, requires_grad=False)

        self.V += (-self.V + input_) * self.alpha_mem
        self.U = self.V - self.rho * self.R + self.bias
        spikes_out = self.threshold(self.U)
        self.R += (-self.R + spikes_out) * self.alpha_ref
        return spikes_out
    
    def set_target(self, By):
       # self.linear.weight.grad = (self.set_der_relu(self.U)*(self.V - By)).T @ self.cached_input

        
       # if self.linear.weight.grad == None:
       #     self.linear.weight.grad = torch.ones_like(self.linear.weight, requires_grad=False)
      
        #print(np.shape((self.V - By).T))
        #print("inputs: ",np.shape(self.cached_input))
        identity = torch.eye((500))
        self.linear.weight.grad = (self.V - By).T @  self.cached_input

        if self.bias is not False:
            self.bias.grad = (self.V - By).sum(0)  # sum over batch, is it right?

    def set_der_relu(self,u):
        return (u >= 0.7).float()    
        
    def reset(self):
        self.R = None
        self.V = None
