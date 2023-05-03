from sklearn.preprocessing import OneHotEncoder
import torch
from torchvision.transforms import ToTensor, Lambda
from neuron_model import TwoCompartmentLIFLayer


def reset_all(network):
    n_reset = 0
    for layer in network.modules():
        if isinstance(layer, TwoCompartmentLIFLayer):
            layer.reset()
            n_reset += 1
    return network

# def img_to_poisson(img, n_steps, max_rate=1.):
#     # assuming 0 <= img <= 1
#     img = img.flatten(start_dim=1)
#     spikes = torch.bernoulli(img.expand((n_steps, -1, -1)) * max_rate)
#     return spikes.moveaxis(0, 1).float()


# class ImgToPoisson:
#     def __init__(self, n_steps, max_rate=1.):
#         self.n_steps = n_steps
#         self.max_rate = max_rate
#         self.totensor = ToTensor()

#     def __call__(self, x):
#         x = self.totensor(x)
#         img = x.flatten()
#         spikes = torch.rand(self.n_steps,
#             *img.shape,
#             device=img.device) < img * self.max_rate
#         return spikes.float()


# def OneHotTransform(n):
#     return Lambda(lambda l: torch.nn.functional.one_hot(torch.tensor(l), n).float())
