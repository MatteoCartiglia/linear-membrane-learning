from neuron_model import TwoCompartmentLIFLayer
from utils import reset_all

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt


DEVICE = "cuda"
N_IN = 784
N_OUT = 10
TOTTIME = 100
MAX_RATE = 0.4
BATCH_SIZE = 1000
T_MEM = 10.
T_REF = 1/(1-0.65)
RHO = 1.0

ds_test = MNIST(root="~/Work/datasets/", transform=ToTensor(), train=False)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

n_per_layer = (120, 120, N_OUT)
bias = True
network = torch.nn.Sequential(
    TwoCompartmentLIFLayer(N_IN, n_per_layer[0], t_mem=T_MEM, t_ref=T_REF, rho=RHO, bias=bias),
    TwoCompartmentLIFLayer(n_per_layer[0], n_per_layer[1], t_mem=T_MEM, t_ref=T_REF, rho=RHO, bias=bias),
    TwoCompartmentLIFLayer(n_per_layer[1], n_per_layer[2], t_mem=T_MEM, t_ref=T_REF, rho=RHO, bias=bias)
).to(DEVICE)
network.load_state_dict(torch.load("3layer_emrepars_120_TR3.pth"))


# ---- Testing ---- #
tot_correct, tot_seen, n_zeros = 0, 0, 0
spikes = torch.zeros((TOTTIME, BATCH_SIZE, 784), device=DEVICE)

pbar = tqdm(dl_test)
for img, label in pbar:
    reset_all(network)
    with torch.no_grad():
        # turn the image into spike trains
        img = img.to(DEVICE)
        img = img.flatten(start_dim=1).expand((TOTTIME, -1, -1))
        spikes.bernoulli_(img * MAX_RATE)

        layer_out = [0. for _ in range(len(network))]
        for t in range(TOTTIME):
            out_t = spikes[t]
            for i in range(len(network)):
                out_t = network[i](out_t)
                layer_out[i] += out_t

        out = layer_out[-1]
        tot_seen += len(label)
        correct = torch.max(out, 1)[1] == label.to(DEVICE)
        n_zeros += (out.sum(1) == 0.).sum().item()
        tot_correct += correct.sum().item()
        pbar.set_postfix(running_accuracy=tot_correct/tot_seen,
                         frac_no_spk=n_zeros/tot_seen)


# ---- This analysis is done only on the LAST BATCH!! ---- #
layer = 1
fr = layer_out[layer].cpu().numpy()

# distribution of firing rates of neurons, average on inputs
plt.violinplot(fr, widths=0.8, showmedians=True, showextrema=False)
plt.show()
