from neuron_model import TwoCompartmentLIFLayer
from utils import reset_all
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torch

DEVICE = "cuda"
N_IN = 784
N_OUT = 10
TOTTIME = 100
MAX_RATE = 0.4
TARGET_RESCALE = 3.
BATCH_SIZE = 500
T_MEM = 10.
T_REF = 1/(1-0.65)
RHO = 1.0


ds = MNIST(root="~/Work/datasets/", transform=ToTensor())
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
ds_test = MNIST(root="~/Work/datasets/", transform=ToTensor(), train=False)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

n_per_layer = (500, 500, N_OUT)
bias = True
network = torch.nn.Sequential(
    TwoCompartmentLIFLayer(N_IN, n_per_layer[0], t_mem=T_MEM, t_ref=T_REF, rho=RHO, bias=bias),
    TwoCompartmentLIFLayer(n_per_layer[0], n_per_layer[1], t_mem=T_MEM, t_ref=T_REF, rho=RHO, bias=bias),

    TwoCompartmentLIFLayer(n_per_layer[1], n_per_layer[2], t_mem=T_MEM, t_ref=T_REF, rho=RHO, bias=bias),

).to(DEVICE)

epochs_per_layer = (3, 3,3)

targets = (
    torch.rand(N_OUT, n_per_layer[0], device=DEVICE),
    torch.rand(N_OUT, n_per_layer[1], device=DEVICE),

    torch.eye(N_OUT, n_per_layer[2], device=DEVICE)
)
torch.nn.init.normal_(targets[0], 0.5, 0.5)
torch.nn.init.normal_(targets[1], 0.5, 0.5)


spikes = torch.zeros((TOTTIME, BATCH_SIZE, 784), device=DEVICE)

# ---- Training ---- #
distance = torch.nn.MSELoss()
for i, layer in enumerate(network):
    # we train only on one layer at a time
    optimizer = torch.optim.RMSprop(
        layer.parameters(), lr=1e-3/TOTTIME, weight_decay=1e-6)
    outs = torch.zeros(TOTTIME, BATCH_SIZE, n_per_layer[i],
                       device=DEVICE, requires_grad=False)

    for epoch in range(epochs_per_layer[i]):
        print(f"----- Training layer {i}, epoch {epoch} -----")
        pbar = tqdm(dl)
        for img, label in pbar:
            # turn the image into spike trains
            img = img.to(DEVICE)
            img = img.flatten(start_dim=1).expand((TOTTIME, -1, -1))
            spikes.bernoulli_(img * MAX_RATE)

            # prepare the target. We get a row of B, this is equivalent to
            # By when y is one-hot encoded.
            By = targets[i][label]

            reset_all(network)
            outs.zero_()
            for t in range(TOTTIME):
                current_input = spikes[t]
                outs[t] = network[:i+1](current_input)
                layer.set_target(By * TARGET_RESCALE)

                optimizer.step()
                optimizer.zero_grad()
            mse = distance(By, outs.mean(0))
            pbar.set_postfix(MSE=mse.item())

    torch.cuda.empty_cache()
torch.save(network.state_dict(), "3layer.pth")


# ---- Testing ---- #
tot_correct, tot_seen, n_zeros = 0, 0, 0

pbar = tqdm(dl_test)
for img, label in pbar:
    reset_all(network)
    with torch.no_grad():
        # turn the image into spike trains
        img = img.to(DEVICE)
        img = img.flatten(start_dim=1).expand((TOTTIME, -1, -1))
        spikes.bernoulli_(img * MAX_RATE)

        out = 0.
        for t in range(TOTTIME):
            out += network(spikes[t])

        tot_seen += len(label)
        correct = torch.max(out, 1)[1] == label.to(DEVICE)
        n_zeros += (out.sum(1) == 0.).sum().item()
        tot_correct += correct.sum().item()
        pbar.set_postfix(running_accuracy=tot_correct/tot_seen,
                         frac_no_spk=n_zeros/tot_seen)
