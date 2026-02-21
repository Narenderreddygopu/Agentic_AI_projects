import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Real data: 2D Gaussian
def sample_real(n=256):
    mean = torch.tensor([2.0, -1.0], device=device)
    cov = torch.tensor([[1.0, 0.3],[0.3, 0.6]], device=device)
    L = torch.linalg.cholesky(cov)
    z = torch.randn(n, 2, device=device)
    return z @ L.T + mean

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(),
                                 nn.Linear(32, 32), nn.ReLU(),
                                 nn.Linear(32, 2))
    def forward(self, z): return self.net(z)

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                                 nn.Linear(32, 32), nn.ReLU(),
                                 nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

g, d = G().to(device), D().to(device)
opt_g = optim.Adam(g.parameters(), lr=1e-3)
opt_d = optim.Adam(d.parameters(), lr=1e-3)
bce = nn.BCELoss()

for step in range(2000):
    # Train D
    real = sample_real(256)
    z = torch.randn(256, 8, device=device)
    fake = g(z).detach()

    opt_d.zero_grad()
    loss_d = bce(d(real), torch.ones(256,1,device=device)) + bce(d(fake), torch.zeros(256,1,device=device))
    loss_d.backward()
    opt_d.step()

    # Train G
    z = torch.randn(256, 8, device=device)
    opt_g.zero_grad()
    fake = g(z)
    loss_g = bce(d(fake), torch.ones(256,1,device=device))
    loss_g.backward()
    opt_g.step()

    if step % 500 == 0:
        print("step", step, "loss_d", round(loss_d.item(),4), "loss_g", round(loss_g.item(),4))

# Plot
real = sample_real(1000).cpu().numpy()
fake = g(torch.randn(1000, 8, device=device)).detach().cpu().numpy()

plt.figure()
plt.scatter(real[:,0], real[:,1], s=5, label="real")
plt.scatter(fake[:,0], fake[:,1], s=5, label="fake")
plt.legend()
plt.title("GAN learns 2D distribution")
plt.show()