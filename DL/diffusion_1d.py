import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Real data: 1D mixture
def sample_real(n=512):
    a = torch.randn(n, device=device) * 0.5 - 2
    b = torch.randn(n, device=device) * 0.5 + 2
    mask = torch.rand(n, device=device) > 0.5
    return torch.where(mask, a, b).unsqueeze(1)

T = 50
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

# Model predicts noise epsilon
model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                      nn.Linear(64, 64), nn.ReLU(),
                      nn.Linear(64, 1)).to(device)

opt = optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

for step in range(2000):
    x0 = sample_real(512)                # clean
    t = torch.randint(0, T, (512,), device=device)
    eps = torch.randn_like(x0)

    a = alpha_bar[t].unsqueeze(1)
    xt = torch.sqrt(a)*x0 + torch.sqrt(1-a)*eps

    inp = torch.cat([xt, (t.float()/T).unsqueeze(1)], dim=1)
    pred_eps = model(inp)
    loss = mse(pred_eps, eps)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 500 == 0:
        print("step", step, "loss", round(loss.item(),4))

print("Trained diffusion toy (noise predictor).")