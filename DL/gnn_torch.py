import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Toy graph (6 nodes) ----------
# Node features (6 nodes, 2 features each)
X = torch.tensor([
    [1.0, 0.0],
    [0.9, 0.1],
    [0.0, 1.0],
    [0.1, 0.9],
    [1.0, 0.2],
    [0.2, 1.0],
])

# Edges (undirected) as pairs
edges = [
    (0, 1), (1, 0),
    (2, 3), (3, 2),
    (0, 4), (4, 0),
    (2, 5), (5, 2)
]

# Labels for node classification
y = torch.tensor([0, 0, 1, 1, 0, 1])  # two classes: 0 or 1

# Train on first 4 nodes
train_mask = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool)

n = X.size(0)

# Build adjacency matrix A
A = torch.zeros(n, n)
for u, v in edges:
    A[u, v] = 1.0

# Add self-loops: A_hat = A + I
I = torch.eye(n)
A_hat = A + I

# Normalize: D^-1/2 * A_hat * D^-1/2
deg = A_hat.sum(dim=1)
D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, X, A_norm):
        # Message passing: aggregate neighbors then linear transform
        return self.lin(A_norm @ X)


class SimpleGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNLayer(2, 8)
        self.gcn2 = GCNLayer(8, 2)

    def forward(self, X, A_norm):
        h = F.relu(self.gcn1(X, A_norm))
        out = self.gcn2(h, A_norm)
        return out


model = SimpleGCN()
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(200):
    model.train()
    logits = model(X, A_norm)
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    opt.zero_grad()
    loss.backward()
    opt.step()

model.eval()
pred = model(X, A_norm).argmax(dim=1)
acc = (pred[train_mask] == y[train_mask]).float().mean().item()

print("Pred:", pred.tolist())
print("Train accuracy:", round(acc, 4))