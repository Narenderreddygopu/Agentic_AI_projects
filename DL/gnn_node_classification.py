import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Toy graph: 6 nodes, edges, node features, node labels
x = torch.tensor([
    [1.0, 0.0],
    [0.9, 0.1],
    [0.0, 1.0],
    [0.1, 0.9],
    [1.0, 0.2],
    [0.2, 1.0],
])

edge_index = torch.tensor([
    [0,1, 1,0, 2,3, 3,2, 0,4, 2,5],
    [1,0, 0,1, 3,2, 2,3, 4,0, 5,2]
], dtype=torch.long)

y = torch.tensor([0,0,1,1,0,1], dtype=torch.long)  # node classes
train_mask = torch.tensor([1,1,1,1,0,0], dtype=torch.bool)

data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 8)
        self.conv2 = GCNConv(8, 2)
    def forward(self, data):
        h = F.relu(self.conv1(data.x, data.edge_index))
        out = self.conv2(h, data.edge_index)
        return out

model = GCN()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(200):
    model.train()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    opt.zero_grad()
    loss.backward()
    opt.step()

model.eval()
pred = model(data).argmax(dim=1)
acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
print("GNN train accuracy:", round(acc, 4))