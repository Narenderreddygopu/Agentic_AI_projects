import torch
import torch.nn as nn
import torch.optim as optim

# Tiny dataset: sentiment (toy)
texts = [
    "i love this movie",
    "this is great",
    "i hate this",
    "this is terrible",
    "amazing and good",
    "bad and boring"
    "bad rating and awful"
    "worst experience ever"
    "i will never watch this again"
]
labels = [1,1,0,0,1,0]  # 1=positive, 0=negative

def build_vocab(texts):
    words = sorted(set(" ".join(texts).split()))
    stoi = {w:i+2 for i,w in enumerate(words)}  # 0=pad, 1=unk
    stoi["<pad>"] = 0
    stoi["<unk>"] = 1
    return stoi

def encode(text, stoi, max_len=6):
    toks = text.split()
    ids = [stoi.get(t, 1) for t in toks][:max_len]
    ids += [0]*(max_len-len(ids))
    return ids

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, max_len=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        # x: [B, T]
        h = self.embed(x) + self.pos[:, :x.size(1), :]
        h = self.encoder(h)              # [B, T, D]
        pooled = h.mean(dim=1)           # simple pooling
        return self.fc(pooled)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stoi = build_vocab(texts)
    X = torch.tensor([encode(t, stoi) for t in texts], dtype=torch.long).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)

    model = TinyTransformer(vocab_size=len(stoi)).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Train few epochs
    model.train()
    for _ in range(200):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

    model.eval()
    pred = model(X).argmax(dim=1)
    acc = (pred == y).float().mean().item()
    print("Transformer toy accuracy:", round(acc, 4))

if __name__ == "__main__":
    main()