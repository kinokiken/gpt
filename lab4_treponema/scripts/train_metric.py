"""Train a projection on labelled embeddings from independent train sources."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('vectors', help='NPY matrix with one embedding per row')
    parser.add_argument('labels', help='JSON list of training class labels')
    parser.add_argument('--out', default='results/metric')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = torch.tensor(np.load(args.vectors), dtype=torch.float32)
    labels = np.asarray(json.loads(Path(args.labels).read_text()))
    if x.ndim != 2 or len(labels) != len(x):
        raise ValueError('Embeddings and labels must have the same row count')
    triplets = []
    for i in range(len(x)):
        positive = np.flatnonzero((labels == labels[i]) & (np.arange(len(x)) != i))
        negative = np.flatnonzero(labels != labels[i])
        if len(positive) and len(negative):
            triplets.append((i, int(rng.choice(positive)), int(rng.choice(negative))))
    if not triplets:
        raise ValueError('Need at least two classes and repeated class examples')
    indices = torch.tensor(triplets)
    head = nn.Sequential(nn.Linear(x.shape[1], 128), nn.ReLU(), nn.Linear(128, 64))
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    history = []
    for epoch in range(args.epochs):
        z = F.normalize(head(x), dim=1)
        loss = F.triplet_margin_loss(z[indices[:, 0]], z[indices[:, 1]],
                                     z[indices[:, 2]], margin=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({'epoch': epoch + 1, 'train_loss': float(loss.detach())})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'input_dim': x.shape[1], 'state_dict': head.state_dict()}, out / 'head.pt')
    (out / 'history.json').write_text(json.dumps(history, indent=2))
    print('Saved training history; independent validation/test evaluation is separate.')

if __name__ == '__main__':
    main()
