import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from itertools import combinations
import matplotlib.pyplot as plt
import random, os

random.seed(42); np.random.seed(42); torch.manual_seed(42)

df = pd.read_csv("kmer_vectors.csv", index_col="species")
species_names = list(df.index)
vectors = {n: df.loc[n].values.astype(np.float32) for n in species_names}
print(f"Loaded {len(vectors)} species, {df.shape[1]} features each\n")

GROUPS = {
    "proboscidea": {"woolly_mammoth","american_mastodon","asian_elephant","african_elephant"},
    "felidae":     {"cave_lion","saber_tooth_cat","tiger"},
    "ursidae":     {"cave_bear","brown_bear"},
    "canidae":     {"dire_wolf","grey_wolf"},
    "outgroup":    {"human","chicken"},
}
def get_group(sp):
    for g, members in GROUPS.items():
        if sp in members: return g
    return "unknown"

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def build_pairs(vectors, species_names):
    pairs = []
    for s1, s2 in combinations(species_names, 2):
        sim = cosine_sim(vectors[s1], vectors[s2])
        g1, g2 = get_group(s1), get_group(s2)
        same_family = 1.0 if g1 == g2 and g1 != "outgroup" else 0.0
        pairs.append((s1, s2, sim, same_family))
    return pairs

def augment_pairs(pairs, vectors, n_augment=12):
    """Add Gaussian noise to vectors to create more training examples."""
    augmented = list(pairs)
    for aug_i in range(n_augment):
        for s1, s2, sim, lbl in pairs:
            noise = 0.003
            v1 = np.clip(vectors[s1] + np.random.normal(0, noise, vectors[s1].shape), 0, None).astype(np.float32)
            v2 = np.clip(vectors[s2] + np.random.normal(0, noise, vectors[s2].shape), 0, None).astype(np.float32)
            t1, t2 = f"{s1}_aug{aug_i}", f"{s2}_aug{aug_i}"
            vectors[t1] = v1; vectors[t2] = v2
            augmented.append((t1, t2, sim, lbl))
    return augmented

all_pairs = build_pairs(vectors, species_names)
aug_pairs = augment_pairs(all_pairs, vectors)
random.shuffle(aug_pairs)
split = int(0.8 * len(aug_pairs))
train_pairs, val_pairs = aug_pairs[:split], aug_pairs[split:]


class DNAPairDataset(Dataset):
    def __init__(self, pairs, vectors):
        self.pairs = pairs; self.vectors = vectors
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        s1, s2, sim, lbl = self.pairs[idx]
        return (torch.tensor(self.vectors[s1]),
                torch.tensor(self.vectors[s2]),
                torch.tensor(sim,  dtype=torch.float32),
                torch.tensor(lbl, dtype=torch.float32))

train_loader = DataLoader(DNAPairDataset(train_pairs, vectors), batch_size=32, shuffle=True)
val_loader   = DataLoader(DNAPairDataset(val_pairs,   vectors), batch_size=32, shuffle=False)

class DNAEncoder(nn.Module):
    """
    Shared encoder tower used by both branches of the Siamese network.
    Input:  4096-dim k-mer frequency vector
    Output: 64-dim unit-sphere embedding
    """
    def __init__(self, input_dim=4096, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, embed_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)   

class SiameseDNA(nn.Module):
    """
    Two sequences → same encoder → cosine similarity of embeddings.
    The key insight: SHARED weights mean the network learns a universal
    'DNA language' — not memorizing individual species.
    """
    def __init__(self, input_dim=4096, embed_dim=64):
        super().__init__()
        self.encoder = DNAEncoder(input_dim, embed_dim)
    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        similarity = (e1 * e2).sum(dim=1)   
        return similarity, e1, e2

model = SiameseDNA(input_dim=df.shape[1])
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")


def contrastive_loss(sim, label, margin=0.3):
    """
    Related pairs   (label=1): penalise if sim < 1  → push embeddings together
    Unrelated pairs (label=0): penalise if sim > margin → push embeddings apart
    """
    pos = label       * (1 - sim) ** 2
    neg = (1 - label) * F.relu(sim - margin) ** 2
    return (pos + neg).mean()

def combined_loss(pred, true_sim, label, alpha=0.6):
    return alpha * contrastive_loss(pred, label) + (1 - alpha) * F.mse_loss(pred, true_sim)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

EPOCHS = 80
history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_val, best_state = float('inf'), None

print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'Acc':>7}")
print("-" * 38)

for epoch in range(1, EPOCHS + 1):
    model.train()
    t_loss = 0
    for v1, v2, sim, lbl in train_loader:
        optimizer.zero_grad()
        pred, _, _ = model(v1, v2)
        loss = combined_loss(pred, sim, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item()

    model.eval(); v_loss = 0; correct = 0; total = 0
    with torch.no_grad():
        for v1, v2, sim, lbl in val_loader:
            pred, _, _ = model(v1, v2)
            v_loss += combined_loss(pred, sim, lbl).item()
            correct += ((pred > 0.5).float() == lbl).sum().item()
            total += len(lbl)

    t_loss /= len(train_loader); v_loss /= len(val_loader)
    acc = correct / total if total > 0 else 0
    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["val_acc"].append(acc)
    scheduler.step()

    if v_loss < best_val:
        best_val = v_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:>6}  {t_loss:>10.4f}  {v_loss:>10.4f}  {acc:>6.1%}")

model.load_state_dict(best_state)
torch.save(model.state_dict(), "siamese_dna_model.pt")
print(f"\nSaved: siamese_dna_model.pt")

def predict_similarity(species1, species2):
    """
    Given two species names (must be in kmer_vectors.csv),
    returns a similarity score 0.0–1.0.
    Can also accept raw k-mer vectors directly.
    """
    model.eval()
    with torch.no_grad():
        v1 = torch.tensor(vectors[species1]).unsqueeze(0)
        v2 = torch.tensor(vectors[species2]).unsqueeze(0)
        sim, _, _ = model(v1, v2)
        return float(sim.item())

print("\n── Predictions ──\n")
print(f"{'Pair':<45}  {'Model':>7}  {'Baseline':>9}  Label")
print("-" * 80)
test_pairs_eval = [
    ("woolly_mammoth",  "asian_elephant",    "HIGH"),
    ("woolly_mammoth",  "american_mastodon", "HIGH"),
    ("cave_lion",       "tiger",             "HIGH"),
    ("cave_bear",       "brown_bear",        "HIGH"),
    ("dire_wolf",       "grey_wolf",         "HIGH"),
    ("cave_lion",       "cave_bear",         "MED"),
    ("woolly_mammoth",  "tiger",             "LOW"),
    ("woolly_mammoth",  "chicken",           "VERY LOW"),
    ("saber_tooth_cat", "grey_wolf",         "LOW"),
]
for s1, s2, expected in test_pairs_eval:
    if s1 in vectors and s2 in vectors:
        pred = predict_similarity(s1, s2)
        base = cosine_sim(vectors[s1], vectors[s2])
        print(f"  {s1+' vs '+s2:<43}  {pred:>7.4f}  {base:>9.4f}  {expected}")

print("\nNext step: build the Streamlit demo app (Stage 5).")
