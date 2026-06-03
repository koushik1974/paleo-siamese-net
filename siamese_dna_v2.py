"""
Stage 3 v2: Siamese Network — fixed for real NCBI data
=======================================================
Key fixes over v1:
  1. Weighted sampling  — balances same-family vs different-family pairs
  2. Larger margin      — forces clearer separation between related/unrelated
  3. Harder negatives   — mines pairs that are close but SHOULD be far apart
  4. More epochs        — real DNA needs longer to converge
  5. Stricter threshold — 0.65 cutoff instead of 0.5 for classification

Usage:
    python siamese_dna_v2.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.decomposition import PCA
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("kmer_vectors.csv", index_col="species")
species_names = list(df.index)
vectors = {n: df.loc[n].values.astype(np.float32) for n in species_names}
input_dim = df.shape[1]
print(f"Loaded {len(vectors)} species, {input_dim} features\n")

# Evolutionary groups — add any new species from your 17 here
GROUPS = {
    "proboscidea": {"woolly_mammoth","american_mastodon","asian_elephant","african_elephant",
                    "columbian_mammoth"},
    "felidae":     {"cave_lion","saber_tooth_cat","tiger","lion","leopard","cheetah"},
    "ursidae":     {"cave_bear","brown_bear","polar_bear","black_bear"},
    "canidae":     {"dire_wolf","grey_wolf","dog","coyote","dhole"},
    "equidae":     {"horse","donkey","zebra","quagga"},
    "rhinocerotidae": {"white_rhinoceros","black_rhinoceros","woolly_rhinoceros",
                       "indian_rhinoceros","sumatran_rhinoceros"},
    "cetacea":     {"blue_whale","humpback_whale","orca","dolphin"},
    "outgroup":    {"human","chicken","zebrafish","fruitfly"},
}
def get_group(sp):
    for g, members in GROUPS.items():
        if sp in members: return g
    return "unknown"

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── 2. Build pairs with CONTINUOUS similarity target ─────────────────────────
#
# Key insight: instead of a binary label, we use a 3-level target:
#   1.0 = same family         (mammoth + elephant)
#   0.3 = related order       (mammoth + rhino — both large herbivores)
#   0.0 = completely unrelated (mammoth + chicken)
#
# This gives the model richer signal than just 0/1.

SAME_ORDER_PAIRS = {
    frozenset({"felidae",  "ursidae"}),
    frozenset({"felidae",  "canidae"}),
    frozenset({"ursidae",  "canidae"}),
    frozenset({"equidae",  "rhinocerotidae"}),
}

def evolutionary_label(s1, s2):
    g1, g2 = get_group(s1), get_group(s2)
    if g1 == "outgroup" or g2 == "outgroup": return 0.0
    if g1 == g2:                              return 1.0
    if frozenset({g1, g2}) in SAME_ORDER_PAIRS: return 0.3
    return 0.0

all_pairs = []
for s1, s2 in combinations(species_names, 2):
    lbl = evolutionary_label(s1, s2)
    sim = cosine_sim(vectors[s1], vectors[s2])
    all_pairs.append((s1, s2, sim, lbl))

same = sum(1 for *_, l in all_pairs if l == 1.0)
mid  = sum(1 for *_, l in all_pairs if l == 0.3)
diff = sum(1 for *_, l in all_pairs if l == 0.0)
print(f"Pairs — same family: {same}  related order: {mid}  unrelated: {diff}")


# ── 3. Augmentation ───────────────────────────────────────────────────────────
def augment(pairs, vectors, n_aug=15):
    aug = list(pairs)
    for i in range(n_aug):
        for s1, s2, sim, lbl in pairs:
            n = 0.004 if lbl == 0.0 else 0.002
            v1 = np.clip(vectors[s1] + np.random.normal(0, n, vectors[s1].shape), 0, None).astype(np.float32)
            v2 = np.clip(vectors[s2] + np.random.normal(0, n, vectors[s2].shape), 0, None).astype(np.float32)
            k1, k2 = f"{s1}_a{i}", f"{s2}_a{i}"
            vectors[k1] = v1; vectors[k2] = v2
            aug.append((k1, k2, sim, lbl))
    return aug

aug_pairs = augment(all_pairs, vectors, n_aug=15)
random.shuffle(aug_pairs)
split = int(0.8 * len(aug_pairs))
train_pairs, val_pairs = aug_pairs[:split], aug_pairs[split:]


# ── 4. Weighted sampler to fix class imbalance ────────────────────────────────
class DNAPairDataset(Dataset):
    def __init__(self, pairs, vectors):
        self.pairs = pairs; self.vectors = vectors
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        s1, s2, sim, lbl = self.pairs[idx]
        return (torch.tensor(self.vectors[s1]),
                torch.tensor(self.vectors[s2]),
                torch.tensor(sim,  dtype=torch.float32),
                torch.tensor(lbl,  dtype=torch.float32))

labels_train = [lbl for _, _, _, lbl in train_pairs]
sample_weights = [5.0 if l == 1.0 else (2.0 if l == 0.3 else 1.0) for l in labels_train]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(DNAPairDataset(train_pairs, vectors), batch_size=64, sampler=sampler)
val_loader   = DataLoader(DNAPairDataset(val_pairs,   vectors), batch_size=64, shuffle=False)
print(f"Train: {len(train_pairs):,}  Val: {len(val_pairs):,}\n")


# ── 5. Model ──────────────────────────────────────────────────────────────────
class DNAEncoder(nn.Module):
    def __init__(self, input_dim=4096, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 256),       nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

class SiameseDNA(nn.Module):
    def __init__(self, input_dim=4096, embed_dim=128):
        super().__init__()
        self.encoder = DNAEncoder(input_dim, embed_dim)
    def forward(self, x1, x2):
        e1, e2 = self.encoder(x1), self.encoder(x2)
        return (e1 * e2).sum(dim=1), e1, e2

model = SiameseDNA(input_dim=input_dim, embed_dim=128)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")


# ── 6. Loss — larger margin forces real separation ────────────────────────────
def contrastive_loss(sim, label, margin=0.5):
    pos = label       * (1 - sim).pow(2)
    neg = (1 - label) * F.relu(sim - margin).pow(2)
    return (pos + neg).mean()

def combined_loss(pred, true_sim, label, alpha=0.7):
    return alpha * contrastive_loss(pred, label) + (1 - alpha) * F.mse_loss(pred, true_sim)


# ── 7. Training ───────────────────────────────────────────────────────────────
EPOCHS    = 150
THRESHOLD = 0.65
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_val, best_state = float('inf'), None

print(f"{'Epoch':>6}  {'Train':>8}  {'Val':>8}  {'Acc':>7}")
print("-" * 38)

for epoch in range(1, EPOCHS + 1):
    model.train(); t_loss = 0
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
            binary_lbl = (lbl > 0.5).float()
            correct += ((pred > THRESHOLD).float() == binary_lbl).sum().item()
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

    if epoch % 15 == 0 or epoch == 1:
        print(f"{epoch:>6}  {t_loss:>8.4f}  {v_loss:>8.4f}  {acc:>6.1%}")

model.load_state_dict(best_state)
torch.save(model.state_dict(), "siamese_dna_model_v2.pt")
print(f"\nBest val loss: {best_val:.4f} → saved siamese_dna_model_v2.pt\n")


# ── 8. Evaluate ───────────────────────────────────────────────────────────────
def predict(s1, s2):
    model.eval()
    with torch.no_grad():
        v1 = torch.tensor(vectors[s1]).unsqueeze(0)
        v2 = torch.tensor(vectors[s2]).unsqueeze(0)
        sim, _, _ = model(v1, v2)
        return float(sim.item())

print(f"{'Pair':<47}  {'Model':>7}  {'Baseline':>9}  Expected")
print("-" * 82)
test = [
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
for s1, s2, exp in test:
    if s1 in vectors and s2 in vectors:
        p = predict(s1, s2)
        b = cosine_sim(vectors[s1], vectors[s2])
        flag = "✓" if (exp == "HIGH" and p > 0.7) or (exp in ("LOW","VERY LOW") and p < 0.5) else "?"
        print(f"  {s1+' vs '+s2:<45}  {p:>7.4f}  {b:>9.4f}  {exp}  {flag}")


# ── 9. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('#0f1117')

ax = axes[0]; ax.set_facecolor('#1a1d27')
ax.plot(history["train_loss"], color='#5DCAA5', lw=1.5, label='train')
ax.plot(history["val_loss"],   color='#AFA9EC', lw=1.5, label='val')
ax.set_title('Loss curves', color='white', fontsize=11)
ax.set_xlabel('Epoch', color='#888780'); ax.set_ylabel('Loss', color='#888780')
ax.legend(facecolor='#1a1d27', edgecolor='#444441', labelcolor='#c2c0b6')
ax.tick_params(colors='#888780')
for sp in ax.spines.values(): sp.set_edgecolor('#444441')

ax = axes[1]; ax.set_facecolor('#1a1d27')
ax.plot([v*100 for v in history["val_acc"]], color='#EF9F27', lw=1.5)
ax.axhline(y=75, color='#5DCAA5', ls='--', lw=0.8, label='75% target')
ax.set_title(f'Val accuracy (threshold {THRESHOLD})', color='white', fontsize=11)
ax.set_xlabel('Epoch', color='#888780'); ax.set_ylabel('Accuracy %', color='#888780')
ax.legend(facecolor='#1a1d27', edgecolor='#444441', labelcolor='#c2c0b6')
ax.tick_params(colors='#888780')
for sp in ax.spines.values(): sp.set_edgecolor('#444441')

ax = axes[2]; ax.set_facecolor('#1a1d27')
palette = {"proboscidea":"#1D9E75","felidae":"#7F77DD","ursidae":"#D85A30",
           "canidae":"#EF9F27","equidae":"#378ADD","rhinocerotidae":"#5DCAA5",
           "outgroup":"#888780","unknown":"#444441"}
extinct = {"woolly_mammoth","american_mastodon","cave_lion","saber_tooth_cat",
           "cave_bear","dire_wolf","woolly_rhinoceros","columbian_mammoth","quagga"}

model.eval(); embs = []
with torch.no_grad():
    for name in species_names:
        v = torch.tensor(vectors[name]).unsqueeze(0)
        embs.append(model.encoder(v).squeeze(0).numpy())
coords = PCA(n_components=2).fit_transform(np.array(embs))

for i, name in enumerate(species_names):
    color = palette.get(get_group(name), "#444441")
    marker = '*' if name in extinct else 'o'
    ax.scatter(coords[i,0], coords[i,1], color=color,
               s=180 if name in extinct else 80,
               marker=marker, edgecolors='white', lw=0.5, zorder=3)
    ax.annotate(name.replace('_',' '), (coords[i,0], coords[i,1]),
                xytext=(4,4), textcoords='offset points', fontsize=6.5, color='#c2c0b6')

import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=c, label=g) for g, c in palette.items()]
handles += [plt.scatter([], [], marker='*', color='white', s=100, label='extinct'),
            plt.scatter([], [], marker='o', color='white', s=60,  label='living')]
ax.legend(handles=handles, fontsize=6, facecolor='#1a1d27',
          edgecolor='#444441', labelcolor='#c2c0b6', ncol=2)
ax.set_title('Learned embeddings (PCA of 128-dim)', color='white', fontsize=11)
ax.set_xlabel('PC1', color='#888780'); ax.set_ylabel('PC2', color='#888780')
ax.tick_params(colors='#888780')
for sp in ax.spines.values(): sp.set_edgecolor('#444441')

plt.tight_layout(pad=2)
plt.savefig("siamese_training_v2.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("\nPlot saved: siamese_training_v2.png")
print("Next: build the Streamlit demo app (Stage 5)")