# 🦣 Extinct Animal DNA Similarity — Paleo Siamese Net

> A Siamese neural network trained on real mitochondrial DNA from NCBI GenBank that predicts evolutionary similarity between extinct and living animals.

🔗 **[Live Demo → paleo-siamese-net-5gw6ee3sjfqvpkjjk4aate.streamlit.app](https://paleo-siamese-net-5gw6ee3sjfqvpkjjk4aate.streamlit.app/)**

---

## What it does

Paste or select any two species — living or extinct — and the model returns an evolutionary similarity score (0–100%) based on their mitochondrial DNA patterns.

- Woolly Mammoth vs Asian Elephant → **99.8%** ✅
- Cave Lion vs Tiger → **97.4%** ✅
- Woolly Mammoth vs Chicken → **56.6%** ✅ (correctly low)
- Saber-tooth Cat vs Grey Wolf → model correctly separates them despite both being large Pleistocene carnivores

The model outperforms the classical k-mer cosine baseline on every tested pair by learning *which* sequence patterns encode evolutionary distance — not just raw base composition.

---

## Why this is interesting

Most genomic similarity tools (BLAST, k-mer cosine) are classical algorithms with no learning component. This project asks: **can a neural network learn what evolutionary relatedness looks like from sequence statistics alone?**

The answer is yes — and measurably better than the baseline.

| Pair | K-mer baseline | Siamese model | Δ uplift |
|------|---------------|---------------|----------|
| Woolly Mammoth vs American Mastodon | 36.9% | 87.2% | +50.3% |
| Cave Bear vs Brown Bear | 45.8% | 90.6% | +44.8% |
| Dire Wolf vs Grey Wolf | 60.9% | 87.2% | +26.3% |
| Cave Lion vs Tiger | 63.8% | 97.4% | +33.6% |

---

## Architecture

```
FASTA sequence (16,000 bp)
        │
        ▼
  K-mer encoding (k=6)
  Sliding window → 4,096-dim frequency vector
        │
        ▼
  ┌─────────────────────┐     ┌─────────────────────┐
  │   DNA Encoder        │     │   DNA Encoder        │  ← shared weights
  │  4096 → 1024 → 256  │     │  4096 → 1024 → 256  │
  │       → 128-dim      │     │       → 128-dim      │
  └──────────┬──────────┘     └──────────┬──────────┘
             │                            │
             └──────── cosine sim ────────┘
                            │
                     similarity score
                          0.0 – 1.0
```

Both sequences pass through the **same encoder** (Siamese = shared weights). The model learns a universal DNA embedding space where evolutionary relatives cluster together.

---

## Dataset

- **44 species** — 8 extinct, 36 living
- **Source:** NCBI GenBank mitochondrial genomes (free, public)
- **Families covered:** Proboscidea, Felidae, Ursidae, Canidae, Rhinocerotidae, Equidae + outgroups
- **Sequence length:** ~16,000 base pairs per species (complete mitochondrial genome)

Extinct species included: Woolly Mammoth, Columbian Mammoth, American Mastodon, Saber-tooth Cat, Cave Lion, Cave Bear, Dire Wolf, Woolly Rhinoceros

---

## Training

| Detail | Value |
|--------|-------|
| Pairs (after augmentation) | 15,136 |
| Same-family pairs | 165 |
| Val accuracy | **96%** |
| Loss | Combined contrastive + MSE |
| Epochs | 150 |
| Optimizer | AdamW + CosineAnnealing LR |
| Class balancing | WeightedRandomSampler (5× upsample) |

Key training decisions:
- **Contrastive loss with margin=0.5** — forces unrelated species embeddings apart on the unit sphere
- **Gaussian augmentation** — 15× augmentation with higher noise on negative pairs
- **3-level labels** (1.0 / 0.3 / 0.0) — same family / related order / unrelated — richer signal than binary

---

## Results

The learned embedding space (PCA of 128-dim vectors) shows species self-organising into evolutionary clusters with no taxonomy labels at inference time:

- Proboscideans (mammoths + elephants) cluster tightly
- Felids (cats, extinct and living) cluster separately
- Outgroups (chicken, crocodile, python) are correctly pushed to the periphery

---

## App features

| Tab | What it does |
|-----|-------------|
| Compare two species | Select or paste any two sequences → similarity % + family prediction + model vs baseline uplift |
| Phylogenetic tree | Dendrogram built from pairwise model scores across selected families |
| Full similarity matrix | Interactive heatmap, all 44 species, exportable |

---

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME
cd YOUR_REPO_NAME

pip install -r requirements.txt
streamlit run app.py
```

---

## Stack

`Python` · `PyTorch` · `Biopython` · `Streamlit` · `scikit-learn` · `scipy` · `NCBI GenBank`

---

## What I learned

- Siamese networks are uniquely suited to similarity tasks because shared weights force the model to learn a *universal* representation rather than memorising individual pairs
- Class imbalance in pair datasets is severe (85% negatives) and requires weighted sampling — not just loss weighting
- Real paleogenomic data from NCBI is surprisingly accessible; the hardest part is curating ground-truth labels from taxonomy
- K-mer frequency vectors at k=6 are a strong baseline for mitochondrial DNA — the Siamese model's value is in learning *which* k-mer patterns matter most for evolutionary distance

---

*Built from scratch using publicly available paleogenomic data. No pre-trained genomics models used.*
