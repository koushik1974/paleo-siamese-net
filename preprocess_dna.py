import os
import numpy as np
import pandas as pd
from itertools import product
from Bio import SeqIO
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


K = 6          
FASTA_DIR = "dna_sequences"
OUTPUT_CSV = "kmer_vectors.csv"



def get_all_kmers(k):
    """All 4^k possible DNA k-mers."""
    return [''.join(p) for p in product('ATGC', repeat=k)]

def sequence_to_kmer_vector(sequence, k=6):
    """
    Sliding window of size k over the sequence.
    Returns L1-normalized frequency array of shape (4^k,).

    Why L1 normalize? So two sequences of different lengths are still
    comparable — we care about *patterns*, not raw counts.
    """
    sequence = sequence.upper().replace('N', '')  
    kmers = get_all_kmers(k)
    kmer_index = {km: i for i, km in enumerate(kmers)}
    counts = np.zeros(len(kmers), dtype=np.float32)

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in kmer_index:
            counts[kmer_index[kmer]] += 1

    total = counts.sum()
    if total > 0:
        counts /= total     

    return counts, kmers



print(f"Loading sequences and computing {K}-mer vectors (4^{K} = {4**K} features)...\n")
print(f"{'Species':<25} {'Length':>12} {'Vector size':>12} {'Unique k-mers':>14}")
print("-" * 65)

vectors = {}
kmer_labels = None

for fname in sorted(os.listdir(FASTA_DIR)):
    if not fname.endswith(".fasta"):
        continue
    name = fname.replace(".fasta", "")
    rec = next(SeqIO.parse(os.path.join(FASTA_DIR, fname), "fasta"))
    seq = str(rec.seq)
    vec, kmer_labels = sequence_to_kmer_vector(seq, k=K)
    vectors[name] = vec
    unique = int((vec > 0).sum())
    print(f"  {name:<23} {len(seq):>12,} {len(vec):>12,} {unique:>14,}")


df = pd.DataFrame(vectors, index=kmer_labels).T
df.index.name = "species"
df.to_csv(OUTPUT_CSV)
print(f"\nSaved: {OUTPUT_CSV}  shape={df.shape}")
print(f"  rows = species, columns = k-mer frequencies")
print(f"  Each row is one training sample for your Siamese network.\n")


names = list(vectors.keys())
n = len(names)
sim_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        a, b = vectors[names[i]], vectors[names[j]]
        sim_matrix[i, j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("── Similarity between key pairs ──")
pairs = [
    ("woolly_mammoth",    "asian_elephant",    "expected HIGH — same order"),
    ("woolly_mammoth",    "american_mastodon", "expected HIGH — close relatives"),
    ("cave_lion",         "tiger",             "expected HIGH — same family"),
    ("cave_lion",         "cave_bear",         "expected LOWER — different orders"),
    ("woolly_mammoth",    "chicken",           "expected LOW — distant outgroup"),
]
for a, b, note in pairs:
    if a in vectors and b in vectors:
        va, vb = vectors[a], vectors[b]
        sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
        print(f"  {a:<22} vs {b:<22} → {sim:.4f}  ({note})")



fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor('#0f1117')


ax1 = axes[0]
im = ax1.imshow(sim_matrix, cmap='RdYlGn', vmin=0.8, vmax=1.0, aspect='auto')
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).set_label('cosine similarity', color='#c2c0b6')
ax1.set_xticks(range(n))
ax1.set_yticks(range(n))
ax1.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7, color='#c2c0b6', rotation=45, ha='right')
ax1.set_yticklabels([n.replace('_', ' ') for n in names], fontsize=8, color='#c2c0b6')
ax1.set_facecolor('#1a1d27')
ax1.set_title('K-mer cosine similarity matrix', color='white', fontsize=12, pad=12)
for spine in ax1.spines.values(): spine.set_visible(False)
ax1.tick_params(colors='#888780')
for i in range(n):
    for j in range(n):
        ax1.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center',
                 fontsize=5.5, color='white' if sim_matrix[i,j] < 0.92 else 'black')


ax2 = axes[1]
mat = np.array([vectors[nm] for nm in names])
pca = PCA(n_components=2)
coords = pca.fit_transform(mat)

groups = {
    "Proboscideans": ["woolly_mammoth","american_mastodon","asian_elephant","african_elephant"],
    "Felids":        ["cave_lion","saber_tooth_cat","tiger"],
    "Ursids":        ["cave_bear","brown_bear"],
    "Canids":        ["dire_wolf","grey_wolf"],
    "Outgroups":     ["human","chicken"],
}
palette = {"Proboscideans":"#1D9E75","Felids":"#7F77DD","Ursids":"#D85A30","Canids":"#EF9F27","Outgroups":"#888780"}
name_to_group = {sp: g for g, sps in groups.items() for sp in sps}
extinct = {"woolly_mammoth","american_mastodon","cave_lion","saber_tooth_cat","cave_bear","dire_wolf"}

ax2.set_facecolor('#1a1d27')
for i, name in enumerate(names):
    group = name_to_group.get(name, "Outgroups")
    color = palette[group]
    marker = '*' if name in extinct else 'o'
    size = 160 if name in extinct else 90
    ax2.scatter(coords[i, 0], coords[i, 1], color=color, s=size, marker=marker,
                edgecolors='white', linewidths=0.5, zorder=3)
    ax2.annotate(name.replace('_', ' '), (coords[i, 0], coords[i, 1]),
                 xytext=(5, 5), textcoords='offset points', fontsize=7, color='#c2c0b6')

handles = [mpatches.Patch(color=c, label=g) for g, c in palette.items()]
handles += [plt.scatter([], [], marker='*', color='white', s=100, label='extinct'),
            plt.scatter([], [], marker='o', color='white', s=60,  label='living')]
ax2.legend(handles=handles, fontsize=7, facecolor='#1a1d27', edgecolor='#444441', labelcolor='#c2c0b6')
ax2.set_title(f'PCA of {K}-mer vectors — species cluster by evolutionary family',
              color='white', fontsize=12, pad=12)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', color='#888780', fontsize=9)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', color='#888780', fontsize=9)
ax2.tick_params(colors='#888780')
for spine in ax2.spines.values(): spine.set_edgecolor('#444441')

plt.tight_layout(pad=2)
plt.savefig("kmer_visualization.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("\nVisualization saved: kmer_visualization.png")
print("\nNext step: feed kmer_vectors.csv into your Siamese network (Stage 3).")
