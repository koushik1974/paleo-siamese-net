import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product, combinations
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, re

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Extinct Animal DNA Similarity",
    page_icon="🦣",
    layout="wide",
)

st.markdown("""
<style>
.big-score { font-size: 4rem; font-weight: 700; text-align: center; line-height: 1; }
.label     { font-size: 1rem; text-align: center; color: #888; margin-bottom: 0.5rem; }
.family-tag{ display:inline-block; padding:3px 12px; border-radius:99px;
             font-size:0.8rem; font-weight:500; }
</style>
""", unsafe_allow_html=True)

# ── Model definition  ───────────────────────────────────
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

# ── Load model + data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("kmer_vectors.csv", index_col="species")
    vectors = {n: df.loc[n].values.astype(np.float32) for n in df.index}
    model = SiameseDNA(input_dim=df.shape[1])
    model.load_state_dict(torch.load("siamese_dna_model_v2.pt", map_location="cpu"))
    model.eval()
    return model, vectors, list(df.index)

model, vectors, species_list = load_model_and_data()

# ── K-mer helper ─────────────────────────────────
def seq_to_vector(sequence, k=6):
    sequence = re.sub(r'[^ATGC]', '', sequence.upper())
    kmers = [''.join(p) for p in product('ATGC', repeat=k)]
    idx   = {km: i for i, km in enumerate(kmers)}
    counts = np.zeros(len(kmers), dtype=np.float32)
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in idx: counts[idx[kmer]] += 1
    total = counts.sum()
    if total > 0: counts /= total
    return counts

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(v1, v2):
    model.eval()
    with torch.no_grad():
        t1 = torch.tensor(v1).unsqueeze(0)
        t2 = torch.tensor(v2).unsqueeze(0)
        sim, _, _ = model(t1, t2)
        return float(sim.item())

# ── Family metadata ───────────────────────────────────────────────────────────
FAMILY = {
    "woolly_mammoth":"Proboscidea","columbian_mammoth":"Proboscidea",
    "american_mastodon":"Proboscidea","asian_elephant":"Proboscidea",
    "african_savanna_elephant":"Proboscidea","african_forest_elephant":"Proboscidea",
    "saber_tooth_cat":"Felidae","cave_lion":"Felidae","tiger":"Felidae",
    "lion":"Felidae","leopard":"Felidae","snow_leopard":"Felidae",
    "cheetah":"Felidae","cougar":"Felidae","domestic_cat":"Felidae",
    "cave_bear":"Ursidae","brown_bear":"Ursidae","polar_bear":"Ursidae",
    "american_black_bear":"Ursidae","giant_panda":"Ursidae","sun_bear":"Ursidae",
    "dire_wolf":"Canidae","grey_wolf":"Canidae","domestic_dog":"Canidae",
    "coyote":"Canidae","african_wild_dog":"Canidae","dhole":"Canidae",
    "woolly_rhinoceros":"Rhinocerotidae","white_rhinoceros":"Rhinocerotidae",
    "black_rhinoceros":"Rhinocerotidae","indian_rhinoceros":"Rhinocerotidae",
    "sumatran_rhinoceros":"Rhinocerotidae",
    "horse":"Equidae","donkey":"Equidae","plains_zebra":"Equidae",
    "human":"Outgroup","bottlenose_dolphin":"Outgroup","blue_whale":"Outgroup",
    "chicken":"Outgroup","nile_crocodile":"Outgroup","komodo_dragon":"Outgroup","python":"Outgroup",
}
EXTINCT = {
    "woolly_mammoth","columbian_mammoth","american_mastodon","saber_tooth_cat",
    "cave_lion","cave_bear","dire_wolf","woolly_rhinoceros",
}
FAMILY_COLORS = {
    "Proboscidea":"#1D9E75","Felidae":"#7F77DD","Ursidae":"#D85A30",
    "Canidae":"#EF9F27","Rhinocerotidae":"#378ADD","Equidae":"#3BAA5C","Outgroup":"#888780",
}

def score_label(s):
    if s >= 0.85: return "Very closely related 🟢", "#1D9E75"
    if s >= 0.70: return "Related 🟡", "#EF9F27"
    if s >= 0.50: return "Distantly related 🟠", "#D85A30"
    return "Not related 🔴", "#A32D2D"

def format_name(n): return n.replace("_", " ").title()

#UI

st.title("🦣 Extinct Animal DNA Similarity")
st.caption("A Siamese neural network trained on mitochondrial DNA from NCBI GenBank")

tab1, tab2, tab3 = st.tabs(["🔬 Compare two species", "🌳 Phylogenetic tree", "📊 Full similarity matrix"])

# ──Compare ────────────────────────────────────────────────────────────
with tab1:
    mode = st.radio("Input mode", ["Choose from database", "Paste raw DNA sequence"], horizontal=True)
    st.divider()

    col1, col2 = st.columns(2)

    if mode == "Choose from database":
        with col1:
            s1 = st.selectbox("Species 1", species_list, index=species_list.index("woolly_mammoth") if "woolly_mammoth" in species_list else 0, format_func=format_name)
            f1 = FAMILY.get(s1, "Unknown")
            e1 = "🦴 Extinct" if s1 in EXTINCT else "🌿 Living"
            st.markdown(f'<span class="family-tag" style="background:{FAMILY_COLORS.get(f1, "#888")}22;color:{FAMILY_COLORS.get(f1, "#888")}">{f1}</span> &nbsp; {e1}', unsafe_allow_html=True)
            v1 = vectors[s1]

        with col2:
            default2 = "asian_elephant" if "asian_elephant" in species_list else species_list[1]
            s2 = st.selectbox("Species 2", species_list, index=species_list.index(default2), format_func=format_name)
            f2 = FAMILY.get(s2, "Unknown")
            e2 = "🦴 Extinct" if s2 in EXTINCT else "🌿 Living"
            st.markdown(f'<span class="family-tag" style="background:{FAMILY_COLORS.get(f2, "#888")}22;color:{FAMILY_COLORS.get(f2, "#888")}">{f2}</span> &nbsp; {e2}', unsafe_allow_html=True)
            v2 = vectors[s2]
    else:
        with col1:
            st.markdown("**Species 1** — paste FASTA or raw sequence")
            seq1 = st.text_area("Sequence 1", height=150, placeholder=">species_name\nATCGATCGATCG...")
        with col2:
            st.markdown("**Species 2** — paste FASTA or raw sequence")
            seq2 = st.text_area("Sequence 2", height=150, placeholder=">species_name\nATCGATCGATCG...")

        if seq1 and seq2:
            raw1 = '\n'.join(l for l in seq1.strip().splitlines() if not l.startswith('>'))
            raw2 = '\n'.join(l for l in seq2.strip().splitlines() if not l.startswith('>'))
            v1 = seq_to_vector(raw1)
            v2 = seq_to_vector(raw2)
            s1, s2 = "custom_1", "custom_2"
        else:
            st.info("Paste both sequences to compare.")
            st.stop()

    st.divider()
    if st.button("🔬 Run comparison", type="primary", use_container_width=True):
        score = predict(v1, v2)
        label, color = score_label(score)

        st.markdown(f'<p class="label">Evolutionary similarity</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="big-score" style="color:{color}">{score*100:.1f}%</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align:center;font-size:1.1rem;margin-top:0.5rem">{label}</p>', unsafe_allow_html=True)
        st.divider()

        c1, c2, c3 = st.columns(3)
        baseline = float(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8))
        c1.metric("Model score",    f"{score*100:.1f}%")
        c2.metric("K-mer baseline", f"{baseline*100:.1f}%")
        c3.metric("Model uplift",   f"+{(score-baseline)*100:.1f}%")

        if mode == "Choose from database":
            same = FAMILY.get(s1,"?") == FAMILY.get(s2,"?")
            st.info(
                f"**{format_name(s1)}** ({FAMILY.get(s1,'?')}) and **{format_name(s2)}** ({FAMILY.get(s2,'?')}) "
                + ("belong to the **same evolutionary family**." if same else "belong to **different families**.")
            )

# ── Phylogenetic tree ───────────────────────────────────────────────────
with tab2:
    st.subheader("Phylogenetic dendrogram — learned by the model")
    st.caption("Computed from pairwise Siamese similarity scores across all species in the database")

    selected_families = st.multiselect(
        "Filter families",
        options=list(FAMILY_COLORS.keys()),
        default=["Proboscidea","Felidae","Ursidae","Canidae","Outgroup"],
    )
    show_species = [s for s in species_list if FAMILY.get(s,"Outgroup") in selected_families]

    if len(show_species) < 3:
        st.warning("Select at least 3 families to draw a tree.")
    else:
        with st.spinner("Computing pairwise similarities..."):
            n = len(show_species)
            sim_mat = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j: sim_mat[i,j] = 1.0
                    elif i < j:
                        s = predict(vectors[show_species[i]], vectors[show_species[j]])
                        sim_mat[i,j] = sim_mat[j,i] = s

        dist_mat = 1.0 - sim_mat
        dist_mat = np.clip(dist_mat, 0, None)
        dist_mat = (dist_mat + dist_mat.T) / 2
        np.fill_diagonal(dist_mat, 0)
        condensed = squareform(dist_mat)
        Z = linkage(condensed, method='average')

        fig, ax = plt.subplots(figsize=(10, max(5, len(show_species)*0.38)))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#0f1117')

        labels = [
            ("🦴 " if s in EXTINCT else "🌿 ") + format_name(s)
            for s in show_species
        ]
        leaf_colors = {
            labels[i]: FAMILY_COLORS.get(FAMILY.get(show_species[i],"Outgroup"), "#888")
            for i in range(len(show_species))
        }

        dn = dendrogram(Z, labels=labels, orientation='left', ax=ax,
                        leaf_font_size=8,
                        link_color_func=lambda k: '#c2c0b6')

        ax.set_xlabel("Evolutionary distance (1 − similarity)", color='#888780', fontsize=9)
        ax.tick_params(colors='#c2c0b6', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#444441')

        yticklabels = ax.get_yticklabels()
        for lbl in yticklabels:
            txt = lbl.get_text()
            clean = txt.replace("🦴 ","").replace("🌿 ","")
            sp_name = clean.lower().replace(" ","_")
            color = FAMILY_COLORS.get(FAMILY.get(sp_name,"Outgroup"),"#888")
            lbl.set_color(color)

        legend_handles = [mpatches.Patch(color=c, label=f) for f,c in FAMILY_COLORS.items()]
        ax.legend(handles=legend_handles, fontsize=7, facecolor='#1a1d27',
                  edgecolor='#444441', labelcolor='#c2c0b6', loc='lower right')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Species that cluster together are predicted to share evolutionary ancestry. Leaf colors = taxonomic family. 🦴 = extinct species.")

# ──Full matrix ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Pairwise similarity matrix")
    fam_filter = st.multiselect("Filter families", list(FAMILY_COLORS.keys()),
                                default=["Proboscidea","Felidae","Ursidae"])
    filtered = [s for s in species_list if FAMILY.get(s,"Outgroup") in fam_filter]

    if len(filtered) < 2:
        st.warning("Select at least 2 families.")
    else:
        with st.spinner("Computing..."):
            rows = []
            for s1_ in filtered:
                row = {}
                for s2_ in filtered:
                    if s1_ == s2_: row[format_name(s2_)] = 1.0
                    else: row[format_name(s2_)] = round(predict(vectors[s1_], vectors[s2_]),3)
                rows.append(row)
            mat_df = pd.DataFrame(rows, index=[format_name(s) for s in filtered])

        st.dataframe(
            mat_df.style.background_gradient(cmap='RdYlGn', vmin=0.3, vmax=1.0)
                        .format("{:.3f}"),
            use_container_width=True
        )
        st.caption("Green = high similarity (closely related). Red = low similarity (distant). Download via the table's built-in export button.")
