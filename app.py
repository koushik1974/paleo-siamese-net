"""
Stage 5: Streamlit Demo App — Extinct Animal DNA Similarity
===========================================================
Install: pip install streamlit torch biopython pandas numpy scikit-learn scipy matplotlib
Run:     streamlit run app.py
Deploy:  push to GitHub → connect on streamlit.io (free)
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import hashlib
import re

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Specimen Comparator — Extinct Animal DNA",
    page_icon="🦴",
    layout="wide",
)

# ── Design tokens ─────────────────────────────────────────────────────────
# A natural-history-museum specimen card, not a dashboard: deep moss/charcoal
# ground, amber resin as the single warm accent, bone-white type, and a
# monospace face reserved for the one thing in this app that is *literally*
# a code — the DNA itself.
BG_DEEP    = "#12160F"
BG_PANEL   = "#1B221A"
BG_PANEL_2 = "#212A20"
BORDER     = "#3B4536"
BONE       = "#EDE8D9"
MUTED      = "#93A08C"
AMBER      = "#C6893F"
AMBER_HI   = "#E2A75B"
MOSS       = "#7C9473"
RUST       = "#B5563B"
OCHRE      = "#D9A441"

FAMILY_COLORS = {
    "Proboscidea":    AMBER,
    "Felidae":        "#8B7EC8",
    "Ursidae":        RUST,
    "Canidae":        OCHRE,
    "Rhinocerotidae": "#5B84A6",
    "Equidae":        MOSS,
    "Outgroup":       MUTED,
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Mono', monospace;
}}
h1, h2, h3, h4, h5, h6, p, span, div, label, button, input, textarea, select {{
    font-family: 'IBM Plex Mono', monospace !important;
}}
.stApp {{
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, #1E2A19 0%, transparent 60%),
        {BG_DEEP};
}}

/* ── Kill the default chrome, tighten the frame ───────────────────────── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}

/* ── Type scale ────────────────────────────────────────────────────────── */
h1, h2, h3 {{
    font-family: 'IBM Plex Mono', monospace !important;
    color: {BONE} !important;
    letter-spacing: -0.01em;
}}
p, span, div, label {{ color: {BONE}; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}

/* ── Specimen letterhead ──────────────────────────────────────────────── */
.specimen-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {AMBER_HI};
    border-top: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 7px 0;
    margin-bottom: 1.1rem;
    display: flex;
    justify-content: space-between;
}}
.specimen-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 2.3rem;
    color: {BONE};
    margin: 0 0 0.15rem 0;
    line-height: 1.2;
    letter-spacing: -0.02em;
}}
.specimen-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 400;
    color: {MUTED};
    font-size: 0.95rem;
    margin-bottom: 1.6rem;
}}

/* ── Tabs styled as catalog dividers ──────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.04em;
    color: {MUTED};
    background: transparent;
    padding: 10px 6px;
}}
.stTabs [aria-selected="true"] {{
    color: {AMBER_HI} !important;
    border-bottom: 2px solid {AMBER} !important;
}}

/* ── Taxonomy chip ─────────────────────────────────────────────────────── */
.family-tag {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 11px 3px 8px;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border: 1px solid;
}}
.family-dot {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}
.status-tag {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: {MUTED};
}}

/* ── Specimen result card ─────────────────────────────────────────────── */
.spec-card {{
    position: relative;
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    padding: 2.4rem 2rem 2rem 2rem;
    margin-top: 0.5rem;
}}
.spec-card::before, .spec-card::after,
.spec-card .cbr, .spec-card .cbl {{
    content: ""; position: absolute; width: 16px; height: 16px;
    border-color: {AMBER}; border-style: solid; opacity: 0.9;
}}
.spec-card::before {{ top: -1px; left: -1px; border-width: 2px 0 0 2px; }}
.spec-card::after   {{ bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }}
.spec-card .cbr {{ top: -1px; right: -1px; border-width: 2px 2px 0 0; }}
.spec-card .cbl {{ bottom: -1px; left: -1px; border-width: 0 0 2px 2px; }}

.spec-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {MUTED};
    text-align: center;
    margin-bottom: 0.4rem;
}}
.spec-score {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 4.2rem;
    text-align: center;
    line-height: 1;
    letter-spacing: -0.02em;
}}
.spec-verdict {{
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    margin-top: 0.6rem;
    color: {BONE};
}}
.dna-ribbon {{
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.35em;
    font-size: 0.68rem;
    color: {BORDER};
    text-align: center;
    margin: 1.4rem 0 0.2rem 0;
    overflow: hidden;
    white-space: nowrap;
}}

/* ── Metrics ───────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {BG_PANEL_2};
    border: 1px solid {BORDER};
    padding: 0.9rem 1rem;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {MUTED} !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    color: {AMBER_HI} !important;
}}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {{
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 0.82rem;
    background: {AMBER} !important;
    color: #12160F !important;
    border: none !important;
    border-radius: 2px !important;
    font-weight: 600;
}}
.stButton > button:hover {{ background: {AMBER_HI} !important; }}

/* ── Inputs / selects ──────────────────────────────────────────────────── */
[data-baseweb="select"] > div, .stTextArea textarea {{
    background: {BG_PANEL_2} !important;
    border-color: {BORDER} !important;
    font-family: 'IBM Plex Mono', monospace !important;
}}

hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)

# ── Model definition (must match training) ───────────────────────────────
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

# ── Load model + data ──────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("kmer_vectors.csv", index_col="species")
    vectors = {n: df.loc[n].values.astype(np.float32) for n in df.index}
    model = SiameseDNA(input_dim=df.shape[1])
    model.load_state_dict(torch.load("siamese_dna_model_v2.pt", map_location="cpu"))
    model.eval()
    return model, vectors, list(df.index)

model, vectors, species_list = load_model_and_data()

# ── K-mer helper (for pasting raw sequences) ──────────────────────────────
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

# ── Predict ────────────────────────────────────────────────────────────────
def predict(v1, v2):
    model.eval()
    with torch.no_grad():
        t1 = torch.tensor(v1).unsqueeze(0)
        t2 = torch.tensor(v2).unsqueeze(0)
        sim, _, _ = model(t1, t2)
        return float(sim.item())

# ── Family / extinction metadata ─────────────────────────────────────────
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

def score_label(s):
    if s >= 0.85: return "Very closely related", MOSS
    if s >= 0.70: return "Related", OCHRE
    if s >= 0.50: return "Distantly related", AMBER
    return "Not related", RUST

def format_name(n): return n.replace("_", " ").title()

def catalog_no(name):
    """Deterministic pseudo-catalog number, purely decorative."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 9000 + 1000
    return f"SPEC-{h}"

def family_chip(species):
    fam = FAMILY.get(species, "Unknown")
    color = FAMILY_COLORS.get(fam, MUTED)
    status = "EXTINCT" if species in EXTINCT else "EXTANT"
    return f"""
    <span class="family-tag" style="border-color:{color}55;color:{color};background:{color}14">
        <span class="family-dot" style="background:{color}"></span>{fam}
    </span>
    &nbsp;&nbsp;<span class="status-tag">{'🦴' if status=='EXTINCT' else '🌿'} {status}</span>
    """

# ─────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="specimen-eyebrow"><span>FIELD COLLECTION · MITOCHONDRIAL DNA</span><span>SIAMESE NEURAL COMPARATOR v2</span></div>
<div class="specimen-title">Extinct Animal DNA Similarity</div>
<div class="specimen-sub">A Siamese neural network trained on mitochondrial sequences from NCBI GenBank, estimating evolutionary kinship between specimens living and long gone.</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔬  COMPARE SPECIMENS", "🌳  PHYLOGENETIC TREE", "📊  SIMILARITY MATRIX"])

# ── Tab 1: Compare ──────────────────────────────────────────────────────
with tab1:
    mode = st.radio("Input mode", ["Choose from database", "Paste raw DNA sequence"], horizontal=True, label_visibility="collapsed")
    st.write("")

    col1, col2 = st.columns(2)

    if mode == "Choose from database":
        with col1:
            s1 = st.selectbox("Specimen 1", species_list, index=species_list.index("woolly_mammoth") if "woolly_mammoth" in species_list else 0, format_func=format_name)
            st.caption(f"Catalog {catalog_no(s1)}")
            st.markdown(family_chip(s1), unsafe_allow_html=True)
            v1 = vectors[s1]

        with col2:
            default2 = "asian_elephant" if "asian_elephant" in species_list else species_list[1]
            s2 = st.selectbox("Specimen 2", species_list, index=species_list.index(default2), format_func=format_name)
            st.caption(f"Catalog {catalog_no(s2)}")
            st.markdown(family_chip(s2), unsafe_allow_html=True)
            v2 = vectors[s2]
    else:
        with col1:
            st.markdown("**Specimen 1** — paste FASTA or raw sequence")
            seq1 = st.text_area("Sequence 1", height=150, placeholder=">species_name\nATCGATCGATCG...", label_visibility="collapsed")
        with col2:
            st.markdown("**Specimen 2** — paste FASTA or raw sequence")
            seq2 = st.text_area("Sequence 2", height=150, placeholder=">species_name\nATCGATCGATCG...", label_visibility="collapsed")

        if seq1 and seq2:
            raw1 = '\n'.join(l for l in seq1.strip().splitlines() if not l.startswith('>'))
            raw2 = '\n'.join(l for l in seq2.strip().splitlines() if not l.startswith('>'))
            v1 = seq_to_vector(raw1)
            v2 = seq_to_vector(raw2)
            s1, s2 = "custom_1", "custom_2"
        else:
            st.info("Paste both sequences to compare.")
            st.stop()

    st.write("")
    run = st.button("RUN COMPARISON", type="primary", use_container_width=True)
    st.write("")

    if run:
        score = predict(v1, v2)
        label, color = score_label(score)
        ribbon = " ".join(np.random.choice(list("ATGC"), 48))

        st.markdown(f"""
        <div class="spec-card">
            <span class="cbr"></span><span class="cbl"></span>
            <div class="spec-label">Evolutionary Similarity Index</div>
            <div class="spec-score" style="color:{color}">{score*100:.1f}%</div>
            <div class="spec-verdict">"{label}" — {format_name(s1)} × {format_name(s2)}</div>
            <div class="dna-ribbon">{ribbon}</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        c1, c2, c3 = st.columns(3)
        baseline = float(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8))
        c1.metric("Model score",    f"{score*100:.1f}%")
        c2.metric("K-mer baseline", f"{baseline*100:.1f}%")
        c3.metric("Model uplift",   f"+{(score-baseline)*100:.1f}%")

        if mode == "Choose from database":
            same = FAMILY.get(s1,"?") == FAMILY.get(s2,"?")
            st.write("")
            st.markdown(f"""
            <div style="border-left:2px solid {AMBER};padding:0.7rem 1rem;background:{BG_PANEL_2};font-family:'IBM Plex Mono',monospace;color:{BONE}">
            <b>{format_name(s1)}</b> <span style="color:{MUTED}">({FAMILY.get(s1,'?')})</span> and
            <b>{format_name(s2)}</b> <span style="color:{MUTED}">({FAMILY.get(s2,'?')})</span>
            {"belong to the <b style='color:%s'>same evolutionary family</b>." % MOSS if same else "belong to <b style='color:%s'>different families</b>." % RUST}
            </div>
            """, unsafe_allow_html=True)

# ── Tab 2: Phylogenetic tree ─────────────────────────────────────────────
with tab2:
    st.markdown("### Phylogenetic dendrogram — learned by the model")
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
        fig.patch.set_facecolor(BG_DEEP)
        ax.set_facecolor(BG_DEEP)

        labels = [
            ("\U0001F9B4 " if s in EXTINCT else "\U0001F33F ") + format_name(s)
            for s in show_species
        ]

        dn = dendrogram(Z, labels=labels, orientation='left', ax=ax,
                        leaf_font_size=8,
                        link_color_func=lambda k: AMBER)

        ax.set_xlabel("Evolutionary distance (1 − similarity)", color=MUTED, fontsize=9)
        ax.tick_params(colors=BONE, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        yticklabels = ax.get_yticklabels()
        for lbl in yticklabels:
            txt = lbl.get_text()
            clean = txt.replace("\U0001F9B4 ","").replace("\U0001F33F ","")
            sp_name = clean.lower().replace(" ","_")
            color = FAMILY_COLORS.get(FAMILY.get(sp_name,"Outgroup"), MUTED)
            lbl.set_color(color)

        legend_handles = [mpatches.Patch(color=c, label=f) for f,c in FAMILY_COLORS.items()]
        ax.legend(handles=legend_handles, fontsize=7, facecolor=BG_PANEL,
                  edgecolor=BORDER, labelcolor=BONE, loc='lower right')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Species that cluster together are predicted to share evolutionary ancestry. Leaf colors = taxonomic family. 🦴 = extinct species.")

# ── Tab 3: Full matrix ────────────────────────────────────────────────────
with tab3:
    st.markdown("### Pairwise similarity matrix")
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

        # Rust → ochre → moss: matches the app's own "not related → related" language.
        specimen_cmap = LinearSegmentedColormap.from_list("specimen", [RUST, OCHRE, MOSS])

        st.dataframe(
            mat_df.style.background_gradient(cmap=specimen_cmap, vmin=0.3, vmax=1.0)
                        .format("{:.3f}"),
            use_container_width=True
        )
        st.caption("Moss green = high similarity (closely related). Rust = low similarity (distant). Download via the table's built-in export button.")
