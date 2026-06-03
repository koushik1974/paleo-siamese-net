"""
Extinct + Living Animal Mitochondrial DNA Downloader
Uses NCBI Entrez via Biopython — free, no account needed.
Run: pip install biopython && python download_dna.py
"""

from Bio import Entrez, SeqIO
import time
import os

# NCBI requires your email (they won't spam you, just for rate limiting)
Entrez.email = "bathulakoushikyadav@gmail.com"  # <-- change this

# ─── Accession numbers ────────────────────────────────────────────────────────
# Mitochondrial genomes (complete, ~16,000 bases each)
SPECIES = {
    # EXTINCT
    "woolly_mammoth":       "NC_007596",   # Mammuthus primigenius
    "american_mastodon":    "NC_005993",   # Mammut americanum
    "cave_bear":            "NC_011112",   # Ursus spelaeus
    "cave_lion":            "NC_043312",   # Panthera spelaea
    "saber_tooth_cat":      "NC_023082",   # Smilodon fatalis (partial)
    "dire_wolf":            "NC_058792",   # Aenocyon dirus
    "woolly_rhinoceros":    "NC_012681",   # Coelodonta antiquitatis
    "dodo":                 "NC_002619",   # Raphus cucullatus

    # LIVING (for comparison — your model needs these as anchors)
    "asian_elephant":       "NC_005129",   # Elephas maximus
    "african_elephant":     "NC_000934",   # Loxodonta africana
    "tiger":                "NC_010641",   # Panthera tigris
    "lion":                 "NC_009833",   # Panthera leo
    "brown_bear":           "NC_003218",   # Ursus arctos
    "grey_wolf":            "NC_008093",   # Canis lupus
    "white_rhinoceros":     "NC_001808",   # Ceratotherium simum
    "chicken":              "NC_001323",   # Gallus gallus (distant outgroup)
    "human":                "NC_012920",   # Homo sapiens (distant outgroup)
}

os.makedirs("dna_sequences", exist_ok=True)

def download_sequence(name, accession):
    filepath = f"dna_sequences/{name}.fasta"
    if os.path.exists(filepath):
        print(f"  [skip] {name} already downloaded")
        return

    try:
        handle = Entrez.efetch(
            db="nucleotide",
            id=accession,
            rettype="fasta",
            retmode="text"
        )
        record = SeqIO.read(handle, "fasta")
        handle.close()

        # Clean up the header to be human-readable
        record.id = name
        record.description = f"{name} | {accession} | {len(record.seq)} bp"

        with open(filepath, "w") as f:
            SeqIO.write(record, f, "fasta")

        print(f"  [ok]   {name:30s} {len(record.seq):,} bp → {filepath}")
        time.sleep(0.4)  # NCBI rate limit: max 3 requests/sec without API key

    except Exception as e:
        print(f"  [ERR]  {name}: {e}")

print("Downloading mitochondrial DNA sequences from NCBI GenBank...\n")
print(f"{'Species':<30} {'Length':<12} File")
print("-" * 60)

for name, accession in SPECIES.items():
    download_sequence(name, accession)

print("\nDone! All files saved to ./dna_sequences/")
print("\nQuick check — sequence lengths:")
for fasta_file in sorted(os.listdir("dna_sequences")):
    if fasta_file.endswith(".fasta"):
        rec = next(SeqIO.parse(f"dna_sequences/{fasta_file}", "fasta"))
        print(f"  {fasta_file.replace('.fasta',''):<30} {len(rec.seq):,} bp")
