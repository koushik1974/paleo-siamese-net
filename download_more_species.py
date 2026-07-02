"""
pip install biopython
python download_more_species.py
"""

from Bio import Entrez, SeqIO
import time, os

Entrez.email = "your_email@example.com"   

SPECIES = {

    "woolly_mammoth":        "NC_007596",
    "columbian_mammoth":     "NC_015529",
    "american_mastodon":     "NC_005993",
    "asian_elephant":        "NC_005129",
    "african_savanna_elephant": "NC_000934",
    "african_forest_elephant":  "NC_011632",  

    "saber_tooth_cat":       "NC_023082",
    "cave_lion":             "NC_043312",
    "tiger":                 "NC_010641",
    "lion":                  "NC_009833",
    "leopard":               "NC_010642",
    "snow_leopard":          "NC_010638",
    "cheetah":               "NC_005212",
    "cougar":                "NC_016422",
    "domestic_cat":          "NC_001700",


    "cave_bear":             "NC_011112",
    "brown_bear":            "NC_003218",
    "polar_bear":            "NC_023722",
    "american_black_bear":   "NC_003489",
    "giant_panda":           "NC_009492",
    "sun_bear":              "NC_009324",

    "dire_wolf":             "NC_058792",
    "grey_wolf":             "NC_008093",
    "domestic_dog":          "NC_002008",
    "coyote":                "NC_008093",   
    "african_wild_dog":      "NC_008440",
    "dhole":                 "NC_008442",


    "woolly_rhinoceros":     "NC_012681",
    "white_rhinoceros":      "NC_001808",
    "black_rhinoceros":      "NC_001808",   
    "indian_rhinoceros":     "NC_001779",
    "sumatran_rhinoceros":   "NC_012684",


    "horse":                 "NC_001640",
    "donkey":                "NC_001788",
    "plains_zebra":          "NC_004394",


    "human":                 "NC_012920",
    "bottlenose_dolphin":    "NC_012059",
    "blue_whale":            "NC_001601",
    "chicken":               "NC_001323",
    "nile_crocodile":        "NC_008241",
    "komodo_dragon":         "NC_016578",
    "python":                "NC_007397",
}

os.makedirs("dna_sequences", exist_ok=True)
print(f"Downloading {len(SPECIES)} species...\n")
print(f"{'Species':<30} {'Length':>10}  File")
print("-" * 60)

ok, skip, err = 0, 0, 0
for name, acc in SPECIES.items():
    path = f"dna_sequences/{name}.fasta"
    if os.path.exists(path):
        rec = next(SeqIO.parse(path, "fasta"))
        print(f"  [skip] {name:<28} {len(rec.seq):>10,}")
        skip += 1
        continue
    try:
        handle = Entrez.efetch(db="nucleotide", id=acc, rettype="fasta", retmode="text")
        rec = SeqIO.read(handle, "fasta")
        handle.close()
        rec.id = name
        rec.description = f"{name} | {acc} | {len(rec.seq)} bp"
        with open(path, "w") as f:
            SeqIO.write(rec, f, "fasta")
        print(f"  [ok]   {name:<28} {len(rec.seq):>10,}")
        ok += 1
        time.sleep(0.4)
    except Exception as e:
        print(f"  [ERR]  {name}: {e}")
        err += 1

print(f"\nDone — {ok} downloaded, {skip} skipped, {err} errors")
print("\nNow re-run:")
print("  python preprocess_dna.py")
print("  python siamese_dna_v2.py")
