"""
Methodenvergleich-Analyse: Korrelation & prozentuale Differenz
==============================================================
Liest eine oder mehrere "<See>_methodenvergleich.csv" (aus dem Wassermasken-
Skript) und berechnet:

  1. Pearson- & Spearman-Korrelation der 5 Methoden
     → Muster-Übereinstimmung. NORMIERUNG IST HIER IRRELEVANT:
       Pearson ist invariant gegenüber z-/Min-Max-Skalierung, das Ergebnis
       auf Absolut- und normierten Werten ist identisch.

  2. Prozentuale Differenz je Methodenpaar, relativ zum jeweils größeren Wert:
        pct = (A - B) / max(A, B) * 100
     - MAPD_%  = mittlere ABSOLUTE prozentuale Differenz (wie groß insgesamt?)
     - Bias_%  = mittlere prozentuale Differenz MIT Vorzeichen
                 (>0: A im Schnitt größer als B, <0: kleiner)
     - max_%   = größte Einzelabweichung im Zeitraum

Ausgabe: Konsole + CSVs + optionale Heatmap.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

# ── EINGABE: eine oder mehrere CSVs ───────────────────────────────
CSV_FILES = [
    r"E:\Bachelorarbeit\openEO\Müritz\Müritz_methodenvergleich.csv",
    # weitere Seen einfach ergänzen, z.B.:
    # r"E:\Bachelorarbeit\openEO\Chiemsee\Chiemsee_methodenvergleich.csv",
]

# ── Daten laden (mehrere Seen werden zusammengeführt) ─────────────
frames = []
for f in CSV_FILES:
    d = pd.read_csv(f)
    d["see"] = Path(f).stem.replace("_methodenvergleich", "")
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

area_cols = [c for c in df.columns if c.startswith("area_")]
print(f"Geladen: {len(df)} Zeilen  |  Methoden: {area_cols}\n")

# ── 1) KORRELATION (normierungs-invariant) ────────────────────────
pearson  = df[area_cols].corr(method="pearson")
spearman = df[area_cols].corr(method="spearman")

print("── Pearson-Korrelation ──")
print(pearson.round(3), "\n")
print("── Spearman-Korrelation ──")
print(spearman.round(3), "\n")

# ── 2) PROZENTUALE DIFFERENZ relativ zum Maximum ──────────────────
def pct_diff_vs_max(a, b):
    """(a - b) / max(|a|,|b|) * 100. Monate, in denen beide 0 sind, -> NaN."""
    denom = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom == 0, np.nan, (a - b) / denom * 100)

rows = []
for i, j in combinations(area_cols, 2):
    pct = pct_diff_vs_max(df[i].values, df[j].values)
    short = lambda c: c.replace("area_", "").replace("_km2", "")
    rows.append({
        "Paar":   f"{short(i)} vs {short(j)}",
        "MAPD_%": round(np.nanmean(np.abs(pct)), 2),   # mittlere abs. %-Differenz
        "Bias_%": round(np.nanmean(pct), 2),           # Richtung der Abweichung
        "max_%":  round(np.nanmax(np.abs(pct)), 2),    # größte Einzelabweichung
    })
diff = pd.DataFrame(rows).sort_values("MAPD_%", ascending=False)

print("── Prozentuale Differenz (rel. zum jeweils größeren Wert) ──")
print(diff.to_string(index=False), "\n")

# ── 3) SPEICHERN ──────────────────────────────────────────────────
out_dir = Path(CSV_FILES[0]).parent
pearson.round(4).to_csv(out_dir / "korrelation_pearson.csv")
spearman.round(4).to_csv(out_dir / "korrelation_spearman.csv")
diff.to_csv(out_dir / "prozentuale_differenz.csv", index=False)
print(f"→ CSVs gespeichert in: {out_dir}")

# ── 4) OPTIONAL: Korrelations-Heatmap ─────────────────────────────
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pearson, vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [c.replace("area_", "").replace("_km2", "") for c in area_cols]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for a in range(len(labels)):
        for b in range(len(labels)):
            ax.text(b, a, f"{pearson.iloc[a, b]:.2f}", ha="center", va="center",
                    color="white" if abs(pearson.iloc[a, b]) > 0.6 else "black")
    fig.colorbar(im, label="Pearson r")
    ax.set_title("Korrelation der Wasserflächen (5 Methoden)")
    fig.tight_layout()
    fig.savefig(out_dir / "korrelation_heatmap.png", dpi=150)
    print(f"→ Heatmap gespeichert: {out_dir / 'korrelation_heatmap.png'}")
except ImportError:
    print("(matplotlib nicht installiert – Heatmap übersprungen)")
