"""
Validierung: Satelliten-Wasserflaeche vs. Pegelstand – alle 4 Seen, alle 5 Methoden
=====================================================================================
Erzeugt pro See+Methode:  Validierung_{See}_{Methode}.png
Erzeugt am Ende:          Validierung_Zusammenfassung.csv  (alle Korrelationen)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURATION  –  hier alle seenspezifischen Einstellungen eintragen
# ══════════════════════════════════════════════════════════════════════════════

BASE_ROOT = r"E:\Bachelorarbeit\openEO"

SEEN_CONFIG = {
    "Forggensee": {
        "pegel_csv":   r"E:\Bachelorarbeit\openEO\Forggensee\Forggensee_Pegelstand.csv",
        "pegel_label": "Rosshaupten Seepegel",
        "skiprows":    8,        # GKD Bayern: 8 Kopfzeilen
        "encoding":    "latin-1",
        "sep":         ";",
        "decimal":     ",",
    },
    "Dümmer": {
        "pegel_csv":   r"E:\Bachelorarbeit\openEO\Dümmer\Dümmer_Pegelstand.csv",
        "pegel_label": "Dümmer Pegelstand",
        "skiprows":    0,
        "encoding":    "latin-1",
        "sep":         ";",
        "decimal":     ",",
    },
    "Groß_Glienicker_See": {
        "pegel_csv":   r"E:\Bachelorarbeit\openEO\Groß_Glienicker_See\Groß_Glienicker_See_Pegelstand.csv",
        "pegel_label": "Groß Glienicker See Pegelstand",
        "skiprows":    0,
        "encoding":    "latin-1",
        "sep":         ";",
        "decimal":     ",",
    },
    "Chiemsee": {
        "pegel_csv":   r"E:\Bachelorarbeit\openEO\Chiemsee\Chiemsee_Pegelstand.csv",
        "pegel_label": "Chiemsee Pegelstand",
        "skiprows":    0,
        "encoding":    "latin-1",
        "sep":         ";",
        "decimal":     ",",
    },
}

METHODEN = [
    ("M1_S1_K0_km2",    "M1 S1-Otsu",   "#2196F3"),
    ("M2_S2_NDWI_km2",  "M2 S2-NDWI",   "#FF9800"),
    ("M3_AND_km2",      "M3 AND",        "#4CAF50"),
    ("M4_OR_km2",       "M4 OR",         "#9C27B0"),
    ("M5_WEIGHTED_km2", "M5 Weighted",   "#F44336"),
]

MONAT_LABELS = ["Jan","Feb","Mär","Apr","Mai","Jun",
                "Jul","Aug","Sep","Okt","Nov","Dez"]


# ══════════════════════════════════════════════════════════════════════════════
#  HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def lade_pegel(cfg):
    """Lädt Pegeldaten und gibt Monatsmittel zurück."""
    try:
        pegel = pd.read_csv(
            cfg["pegel_csv"],
            sep=cfg["sep"],
            skiprows=cfg["skiprows"],
            decimal=cfg["decimal"],
            encoding=cfg["encoding"],
            header=0,
        )
        pegel.columns = pegel.columns.str.strip()
        pegel = pegel.iloc[:, [0, 1]]
        pegel.columns = ["Datum", "Mittelwert"]
        pegel["Datum"]      = pd.to_datetime(pegel["Datum"], dayfirst=True, errors="coerce")
        pegel["Mittelwert"] = pd.to_numeric(pegel["Mittelwert"], errors="coerce")
        pegel = pegel.dropna()
    except Exception as e:
        print(f"    FEHLER beim Laden der Pegeldaten: {e}")
        return None

    pegel["monat"] = pegel["Datum"].dt.to_period("M").dt.to_timestamp()
    monthly = pegel.groupby("monat")["Mittelwert"].mean().reset_index()
    monthly.columns = ["datum", "pegel_m"]
    return monthly


def normiere(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn) if mx > mn else x * 0


def bewerte(r):
    r = abs(r)
    if r > 0.85: return "sehr gut"
    if r > 0.70: return "gut"
    if r > 0.50: return "maessig"
    return "schwach"


def erstelle_plot(df, see_name, methode, methode_label, methode_farbe,
                  pearson_r, pearson_p, spearman_r, spearman_p,
                  pegel_label, out_path, delta_col=None):
    """Erstellt das 4-Panel-Validierungsdiagramm und speichert es."""

    df = df.copy()
    df["pegel_norm"] = normiere(df["pegel_m"])
    df["sat_norm"]   = normiere(df[methode])
    df["monat_nr"]   = df["datum"].dt.month

    fig = plt.figure(figsize=(14, 13))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.35)

    # ── Plot 1: Normierte Zeitreihen ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df["datum"], df["pegel_norm"], color="#1a3a5c", linewidth=1.5,
             label=f"{pegel_label} (normiert)", zorder=3)
    ax1.plot(df["datum"], df["sat_norm"], color=methode_farbe, linewidth=1.5,
             linestyle="--", label=f"Satellit {methode_label} (normiert)", zorder=2)
    ax1.fill_between(df["datum"], df["pegel_norm"], df["sat_norm"],
                     alpha=0.15, color="grey", label="Abweichung")
    ax1.set_title("Pegelstand vs. Satelliten-Wasserflaeche (normiert 0–1)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Normierter Wert [0–1]")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(df["datum"].min(), df["datum"].max())

    # ── Plot 2: Absolute Zeitreihen (zwei Y-Achsen) ───────────────────────────
    ax2  = fig.add_subplot(gs[1, :])
    ax2b = ax2.twinx()
    l1, = ax2.plot(df["datum"], df["pegel_m"], color="#1a3a5c",
                   linewidth=1.5, label="Pegelstand [m ü. NN]")
    l2, = ax2b.plot(df["datum"], df[methode], color=methode_farbe,
                    linewidth=1.5, linestyle="--",
                    label=f"Wasserflaeche {methode_label} [km²]")
    ax2.set_ylabel("Pegelstand [m ü. NN]", color="#1a3a5c")
    ax2b.set_ylabel("Wasserflaeche [km²]",  color=methode_farbe)
    ax2b.set_ylim(bottom=0)   # Satellit-Achse bei 0 → Ausreißer proportional vergleichbar
    ax2.tick_params(axis="y", colors="#1a3a5c")
    ax2b.tick_params(axis="y", colors=methode_farbe)
    ax2.set_title("Absolute Zeitreihen (zwei Y-Achsen)", fontsize=12, fontweight="bold")
    ax2.legend(handles=[l1, l2], fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(df["datum"].min(), df["datum"].max())

    # ── Plot 3: Scatter + Regression ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.scatter(df["pegel_m"], df[methode], alpha=0.5, s=20,
                color=methode_farbe, zorder=3)
    m_koef, b_koef = np.polyfit(df["pegel_m"], df[methode], 1)
    x_line = np.linspace(df["pegel_m"].min(), df["pegel_m"].max(), 100)
    ax3.plot(x_line, m_koef * x_line + b_koef, color="#e74c3c", linewidth=2,
             label=f"Regression (r={pearson_r:.2f})")
    ax3.set_xlabel("Pegelstand [m ü. NN]")
    ax3.set_ylabel("Wasserflaeche [km²]")
    ax3.set_title("Scatter: Pegel vs. Flaeche", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Saisonales Muster ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    seas_pegel = df.groupby("monat_nr")["pegel_norm"].mean()
    seas_sat   = df.groupby("monat_nr")["sat_norm"].mean()
    monate_idx = seas_pegel.index - 1   # 1-basiert → 0-basiert
    ax4.bar(monate_idx - 0.2, seas_pegel.values, 0.35,
            label="Pegel (normiert)", color="#1a3a5c", alpha=0.8)
    ax4.bar(monate_idx + 0.2, seas_sat.values,   0.35,
            label="Satellit (normiert)", color=methode_farbe, alpha=0.8)
    ax4.set_xticks(range(12))
    ax4.set_xticklabels(MONAT_LABELS, fontsize=8)
    ax4.set_title("Saisonales Muster (Monatsmittel)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Normierter Mittelwert")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    # ── Plot 5: Relative Flächenabweichung ΔA/A [%] ───────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    print(f"      [DEBUG ax5] delta_col={delta_col!r}  "
          f"not_none={delta_col is not None}  "
          f"in_df={delta_col in df.columns if delta_col else '-'}  "
          f"gt0={df[methode].gt(0).any()}")
    if delta_col is not None and delta_col in df.columns and df[methode].gt(0).any():
        abw_pct      = (df[delta_col] / df[methode] * 100).clip(upper=50)
        mittlere_abw = abw_pct.mean()

        ax5.fill_between(df["datum"], 0, abw_pct,
                         alpha=0.25, color=methode_farbe)
        ax5.plot(df["datum"], abw_pct,
                 color=methode_farbe, linewidth=1.3, label="ΔA/A [%]")
        ax5.axhline(mittlere_abw, color="#e74c3c", linewidth=1.3,
                    linestyle="--", label=f"Mittel: {mittlere_abw:.1f} %")
        ax5.set_ylabel("ΔA / A [%]")
        ax5.legend(fontsize=9)
    else:
        ax5.text(0.5, 0.5,
                 f"Keine Δ-Spalte verfügbar (erwartet: '{delta_col}')",
                 ha="center", va="center", transform=ax5.transAxes,
                 fontsize=10, color="#aaa")

    ax5.set_title(f"Relative Flächenabweichung ΔA/A durch Randpixel [%]",
                  fontsize=11, fontweight="bold")
    ax5.set_xlim(df["datum"].min(), df["datum"].max())
    ax5.grid(True, alpha=0.3)

    # Kennzahlen als Fußzeile
    fig.text(0.5, 0.005,
        f"Pearson r = {pearson_r:.3f} (p={pearson_p:.4f})  |  "
        f"Spearman r = {spearman_r:.3f} (p={spearman_p:.4f})  |  "
        f"Bewertung: {bewerte(pearson_r)}  |  n = {len(df)} Monate",
        ha="center", fontsize=9, style="italic", color="#444444")

    plt.suptitle(
        f"Validierung {see_name}: {methode_label} vs. Pegelstand",
        fontsize=14, fontweight="bold", y=1.01)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)   # kein plt.show() – sonst pausiert die Schleife


# ══════════════════════════════════════════════════════════════════════════════
#  HAUPTSCHLEIFE
# ══════════════════════════════════════════════════════════════════════════════

zusammenfassung = []   # sammelt alle Korrelationsergebnisse

for see_name, cfg in SEEN_CONFIG.items():
    print(f"\n{'═'*60}")
    print(f"  {see_name}")
    print(f"{'═'*60}")

    see_dir = os.path.join(BASE_ROOT, see_name)
    sat_csv = os.path.join(see_dir, f"{see_name}_methodenvergleich.csv")

    # Pegeldaten laden (einmal pro See)
    pegel = lade_pegel(cfg)
    if pegel is None:
        print("  [SKIP] Pegeldaten nicht ladbar.")
        continue
    print(f"  Pegeldaten: {len(pegel)} Monatsmittel "
          f"({pegel['datum'].min().date()} – {pegel['datum'].max().date()})")

    # Satellitendaten laden (einmal pro See)
    if not os.path.exists(sat_csv):
        print(f"  [SKIP] Satelliten-CSV nicht gefunden: {sat_csv}")
        continue
    sat = pd.read_csv(sat_csv, sep=";", decimal=",")
    sat["datum"] = pd.to_datetime(sat["datum"]).dt.to_period("M").dt.to_timestamp()
    print(f"  Satellitendaten: {len(sat)} Monate")

    # Ausgabe-Ordner
    out_dir = os.path.join(see_dir, "Validierung")
    os.makedirs(out_dir, exist_ok=True)

    # ── Innere Schleife: alle 5 Methoden ─────────────────────────────────────
    for methode, methode_label, methode_farbe in METHODEN:

        if methode not in sat.columns:
            print(f"  [SKIP] Spalte '{methode}' nicht in CSV.")
            continue

        # Delta-Spalte (Randpixel-Unsicherheit) mitladen wenn vorhanden
        m_kurz    = methode.split("_")[0]       # "M1", "M2", …
        delta_col = f"{m_kurz}_delta_km2"
        merge_cols = ["datum", methode] + ([delta_col] if delta_col in sat.columns else [])

        # Zusammenführen
        df = pd.merge(
            pegel,
            sat[merge_cols],
            on="datum", how="inner"
        ).dropna(subset=["pegel_m", methode]).sort_values("datum").reset_index(drop=True)

        if len(df) < 10:
            print(f"  [SKIP] {methode_label}: nur {len(df)} gemeinsame Monate.")
            continue

        # Korrelation
        pearson_r,  pearson_p  = stats.pearsonr(df["pegel_m"], df[methode])
        spearman_r, spearman_p = stats.spearmanr(df["pegel_m"], df[methode])

        print(f"  {methode_label:<18}  "
              f"Pearson r={pearson_r:+.3f}  "
              f"Spearman r={spearman_r:+.3f}  "
              f"n={len(df)}  [{bewerte(pearson_r)}]")

        # Ergebnis sammeln
        zusammenfassung.append({
            "See":        see_name,
            "Methode":    methode_label,
            "Pearson_r":  round(pearson_r,  4),
            "Pearson_p":  round(pearson_p,  4),
            "Spearman_r": round(spearman_r, 4),
            "Spearman_p": round(spearman_p, 4),
            "n":          len(df),
            "Bewertung":  bewerte(pearson_r),
        })

        # Diagramm erstellen
        out_png = os.path.join(out_dir, f"Validierung_{see_name}_{methode_label.replace(' ','_')}.png")
        erstelle_plot(
            df, see_name, methode, methode_label, methode_farbe,
            pearson_r, pearson_p, spearman_r, spearman_p,
            cfg["pegel_label"], out_png,
            delta_col=delta_col if delta_col in sat.columns else None
        )
        print(f"    → {os.path.basename(out_png)}")

# ══════════════════════════════════════════════════════════════════════════════
#  ZUSAMMENFASSUNG
# ══════════════════════════════════════════════════════════════════════════════

if zusammenfassung:
    df_zus = pd.DataFrame(zusammenfassung)

    # Pivot-Tabelle: Seen als Zeilen, Methoden als Spalten
    pivot = df_zus.pivot(index="See", columns="Methode", values="Pearson_r")

    print(f"\n{'═'*60}")
    print("  Pearson r – Übersicht (alle Seen × Methoden)")
    print(f"{'═'*60}")
    print(pivot.to_string())

    # CSV speichern
    zus_csv = os.path.join(BASE_ROOT, "Validierung_Zusammenfassung.csv")
    df_zus.to_csv(zus_csv, sep=";", decimal=",", index=False)
    print(f"\n  Zusammenfassung gespeichert: {zus_csv}")

print("\nFertig!")

