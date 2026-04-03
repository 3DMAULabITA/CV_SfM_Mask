"""
modules/report.py
─────────────────────────────────────────────────────────────
Modulo Report – Generazione metriche, grafici e dashboard

Produce:
  • processing_report.json / .csv  – tabella completa per immagine
  • chart_iqa.png          – qualità immagini (nitidezza, esposizione)
  • chart_segmentation.png – distribuzione classi rilevate e tempi
  • chart_masks.png        – distribuzione area mascherata
  • dashboard.png          – pannello riepilogativo 4-in-1
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend non-interattivo (sicuro su Windows)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Palette colori coerente con l'articolo
PALETTE_MAIN  = "#2C3E50"
PALETTE_ACC1  = "#E74C3C"   # rosso – oggetti dinamici
PALETTE_ACC2  = "#2ECC71"   # verde – qualità OK
PALETTE_ACC3  = "#3498DB"   # blu – info
PALETTE_WARN  = "#F39C12"   # arancione – warning IQA
PALETTE_GRID  = "#ECF0F1"

# Nome visualizzabile per ciascuna classe COCO
CLASS_LABELS = {
    0: "Persona", 1: "Bicicletta", 2: "Auto", 3: "Moto",
    5: "Bus", 7: "Camion", 14: "Uccello", 15: "Gatto",
    16: "Cane", 17: "Cavallo",
}


class ReportGenerator:
    """
    Genera report testuali e visualizzazioni delle metriche di pipeline.

    Parametri
    ----------
    report_dir : Path
        Cartella di output per tutti i report.
    chart_dpi : int
        Risoluzione DPI dei grafici PNG.
    """

    def __init__(self, report_dir: Path, chart_dpi: int = 150):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = chart_dpi

        # Stile globale matplotlib
        plt.rcParams.update({
            "font.family":       "DejaVu Sans",
            "axes.facecolor":    "white",
            "axes.edgecolor":    PALETTE_MAIN,
            "axes.grid":         True,
            "grid.color":        PALETTE_GRID,
            "grid.linewidth":    0.8,
            "figure.facecolor":  "white",
            "axes.spines.top":   False,
            "axes.spines.right": False,
        })

    # ── Salvataggio dati ─────────────────────────────────────────────

    def save_json(self, data: list[dict], filename: str = "processing_report.json"):
        out = self.report_dir / filename
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return out

    def save_csv(self, data: list[dict], filename: str = "processing_report.csv"):
        out = self.report_dir / filename
        if not data:
            return out
        keys = list(data[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
        return out

    # ── Grafico 1: IQA ──────────────────────────────────────────────

    def chart_iqa(self, iqa_results: list[dict]) -> Optional[Path]:
        """
        Pannello 2x2 con metriche di qualità immagini:
          - Distribuzione nitidezza (istogramma)
          - Scatter nitidezza vs punteggio composito
          - Esposizione per immagine (stacked bar)
          - Torta pass/fail IQA
        """
        if not iqa_results:
            return None

        filenames    = [r["filename"] for r in iqa_results]
        sharpness    = [r["sharpness_score"]     for r in iqa_results]
        composite    = [r["composite_score"]     for r in iqa_results]
        under        = [r["underexposure_pct"]   for r in iqa_results]
        over         = [r["overexposure_pct"]    for r in iqa_results]
        passed       = [r["passed"]              for r in iqa_results]

        n_pass = sum(passed)
        n_fail = len(passed) - n_pass

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Modulo 1 – Valutazione Qualità Immagini (IQA)",
                     fontsize=14, fontweight="bold", color=PALETTE_MAIN, y=1.01)

        # 1a. Istogramma nitidezza
        ax = axes[0, 0]
        ax.hist(sharpness, bins=20, color=PALETTE_ACC3, edgecolor="white", alpha=0.85)
        ax.axvline(80, color=PALETTE_ACC1, linestyle="--", lw=1.5, label="Soglia (80)")
        ax.set_title("Distribuzione Nitidezza (Varianza Laplaciano)", fontsize=10)
        ax.set_xlabel("Punteggio Nitidezza")
        ax.set_ylabel("N. Immagini")
        ax.legend(fontsize=8)

        # 1b. Scatter nitidezza vs composito
        ax = axes[0, 1]
        colors = [PALETTE_ACC2 if p else PALETTE_ACC1 for p in passed]
        ax.scatter(sharpness, composite, c=colors, s=60, alpha=0.75, edgecolors="white", lw=0.5)
        ax.set_title("Nitidezza vs Punteggio Composito", fontsize=10)
        ax.set_xlabel("Nitidezza")
        ax.set_ylabel("Punteggio Composito (0–1)")
        # Legenda manuale
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE_ACC2, markersize=9, label="Superata"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE_ACC1, markersize=9, label="Non superata"),
        ]
        ax.legend(handles=handles, fontsize=8)

        # 1c. Esposizione per immagine (solo prime 40 per leggibilità)
        ax = axes[1, 0]
        max_show = min(40, len(filenames))
        x_idx = np.arange(max_show)
        ax.bar(x_idx, under[:max_show], label="Sottoesposizione %", color=PALETTE_MAIN, alpha=0.8)
        ax.bar(x_idx, over[:max_show], bottom=under[:max_show],
               label="Sovraesposizione %", color=PALETTE_WARN, alpha=0.8)
        ax.axhline(30, color=PALETTE_ACC1, linestyle="--", lw=1, label="Soglia 30%")
        ax.set_title("Esposizione per Immagine (prime 40)", fontsize=10)
        ax.set_xlabel("Indice Immagine")
        ax.set_ylabel("% Pixel Fuori Range")
        ax.legend(fontsize=8)

        # 1d. Torta Pass/Fail
        ax = axes[1, 1]
        if n_pass + n_fail > 0:
            wedges, texts, autotexts = ax.pie(
                [n_pass, n_fail] if n_fail > 0 else [n_pass],
                labels=["Superata", "Non superata"] if n_fail > 0 else ["Superata"],
                colors=[PALETTE_ACC2, PALETTE_ACC1] if n_fail > 0 else [PALETTE_ACC2],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            for at in autotexts:
                at.set_fontsize(10)
        ax.set_title(f"Esito IQA ({len(passed)} immagini totali)", fontsize=10)

        plt.tight_layout()
        out = self.report_dir / "chart_iqa.png"
        fig.savefig(str(out), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    # ── Grafico 2: Segmentazione ─────────────────────────────────────

    def chart_segmentation(self, seg_data: list[dict]) -> Optional[Path]:
        """
        Pannello 2x2 con metriche di segmentazione:
          - Conteggio oggetti per immagine
          - Distribuzione classi rilevate (bar chart)
          - Distribuzione tempi di inferenza
          - Confidenza media per classe
        """
        if not seg_data:
            return None

        n_objects       = [r["n_objects"]       for r in seg_data]
        processing_ms   = [r["processing_ms"]   for r in seg_data]

        # Costruisce distribuzione classi
        class_counts: dict[str, int] = {}
        class_confidences: dict[str, list] = {}
        for r in seg_data:
            for det in r.get("detections", []):
                cn = CLASS_LABELS.get(det["class_id"], det.get("class_name", "?"))
                class_counts[cn]      = class_counts.get(cn, 0) + 1
                class_confidences.setdefault(cn, []).append(det["confidence"])

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Modulo 2 – Rilevamento e Segmentazione Oggetti",
                     fontsize=14, fontweight="bold", color=PALETTE_MAIN, y=1.01)

        # 2a. Oggetti per immagine
        ax = axes[0, 0]
        ax.bar(range(len(n_objects)), n_objects, color=PALETTE_ACC1, alpha=0.8)
        ax.axhline(np.mean(n_objects), color=PALETTE_MAIN, linestyle="--",
                   lw=1.5, label=f"Media: {np.mean(n_objects):.1f}")
        ax.set_title("Numero Oggetti Rilevati per Immagine", fontsize=10)
        ax.set_xlabel("Indice Immagine")
        ax.set_ylabel("N. Oggetti")
        ax.legend(fontsize=8)

        # 2b. Distribuzione classi
        ax = axes[0, 1]
        if class_counts:
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            cls_names, cls_vals = zip(*sorted_classes)
            bars = ax.barh(cls_names, cls_vals, color=PALETTE_ACC3, alpha=0.85)
            for bar, val in zip(bars, cls_vals):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        str(val), va="center", fontsize=8)
        ax.set_title("Distribuzione Classi Rilevate", fontsize=10)
        ax.set_xlabel("Conteggio Totale")

        # 2c. Tempi di inferenza
        ax = axes[1, 0]
        ax.hist(processing_ms, bins=20, color=PALETTE_ACC2, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(processing_ms), color=PALETTE_ACC1, linestyle="--",
                   lw=1.5, label=f"Media: {np.mean(processing_ms):.0f} ms")
        ax.set_title("Distribuzione Tempi di Elaborazione", fontsize=10)
        ax.set_xlabel("Tempo (ms)")
        ax.set_ylabel("N. Immagini")
        ax.legend(fontsize=8)

        # 2d. Confidenza media per classe
        ax = axes[1, 1]
        if class_confidences:
            cls_names_c  = list(class_confidences.keys())
            cls_means    = [np.mean(v) for v in class_confidences.values()]
            cls_stds     = [np.std(v)  for v in class_confidences.values()]
            y_pos = np.arange(len(cls_names_c))
            ax.barh(y_pos, cls_means, xerr=cls_stds,
                    color=PALETTE_WARN, alpha=0.85, capsize=4)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cls_names_c)
            ax.set_xlim(0, 1.05)
            ax.axvline(0.45, color=PALETTE_ACC1, linestyle="--",
                       lw=1, label="Soglia confidenza")
            ax.legend(fontsize=8)
        ax.set_title("Confidenza Media per Classe (± std)", fontsize=10)
        ax.set_xlabel("Confidenza")

        plt.tight_layout()
        out = self.report_dir / "chart_segmentation.png"
        fig.savefig(str(out), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    # ── Grafico 3: Maschere ──────────────────────────────────────────

    def chart_masks(self, export_data: list[dict]) -> Optional[Path]:
        """
        Pannello con metriche delle maschere generate:
          - Distribuzione % area mascherata
          - Area mascherata per immagine (sorted)
          - Istogramma oggetti per immagine
          - Cumulativa area mascherata
        """
        if not export_data:
            return None

        masked_pcts = [r["masked_pct"] for r in export_data]
        n_objects   = [r["n_objects"]  for r in export_data]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Modulo 3 – Analisi Maschere Generate",
                     fontsize=14, fontweight="bold", color=PALETTE_MAIN, y=1.01)

        # 3a. Istogramma % area mascherata
        ax = axes[0, 0]
        ax.hist(masked_pcts, bins=20, color=PALETTE_ACC1, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(masked_pcts), color=PALETTE_MAIN, linestyle="--",
                   lw=1.5, label=f"Media: {np.mean(masked_pcts):.1f}%")
        ax.set_title("Distribuzione Area Mascherata (%)", fontsize=10)
        ax.set_xlabel("% Area Mascherata")
        ax.set_ylabel("N. Immagini")
        ax.legend(fontsize=8)

        # 3b. Area per immagine (ordinata)
        ax = axes[0, 1]
        sorted_pcts = sorted(masked_pcts)
        ax.fill_between(range(len(sorted_pcts)), sorted_pcts,
                        alpha=0.4, color=PALETTE_ACC1)
        ax.plot(sorted_pcts, color=PALETTE_ACC1, lw=1.5)
        ax.axhline(np.mean(masked_pcts), color=PALETTE_MAIN, linestyle="--",
                   lw=1, label=f"Media: {np.mean(masked_pcts):.1f}%")
        ax.set_title("Area Mascherata per Immagine (ordinata)", fontsize=10)
        ax.set_xlabel("Immagine (ordinata)")
        ax.set_ylabel("% Area Mascherata")
        ax.legend(fontsize=8)

        # 3c. Box plot oggetti per immagine
        ax = axes[1, 0]
        bp = ax.boxplot(n_objects, vert=True, patch_artist=True,
                        medianprops={"color": PALETTE_ACC1, "lw": 2})
        bp["boxes"][0].set_facecolor(PALETTE_ACC3)
        bp["boxes"][0].set_alpha(0.6)
        ax.set_title("Distribuzione Oggetti per Immagine", fontsize=10)
        ax.set_ylabel("N. Oggetti")
        ax.set_xticklabels(["Dataset"])
        stats_txt = (f"Min: {min(n_objects)}\nMax: {max(n_objects)}\n"
                     f"Media: {np.mean(n_objects):.1f}\nMediana: {np.median(n_objects):.1f}")
        ax.text(1.35, np.median(n_objects), stats_txt, fontsize=8,
                va="center", color=PALETTE_MAIN,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": PALETTE_GRID, "alpha": 0.8})

        # 3d. Cumulativa area mascherata
        ax = axes[1, 1]
        sorted_arr = np.sort(masked_pcts)
        cdf = np.arange(1, len(sorted_arr)+1) / len(sorted_arr)
        ax.plot(sorted_arr, cdf * 100, color=PALETTE_ACC3, lw=2)
        ax.fill_between(sorted_arr, cdf * 100, alpha=0.2, color=PALETTE_ACC3)
        ax.set_title("Distribuzione Cumulativa Area Mascherata", fontsize=10)
        ax.set_xlabel("% Area Mascherata")
        ax.set_ylabel("% Immagini (CDF)")
        ax.set_ylim(0, 105)

        plt.tight_layout()
        out = self.report_dir / "chart_masks.png"
        fig.savefig(str(out), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    # ── Dashboard riepilogativa ──────────────────────────────────────

    def chart_dashboard(
        self,
        iqa_results:  list[dict],
        seg_data:     list[dict],
        export_data:  list[dict],
        total_time_s: float,
    ) -> Optional[Path]:
        """
        Pannello riepilogativo con KPI principali.
        """
        if not export_data:
            return None

        n_imgs       = len(export_data)
        n_pass_iqa   = sum(1 for r in iqa_results if r["passed"]) if iqa_results else n_imgs
        n_objects    = sum(r["n_objects"] for r in export_data)
        avg_masked   = np.mean([r["masked_pct"] for r in export_data])
        avg_time_ms  = np.mean([r.get("processing_ms", 0) for r in seg_data]) if seg_data else 0
        masked_pcts  = [r["masked_pct"] for r in export_data]

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor("#F8F9FA")

        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

        # ── KPI tiles (riga 0) ──────────────────────────────────────
        kpis = [
            ("Immagini\nElaborate",  f"{n_imgs}",              PALETTE_ACC3),
            ("IQA Superata",         f"{n_pass_iqa}/{n_imgs}", PALETTE_ACC2),
            ("Oggetti\nRilevati",    f"{n_objects}",           PALETTE_ACC1),
            ("Area Masch. Media",    f"{avg_masked:.1f}%",     PALETTE_WARN),
        ]
        for col, (label, value, color) in enumerate(kpis):
            ax_kpi = fig.add_subplot(gs[0, col])
            ax_kpi.set_facecolor(color)
            ax_kpi.set_xticks([]); ax_kpi.set_yticks([])
            for spine in ax_kpi.spines.values():
                spine.set_visible(False)
            ax_kpi.text(0.5, 0.65, value, ha="center", va="center",
                        transform=ax_kpi.transAxes,
                        fontsize=22, fontweight="bold", color="white")
            ax_kpi.text(0.5, 0.25, label, ha="center", va="center",
                        transform=ax_kpi.transAxes,
                        fontsize=10, color="white", alpha=0.9)

        # ── Grafico area mascherata per immagine (riga 1, span 3) ───
        ax_main = fig.add_subplot(gs[1, :3])
        x = np.arange(n_imgs)
        colors_bar = [PALETTE_ACC1 if p > avg_masked else PALETTE_ACC3 for p in masked_pcts]
        ax_main.bar(x, masked_pcts, color=colors_bar, alpha=0.8)
        ax_main.axhline(avg_masked, color=PALETTE_MAIN, lw=1.5, linestyle="--",
                        label=f"Media {avg_masked:.1f}%")
        ax_main.set_title("Area Mascherata per Immagine", fontsize=11, fontweight="bold")
        ax_main.set_xlabel("Indice Immagine")
        ax_main.set_ylabel("% Area Mascherata")
        ax_main.legend(fontsize=9)

        # ── Torta IQA (riga 1, col 3) ───────────────────────────────
        ax_pie = fig.add_subplot(gs[1, 3])
        n_fail_iqa = n_imgs - n_pass_iqa
        if n_fail_iqa > 0:
            ax_pie.pie([n_pass_iqa, n_fail_iqa],
                       labels=["Pass", "Fail"], autopct="%1.0f%%",
                       colors=[PALETTE_ACC2, PALETTE_ACC1],
                       wedgeprops={"edgecolor": "white", "lw": 2},
                       startangle=90)
        else:
            ax_pie.pie([1], labels=["Tutte OK"],
                       colors=[PALETTE_ACC2],
                       wedgeprops={"edgecolor": "white", "lw": 2})
        ax_pie.set_title("Esito IQA", fontsize=11, fontweight="bold")

        # ── Istogramma oggetti per immagine (riga 2, col 0-1) ───────
        ax_hist = fig.add_subplot(gs[2, :2])
        n_obj_list = [r["n_objects"] for r in export_data]
        ax_hist.hist(n_obj_list, bins=max(1, max(n_obj_list)+1),
                     color=PALETTE_ACC1, edgecolor="white", alpha=0.85, align="left")
        ax_hist.set_title("Distribuzione N. Oggetti per Immagine", fontsize=11, fontweight="bold")
        ax_hist.set_xlabel("N. Oggetti Rilevati")
        ax_hist.set_ylabel("N. Immagini")

        # ── Riepilogo testuale (riga 2, col 2-3) ────────────────────
        ax_txt = fig.add_subplot(gs[2, 2:])
        ax_txt.set_facecolor("#EBF5FB")
        ax_txt.set_xticks([]); ax_txt.set_yticks([])
        for spine in ax_txt.spines.values():
            spine.set_edgecolor(PALETTE_ACC3)

        mins = int(total_time_s // 60)
        secs = int(total_time_s % 60)

        summary = (
            f"RIEPILOGO ELABORAZIONE\n"
            f"{'─'*35}\n"
            f"Immagini totali:      {n_imgs:>6}\n"
            f"IQA superata:         {n_pass_iqa:>6}\n"
            f"IQA non superata:     {n_fail_iqa:>6}\n"
            f"Oggetti totali:       {n_objects:>6}\n"
            f"Area media mascherata:{avg_masked:>5.1f}%\n"
            f"Tempo tot. elab.:     {mins:>4}m {secs:02d}s\n"
            f"Tempo medio/img:      {avg_time_ms:>5.0f} ms\n"
        )
        ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                    fontsize=10, va="top", fontfamily="monospace",
                    color=PALETTE_MAIN)

        fig.suptitle("SfM Masker – Dashboard Riepilogativo",
                     fontsize=16, fontweight="bold", color=PALETTE_MAIN, y=1.01)

        out = self.report_dir / "dashboard.png"
        fig.savefig(str(out), dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return out

    # ── Metodo pubblico principale ───────────────────────────────────

    def generate_all(
        self,
        iqa_results:  list[dict],
        seg_data:     list[dict],
        export_data:  list[dict],
        total_time_s: float,
        fmt:          str = "both",
    ) -> dict[str, Path | None]:
        """
        Genera tutti i report e i grafici.

        Restituisce un dizionario con i percorsi degli artefatti generati.
        """
        # Costruisce record unificato per JSON/CSV
        unified = []
        for i, ed in enumerate(export_data):
            rec = {**ed}
            if i < len(iqa_results):
                iq = iqa_results[i]
                rec.update({
                    "sharpness_score":   iq.get("sharpness_score"),
                    "composite_score":   iq.get("composite_score"),
                    "iqa_passed":        iq.get("passed"),
                })
            if i < len(seg_data):
                sd = seg_data[i]
                rec["processing_ms"] = sd.get("processing_ms", 0)
            unified.append(rec)

        outputs = {}

        if fmt in ("json", "both"):
            outputs["json"] = self.save_json(unified)
        if fmt in ("csv", "both"):
            outputs["csv"] = self.save_csv(unified)

        outputs["chart_iqa"]          = self.chart_iqa(iqa_results)
        outputs["chart_segmentation"] = self.chart_segmentation(seg_data)
        outputs["chart_masks"]        = self.chart_masks(export_data)
        outputs["dashboard"]          = self.chart_dashboard(
            iqa_results, seg_data, export_data, total_time_s
        )

        return outputs
