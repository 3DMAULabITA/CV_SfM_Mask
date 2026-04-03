"""
iqa_analysis.py
═══════════════════════════════════════════════════════════════════════════════
Standalone script for Image Quality Assessment (IQA)
-------------------------------------------------------------------------------
Automatically reads all images in the DATASET folder and
computes three complementary metrics for each image:

  1. SHARPNESS   – variance of the Laplacian operator on the luminance channel
                   (Pech-Pacheco et al., ICPR 2000)
  2. EXPOSURE    – proportion of underexposed (< 30) and overexposed (> 225)
                   pixels from the normalised luminance histogram
  3. RESOLUTION  – comparison of image dimensions against a configurable
                   minimum megapixel threshold

Output saved in the IQA_REPORT/ folder:
  • iqa_report.csv            – full per-image metrics table
  • iqa_report.json           – same content in JSON format
  • chart_sharpness.png       – sharpness score distribution + threshold
  • chart_exposure.png        – under/overexposure per image
  • chart_resolution.png      – megapixel distribution across the dataset
  • chart_composite.png       – composite score and pass/fail map
  • dashboard_iqa.png         – 4-in-1 summary dashboard

Usage
-----
  python iqa_analysis.py                        # default parameters
  python iqa_analysis.py --dataset ./MY_PHOTOS  # custom dataset folder
  python iqa_analysis.py --sharpness 120        # stricter sharpness threshold
  python iqa_analysis.py --min-mp 2.0           # require at least 2 MP
  python iqa_analysis.py --output ./RESULTS     # custom output folder

Dependencies
------------
  pip install opencv-python numpy matplotlib seaborn pandas rich tqdm
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich import box
from tqdm import tqdm

# ── Global chart configuration ──────────────────────────────────────────────
C_OK     = "#2ECC71"   # green  – pass
C_FAIL   = "#E74C3C"   # red    – fail
C_WARN   = "#F39C12"   # orange – warning
C_BLUE   = "#3498DB"   # blue   – info
C_DARK   = "#2C3E50"   # dark blue
C_GRID   = "#ECF0F1"   # light grey

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    "white",
    "axes.edgecolor":    C_DARK,
    "axes.grid":         True,
    "grid.color":        C_GRID,
    "grid.linewidth":    0.8,
    "figure.facecolor":  "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})

console = Console()

# ── Supported image extensions ──────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ══════════════════════════════════════════════════════════════════════════════
# Result data structure
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IQARecord:
    """Complete IQA evaluation result for a single image."""

    # Identification
    filename:            str
    filepath:            str

    # Dimensions
    width_px:            int
    height_px:           int
    megapixels:          float

    # Metric 1 – Sharpness
    sharpness_score:     float   # Laplacian variance (↑ better)
    sharpness_pass:      bool

    # Metric 2 – Exposure
    underexposure_pct:   float   # % pixels < 30 on luminance channel
    overexposure_pct:    float   # % pixels > 225 on luminance channel
    exposure_pass:       bool

    # Metric 3 – Resolution
    resolution_pass:     bool

    # Composite score
    composite_score:     float   # 0–1 (↑ better)

    # Overall result
    overall_pass:        bool
    failure_reasons:     list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["failure_reasons"] = "; ".join(self.failure_reasons) if self.failure_reasons else ""
        return d


# ══════════════════════════════════════════════════════════════════════════════
# IQA analysis engine
# ══════════════════════════════════════════════════════════════════════════════

class IQAAnalyzer:
    """
    Computes the three IQA metrics for each image.

    Parameters
    ----------
    sharpness_threshold : float
        Laplacian variance threshold. Below this value → blurry image.
        Typical range: 60–150. Default: 80.
    underexposure_max_pct : float
        Maximum allowed fraction of underexposed pixels (value < 30).
        Default: 30 %.
    overexposure_max_pct : float
        Maximum allowed fraction of overexposed pixels (value > 225).
        Default: 30 %.
    min_resolution_mp : float
        Minimum required resolution in megapixels. Default: 1.0 MP.
    """

    def __init__(
        self,
        sharpness_threshold:    float = 80.0,
        underexposure_max_pct:  float = 30.0,
        overexposure_max_pct:   float = 30.0,
        min_resolution_mp:      float = 1.0,
    ):
        self.sharpness_threshold   = sharpness_threshold
        self.underexposure_max_pct = underexposure_max_pct
        self.overexposure_max_pct  = overexposure_max_pct
        self.min_resolution_mp     = min_resolution_mp

    # ── Metric 1: Sharpness ──────────────────────────────────────────────────

    def _sharpness(self, gray: np.ndarray) -> float:
        """
        Estimates sharpness as the variance of the Laplacian on the luminance channel.

        The Laplacian detects high-frequency intensity variations:
        sharp images yield high variance, blurry images yield low variance.
        Reference: Pech-Pacheco et al., ICPR 2000.
        """
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    # ── Metric 2: Exposure ───────────────────────────────────────────────────

    def _exposure(self, gray: np.ndarray) -> tuple[float, float]:
        """
        Computes the percentages of underexposed and overexposed pixels.

        Underexposed : pixels with value < 30  (deep shadow region)
        Overexposed  : pixels with value > 225 (blown highlight region)

        Returns (underexposure_pct, overexposure_pct) as percentages 0–100.
        """
        total = gray.size
        under = float(np.sum(gray < 30)  / total * 100)
        over  = float(np.sum(gray > 225) / total * 100)
        return under, over

    # ── Metric 3: Resolution ─────────────────────────────────────────────────

    def _resolution(self, h: int, w: int) -> float:
        """Computes resolution in megapixels."""
        return (h * w) / 1_000_000

    # ── Composite score ──────────────────────────────────────────────────────

    def _composite(
        self,
        sharpness: float,
        under: float,
        over: float,
        mp: float,
    ) -> float:
        """
        Combines the three metrics into a normalised composite index 0–1.

        Weights:
          • Sharpness   50 % – highest impact on feature matching quality
          • Exposure    35 % – direct influence on feature discriminability
          • Resolution  15 % – binary criterion, lower weight

        Sharpness is normalised with a sigmoid centred on the threshold,
        so that sub-threshold values are penalised gradually rather than hard-cut.
        """
        # Sharpness: sigmoid centred on threshold (scale 20 for smooth transition)
        sharp_norm = 1.0 / (1.0 + np.exp(-(sharpness - self.sharpness_threshold) / 20.0))

        # Exposure: penalty proportional to out-of-range pixel fraction
        exp_score = max(0.0, 1.0 - (under + over) / 100.0 * 1.5)

        # Resolution: linear up to twice the minimum threshold
        res_score = min(1.0, mp / max(self.min_resolution_mp * 2, 0.001))

        score = 0.50 * sharp_norm + 0.35 * exp_score + 0.15 * res_score
        return round(float(score), 4)

    # ── Single image analysis ────────────────────────────────────────────────

    def analyze(self, image_path: Path) -> IQARecord | None:
        """
        Runs the full IQA analysis on a single image.

        Returns IQARecord or None if the file cannot be read.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w  = img.shape[:2]
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute metrics
        sharpness       = self._sharpness(gray)
        under, over     = self._exposure(gray)
        mp              = self._resolution(h, w)
        composite       = self._composite(sharpness, under, over, mp)

        # Threshold evaluation
        sharp_pass  = sharpness >= self.sharpness_threshold
        exp_pass    = (under <= self.underexposure_max_pct and
                       over  <= self.overexposure_max_pct)
        res_pass    = mp >= self.min_resolution_mp

        reasons = []
        if not sharp_pass:
            reasons.append(
                f"sharpness {sharpness:.1f} < threshold {self.sharpness_threshold:.0f}"
            )
        if under > self.underexposure_max_pct:
            reasons.append(
                f"underexposure {under:.1f}% > {self.underexposure_max_pct:.0f}%"
            )
        if over > self.overexposure_max_pct:
            reasons.append(
                f"overexposure {over:.1f}% > {self.overexposure_max_pct:.0f}%"
            )
        if not res_pass:
            reasons.append(
                f"resolution {mp:.2f} MP < {self.min_resolution_mp:.1f} MP"
            )

        return IQARecord(
            filename          = image_path.name,
            filepath          = str(image_path.resolve()),
            width_px          = w,
            height_px         = h,
            megapixels        = round(mp, 3),
            sharpness_score   = round(sharpness, 2),
            sharpness_pass    = sharp_pass,
            underexposure_pct = round(under, 2),
            overexposure_pct  = round(over, 2),
            exposure_pass     = exp_pass,
            resolution_pass   = res_pass,
            composite_score   = composite,
            overall_pass      = len(reasons) == 0,
            failure_reasons   = reasons,
        )

    # ── Full dataset analysis ────────────────────────────────────────────────

    def analyze_dataset(self, image_paths: list[Path]) -> list[IQARecord]:
        """Analyses a complete dataset with a progress bar."""
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=45),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("IQA Analysis", total=len(image_paths))
            for p in image_paths:
                progress.update(task, description=f"[cyan]{p.name[:40]}")
                r = self.analyze(p)
                if r is not None:
                    results.append(r)
                else:
                    console.print(f"  [red]✗ Cannot read: {p.name}[/red]")
                progress.advance(task)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Report generator
# ══════════════════════════════════════════════════════════════════════════════

class IQAReportGenerator:
    """
    Generates textual reports (CSV, JSON) and charts (PNG) from IQA results.

    Parameters
    ----------
    output_dir : Path
        Destination folder for all generated files.
    dpi : int
        Resolution in DPI for PNG charts. Default: 150.
    """

    def __init__(self, output_dir: Path, dpi: int = 150):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    # ── Textual reports ──────────────────────────────────────────────────────

    def save_csv(self, records: list[IQARecord]) -> Path:
        path = self.out / "iqa_report.csv"
        rows = [r.to_dict() for r in records]
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        return path

    def save_json(self, records: list[IQARecord]) -> Path:
        path = self.out / "iqa_report.json"
        data = [r.to_dict() for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    # ── Chart 1: Sharpness ───────────────────────────────────────────────────

    def chart_sharpness(
        self,
        records: list[IQARecord],
        threshold: float,
    ) -> Path:
        """
        Double panel:
          - Histogram of sharpness score distribution
          - Per-image sharpness scatter plot (coloured by pass/fail)
        """
        scores  = [r.sharpness_score for r in records]
        passed  = [r.sharpness_pass  for r in records]
        names   = [r.filename        for r in records]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Metric 1 – Sharpness (Laplacian Variance)",
                     fontsize=13, fontweight="bold", color=C_DARK)

        # Histogram
        colors_hist = [C_OK if s >= threshold else C_FAIL for s in scores]
        n, bins, patches = ax1.hist(scores, bins=min(25, len(scores)),
                                    color=C_BLUE, edgecolor="white", alpha=0.80)
        # Colour bars below threshold in red
        for patch, left in zip(patches, bins[:-1]):
            if left < threshold:
                patch.set_facecolor(C_FAIL)
                patch.set_alpha(0.75)

        ax1.axvline(threshold, color=C_DARK, linestyle="--", lw=2,
                    label=f"Threshold = {threshold:.0f}")
        ax1.set_title("Sharpness Score Distribution")
        ax1.set_xlabel("Laplacian Variance")
        ax1.set_ylabel("No. of images")
        ax1.legend(fontsize=9)

        # Pass/fail annotation
        n_pass = sum(passed)
        n_fail = len(passed) - n_pass
        ax1.text(0.97, 0.95,
                 f"Pass: {n_pass}  |  Fail: {n_fail}",
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=9, color=C_DARK,
                 bbox={"boxstyle": "round,pad=0.3", "facecolor": C_GRID, "alpha": 0.9})

        # Scatter per immagine
        x_idx   = range(len(records))
        colors  = [C_OK if p else C_FAIL for p in passed]
        ax2.scatter(x_idx, scores, c=colors, s=55, alpha=0.80,
                    edgecolors="white", linewidths=0.5, zorder=3)
        ax2.axhline(threshold, color=C_DARK, linestyle="--", lw=1.5,
                    label=f"Threshold = {threshold:.0f}", zorder=2)
        ax2.fill_between(x_idx, 0, threshold,
                         color=C_FAIL, alpha=0.06, zorder=1)
        ax2.set_title("Sharpness per Image")
        ax2.set_xlabel("Image index")
        ax2.set_ylabel("Laplacian Variance")
        ax2.legend(fontsize=9)

        # Manual legend
        from matplotlib.lines import Line2D
        leg_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OK,
                   markersize=9, label="Pass"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FAIL,
                   markersize=9, label="Fail"),
        ]
        ax2.legend(handles=leg_handles, fontsize=9)

        plt.tight_layout()
        path = self.out / "chart_sharpness.png"
        fig.savefig(str(path), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── Chart 2: Exposure ────────────────────────────────────────────────────

    def chart_exposure(
        self,
        records: list[IQARecord],
        under_thr: float,
        over_thr: float,
    ) -> Path:
        """
        Double panel:
          - Stacked bar chart of under/overexposure per image
          - 2D scatter (under vs over) with acceptability zones
        """
        under   = [r.underexposure_pct for r in records]
        over    = [r.overexposure_pct  for r in records]
        passed  = [r.exposure_pass     for r in records]
        x_idx   = np.arange(len(records))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Metric 2 – Exposure (Luminance Histogram)",
                     fontsize=13, fontweight="bold", color=C_DARK)

        # Stacked bar (show first 50 for readability)
        max_show = min(50, len(records))
        ax1.bar(x_idx[:max_show], under[:max_show],
                label=f"Underexposure (< 30)", color=C_DARK, alpha=0.75)
        ax1.bar(x_idx[:max_show], over[:max_show],
                bottom=under[:max_show],
                label=f"Overexposure (> 225)", color=C_WARN, alpha=0.75)
        ax1.axhline(under_thr, color=C_FAIL, linestyle="--", lw=1.5,
                    label=f"Threshold {under_thr:.0f}%")
        if over_thr != under_thr:
            ax1.axhline(over_thr, color=C_FAIL, linestyle=":", lw=1.5,
                        label=f"Threshold {over_thr:.0f}%")
        ax1.set_title(f"Out-of-range pixels per image (first {max_show})")
        ax1.set_xlabel("Image index")
        ax1.set_ylabel("% Pixels")
        ax1.legend(fontsize=8)

        # Scatter under vs over
        colors = [C_OK if p else C_FAIL for p in passed]
        sc = ax2.scatter(under, over, c=colors, s=60, alpha=0.80,
                         edgecolors="white", linewidths=0.5, zorder=3)
        # Acceptable zone rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), under_thr, over_thr,
                          linewidth=1.5, edgecolor=C_OK,
                          facecolor=C_OK, alpha=0.08, zorder=1)
        ax2.add_patch(rect)
        ax2.axvline(under_thr, color=C_FAIL, linestyle="--", lw=1.2, alpha=0.7)
        ax2.axhline(over_thr,  color=C_FAIL, linestyle="--", lw=1.2, alpha=0.7,
                    label=f"Threshold {over_thr:.0f}%")
        ax2.set_title("Underexposure vs Overexposure Map")
        ax2.set_xlabel(f"Underexposure % (threshold: {under_thr:.0f}%)")
        ax2.set_ylabel(f"Overexposure % (threshold: {over_thr:.0f}%)")

        from matplotlib.lines import Line2D
        leg_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OK,
                   markersize=9, label="Exposure OK"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FAIL,
                   markersize=9, label="Exposure KO"),
        ]
        ax2.legend(handles=leg_handles, fontsize=9)

        plt.tight_layout()
        path = self.out / "chart_exposure.png"
        fig.savefig(str(path), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── Chart 3: Resolution ──────────────────────────────────────────────────

    def chart_resolution(
        self,
        records: list[IQARecord],
        min_mp: float,
    ) -> Path:
        """
        Double panel:
          - Megapixel distribution histogram
          - Pixel dimensions scatter (W × H) with MP area
        """
        mps     = [r.megapixels  for r in records]
        widths  = [r.width_px    for r in records]
        heights = [r.height_px   for r in records]
        passed  = [r.resolution_pass for r in records]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Metric 3 – Resolution (Megapixels)",
                     fontsize=13, fontweight="bold", color=C_DARK)

        # Histogram MP
        colors_hist = [C_OK if m >= min_mp else C_FAIL for m in mps]
        ax1.hist(mps, bins=min(20, len(mps)),
                 color=C_BLUE, edgecolor="white", alpha=0.80)
        ax1.axvline(min_mp, color=C_FAIL, linestyle="--", lw=2,
                    label=f"Threshold = {min_mp:.1f} MP")
        ax1.set_title("Dataset Megapixel Distribution")
        ax1.set_xlabel("Megapixels")
        ax1.set_ylabel("No. of images")
        ax1.legend(fontsize=9)

        # In-chart statistics
        stats_txt = (f"Min:    {min(mps):.2f} MP\n"
                     f"Max:    {max(mps):.2f} MP\n"
                     f"Mean:   {np.mean(mps):.2f} MP\n"
                     f"Median: {np.median(mps):.2f} MP")
        ax1.text(0.97, 0.95, stats_txt,
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=8, fontfamily="monospace", color=C_DARK,
                 bbox={"boxstyle": "round,pad=0.4", "facecolor": C_GRID, "alpha": 0.9})

        # Scatter W × H
        colors = [C_OK if p else C_FAIL for p in passed]
        sizes  = [max(20, mp * 8) for mp in mps]   # point size ∝ MP
        ax2.scatter(widths, heights, c=colors, s=sizes,
                    alpha=0.75, edgecolors="white", linewidths=0.5)

        # MP isocurve (W*H = min_mp * 1e6)
        w_range = np.linspace(min(widths) * 0.9, max(widths) * 1.1, 300)
        h_iso   = (min_mp * 1_000_000) / w_range
        ax2.plot(w_range, h_iso, color=C_FAIL, linestyle="--", lw=1.5,
                 label=f"Isocurve {min_mp:.1f} MP")
        ax2.fill_between(w_range, 0, h_iso, color=C_FAIL, alpha=0.05)

        ax2.set_title("Image Dimensions Distribution (Width × Height)")
        ax2.set_xlabel("Width (pixels)")
        ax2.set_ylabel("Height (pixels)")
        ax2.legend(fontsize=9)

        from matplotlib.lines import Line2D
        leg_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OK,
                   markersize=9, label="Resolution OK"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FAIL,
                   markersize=9, label="Resolution KO"),
        ]
        ax2.legend(handles=leg_handles, fontsize=9)

        plt.tight_layout()
        path = self.out / "chart_resolution.png"
        fig.savefig(str(path), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── Chart 4: Composite score ─────────────────────────────────────────────

    def chart_composite(self, records: list[IQARecord]) -> Path:
        """
        Double panel:
          - Composite score distribution (histogram + KDE)
          - Metric correlation heatmap
        """
        composite = [r.composite_score   for r in records]
        sharpness = [r.sharpness_score    for r in records]
        under     = [r.underexposure_pct  for r in records]
        over      = [r.overexposure_pct   for r in records]
        mps       = [r.megapixels         for r in records]
        passed    = [r.overall_pass       for r in records]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("IQA Composite Score and Metric Correlations",
                     fontsize=13, fontweight="bold", color=C_DARK)

        # Composite score histogram with KDE
        ax1.hist(composite, bins=min(20, len(composite)),
                 color=C_BLUE, edgecolor="white", alpha=0.70,
                 density=True, label="Distribution")

        try:
            from scipy.stats import gaussian_kde
            kde_x = np.linspace(0, 1, 200)
            kde   = gaussian_kde(composite)
            ax1.plot(kde_x, kde(kde_x), color=C_DARK, lw=2, label="KDE")
        except ImportError:
            pass

        ax1.axvline(0.5, color=C_FAIL, linestyle="--", lw=1.5,
                    label="Acceptable quality (0.5)")
        ax1.axvline(np.mean(composite), color=C_WARN, linestyle="-.", lw=1.5,
                    label=f"Mean: {np.mean(composite):.2f}")
        ax1.set_title("Composite Score Distribution (0–1)")
        ax1.set_xlabel("Composite score")
        ax1.set_ylabel("Density")
        ax1.set_xlim(0, 1)
        ax1.legend(fontsize=8)

        n_pass = sum(passed)
        n_fail = len(passed) - n_pass
        ax1.text(0.03, 0.95,
                 f"Pass: {n_pass}  |  Fail: {n_fail}",
                 transform=ax1.transAxes, ha="left", va="top",
                 fontsize=9, color=C_DARK,
                 bbox={"boxstyle": "round,pad=0.3", "facecolor": C_GRID, "alpha": 0.9})

        # Correlation heatmap
        df = pd.DataFrame({
            "Sharpness":     sharpness,
            "Underexposure": under,
            "Overexposure":  over,
            "Megapixels":    mps,
            "Comp. Score":   composite,
        })
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, ax=ax2, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 9},
        )
        ax2.set_title("Metric Correlation Matrix")

        plt.tight_layout()
        path = self.out / "chart_composite.png"
        fig.savefig(str(path), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── Summary dashboard ────────────────────────────────────────────────────

    def dashboard(
        self,
        records: list[IQARecord],
        thresholds: dict,
        elapsed_s: float,
    ) -> Path:
        """
        Six-section summary panel:
          Main KPIs + 4 compact charts + text summary.
        """
        n_total   = len(records)
        n_pass    = sum(r.overall_pass for r in records)
        n_fail    = n_total - n_pass
        avg_sharp = np.mean([r.sharpness_score   for r in records])
        avg_comp  = np.mean([r.composite_score   for r in records])
        avg_under = np.mean([r.underexposure_pct for r in records])
        avg_over  = np.mean([r.overexposure_pct  for r in records])
        avg_mp    = np.mean([r.megapixels         for r in records])
        masked_pcts = [r.composite_score for r in records]

        fig = plt.figure(figsize=(18, 11))
        fig.patch.set_facecolor("#F0F3F4")

        gs = gridspec.GridSpec(
            3, 4, figure=fig,
            hspace=0.50, wspace=0.35,
            top=0.90, bottom=0.06, left=0.05, right=0.97
        )

        # ── Titolo ──────────────────────────────────────────────────────────
        fig.suptitle(
            "SfM Masker – Image Quality Assessment (IQA) Dashboard",
            fontsize=15, fontweight="bold", color=C_DARK, y=0.97
        )

        # ── KPI tiles (row 0) ────────────────────────────────────────────────
        kpis = [
            ("Images\nAnalysed",       f"{n_total}",            C_BLUE),
            ("IQA\nPassed",           f"{n_pass}/{n_total}",   C_OK),
            ("IQA\nFailed",           f"{n_fail}/{n_total}",   C_FAIL if n_fail>0 else C_OK),
            ("Avg Composite\nScore",  f"{avg_comp:.2f}",       C_WARN),
        ]
        for col, (label, value, color) in enumerate(kpis):
            ax_kpi = fig.add_subplot(gs[0, col])
            ax_kpi.set_facecolor(color)
            ax_kpi.set_xticks([]); ax_kpi.set_yticks([])
            for sp in ax_kpi.spines.values():
                sp.set_visible(False)
            ax_kpi.text(0.5, 0.62, value,
                        ha="center", va="center",
                        transform=ax_kpi.transAxes,
                        fontsize=24, fontweight="bold", color="white")
            ax_kpi.text(0.5, 0.22, label,
                        ha="center", va="center",
                        transform=ax_kpi.transAxes,
                        fontsize=9, color="white", alpha=0.92)

        # ── Sharpness per image (row 1, span 2) ──────────────────────────────
        ax_sharp = fig.add_subplot(gs[1, :2])
        scores   = [r.sharpness_score for r in records]
        colors   = [C_OK if r.sharpness_pass else C_FAIL for r in records]
        ax_sharp.bar(range(n_total), scores, color=colors, alpha=0.80)
        ax_sharp.axhline(thresholds["sharpness"], color=C_DARK,
                         linestyle="--", lw=1.5,
                         label=f"Threshold {thresholds['sharpness']:.0f}")
        ax_sharp.set_title("Sharpness per Image", fontsize=10, fontweight="bold")
        ax_sharp.set_xlabel("Image index"); ax_sharp.set_ylabel("Laplacian Variance")
        ax_sharp.legend(fontsize=8)

        # ── Exposure (row 1, span 2) ──────────────────────────────────────────
        ax_exp = fig.add_subplot(gs[1, 2:])
        under  = [r.underexposure_pct for r in records]
        over   = [r.overexposure_pct  for r in records]
        x_idx  = np.arange(n_total)
        ax_exp.bar(x_idx, under, label="Underexposure %", color=C_DARK, alpha=0.75)
        ax_exp.bar(x_idx, over, bottom=under,
                   label="Overexposure %",  color=C_WARN, alpha=0.75)
        ax_exp.axhline(thresholds["underexposure"], color=C_FAIL,
                       linestyle="--", lw=1.2, label=f"Threshold {thresholds['underexposure']:.0f}%")
        ax_exp.set_title("Exposure per Image", fontsize=10, fontweight="bold")
        ax_exp.set_xlabel("Image index"); ax_exp.set_ylabel("% Out-of-range pixels")
        ax_exp.legend(fontsize=8)

        # ── Pass/fail pie chart (row 2, col 0) ───────────────────────────────
        ax_pie = fig.add_subplot(gs[2, 0])
        if n_fail > 0:
            ax_pie.pie(
                [n_pass, n_fail],
                labels=["Pass", "Fail"],
                colors=[C_OK, C_FAIL],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"edgecolor": "white", "lw": 2}
            )
        else:
            ax_pie.pie([1], labels=["All OK"],
                       colors=[C_OK], wedgeprops={"edgecolor": "white", "lw": 2})
        ax_pie.set_title("Overall IQA Result", fontsize=10, fontweight="bold")

        # ── Composite score per image (row 2, col 1-2) ───────────────────────
        ax_comp = fig.add_subplot(gs[2, 1:3])
        comp_scores = [r.composite_score for r in records]
        comp_colors = [C_OK if r.overall_pass else C_FAIL for r in records]
        ax_comp.fill_between(range(n_total), comp_scores,
                             alpha=0.25,
                             color=[C_OK if r.overall_pass else C_FAIL
                                    for r in records][0])
        ax_comp.scatter(range(n_total), comp_scores,
                        c=comp_colors, s=40, alpha=0.85,
                        edgecolors="white", linewidths=0.4, zorder=3)
        ax_comp.plot(range(n_total), comp_scores,
                     color=C_BLUE, lw=0.8, alpha=0.5, zorder=2)
        ax_comp.axhline(0.5, color=C_FAIL, linestyle="--", lw=1.2,
                        label="Acceptability threshold (0.5)")
        ax_comp.axhline(avg_comp, color=C_WARN, linestyle="-.", lw=1.2,
                        label=f"Mean: {avg_comp:.2f}")
        ax_comp.set_ylim(0, 1.05)
        ax_comp.set_title("Composite Score per Image", fontsize=10, fontweight="bold")
        ax_comp.set_xlabel("Image index"); ax_comp.set_ylabel("Score (0–1)")
        ax_comp.legend(fontsize=8)

        # ── Text summary (row 2, col 3) ──────────────────────────────────────
        ax_txt = fig.add_subplot(gs[2, 3])
        ax_txt.set_facecolor("#EBF5FB")
        ax_txt.set_xticks([]); ax_txt.set_yticks([])
        for sp in ax_txt.spines.values():
            sp.set_edgecolor(C_BLUE); sp.set_linewidth(1.2)

        mins = int(elapsed_s // 60); secs = int(elapsed_s % 60)
        mps_min = min(r.megapixels for r in records)
        mps_max = max(r.megapixels for r in records)

        summary = (
            f"DATASET STATISTICS\n"
            f"{'─'*28}\n"
            f"Total images:     {n_total:>5}\n"
            f"IQA passed:       {n_pass:>5}\n"
            f"IQA failed:       {n_fail:>5}\n"
            f"─────────────────────────────\n"
            f"Avg sharpness:    {avg_sharp:>7.1f}\n"
            f"Avg underexp.:    {avg_under:>6.1f}%\n"
            f"Avg overexp.:     {avg_over:>6.1f}%\n"
            f"Avg resolution:   {avg_mp:>5.2f} MP\n"
            f"MP min/max: {mps_min:.1f} / {mps_max:.1f}\n"
            f"─────────────────────────────\n"
            f"Analysis time: {mins}m {secs:02d}s"
        )
        ax_txt.text(0.07, 0.95, summary,
                    transform=ax_txt.transAxes, fontsize=8.5,
                    va="top", fontfamily="monospace", color=C_DARK)

        path = self.out / "dashboard_iqa.png"
        fig.savefig(str(path), dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    # ── Metodo pubblico principale ───────────────────────────────────────────

    def generate_all(
        self,
        records: list[IQARecord],
        thresholds: dict,
        elapsed_s: float,
    ) -> dict[str, Path]:
        """Generates all reports and charts. Returns output paths."""
        console.rule("[bold cyan]Generating Reports")
        out = {}
        out["csv"]       = self.save_csv(records)
        out["json"]      = self.save_json(records)
        out["sharpness"] = self.chart_sharpness(records, thresholds["sharpness"])
        out["exposure"]  = self.chart_exposure(records,
                                               thresholds["underexposure"],
                                               thresholds["overexposure"])
        out["resolution"]= self.chart_resolution(records, thresholds["min_mp"])
        out["composite"] = self.chart_composite(records)
        out["dashboard"] = self.dashboard(records, thresholds, elapsed_s)

        for name, path in out.items():
            if path and path.exists():
                console.print(f"  [green]✓[/green] {name:<12} → {path.name}")
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Console output
# ══════════════════════════════════════════════════════════════════════════════

def print_results_table(records: list[IQARecord]):
    """Prints the summary table to the console."""
    table = Table(
        title="IQA Analysis Results",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Image",           max_width=32, style="white")
    table.add_column("Sharpness",       justify="right", style="cyan")
    table.add_column("Sharp ✓",        justify="center")
    table.add_column("Under %",         justify="right", style="yellow")
    table.add_column("Over %",          justify="right", style="yellow")
    table.add_column("Exp. ✓",         justify="center")
    table.add_column("MP",              justify="right", style="magenta")
    table.add_column("Res. ✓",         justify="center")
    table.add_column("Score",           justify="right", style="green")
    table.add_column("Result",          justify="center")

    for r in records:
        ok  = lambda b: "[green]✓[/green]" if b else "[red]✗[/red]"
        result = "[green]PASS[/green]" if r.overall_pass else "[red]FAIL[/red]"
        table.add_row(
            r.filename[:32],
            f"{r.sharpness_score:.1f}",
            ok(r.sharpness_pass),
            f"{r.underexposure_pct:.1f}",
            f"{r.overexposure_pct:.1f}",
            ok(r.exposure_pass),
            f"{r.megapixels:.2f}",
            ok(r.resolution_pass),
            f"{r.composite_score:.3f}",
            result,
        )
    console.print(table)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone IQA analysis for SfM photogrammetric datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",    default="DATASET",
                   help="Folder containing the images to analyse")
    p.add_argument("--output",     default="IQA_REPORT",
                   help="Output folder for reports and charts")
    p.add_argument("--sharpness",  type=float, default=80.0,
                   help="Laplacian variance threshold for sharpness")
    p.add_argument("--under",      type=float, default=30.0,
                   help="Maximum allowed underexposed pixels (%%)"),
    p.add_argument("--over",       type=float, default=30.0,
                   help="Maximum allowed overexposed pixels (%%)"),
    p.add_argument("--min-mp",     type=float, default=1.0,
                   help="Minimum resolution in megapixels")
    p.add_argument("--dpi",        type=int,   default=150,
                   help="DPI resolution for PNG charts")
    return p.parse_args()


def main():
    console.print(
        Panel(
            "[bold cyan]IQA Analysis[/bold cyan] – "
            "Image Quality Assessment for SfM Workflows\n"
            "[dim]Metrics: Sharpness · Exposure · Resolution[/dim]",
            border_style="cyan"
        )
    )

    args = parse_args()

    dataset_dir = Path(args.dataset)
    output_dir  = Path(args.output)

    # Image collection
    if not dataset_dir.exists():
        console.print(f"[bold red]ERROR: folder '{dataset_dir}' not found.[/bold red]")
        console.print("Create the DATASET folder and place the images to be analysed inside.")
        sys.exit(1)

    images = sorted([
        p for p in dataset_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not images:
        console.print(f"[bold red]ERROR: no images found in '{dataset_dir}'.[/bold red]")
        sys.exit(1)

    console.print(f"\n  [green]✓[/green] Dataset:    [bold]{dataset_dir.resolve()}[/bold]")
    console.print(f"  [green]✓[/green] Images:     [bold]{len(images)}[/bold]")
    console.print(f"  [cyan]→[/cyan] Output:     [bold]{output_dir.resolve()}[/bold]")
    console.print()
    console.print(f"  Configured thresholds:")
    console.print(f"    Sharpness (Laplacian):   ≥ {args.sharpness:.0f}")
    console.print(f"    Underexposure:           ≤ {args.under:.0f}%")
    console.print(f"    Overexposure:            ≤ {args.over:.0f}%")
    console.print(f"    Minimum resolution:      ≥ {args.min_mp:.1f} MP")
    console.print()

    thresholds = {
        "sharpness":    args.sharpness,
        "underexposure": args.under,
        "overexposure":  args.over,
        "min_mp":        args.min_mp,
    }

    # Analysis
    console.rule("[bold]Image Analysis")
    t0      = time.perf_counter()
    analyzer = IQAAnalyzer(
        sharpness_threshold   = args.sharpness,
        underexposure_max_pct = args.under,
        overexposure_max_pct  = args.over,
        min_resolution_mp     = args.min_mp,
    )
    records  = analyzer.analyze_dataset(images)
    elapsed  = time.perf_counter() - t0

    if not records:
        console.print("[bold red]No images were successfully processed.[/bold red]")
        sys.exit(1)

    # Console table
    console.print()
    print_results_table(records)

    # Reports and charts
    reporter = IQAReportGenerator(output_dir, dpi=args.dpi)
    outputs  = reporter.generate_all(records, thresholds, elapsed)

    # Final summary
    n_pass = sum(r.overall_pass for r in records)
    n_fail = len(records) - n_pass
    mins   = int(elapsed // 60); secs = int(elapsed % 60)

    content = (
        f"[green]✓[/green] Images analysed:      [bold]{len(records)}[/bold]\n"
        f"[green]✓[/green] IQA passed:            [bold]{n_pass}[/bold]  "
        f"[red]✗[/red] Failed: [bold]{n_fail}[/bold]\n"
        f"[green]✓[/green] Avg composite score:   [bold]{np.mean([r.composite_score for r in records]):.3f}[/bold]\n"
        f"[green]✓[/green] Analysis time:         [bold]{mins}m {secs:02d}s[/bold]\n\n"
        f"[cyan]Reports saved in:[/cyan] [bold]{output_dir.resolve()}[/bold]\n"
        f"  ├─ iqa_report.csv\n"
        f"  ├─ iqa_report.json\n"
        f"  ├─ chart_sharpness.png\n"
        f"  ├─ chart_exposure.png\n"
        f"  ├─ chart_resolution.png\n"
        f"  ├─ chart_composite.png\n"
        f"  └─ dashboard_iqa.png"
    )
    console.print(Panel(content, title="[bold green]Analysis Complete", border_style="green"))


if __name__ == "__main__":
    main()
