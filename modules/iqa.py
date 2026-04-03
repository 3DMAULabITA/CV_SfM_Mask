"""
modules/iqa.py
─────────────────────────────────────────────────────────────
Modulo 1 – Image Quality Assessment (IQA)

Valuta automaticamente la qualità di ciascuna immagine del
dataset tramite tre criteri:
  1. Nitidezza  – varianza del Laplaciano sul canale luminanza
  2. Esposizione – proporzione pixel sotto/sovraesposti
  3. Risoluzione – numero di megapixel

Restituisce un dizionario di metriche per immagine e un flag
di superamento della soglia complessiva.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class IQAResult:
    """Risultato della valutazione qualità per una singola immagine."""
    filename:            str
    width:               int
    height:              int
    megapixels:          float
    sharpness_score:     float       # varianza Laplaciano (↑ meglio)
    underexposure_pct:   float       # % pixel < 30 (↓ meglio)
    overexposure_pct:    float       # % pixel > 225 (↓ meglio)
    composite_score:     float       # indice composito 0–1 (↑ meglio)
    passed:              bool        # True se supera tutte le soglie
    failure_reasons:     list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class IQAModule:
    """
    Valutazione automatica della qualità delle immagini.

    Parametri
    ----------
    sharpness_threshold : float
        Soglia varianza Laplaciano. Valori < soglia → immagine sfocata.
        Tipicamente 80–150 per immagini fotogrammetriche.
    underexposure_max_pct : float
        Frazione massima ammessa di pixel sottoesposti (val < 30).
    overexposure_max_pct : float
        Frazione massima ammessa di pixel sovraesposti (val > 225).
    min_resolution_mp : float
        Risoluzione minima in megapixel.
    """

    def __init__(
        self,
        sharpness_threshold:    float = 80.0,
        underexposure_max_pct:  float = 0.30,
        overexposure_max_pct:   float = 0.30,
        min_resolution_mp:      float = 1.0,
    ):
        self.sharpness_threshold    = sharpness_threshold
        self.underexposure_max_pct  = underexposure_max_pct
        self.overexposure_max_pct   = overexposure_max_pct
        self.min_resolution_mp      = min_resolution_mp

    # ── Metriche elementari ──────────────────────────────────────────

    @staticmethod
    def _compute_sharpness(gray: np.ndarray) -> float:
        """
        Stima la nitidezza tramite la varianza del Laplaciano.
        Riferimento: Pech-Pacheco et al., ICPR 2000.
        """
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _compute_exposure(gray: np.ndarray) -> tuple[float, float]:
        """
        Calcola le percentuali di pixel sottoesposti e sovraesposti.
        """
        total = gray.size
        under = float(np.sum(gray < 30)  / total)
        over  = float(np.sum(gray > 225) / total)
        return under, over

    @staticmethod
    def _compute_megapixels(h: int, w: int) -> float:
        return (h * w) / 1_000_000

    # ── Indice composito ────────────────────────────────────────────

    def _composite_score(
        self,
        sharpness: float,
        under: float,
        over:  float,
        mp:    float,
    ) -> float:
        """
        Combina le tre metriche in un indice 0–1.
        Pesi calibrati empiricamente (somma = 1).
        """
        # Normalizzazione nitidezza: sigmoide centrata sulla soglia
        sharp_norm = 1.0 / (1.0 + np.exp(-(sharpness - self.sharpness_threshold) / 20.0))

        # Esposizione: penalità per ogni % di pixel fuori range
        exp_score = max(0.0, 1.0 - (under + over) * 1.5)

        # Risoluzione: lineare fino al doppio della soglia minima
        res_score = min(1.0, mp / (self.min_resolution_mp * 2))

        # Pesi: nitidezza 50%, esposizione 35%, risoluzione 15%
        return round(0.50 * sharp_norm + 0.35 * exp_score + 0.15 * res_score, 4)

    # ── Metodo principale ────────────────────────────────────────────

    def evaluate(self, image_path: Path) -> Optional[IQAResult]:
        """
        Esegue la valutazione IQA su una singola immagine.

        Parametri
        ----------
        image_path : Path
            Percorso all'immagine da analizzare.

        Restituisce
        -----------
        IQAResult oppure None se il file non è leggibile.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sharpness       = self._compute_sharpness(gray)
        under, over     = self._compute_exposure(gray)
        mp              = self._compute_megapixels(h, w)
        composite       = self._composite_score(sharpness, under, over, mp)

        # Verifica soglie
        reasons = []
        if sharpness < self.sharpness_threshold:
            reasons.append(f"nitidezza {sharpness:.1f} < soglia {self.sharpness_threshold}")
        if under > self.underexposure_max_pct:
            reasons.append(f"sottoesposizione {under*100:.1f}% > {self.underexposure_max_pct*100:.0f}%")
        if over > self.overexposure_max_pct:
            reasons.append(f"sovraesposizione {over*100:.1f}% > {self.overexposure_max_pct*100:.0f}%")
        if mp < self.min_resolution_mp:
            reasons.append(f"risoluzione {mp:.2f} MP < {self.min_resolution_mp} MP")

        return IQAResult(
            filename          = image_path.name,
            width             = w,
            height            = h,
            megapixels        = round(mp, 2),
            sharpness_score   = round(sharpness, 2),
            underexposure_pct = round(under * 100, 2),
            overexposure_pct  = round(over  * 100, 2),
            composite_score   = composite,
            passed            = len(reasons) == 0,
            failure_reasons   = reasons,
        )

    def evaluate_dataset(self, image_paths: list[Path]) -> list[IQAResult]:
        """
        Valuta un dataset completo. Restituisce la lista dei risultati.
        """
        results = []
        for p in image_paths:
            r = self.evaluate(p)
            if r is not None:
                results.append(r)
        return results
