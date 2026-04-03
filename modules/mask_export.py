"""
modules/mask_export.py
─────────────────────────────────────────────────────────────
Modulo 3 – Post-processing morfologico ed esportazione maschere

Operazioni in sequenza:
  1. Aggregazione maschere per-istanza → maschera globale
  2. Dilatazione  – espande i bordi per margine di sicurezza
  3. Rimozione piccoli componenti – elimina falsi positivi
  4. Closing morfologico – riempie lacune interne
  5. Export in formato Metashape / OpenDroneMap / RealityCapture
  6. Generazione preview con overlay colorato
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from modules.segmentation import SegmentationResult


class MaskExporter:
    """
    Post-processing morfologico e salvataggio maschere.

    Parametri
    ----------
    dilation_kernel_size : int
        Lato del kernel quadrato per la dilatazione.
    dilation_iterations : int
        Numero di iterazioni di dilatazione.
    min_component_area : int
        Area minima (pixel²) delle componenti connesse da mantenere.
    closing_kernel_size : int
        Lato del kernel per il closing morfologico (riempie buchi).
    platform : str
        "metashape" | "odm" | "realitycapture" | "all"
    generate_preview : bool
        Se True, salva immagine di anteprima con overlay colorato.
    preview_alpha : float
        Opacità dell'overlay colorato (0–1).
    preview_color : tuple
        Colore BGR dell'overlay (default: rosso).
    """

    def __init__(
        self,
        dilation_kernel_size: int   = 5,
        dilation_iterations:  int   = 2,
        min_component_area:   int   = 200,
        closing_kernel_size:  int   = 15,
        platform:             str   = "all",
        generate_preview:     bool  = True,
        preview_alpha:        float = 0.45,
        preview_color:        tuple = (0, 0, 255),   # BGR rosso
    ):
        self.dilation_kernel_size = dilation_kernel_size
        self.dilation_iterations  = dilation_iterations
        self.min_component_area   = min_component_area
        self.closing_kernel_size  = closing_kernel_size
        self.platform             = platform
        self.generate_preview     = generate_preview
        self.preview_alpha        = preview_alpha
        self.preview_color        = tuple(preview_color)

        # Kernel morfologici
        self._dil_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_kernel_size, dilation_kernel_size),
        )
        self._cls_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (closing_kernel_size, closing_kernel_size),
        )

    # ── Post-processing ─────────────────────────────────────────────

    def postprocess(self, raw_mask: np.ndarray) -> np.ndarray:
        """
        Applica la catena morfologica alla maschera grezza aggregata.

        Parametri
        ----------
        raw_mask : np.ndarray
            Maschera binaria HxW uint8 (valori 0 o 255).

        Restituisce
        -----------
        Maschera post-processata (stessa forma e tipo).
        """
        mask = raw_mask.copy()

        # 1. Dilatazione – espande i bordi
        if self.dilation_kernel_size > 0 and self.dilation_iterations > 0:
            mask = cv2.dilate(mask, self._dil_kernel,
                              iterations=self.dilation_iterations)

        # 2. Rimozione componenti piccoli (rumore / falsi positivi)
        if self.min_component_area > 0:
            mask = self._remove_small_components(mask)

        # 3. Closing – riempie lacune interne
        if self.closing_kernel_size > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._cls_kernel)

        return mask

    @staticmethod
    def _remove_small_components(
        mask: np.ndarray,
        min_area: int = 200,
    ) -> np.ndarray:
        """Rimuove componenti connesse con area < min_area pixel²."""
        # connectedComponentsWithStats restituisce anche statistiche per area
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        out = np.zeros_like(mask)
        for lbl in range(1, n_labels):            # 0 = sfondo
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                out[labels == lbl] = 255
        return out

    # ── Export per piattaforma ───────────────────────────────────────

    def _export_metashape(
        self,
        mask: np.ndarray,
        output_dir: Path,
        stem: str,
        ext: str,
    ) -> Path:
        """
        Metashape: PNG 8-bit, bianco = area mascherata, nero = valida.
        Nome file: <stem>_mask.png
        """
        out_dir = output_dir / "METASHAPE"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_mask.png"
        cv2.imwrite(str(out_path), mask)
        return out_path

    def _export_odm(
        self,
        mask: np.ndarray,
        output_dir: Path,
        stem: str,
        ext: str,
    ) -> Path:
        """
        OpenDroneMap: PNG co-localizzato con il nome <stem>_mask.png
        nella stessa cartella dell'immagine o in una sub-dir dedicata.
        Bianco = area da escludere.
        """
        out_dir = output_dir / "ODM"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_mask.png"
        cv2.imwrite(str(out_path), mask)
        return out_path

    def _export_realitycapture(
        self,
        mask: np.ndarray,
        output_dir: Path,
        stem: str,
        ext: str,
    ) -> Path:
        """
        RealityCapture: TIFF con canale maschera.
        Nero = area da mascherare (convenzione inversa).
        """
        out_dir = output_dir / "REALITYCAPTURE"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.mask.tif"
        # RealityCapture usa convenzione invertita: 0 = mascherato
        rc_mask = cv2.bitwise_not(mask)
        cv2.imwrite(str(out_path), rc_mask)
        return out_path

    def _export_generic(
        self,
        mask: np.ndarray,
        output_dir: Path,
        stem: str,
    ) -> Path:
        """
        Export generico: PNG nella cartella MASKS principale.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{stem}_mask.png"
        cv2.imwrite(str(out_path), mask)
        return out_path

    # ── Preview con overlay ─────────────────────────────────────────

    def generate_preview_image(
        self,
        original_path: Path,
        mask: np.ndarray,
        preview_dir: Path,
    ) -> Optional[Path]:
        """
        Genera un'immagine di anteprima con overlay colorato sulle aree
        mascherate e contorni evidenziati.
        """
        img = cv2.imread(str(original_path))
        if img is None:
            return None

        preview_dir.mkdir(parents=True, exist_ok=True)

        # Ridimensiona la maschera se necessario
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Overlay colorato semi-trasparente
        overlay = img.copy()
        colored_region = np.zeros_like(img)
        colored_region[mask > 0] = self.preview_color
        overlay = cv2.addWeighted(
            img, 1.0,
            colored_region, self.preview_alpha,
            0,
        )

        # Contorno della maschera
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        stem = original_path.stem
        out_path = preview_dir / f"{stem}_preview.jpg"
        cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return out_path

    # ── Metodo principale ────────────────────────────────────────────

    def export(
        self,
        seg_result: SegmentationResult,
        original_path: Path,
        masks_output_dir: Path,
        preview_dir: Optional[Path] = None,
    ) -> dict:
        """
        Esegue post-processing ed esportazione per una singola immagine.

        Restituisce un dizionario con i percorsi dei file generati
        e le metriche della maschera finale.
        """
        stem = original_path.stem
        ext  = original_path.suffix

        # Maschera grezza aggregata
        raw_mask = seg_result.combined_mask

        # Post-processing morfologico
        final_mask = self.postprocess(raw_mask)

        # Statistiche maschera finale
        total_px   = seg_result.image_h * seg_result.image_w
        masked_px  = int(np.sum(final_mask > 0))
        masked_pct = round(masked_px / total_px * 100, 2) if total_px > 0 else 0.0

        exported_files = {}

        # Export per piattaforma
        if self.platform in ("metashape", "all"):
            p = self._export_metashape(final_mask, masks_output_dir, stem, ext)
            exported_files["metashape"] = str(p)

        if self.platform in ("odm", "all"):
            p = self._export_odm(final_mask, masks_output_dir, stem, ext)
            exported_files["odm"] = str(p)

        if self.platform in ("realitycapture", "all"):
            p = self._export_realitycapture(final_mask, masks_output_dir, stem, ext)
            exported_files["realitycapture"] = str(p)

        # Export generico nella cartella principale MASKS
        p = self._export_generic(final_mask, masks_output_dir, stem)
        exported_files["generic"] = str(p)

        # Preview
        preview_path = None
        if self.generate_preview and preview_dir is not None:
            pp = self.generate_preview_image(original_path, final_mask, preview_dir)
            if pp:
                preview_path = str(pp)
                exported_files["preview"] = preview_path

        return {
            "filename":         original_path.name,
            "masked_pixels":    masked_px,
            "total_pixels":     total_px,
            "masked_pct":       masked_pct,
            "n_objects":        seg_result.n_objects,
            "exported_files":   exported_files,
        }
