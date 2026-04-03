"""
modules/segmentation.py
─────────────────────────────────────────────────────────────
Modulo 2 – Detect-Then-Segment

Stadio A: rilevamento e segmentazione a istanza con YOLOv8-seg
           (pre-addestrato su MS COCO, 80 classi)

Stadio B: raffinamento opzionale dei bordi con SAM2
           (attivabile tramite config: use_sam2: true)

Output per immagine:
  - lista di maschere binarie per oggetto rilevato
  - metadati (classe, confidenza, area)
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Mappa ID classe COCO → nome leggibile
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow",
}


@dataclass
class DetectionResult:
    """Risultato di rilevamento per un singolo oggetto in un'immagine."""
    class_id:    int
    class_name:  str
    confidence:  float
    bbox:        tuple          # (x1, y1, x2, y2) pixel assoluti
    mask:        np.ndarray     # maschera binaria HxW uint8 (0/255)
    area_px:     int            # area maschera in pixel
    area_pct:    float          # area maschera in % area immagine


@dataclass
class SegmentationResult:
    """Risultato di segmentazione per una singola immagine."""
    filename:        str
    image_h:         int
    image_w:         int
    detections:      list[DetectionResult] = field(default_factory=list)
    processing_ms:   float = 0.0           # tempo di inferenza in ms
    sam2_used:       bool  = False

    @property
    def n_objects(self) -> int:
        return len(self.detections)

    @property
    def combined_mask(self) -> np.ndarray:
        """Maschera binaria aggregata di tutti gli oggetti rilevati."""
        mask = np.zeros((self.image_h, self.image_w), dtype=np.uint8)
        for d in self.detections:
            mask = cv2.bitwise_or(mask, d.mask)
        return mask

    @property
    def total_masked_pct(self) -> float:
        """Percentuale totale di area mascherata sull'immagine."""
        total_px = self.image_h * self.image_w
        if total_px == 0:
            return 0.0
        return float(self.combined_mask.sum()) / (255 * total_px) * 100


class SegmentationModule:
    """
    Pipeline detect-then-segment per la generazione di maschere di occlusione.

    Parametri
    ----------
    yolo_model : str
        Nome o percorso del modello YOLOv8-seg (es. "yolov8x-seg.pt").
    confidence_threshold : float
        Soglia minima di confidenza per i rilevamenti.
    iou_threshold : float
        Soglia IoU per la Non-Maximum Suppression.
    imgsz : int
        Dimensione lato lungo per il resize interno YOLO.
    target_classes : list[int]
        ID classi COCO da includere nella maschera.
    use_sam2 : bool
        Se True, attiva il raffinamento bordi con SAM2.
    sam2_checkpoint : str
        Percorso al checkpoint SAM2.
    sam2_config : str
        Percorso al file di configurazione SAM2.
    device : str
        "cuda", "cpu" o "auto".
    """

    def __init__(
        self,
        yolo_model:          str        = "yolov8x-seg.pt",
        confidence_threshold: float     = 0.45,
        iou_threshold:       float      = 0.45,
        imgsz:               int        = 1280,
        target_classes:      list[int]  = None,
        use_sam2:            bool       = False,
        sam2_checkpoint:     str        = "sam2_hiera_large.pt",
        sam2_config:         str        = "sam2_hiera_l.yaml",
        device:              str        = "auto",
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold        = iou_threshold
        self.imgsz                = imgsz
        self.target_classes       = target_classes or [0, 1, 2, 3, 5, 7]
        self.use_sam2             = use_sam2
        self.sam2_checkpoint      = sam2_checkpoint
        self.sam2_config          = sam2_config

        # Rilevazione automatica dispositivo
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"  [Segmentazione] Dispositivo: {self.device.upper()}")
        if self.device == "cuda":
            print(f"  [Segmentazione] GPU: {torch.cuda.get_device_name(0)}")

        # Caricamento modello YOLO
        self._load_yolo(yolo_model)

        # Caricamento opzionale SAM2
        self._sam2_predictor = None
        if self.use_sam2:
            self._load_sam2()

    # ── Caricamento modelli ──────────────────────────────────────────

    def _load_yolo(self, model_name: str):
        """Carica il modello YOLOv8-seg (scarica automaticamente se assente)."""
        from ultralytics import YOLO
        print(f"  [YOLO] Caricamento modello: {model_name}")
        self._yolo = YOLO(model_name)
        self._yolo.to(self.device)
        print(f"  [YOLO] Pronto.")

    def _load_sam2(self):
        """
        Tenta il caricamento di SAM2.
        Se il pacchetto non è installato, disabilita silenziosamente.
        """
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            print(f"  [SAM2] Caricamento checkpoint: {self.sam2_checkpoint}")
            sam2_model = build_sam2(
                self.sam2_config,
                self.sam2_checkpoint,
                device=self.device,
            )
            self._sam2_predictor = SAM2ImagePredictor(sam2_model)
            print(f"  [SAM2] Pronto.")
        except ImportError:
            print("  [SAM2] ATTENZIONE: pacchetto sam2 non trovato. "
                  "Disabilitato automaticamente. Installa con: pip install sam2")
            self.use_sam2 = False
        except FileNotFoundError:
            print(f"  [SAM2] ATTENZIONE: checkpoint '{self.sam2_checkpoint}' non trovato. "
                  "Disabilitato automaticamente.")
            self.use_sam2 = False

    # ── Inferenza ────────────────────────────────────────────────────

    def _yolo_inference(self, img_bgr: np.ndarray) -> list[dict]:
        """
        Esegue l'inferenza YOLOv8-seg su una singola immagine.

        Restituisce lista di dict con:
          class_id, class_name, confidence, bbox, mask_raw
        """
        results = self._yolo.predict(
            img_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=self.target_classes,
            verbose=False,
            device=self.device,
        )

        detections = []
        h, w = img_bgr.shape[:2]

        for r in results:
            if r.masks is None:
                continue
            for i, (box, mask_data) in enumerate(
                zip(r.boxes, r.masks.data)
            ):
                cls_id  = int(box.cls.item())
                conf    = float(box.conf.item())
                xyxy    = box.xyxy[0].cpu().numpy().astype(int)

                # Ridimensiona la maschera alla risoluzione originale
                mask_np = mask_data.cpu().numpy()
                mask_rs = cv2.resize(
                    mask_np, (w, h), interpolation=cv2.INTER_LINEAR
                )
                mask_bin = (mask_rs > 0.5).astype(np.uint8) * 255

                detections.append({
                    "class_id":   cls_id,
                    "class_name": COCO_NAMES.get(cls_id, f"class_{cls_id}"),
                    "confidence": conf,
                    "bbox":       tuple(xyxy.tolist()),
                    "mask_raw":   mask_bin,
                })
        return detections

    def _sam2_refine(
        self,
        img_bgr: np.ndarray,
        detections: list[dict],
    ) -> list[dict]:
        """
        Raffina i bordi delle maschere usando SAM2 con i bounding box
        di YOLO come prompt geometrici.
        """
        if self._sam2_predictor is None or not detections:
            return detections

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            self._sam2_predictor.set_image(img_rgb)

            refined = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                # SAM2 accetta box nel formato [x1, y1, x2, y2]
                box_np = np.array([x1, y1, x2, y2], dtype=np.float32)

                masks_sam, scores, _ = self._sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_np[None, :],
                    multimask_output=False,
                )
                # Prende la maschera con score più alto
                best_mask = masks_sam[np.argmax(scores)]
                mask_bin  = (best_mask > 0.5).astype(np.uint8) * 255

                det_refined      = det.copy()
                det_refined["mask_raw"] = mask_bin
                refined.append(det_refined)

        return refined

    # ── Metodo principale ────────────────────────────────────────────

    def process_image(
        self,
        image_path: Path,
    ) -> Optional[SegmentationResult]:
        """
        Elabora una singola immagine: rilevamento + segmentazione.

        Restituisce SegmentationResult oppure None se il file non è leggibile.
        """
        import time

        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w = img.shape[:2]
        t0 = time.perf_counter()

        # Stadio A: YOLO
        raw_detections = self._yolo_inference(img)

        # Stadio B: SAM2 (opzionale)
        sam2_used = False
        if self.use_sam2 and raw_detections and self._sam2_predictor is not None:
            raw_detections = self._sam2_refine(img, raw_detections)
            sam2_used = True

        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_px   = h * w

        # Costruisce DetectionResult per ciascun oggetto
        detection_results = []
        for det in raw_detections:
            area_px  = int(np.sum(det["mask_raw"] > 0))
            area_pct = round(area_px / total_px * 100, 2) if total_px > 0 else 0.0
            detection_results.append(DetectionResult(
                class_id   = det["class_id"],
                class_name = det["class_name"],
                confidence = round(det["confidence"], 3),
                bbox       = det["bbox"],
                mask       = det["mask_raw"],
                area_px    = area_px,
                area_pct   = area_pct,
            ))

        return SegmentationResult(
            filename      = image_path.name,
            image_h       = h,
            image_w       = w,
            detections    = detection_results,
            processing_ms = round(elapsed_ms, 1),
            sam2_used     = sam2_used,
        )
