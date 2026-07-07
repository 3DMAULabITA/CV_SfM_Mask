# Integrating Computer Vision Models into SfM Photogrammetry: An Automated Pipeline for Occlusion Mask Generation


## Overview

SfM Masker is a modular pipeline designed to automatically detect and remove dynamic objects (e.g., people, vehicles) from photogrammetric datasets.  
It integrates state-of-the-art computer vision models (YOLOv8 and optional SAM2) to generate high-quality occlusion masks compatible with major SfM software.

## Abstract
This paper presents an automated and scalable pipeline for occlusion mask generation in Structure-from-Motion (SfM) photogrammetric workflows, addressing a well-known limitation related to dynamic elements, such as pedestrians, vehicles, and other transient objects, that introduce geometric artifacts and reduce the metric reliability of 3D reconstructions. To overcome the inefficiencies and subjectivity of manual masking, the proposed approach integrates pre-trained computer vision models into the preprocessing stage, enabling a reproducible solution that does not require domain-specific training data. The pipeline is structured into three modular components. The first module performs automated image quality assessment based on sharpness, exposure, and resolution, ensuring dataset consistency and reducing error propagation during feature extraction. The second module implements a detect-then-segment strategy, combining instance-level detection via YOLOv8-seg with optional pixel-wise refinement using the Segment Anything Model (SAM2), improving segmentation accuracy in high-resolution and close-range scenarios. The third module handles automatic mask generation and formatting for compatibility with major SfM platforms, including morphology-based post-processing to enhance robustness. A proof-of-concept implementation was developed in Python using exclusively pre-trained models, allowing immediate applicability while minimizing computational and operational overhead. Experimental results highlight the efficiency of the automated masking process, while maintaining improved or comparable reconstruction quality. The approach proves particularly effective in architectural and cultural heritage surveys conducted in complex and uncontrolled environments.

## Project Structure Folder

```text
<root project>/
├── main.py                # Entry point
├── config.yaml            # Pipeline configuration
├── environment.yml        # Optional environment definition
├── README.md
├── DATASET/               # Place input images here
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── OUTPUT/                # Automatically generated
│   ├── MASKS/
│   │   ├── image_001_mask.png
│   │   ├── METASHAPE/
│   │   ├── ODM/
│   │   └── REALITYCAPTURE/
│   ├── PREVIEW/
│   └── REPORT/
│       ├── processing_report.json
│       ├── processing_report.csv
│       ├── chart_iqa.png
│       ├── chart_segmentation.png
│       ├── chart_masks.png
│       └── dashboard.png
└── modules/
    ├── iqa.py
    ├── segmentation.py
    ├── mask_export.py
    └── report.py
```

### INPUT

- `DATASET/`
  - Contains input images
  - Supported formats: `.jpg`, `.png`, `.tif`
  - Images should be consistent in resolution and acquisition conditions

---

### OUTPUT

Generated automatically after running the pipeline:

- `OUTPUT/MASKS/`
  - Binary masks for each image
  - Platform-specific subfolders:
    - `METASHAPE/`
    - `ODM/`
    - `REALITYCAPTURE/`

- `OUTPUT/PREVIEW/`
  - Visual overlays of masks for quick inspection

- `OUTPUT/REPORT/`
  - Processing statistics and diagnostics:
    - JSON and CSV reports
    - Charts and visual summaries

---

##  Installation 

SfM Masker can be installed in any Python environment.

### Requirements

- Python ≥ 3.10
- PyTorch with CUDA (recommended for GPU acceleration)
- Ultralytics (YOLOv8)
- OpenCV
- NumPy

---

### Install dependencies

Using pip:

```bash
pip install -r requirements.txt

## Installation CHECKPOINT

curl -L -o sam2_hiera_large.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

segmentation:
  use_sam2: true
  sam2_checkpoint: "checkpoints/sam2_hiera_large.pt"


## USAGE
python main.py
```
---

## Cited

Tavolare, R. (2027). Integrating Computer Vision Models into SfM Photogrammetry: An Automated Pipeline for Occlusion Mask Generation. In: Gervasi, O., et al. Computational Science and Its Applications – ICCSA 2026 Workshops. ICCSA 2026. Lecture Notes in Computer Science, vol 16762. Springer, Cham. https://doi.org/10.1007/978-3-032-30524-4_8
