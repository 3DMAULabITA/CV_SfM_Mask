# Integrating Computer Vision Models into SfM Photogrammetry: An Automated Pipeline for Occlusion Mask Generation


## Overview

SfM Masker is a modular pipeline designed to automatically detect and remove dynamic objects (e.g., people, vehicles) from photogrammetric datasets.  
It integrates state-of-the-art computer vision models (YOLOv8 and optional SAM2) to generate high-quality occlusion masks compatible with major SfM software.

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

Tavolare, R.: Integrating Computer Vision Models into SfM Photogrammetry: An Automated Pipeline for Occlusion Mask Generation. Under review for the 2026 International Conference on Computational Science and Its Applications (ICCSA 2026).
