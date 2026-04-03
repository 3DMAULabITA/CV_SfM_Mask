"""
main.py
═══════════════════════════════════════════════════════════════
SfM Masker – Pipeline automatizzata per la generazione di
maschere di occlusione in workflow Structure-from-Motion

Autori: [INSERIRE AUTORI]
Versione: 1.0.0

Utilizzo
--------
  conda activate sfm_masker
  python main.py                          # usa config.yaml
  python main.py --config my_config.yaml  # configurazione custom
  python main.py --dataset ./MIEI_FOTO    # sovrascrive percorso dataset
  python main.py --no-sam2               # disabilita SAM2
  python main.py --platform metashape    # solo export Metashape

Struttura output
----------------
  OUTPUT/
  ├── MASKS/
  │   ├── <immagine>_mask.png            ← maschera generica
  │   ├── METASHAPE/<immagine>_mask.png  ← formato Metashape
  │   ├── ODM/<immagine>_mask.png        ← formato OpenDroneMap
  │   └── REALITYCAPTURE/<immagine>.mask.tif
  ├── PREVIEW/
  │   └── <immagine>_preview.jpg         ← immagine + overlay
  └── REPORT/
      ├── processing_report.json
      ├── processing_report.csv
      ├── chart_iqa.png
      ├── chart_segmentation.png
      ├── chart_masks.png
      └── dashboard.png
"""

import argparse
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeRemainingColumn, TimeElapsedColumn,
)
from rich.table import Table
from rich import box

# Moduli della pipeline
from modules.iqa          import IQAModule
from modules.segmentation import SegmentationModule
from modules.mask_export  import MaskExporter
from modules.report       import ReportGenerator

console = Console()

BANNER = """
 ███████╗███████╗███╗   ███╗    ███╗   ███╗ █████╗ ███████╗██╗  ██╗███████╗██████╗
 ██╔════╝██╔════╝████╗ ████║    ████╗ ████║██╔══██╗██╔════╝██║ ██╔╝██╔════╝██╔══██╗
 ███████╗█████╗  ██╔████╔██║    ██╔████╔██║███████║███████╗█████╔╝ █████╗  ██████╔╝
 ╚════██║██╔══╝  ██║╚██╔╝██║    ██║╚██╔╝██║██╔══██║╚════██║██╔═██╗ ██╔══╝  ██╔══██╗
 ███████║██║     ██║ ╚═╝ ██║    ██║ ╚═╝ ██║██║  ██║███████║██║  ██╗███████╗██║  ██║
 ╚══════╝╚═╝     ╚═╝     ╚═╝    ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
"""


# ── Utilità ──────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Carica il file YAML di configurazione."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_images(dataset_dir: Path, extensions: list[str]) -> list[Path]:
    """Raccoglie tutti i file immagine nella cartella dataset."""
    images = []
    for ext in extensions:
        images.extend(dataset_dir.glob(f"*{ext}"))
        images.extend(dataset_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def print_summary_table(iqa_results, seg_data, export_data):
    """Stampa una tabella riepilogativa a console."""
    table = Table(
        title="Riepilogo Elaborazione",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Immagine",   style="white",   max_width=30)
    table.add_column("IQA",        style="green",   justify="center")
    table.add_column("Nitidezza",  style="cyan",    justify="right")
    table.add_column("Oggetti",    style="yellow",  justify="center")
    table.add_column("Masch. %",   style="red",     justify="right")
    table.add_column("Tempo (ms)", style="magenta", justify="right")

    for i, ed in enumerate(export_data):
        iqa_r = iqa_results[i] if i < len(iqa_results) else {}
        seg_r = seg_data[i]    if i < len(seg_data)    else {}

        passed   = iqa_r.get("passed", True)
        iqa_icon = "[green]✓[/green]" if passed else "[red]✗[/red]"
        sharp    = f"{iqa_r.get('sharpness_score', 0):.1f}"
        n_obj    = str(ed.get("n_objects", 0))
        mpct     = f"{ed.get('masked_pct', 0):.1f}%"
        ms       = f"{seg_r.get('processing_ms', 0):.0f}"

        table.add_row(
            ed.get("filename", "")[:30],
            iqa_icon, sharp, n_obj, mpct, ms
        )

    console.print(table)


# ── Pipeline principale ───────────────────────────────────────────────────────

def run_pipeline(cfg: dict) -> None:
    """Esegue l'intera pipeline SfM Masker."""

    t_start = time.perf_counter()

    # ── Percorsi ─────────────────────────────────────────────────────
    dataset_dir   = Path(cfg["paths"]["dataset_dir"])
    output_dir    = Path(cfg["paths"]["output_dir"])
    masks_dir     = Path(cfg["paths"]["masks_dir"])
    preview_dir   = Path(cfg["paths"]["preview_dir"])
    report_dir    = Path(cfg["paths"]["report_dir"])

    # Crea cartelle di output
    for d in [output_dir, masks_dir, preview_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Raccolta immagini ────────────────────────────────────────────
    console.print(f"\n[bold cyan]📁 Dataset:[/bold cyan] {dataset_dir.resolve()}")

    if not dataset_dir.exists():
        console.print(f"[bold red]ERRORE: la cartella '{dataset_dir}' non esiste.[/bold red]")
        console.print("Crea la cartella DATASET e inserisci le immagini da elaborare.")
        sys.exit(1)

    images = collect_images(dataset_dir, cfg["image_extensions"])
    if not images:
        console.print("[bold red]ERRORE: nessuna immagine trovata nella cartella DATASET.[/bold red]")
        sys.exit(1)

    console.print(f"[bold green]✓[/bold green] {len(images)} immagini trovate\n")

    # ── Inizializzazione moduli ──────────────────────────────────────
    console.rule("[bold]Inizializzazione Moduli")

    # Modulo 1: IQA
    iqa_cfg = cfg["iqa"]
    iqa_module = IQAModule(
        sharpness_threshold   = iqa_cfg["sharpness_threshold"],
        underexposure_max_pct = iqa_cfg["underexposure_max_pct"],
        overexposure_max_pct  = iqa_cfg["overexposure_max_pct"],
        min_resolution_mp     = iqa_cfg["min_resolution_mp"],
    ) if iqa_cfg["enabled"] else None

    # Modulo 2: Segmentazione
    seg_cfg    = cfg["segmentation"]
    hw_cfg     = cfg["hardware"]
    seg_module = SegmentationModule(
        yolo_model           = seg_cfg["yolo_model"],
        confidence_threshold = seg_cfg["confidence_threshold"],
        iou_threshold        = seg_cfg["iou_threshold"],
        imgsz                = seg_cfg["imgsz"],
        target_classes       = seg_cfg["target_classes"],
        use_sam2             = seg_cfg["use_sam2"],
        sam2_checkpoint      = seg_cfg["sam2_checkpoint"],
        sam2_config          = seg_cfg["sam2_config"],
        device               = hw_cfg["device"],
    )

    # Modulo 3: Export
    morph_cfg   = cfg["morphology"]
    export_cfg  = cfg["export"]
    exporter    = MaskExporter(
        dilation_kernel_size = morph_cfg["dilation_kernel_size"],
        dilation_iterations  = morph_cfg["dilation_iterations"],
        min_component_area   = morph_cfg["min_component_area"],
        closing_kernel_size  = morph_cfg["closing_kernel_size"],
        platform             = export_cfg["platform"],
        generate_preview     = export_cfg["generate_preview"],
        preview_alpha        = export_cfg["preview_alpha"],
        preview_color        = export_cfg["preview_color"],
    )

    # Report
    rep_cfg   = cfg["report"]
    reporter  = ReportGenerator(report_dir, chart_dpi=rep_cfg["chart_dpi"])

    # ── Loop principale ──────────────────────────────────────────────
    console.rule("[bold]Elaborazione Immagini")

    iqa_results  = []    # list[dict]
    seg_data_all = []    # list[dict] – include detections per grafici
    export_data  = []    # list[dict]

    skip_failed  = iqa_cfg.get("skip_failed", False)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Elaborazione", total=len(images))

        for img_path in images:
            progress.update(task, description=f"[cyan]{img_path.name[:35]}")

            # ── Modulo 1: IQA ──────────────────────────────────────
            iqa_result_obj = None
            if iqa_module is not None:
                iqa_result_obj = iqa_module.evaluate(img_path)
                if iqa_result_obj:
                    iqa_results.append(iqa_result_obj.to_dict())
                    if skip_failed and not iqa_result_obj.passed:
                        console.print(
                            f"  [yellow]⚠ Saltata (IQA fallita): {img_path.name}[/yellow]"
                        )
                        progress.advance(task)
                        continue

            # ── Modulo 2: Segmentazione ────────────────────────────
            seg_result = seg_module.process_image(img_path)
            if seg_result is None:
                console.print(f"  [red]✗ Impossibile leggere: {img_path.name}[/red]")
                progress.advance(task)
                continue

            # Serializza detections per il report
            seg_dict = {
                "filename":      seg_result.filename,
                "n_objects":     seg_result.n_objects,
                "processing_ms": seg_result.processing_ms,
                "sam2_used":     seg_result.sam2_used,
                "masked_pct":    seg_result.total_masked_pct,
                "detections": [
                    {
                        "class_id":   d.class_id,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "area_pct":   d.area_pct,
                    }
                    for d in seg_result.detections
                ],
            }
            seg_data_all.append(seg_dict)

            # ── Modulo 3: Export maschera ──────────────────────────
            export_result = exporter.export(
                seg_result      = seg_result,
                original_path   = img_path,
                masks_output_dir= masks_dir,
                preview_dir     = preview_dir if export_cfg["generate_preview"] else None,
            )
            export_data.append(export_result)

            progress.advance(task)

    # ── Generazione report ───────────────────────────────────────────
    console.rule("[bold]Generazione Report e Grafici")

    total_time = time.perf_counter() - t_start

    report_outputs = reporter.generate_all(
        iqa_results  = iqa_results,
        seg_data     = seg_data_all,
        export_data  = export_data,
        total_time_s = total_time,
        fmt          = rep_cfg["format"],
    )

    # ── Stampa tabella riepilogativa ────────────────────────────────
    print_summary_table(iqa_results, seg_data_all, export_data)

    # ── Riepilogo finale ─────────────────────────────────────────────
    mins = int(total_time // 60)
    secs = int(total_time % 60)

    n_total_obj  = sum(r.get("n_objects", 0) for r in export_data)
    avg_masked   = (sum(r.get("masked_pct", 0) for r in export_data) /
                    max(len(export_data), 1))

    panel_content = (
        f"[green]✓[/green] Immagini elaborate:    [bold]{len(export_data)}[/bold]\n"
        f"[green]✓[/green] Oggetti rilevati:     [bold]{n_total_obj}[/bold]\n"
        f"[green]✓[/green] Area media mascherata: [bold]{avg_masked:.1f}%[/bold]\n"
        f"[green]✓[/green] Tempo totale:          [bold]{mins}m {secs:02d}s[/bold]\n\n"
        f"[cyan]Output salvato in:[/cyan] [bold]{output_dir.resolve()}[/bold]\n"
        f"  • Maschere:  {masks_dir.name}/\n"
        f"  • Preview:   {preview_dir.name}/\n"
        f"  • Report:    {report_dir.name}/\n"
        f"    ├─ processing_report.json\n"
        f"    ├─ processing_report.csv\n"
        f"    ├─ chart_iqa.png\n"
        f"    ├─ chart_segmentation.png\n"
        f"    ├─ chart_masks.png\n"
        f"    └─ dashboard.png"
    )
    console.print(Panel(panel_content, title="[bold green]Elaborazione Completata",
                        border_style="green"))


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SfM Masker – Pipeline automatizzata per maschere di occlusione SfM"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Percorso al file di configurazione YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Sovrascrive il percorso della cartella dataset"
    )
    parser.add_argument(
        "--output", default=None,
        help="Sovrascrive il percorso della cartella output"
    )
    parser.add_argument(
        "--platform", default=None,
        choices=["metashape", "odm", "realitycapture", "all"],
        help="Piattaforma SfM per l'export delle maschere"
    )
    parser.add_argument(
        "--no-sam2", action="store_true",
        help="Disabilita il raffinamento SAM2"
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Non generare le immagini di preview"
    )
    parser.add_argument(
        "--confidence", type=float, default=None,
        help="Sovrascrive la soglia di confidenza YOLO (0.0–1.0)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    console.print(BANNER, style="bold cyan")
    console.print("[bold]Pipeline per la Generazione Automatizzata di Maschere SfM[/bold]",
                  justify="center")
    console.print("Versione 1.0.0  |  Compatibile con Metashape, OpenDroneMap, RealityCapture\n",
                  justify="center", style="dim")

    args = parse_args()

    # Carica configurazione
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        console.print(f"[bold red]ERRORE: file di configurazione '{args.config}' non trovato.[/bold red]")
        sys.exit(1)

    # Override da argomenti CLI
    if args.dataset:
        cfg["paths"]["dataset_dir"] = args.dataset
        cfg["paths"]["output_dir"]  = str(Path(args.dataset).parent / "OUTPUT")
        cfg["paths"]["masks_dir"]   = str(Path(args.dataset).parent / "OUTPUT" / "MASKS")
        cfg["paths"]["preview_dir"] = str(Path(args.dataset).parent / "OUTPUT" / "PREVIEW")
        cfg["paths"]["report_dir"]  = str(Path(args.dataset).parent / "OUTPUT" / "REPORT")
    if args.output:
        cfg["paths"]["output_dir"]  = args.output
        cfg["paths"]["masks_dir"]   = str(Path(args.output) / "MASKS")
        cfg["paths"]["preview_dir"] = str(Path(args.output) / "PREVIEW")
        cfg["paths"]["report_dir"]  = str(Path(args.output) / "REPORT")
    if args.no_sam2:
        cfg["segmentation"]["use_sam2"] = False
    if args.no_preview:
        cfg["export"]["generate_preview"] = False
    if args.platform:
        cfg["export"]["platform"] = args.platform
    if args.confidence is not None:
        cfg["segmentation"]["confidence_threshold"] = args.confidence

    # Esegui pipeline
    run_pipeline(cfg)
