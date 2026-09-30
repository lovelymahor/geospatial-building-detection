# Geospatial Building Detection & Dataset Benchmark

A reproducible geospatial computer vision project for building footprint detection using the SpaceNet 2 AOI 3 Paris dataset.

## Project Overview

This project investigates building detection from high-resolution satellite imagery through:

- Dataset quality and geographic analysis
- GeoJSON building polygon validation
- Raster mask generation
- Patch-based dataset preparation
- U-Net semantic segmentation
- Controlled loss-function experiments
- Quantitative model evaluation
- Threshold analysis
- Error analysis
- Building-density analysis
- Geographic performance analysis

The project emphasizes reproducibility, quantitative benchmarking, and understanding model failure cases.

## Dataset

**Dataset:** SpaceNet 2 Building Detection  
**Area of Interest:** AOI 3 Paris

| Property | Value |
|---|---:|
| Total image tiles | 1,148 |
| Training tiles | 802 |
| Validation tiles | 173 |
| Test tiles | 173 |
| Image size | 650 × 650 |
| Image bands | 3 |
| Building polygons | 16,478 |
| Patch size | 256 × 256 |
| Patches per tile | 9 |

## Data Processing

The preprocessing pipeline includes:

1. Image and annotation pairing
2. GeoJSON geometry validation
3. Invalid geometry repair using `make_valid`
4. Rasterization of building polygons into binary masks
5. Training-only normalization
6. 256 × 256 patch extraction
7. Tile-level train/validation/test splitting
8. Training-time augmentation

Each 650 × 650 tile is divided into nine overlapping patches using starting coordinates:

`[0, 256, 394]`

## Model

**U-Net with ResNet18 encoder**

Configuration:

- Input channels: 3
- Output classes: 1
- Encoder: ResNet18
- Encoder pretrained weights: None
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 16
- Training epochs: 10
- Patch size: 256 × 256

## Model Benchmark

Three controlled loss-function experiments were evaluated.

| Model | Loss | Best Epoch | IoU | Dice/F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Baseline U-Net | BCE + Dice | 7 | 0.5327 | 0.6951 | 0.6029 | 0.8207 |
| U-Net | Focal + Dice | 7 | 0.5340 | 0.6963 | 0.7900 | 0.6224 |
| U-Net | Weighted BCE + Dice | 8 | **0.5631** | **0.7205** | 0.7194 | 0.7216 |

The weighted BCE + Dice experiment used a positive-class weight of `2.0`.

The reported test metrics are global pixel-level metrics calculated after reconstructing full-tile predictions from overlapping patches.

## Threshold Analysis

Threshold calibration was evaluated separately from the standard 0.5 threshold.

For the baseline model:

| Threshold | IoU | Dice/F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.2 | 0.4783 | 0.6471 | 0.5098 | 0.8853 |
| 0.3 | 0.5055 | 0.6716 | 0.5500 | 0.8621 |
| 0.4 | 0.5218 | 0.6858 | 0.5786 | 0.8417 |
| 0.5 | 0.5327 | 0.6951 | 0.6029 | 0.8207 |
| 0.6 | 0.5393 | 0.7007 | 0.6253 | 0.7968 |
| 0.7 | 0.5425 | 0.7034 | 0.6490 | 0.7678 |
| 0.8 | 0.5411 | 0.7022 | 0.6824 | 0.7232 |

For the weighted BCE + Dice model, threshold `0.5` produced the highest IoU and Dice/F1 among the evaluated thresholds.

## Error Analysis

Performance varied with building density.

### Baseline building-density analysis

| Building coverage | Mean IoU |
|---|---:|
| 0–1% | 0.2895 |
| 1–5% | 0.4067 |
| 5–10% | 0.4959 |
| 10–20% | 0.5437 |
| >20% | 0.5601 |

Sparse scenes were generally more difficult.

The weighted BCE + Dice experiment improved performance in most sparse and moderate-density groups while increasing precision. Recall decreased in several groups, particularly in the highest-density group.

Qualitative analysis identified:

- Missed buildings in sparse scenes
- False-positive segmentation
- Underprediction in some dense scenes
- High-precision but low-recall cases

## Geospatial Analysis

Dataset-wide geographic extent:

- Longitude: approximately 2.1892° to 2.2998°
- Latitude: approximately 48.9778° to 49.0574°

Building polygon statistics:

- 16,478 building polygons
- Approximately 1.992 km² total building area
- Geographic bounding-box area: approximately 71.43 km²
- Projected CRS for area analysis: EPSG:32631

A 1 km × 1 km building-density grid was generated.

The project also includes:

- Train/validation/test spatial distribution
- Geographic extent by split
- Geographic IoU distribution
- Building density versus model performance
- Geographic coordinate versus model performance

Within this AOI, linear correlations between geographic coordinates and IoU were weak.

## Repository Structure

```text
geospatial-building-detection/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_label_visualization.ipynb
│   ├── 03_geospatial_analysis.ipynb
│   ├── 04_baseline_training.ipynb
│   └── 05_model_evaluation.ipynb
├── src/
│   ├── data/
│   ├── dataset/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── visualization/
├── tests/
├── results/
│   ├── figures/
│   └── metrics/
└── docs/
    └── methodology.md
