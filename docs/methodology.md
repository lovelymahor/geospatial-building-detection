# Project Methodology

## 1. Project Objective

This project develops and evaluates a building-footprint segmentation pipeline using satellite imagery from the SpaceNet 2 AOI 3 Paris dataset.

The methodology focuses on:

- Dataset validation
- Geospatial analysis
- Building mask generation
- Patch-based preprocessing
- Baseline segmentation
- Controlled loss-function experiments
- Quantitative evaluation
- Threshold analysis
- Error analysis
- Geographic performance analysis

The objective is to establish a reproducible benchmark and understand the model's strengths and failure cases.

---

## 2. Dataset

### Dataset

SpaceNet 2 Building Detection

### Area of Interest

AOI 3 Paris

### Dataset statistics

| Property | Value |
|---|---:|
| Total tiles | 1,148 |
| Training tiles | 802 |
| Validation tiles | 173 |
| Test tiles | 173 |
| Image dimensions | 650 × 650 |
| Image bands | 3 |
| Building polygons | 16,478 |
| Patch size | 256 × 256 |
| Patches per tile | 9 |

The imagery consists of RGB-PanSharpen GeoTIFF files and the building annotations are provided as GeoJSON polygons.

---

## 3. Dataset Validation

Image and annotation files were matched using their image identifiers.

Validation confirmed:

- 1,148 RGB images
- 1,148 GeoJSON annotation files
- No missing image-label pairs
- All images are 650 × 650 pixels
- All images contain three bands
- Image data type is uint16
- Source geographic CRS is EPSG:4326

Three invalid polygon geometries were detected.

They contained ring self-intersection errors and were repaired using Shapely `make_valid` during mask generation.

The original annotation files were not modified.

---

## 4. Building Mask Generation

Building polygons were converted into binary raster masks.

For each tile:

1. The RGB image dimensions were read.
2. The corresponding GeoJSON file was loaded.
3. Invalid geometries were repaired when required.
4. Polygon and MultiPolygon geometries were handled.
5. Building geometries were rasterized.
6. Building pixels were assigned value `1`.
7. Background pixels were assigned value `0`.

Dataset-wide mask analysis found:

- Total masks: 1,148
- Non-empty masks: 633
- Empty masks: 515
- Total building pixels: 33,607,696
- Mean raster building coverage: approximately 6.93%
- Median raster building coverage: approximately 0.79%
- Maximum raster building coverage: approximately 59.58%

---

## 5. Dataset Splitting

The dataset was divided at the tile level:

- Training: 802 tiles
- Validation: 173 tiles
- Test: 173 tiles

Each split contains both building-containing and empty tiles.

The split was performed before patch extraction to prevent patches from the same source tile from appearing across different splits.

The resulting split information is stored in:

`split.csv`

---

## 6. Image Normalization

Normalization statistics were calculated using training images only.

A sample of 5,000 pixels was collected from each training image, resulting in approximately 4.01 million sampled pixels.

Training-set statistics:

| Channel | Mean | Std |
|---|---:|---:|
| Red | 194.1930 | 102.4941 |
| Green | 263.0428 | 105.2618 |
| Blue | 238.3510 | 82.3202 |

Images were normalized using:

`(image - mean) / std`

The raw uint16 imagery was not divided by 255.

Normalization statistics are stored in:

`normalization.json`

---

## 7. Patch Extraction

The original images are 650 × 650 pixels.

A patch size of 256 × 256 was selected.

Patch starting coordinates were:

`[0, 256, 394]`

This produces:

`3 × 3 = 9 patches per tile`

The final patch position overlaps the preceding patch to ensure complete coverage of the 650 × 650 tile.

Total patches:

| Split | Tiles | Patches |
|---|---:|---:|
| Train | 802 | 7,218 |
| Validation | 173 | 1,557 |
| Test | 173 | 1,557 |

Training patches contained:

- 3,173 positive patches
- 4,045 empty patches

---

## 8. Data Augmentation

Training patches were augmented using:

- Horizontal flip
- Vertical flip
- Random 90-degree rotation

Validation and test data were not augmented.

The same preprocessing and augmentation configuration was used across the controlled model experiments.

---

## 9. Model Architecture

The segmentation model is a U-Net with a ResNet18 encoder.

Configuration:

- Architecture: U-Net
- Encoder: ResNet18
- Encoder weights: None
- Input channels: 3
- Output classes: 1
- Activation: raw logits

Total trainable parameters:

14,328,209

The model was trained from scratch without pretrained encoder weights.

---

## 10. Training Configuration

Common training configuration:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 16 |
| Patch size | 256 × 256 |
| Epochs | 10 |
| Device | CUDA GPU |
| Decision threshold | 0.5 |

The best checkpoint was selected using validation IoU.

---

## 11. Loss Functions

Three controlled experiments were performed.

### 11.1 Baseline BCE + Dice

The baseline loss combines:

- Binary Cross Entropy with logits
- Dice loss

`Loss = BCE + Dice`

The best validation checkpoint occurred at epoch 7.

---

### 11.2 Focal + Dice

The second experiment replaced BCE with focal loss.

Configuration:

- Alpha: 0.25
- Gamma: 2.0

`Loss = Focal + Dice`

The best validation checkpoint occurred at epoch 7.

---

### 11.3 Weighted BCE + Dice

The third experiment used positive-class weighting.

Configuration:

- Positive class weight: 2.0

`Loss = Weighted BCE + Dice`

The best validation checkpoint occurred at epoch 8.

All other major training variables were kept fixed so that the loss function was the primary experimental change.

---

## 12. Evaluation Metrics

The following pixel-level metrics were calculated:

### Intersection over Union

`IoU = TP / (TP + FP + FN)`

### Dice / F1

`Dice = 2TP / (2TP + FP + FN)`

### Precision

`Precision = TP / (TP + FP)`

### Recall

`Recall = TP / (TP + FN)`

False-positive rate and false-negative rate were also calculated during error analysis.

---

## 13. Full-Tile Reconstruction

Models operate on 256 × 256 patches, but the final evaluation was performed at the original 650 × 650 tile level.

For every test tile:

1. Nine patches were extracted.
2. Each patch was passed through the model.
3. Sigmoid probabilities were generated.
4. Overlapping predictions were accumulated.
5. Probability values in overlapping regions were averaged.
6. A binary mask was generated using the selected threshold.

This provides a complete prediction for each original tile.

---

## 14. Baseline Results

The baseline BCE + Dice model produced the following global test metrics:

| Metric | Value |
|---|---:|
| IoU | 0.5327 |
| Dice/F1 | 0.6951 |
| Precision | 0.6029 |
| Recall | 0.8207 |

The baseline produced relatively high recall while also generating a substantial number of false-positive pixels.

---

## 15. Focal + Dice Results

The Focal + Dice experiment produced:

| Metric | Value |
|---|---:|
| IoU | 0.5340 |
| Dice/F1 | 0.6963 |
| Precision | 0.7900 |
| Recall | 0.6224 |

Compared with the baseline, this experiment produced higher precision and lower recall.

---

## 16. Weighted BCE + Dice Results

The Weighted BCE + Dice experiment produced:

| Metric | Value |
|---|---:|
| IoU | 0.5631 |
| Dice/F1 | 0.7205 |
| Precision | 0.7194 |
| Recall | 0.7216 |

This experiment used a positive-class weight of 2.0 and was selected as the final model configuration for the benchmark.

---

## 17. Model Benchmark

| Model | Loss | Best Epoch | IoU | Dice/F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Baseline U-Net | BCE + Dice | 7 | 0.5327 | 0.6951 | 0.6029 | 0.8207 |
| U-Net | Focal + Dice | 7 | 0.5340 | 0.6963 | 0.7900 | 0.6224 |
| U-Net | Weighted BCE + Dice | 8 | 0.5631 | 0.7205 | 0.7194 | 0.7216 |

These metrics are global pixel-level test results obtained after full-tile reconstruction.

---

## 18. Threshold Analysis

Thresholds from 0.2 to 0.8 were evaluated.

For the baseline model:

| Threshold | IoU | Dice | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.20 | 0.4783 | 0.6471 | 0.5098 | 0.8853 |
| 0.30 | 0.5055 | 0.6716 | 0.5500 | 0.8621 |
| 0.40 | 0.5218 | 0.6858 | 0.5786 | 0.8417 |
| 0.50 | 0.5327 | 0.6951 | 0.6029 | 0.8207 |
| 0.60 | 0.5393 | 0.7007 | 0.6253 | 0.7968 |
| 0.70 | 0.5425 | 0.7034 | 0.6490 | 0.7678 |
| 0.80 | 0.5411 | 0.7022 | 0.6824 | 0.7232 |

For the weighted BCE + Dice model, threshold 0.5 produced the highest IoU and Dice among the tested thresholds.

Threshold calibration was therefore reported separately from the standard baseline result.

---

## 19. Error Analysis

For the 95 building-containing test tiles, the weighted BCE + Dice model showed:

- Mean false-positive rate: 4.61%
- Median false-positive rate: 4.31%
- Mean false-negative rate: 26.72%
- Median false-negative rate: 19.29%

Qualitative analysis identified several types of errors:

### Sparse scenes

Very small building regions can be missed because they occupy a small fraction of the image.

### False positives

The model sometimes labels visually similar non-building regions as buildings.

### Dense scenes

Some dense scenes contain substantial underprediction, even when predicted building pixels have relatively high precision.

### Coverage cancellation

Mean predicted building coverage can be close to ground-truth coverage even when individual pixels are incorrect. Therefore coverage difference alone is not sufficient for evaluating segmentation quality.

---

## 20. Building Density Analysis

Test tiles were grouped by ground-truth building coverage.

Baseline mean IoU:

| Building coverage | Mean IoU |
|---|---:|
| 0–1% | 0.2895 |
| 1–5% | 0.4067 |
| 5–10% | 0.4959 |
| 10–20% | 0.5437 |
| >20% | 0.5601 |

The baseline generally achieved higher IoU in tiles containing more building area.

The weighted BCE + Dice experiment improved IoU in the 0–1%, 1–5%, 5–10%, and 10–20% groups, while the >20% group showed a decrease.

The smallest density groups contain relatively few test tiles, so these group-level comparisons are descriptive.

---

## 21. Geospatial Analysis

### Geographic extent

All 1,148 annotation files were examined.

The geographic extent is approximately:

- Minimum longitude: 2.1892°
- Maximum longitude: 2.2998°
- Minimum latitude: 48.9778°
- Maximum latitude: 49.0574°

### Projected analysis

Building areas were calculated using:

**EPSG:32631 — UTM Zone 31N**

Results:

- Bounding-box width: approximately 8.07 km
- Bounding-box height: approximately 8.85 km
- Bounding-box area: approximately 71.43 km²
- Total building area: approximately 1.992 km²

A 1 km × 1 km density grid was created using projected building centroids.

---

## 22. Spatial Distribution

Train, validation, and test tile locations were mapped geographically.

All three splits cover essentially the same overall geographic extent within AOI 3 Paris.

This indicates that geographic extent alone does not reveal whether the splits have identical spatial distributions.

Spatial distribution and geographic IoU maps are included in:

`results/figures/geospatial/`

---

## 23. Geographic Correlation Analysis

For the 95 building-containing test tiles, Pearson correlations were calculated.

### Building coverage versus model performance

| Relationship | Pearson r |
|---|---:|
| Coverage vs IoU | 0.0635 |
| Coverage vs Dice | 0.0780 |
| Coverage vs Precision | 0.3516 |
| Coverage vs Recall | -0.0909 |

### Geographic coordinates versus IoU

| Relationship | Pearson r |
|---|---:|
| Longitude vs IoU | -0.0648 |
| Latitude vs IoU | -0.0671 |

These results do not indicate strong linear relationships between geographic coordinates and IoU within this AOI.

The analysis is limited to a single geographic region and should not be interpreted as evidence about performance across other cities or countries.

---

## 24. Reproducibility

The project stores:

- `splits.csv`
- `normalization.json`
- `patch_stats.csv`
- `training_config.json`
- Model benchmark metrics
- Threshold analysis
- Error analysis
- Building-density analysis
- Geospatial analysis
- Visualization outputs

Large source datasets, cached masks, and model checkpoints are excluded from version control.

---

## 25. Limitations

1. Only one SpaceNet AOI was evaluated.
2. The test set contains 173 tiles.
3. Geographic correlation analysis is limited to one region.
4. Some building-density groups contain relatively few samples.
5. The models were trained from scratch without pretrained encoder weights.
6. Evaluation focuses on semantic building segmentation rather than instance separation.
7. The benchmark does not establish cross-city or cross-sensor generalization.

---

## 26. Future Work

Potential extensions include:

- Evaluation on additional SpaceNet AOIs
- Cross-city generalization
- Pretrained encoder experiments
- Multi-scale architectures
- Boundary-aware segmentation losses
- Instance-level building detection
- Post-processing and connected-component analysis
- Systematic hyperparameter optimization
- Larger-scale geospatial validation

---

## 27. Final Configuration

The final benchmark configuration is:

| Parameter | Value |
|---|---|
| Dataset | SpaceNet 2 AOI 3 Paris |
| Architecture | U-Net |
| Encoder | ResNet18 |
| Loss | Weighted BCE + Dice |
| Positive class weight | 2.0 |
| Patch size | 256 × 256 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Epochs | 10 |
| Best epoch | 8 |
| Test threshold | 0.5 |
| Test IoU | 0.5631 |
| Test Dice/F1 | 0.7205 |
| Test Precision | 0.7194 |
| Test Recall | 0.7216 |
'''

print("Methodology file created successfully.")
``` id="v0s7ka"

Then verify:

```python id="a6n2cf"
!ls -lh /content/drive/MyDrive/geospatial-building-detection/docs/methodology.md
!git status --short
