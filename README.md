# EllipseNet

Applied Machine Learning final project, Spring 2026.

EllipseNet is a from-scratch NumPy object detector for Pascal VOC 2012 that
predicts oriented ellipses instead of rectangular boxes. The current submission
focuses on a YOLOv1-style detector: a `448 x 448` image is mapped to a
`7 x 7 x 32` prediction tensor, where each grid cell predicts two ellipse
hypotheses and one shared 20-class VOC distribution.

Repository: <https://github.com/HailemariamMersha/Ellipsenet>

## Submission Files

- `EllipseNet_Complete/EllipseNet_YOLOV1_Submission.ipynb`  
  Main submission notebook. It contains the complete NumPy implementation,
  executed experiment cells, figures, and analysis.

- `EllipseNet_Complete/EllipseNet_YOLOV1_Submission_Report.pdf`  
  Final report PDF for submission.

- `EllipseNet_Complete/EllipseNet_Final.ipynb`  
  Earlier polished NumPy detector notebook kept for reference.

- `EllipseNet_Complete/EllipseNet_Clean.ipynb` and
  `EllipseNet_Complete/EllipseNet_Clean2.ipynb`  
  Development notebooks retained as intermediate experiment history.

- `EllipseNet_Complete/requirements.txt`  
  Minimal runtime dependencies for the NumPy notebooks.

## Model Summary

The submission model keeps the core YOLOv1 design:

```text
Input image:          448 x 448 x 3
Grid:                 S = 7
Predictors per cell:  B = 2
Classes:              C = 20 Pascal VOC classes
Output per cell:      B * 6 + C = 32
Output tensor:        7 x 7 x 32
```

Each predictor outputs:

```text
cx, cy, a, b, theta, confidence
```

where `cx, cy` are the normalized ellipse center, `a, b` are normalized
semi-axes, `theta` is the ellipse orientation in radians, and `confidence` is
the model's learned objectness/localization-quality estimate. The class logits
are shared by the grid cell, matching the YOLOv1-style output layout.

## What Is Implemented

- Pascal VOC XML parsing and bounding-box-to-ellipse annotation conversion.
- On-the-fly rotation augmentation with updated ellipse centers and angles.
- NumPy-only neural network layers:
  - convolution,
  - `1 x 1` convolution bottlenecks,
  - max pooling,
  - dense layers,
  - LeakyReLU,
  - manual backward passes.
- Adam optimizer implemented directly over NumPy arrays.
- YOLOv1-style grid assignment and responsible-predictor selection.
- Ellipse-aware loss with localization, confidence, no-objectness,
  classification, and orientation terms.
- Ellipse IoU approximation for evaluation and non-maximum suppression.
- mAP evaluation with confidence-threshold sweeps.
- Hyperparameter search, small-set training, and overfit sanity check.
- Qualitative panels for correct predictions, missed predictions, and false
  positives.

## Dataset Layout

The notebooks expect a Pascal VOC 2012 directory containing:

```text
JPEGImages/
Annotations/
ImageSets/
```

The local default path used in the notebooks is:

```text
../archive/VOC2012_train_val/VOC2012_train_val
```

The `archive/` directory is intentionally ignored by Git because the dataset is
large. If your VOC copy is somewhere else, update `VOC_SEARCH_ROOT` in the first
configuration cell of the notebook.

## Running the Notebook

Install the minimal dependencies:

```bash
python -m pip install -r EllipseNet_Complete/requirements.txt
```

Then open the submission notebook:

```bash
jupyter notebook EllipseNet_Complete/EllipseNet_YOLOV1_Submission.ipynb
```

The default workflow runs the small staged experiments rather than the expensive
full-dataset run:

```python
RUN_HPARAM_SEARCH = True
RUN_SMALL_TRAIN = True
RUN_OVERFIT_TRAIN = True
RUN_FULL_TRAIN = False
```

To run the full Pascal VOC training stage, set:

```python
RUN_FULL_TRAIN = True
```

The full run is configured for:

```python
FULL_EPOCHS_FINAL = 30
```

## Experiments

The submission notebook includes:

- Hyperparameter search over learning rate and selected loss weights.
- A selected baseline small-set run.
- A 20-image overfit sanity check to verify the training pipeline.
- Threshold sweeps for detection confidence.
- Saved qualitative visualizations for:
  - correct predictions,
  - missed predictions,
  - false positives.

Generated experiment artifacts are written under:

```text
EllipseNet_Complete/checkpoints_yolov1_final/
EllipseNet_Complete/figures/
EllipseNet_Complete/outputs/
```

These generated folders are ignored by Git to avoid committing large or
machine-specific experiment output. The important figures and results are
already embedded in the notebook and final PDF report.

## Notes for Grading

The submission notebook and report are self-contained for review. The PDF
contains the final write-up with architecture details, formulas, results, and
figures. The notebook contains the executable implementation and the recorded
experiment outputs used to generate the report.

The full-dataset and pretrained-backbone experiments are computationally
expensive on a laptop and may continue separately from the staged submission
workflow. The completed staged experiments demonstrate the full detection
pipeline end to end.
