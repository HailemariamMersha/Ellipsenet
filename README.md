# EllipseNet

Final project for Applied Machine Learning, Spring 2026.

EllipseNet is a YOLO-style object detector that predicts oriented ellipses
instead of rectangular bounding boxes on Pascal VOC 2012.

## Main Files

- `EllipseNet_Complete/EllipseNet_Final.ipynb`  
  Main NumPy-from-scratch notebook.
- `EllipseNet_Complete/EllipseNet_Pretrained.ipynb`  
  Advanced version using a pretrained ResNet-18 backbone.
- `EllipseNet_Complete/EllipseNet_Report.pdf`  
  Final project report with mathematical details and results.
- `EllipseNet_Complete/EllipseNet_Report.tex`  
  LaTeX source for the report.

## Final NumPy Model

- Dataset: Pascal VOC 2012
- Classes: 20 VOC classes
- Input size: `224 x 224`
- Grid: `S = 7`
- Predictors per cell: `B = 2`
- Output per predictor:

```text
[objectness, cx, cy, a, b, theta, 20 class logits]
```

The output tensor per image is:

```text
7 x 7 x 2 x 26
```

The model is implemented with NumPy only, including convolution, pooling,
backpropagation, Adam updates, loss computation, ellipse IoU, NMS, evaluation,
and visualization.

## Project Features

- VOC bounding boxes converted to ellipse annotations
- Rotation augmentation with updated ellipse centers and angles
- YOLO-style grid prediction
- Ellipse-aware output head
- Combined localization, objectness, no-objectness, class, and angle losses
- Monte Carlo ellipse IoU for NMS and mAP evaluation
- Hyperparameter tests for learning rate and thresholds
- Overfitting sanity check
- Prediction visualizations
- Optional pretrained ResNet-18 backbone experiment

## Running

Install the minimal NumPy notebook dependencies:

```bash
python -m pip install -r EllipseNet_Complete/requirements.txt
```

Then open:

```bash
jupyter notebook EllipseNet_Complete/EllipseNet_Final.ipynb
```

The notebooks expect the Pascal VOC directory to contain:

```text
JPEGImages/
Annotations/
ImageSets/
```

For the pretrained notebook, install PyTorch and torchvision if needed:

```bash
python -m pip install torch torchvision
```

## Notes

Generated checkpoints, figures, and output logs are ignored by Git. The report
PDF includes the selected result figures.
