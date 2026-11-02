# ML Coffee and Iris

RAS 585 assignment on clustering, density estimation, classification, and object recognition. The experiments live in two Jupyter notebooks under `scr/`, and the saved figures and tables are in `results/`.

`scr/main.ipynb` is the handwritten notebook. `scr/Assignment2_AI_Generated.ipynb` is the AI-generated version, with comments on where the two implementations differ.

## What it covers

**Image compression.** The scikit-image coffee photo is split into non-overlapping windows and compressed with single-linkage agglomerative clustering and with k-means. Both methods use cluster counts of 6, 10, and 14 and window sizes of 20, 50, and 80. Each run records a reconstruction and a compression rate.

**Iris.** A four-dimensional Gaussian kernel density estimate is fit on the setosa samples at bandwidths 1.5, 0.5, and 0.1. The full dataset is split 50/50 with stratified sampling, then classified with k-nearest neighbors (`k` of 3, 7, 11, and 15), a Gini classification tree, and linear discriminant analysis.

**Object recognition.** A pretrained YOLO26n model is run on `scr/Assignment 2_image.jpg`. The notebook also compares convolutional networks with multilayer perceptrons.

## Setup

Python 3.14 or newer. Dependencies are declared in `pyproject.toml`.

```bash
uv sync
uv run jupyter lab scr/main.ipynb
```

Run the notebooks with `scr/` as the working directory. Image and result paths are relative to that folder (`../results/`, `Assignment 2_image.jpg`).

`yolo26n.pt` is not stored in git. Ultralytics downloads it on the first YOLO run. The AI notebook looks for it at `../yolo26n.pt`.

## Layout

```
scr/main.ipynb                      handwritten experiments and write-up
scr/Assignment2_AI_Generated.ipynb  AI-generated experiments, with comparison comments
scr/Assignment 2_image.jpg          image used for YOLO
results/                            handwritten reconstructions and results.csv
results/ai/                         AI reconstructions and metric tables
runs/detect/                        annotated images saved by Ultralytics
```

The handwritten notebook writes JPEGs under `results/hierarchical/` and `results/kmeans/`. The AI notebook writes PNGs and its own CSVs under `results/ai/`.

## Why the two notebooks disagree

The same `k` and `c` do not always produce the same reconstruction.

- The handwritten code reflection-pads the coffee image so every pixel is kept. The AI code crops to a size divisible by the window, so a window of 80 drops the last 40 columns. Windows of 20 and 50 divide 400×600 evenly, so both notebooks see the same pixels there.
- Cluster means are stored differently. The handwritten code truncates the mean to an integer. The AI code rounds to the nearest integer. Hierarchical error at window sizes 20 and 50 differs only by that rounding.
- K-means is a different fit. The handwritten call uses scikit-learn’s default `n_init` (`"auto"`), which is one k-means++ start. The AI notebook sets `n_init=10` and keeps the best start. The same `random_state` does not make those identical.
