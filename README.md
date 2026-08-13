# Physics-Informed 3D Seismic Reconstruction

## Overview

This repository implements a **Physics-Informed 3D Encoder–Decoder Framework with Predictive Uncertainty for Seismic Data Reconstruction in Complex Geological Settings**.

The framework combines:

- 3D Encoder–Decoder Neural Networks
- Physics-Informed Constraints
- Monte Carlo Dropout Uncertainty Estimation
- Seismic Reconstruction Metrics
- Robustness Evaluation
- Statistical Validation
- Automated Thesis-Ready Reporting

The objective is to reconstruct missing seismic traces while quantifying prediction uncertainty and preserving geological structures.

---

## Research Objectives

1. Reconstruct missing seismic traces in 3D seismic volumes.
2. Incorporate physics-informed constraints into training.
3. Estimate predictive uncertainty.
4. Evaluate robustness under varying geological complexity, noise levels, and missing-data percentages.
5. Compare performance against conventional interpolation methods.
6. Generate thesis-ready tables, figures, and reports automatically.

---

## Framework Architecture

Input Seismic Volume
↓
3D Encoder
↓
Physics-Informed Bottleneck
↓
3D Decoder
↓
Reconstruction Output
↓
Predictive Uncertainty Estimation

---

## Project Structure

```text
dataset/
models/
losses/
trainer/
inference/
evaluation/
train/
test/
utils/
outputs/
```

---

## Installation

### Create virtual environment

```bash
python -m venv .venv
```

### Activate environment

Windows:

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Supported Datasets

### Synthetic Dataset

Generated automatically using:

- Horizontal Layers
- Dipping Layers
- Fault Structures
- Geological Complexity Variations

### F3 Netherlands Dataset

SEG Open Data Repository

SEGY format supported through:

- segyio

### Marmousi2 Dataset

Can be imported as an additional benchmark dataset.

---

## Training

### Synthetic Dataset

```bash
python -m train.train_model
```

### F3 Dataset

```bash
python -m train.train_f3
```

---

## Evaluation

### Quantitative Evaluation

```bash
python -m evaluation.evaluate_model
```

### Uncertainty Analysis

```bash
python -m evaluation.uncertainty_analysis
```

### Ablation Study

```bash
python -m evaluation.ablation_study
```

### Full Evaluation Pipeline

```bash
python -m evaluation.run_full_evaluation
```

---

## Automated Pipeline

Complete workflow:

### Step 1

Select dataset in:

```text
utils/config.py
```

### Step 2

Train model

```bash
python -m train.train_model
```

or

```bash
python -m train.train_f3
```

### Step 3

Run complete evaluation

```bash
python -m evaluation.run_full_evaluation
```

### Step 4

Generate final report

```bash
python -m evaluation.generate_final_report
```

---

## Evaluation Metrics

The framework reports:

- MAE
- RMSE
- PSNR
- SNR
- SSIM

Uncertainty metrics:

- Mean Predictive Uncertainty
- Uncertainty Calibration
- Uncertainty–Error Correlation

---

## Outputs

Generated automatically in:

```text
outputs/
```

Including:

```text
outputs/checkpoints/
outputs/reports/
outputs/figures/
outputs/reconstructions/
```

---

## Author

**Ormin Joseph**

- Lecturer, Department of Science Laboratory Technology
- Federal Polytechnic Bali, Taraba State, Nigeria
- Member, Nigerian Institute of Physics

---

## Research Topic

**Physics-Informed 3D Encoder–Decoder Framework with Predictive Uncertainty for Seismic Data Reconstruction in Complex Geological Settings**