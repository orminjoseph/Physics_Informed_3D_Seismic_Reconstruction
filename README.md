# Physics-Informed 3D Seismic Reconstruction

Physics-Informed 3D Encoder–Decoder Framework with Predictive Uncertainty for Seismic Data Reconstruction in Complex Geological Settings.

## Installation

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

## Training

Synthetic dataset:

python -m train.train_model

F3 dataset:

python -m train.train_f3

## Evaluation

python -m evaluation.run_full_evaluation

## Project Structure

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