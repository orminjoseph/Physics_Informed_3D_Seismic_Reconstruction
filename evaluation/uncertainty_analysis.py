"""
=========================================================
Uncertainty Analysis
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

This module evaluates and visualizes predictive uncertainty
for the trained seismic reconstruction model.

Uncertainty decomposition:

    Predictive Variance
        |
        +---- Aleatoric Variance
        |       |
        |       +---- Mean(exp(log_variance))
        |
        +---- Epistemic Variance
                |
                +---- MC Dropout Reconstruction Variance

Mathematically:

    sigma_predictive^2
        =
    sigma_aleatoric^2
        +
    sigma_epistemic^2

Aleatoric uncertainty:
    Model-predicted heteroscedastic observation uncertainty.

Epistemic uncertainty:
    Model uncertainty estimated from Monte Carlo Dropout.

This module produces:

1. uncertainty_analysis.png

The figure provides a visual representation of:

    - Mean reconstruction
    - Aleatoric uncertainty
    - Epistemic uncertainty
    - Predictive uncertainty

Author: Ormin Joseph
=========================================================
"""

import os

import torch
import matplotlib.pyplot as plt

from models.network import Network3D
from models.mc_dropout import MCDropout3D

from dataset.build_dataset import build_dataset

from utils.config import (
    CHECKPOINT_DIR,
    REPORT_DIR,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    DEVICE,
)


# =========================================================
# CONFIGURATION
# =========================================================

# Number of Monte Carlo Dropout forward passes.
NUM_MC_SAMPLES = 20

# Numerical stability constant.
EPSILON = 1e-8

# Log-variance safety limits.
LOG_VARIANCE_MIN = -10.0
LOG_VARIANCE_MAX = 10.0


# =========================================================
# OUTPUT PATHS
# =========================================================

# Trained model checkpoint.
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth",
)

# Directory for uncertainty figures.
OUTPUT_DIRECTORY = os.path.join(
    REPORT_DIR,
    "uncertainty",
)


# =========================================================
# DEVICE
# =========================================================

# Use configured device.
DEVICE = torch.device(DEVICE)


# =========================================================
# MODEL LOADING
# =========================================================

def load_model():
    """
    Load the trained Physics-Informed 3D network.

    Returns
    -------
    model : torch.nn.Module
        Loaded trained network in evaluation mode.
    """

    # Check that the checkpoint exists.
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    # Create the network using the same architecture
    # configuration used during training.
    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
    )

    # Load checkpoint onto the selected device.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    # Support the standard checkpoint structure.
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Assume the dictionary itself is the state dictionary.
            state_dict = checkpoint

    else:
        raise TypeError(
            "Unsupported checkpoint format."
        )

    # Load trained model parameters.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    # Move model to selected device.
    model = model.to(DEVICE)

    # Put model into evaluation mode.
    model.eval()

    return model


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_input_tensor(x):
    """
    Validate a seismic input tensor.

    Expected shape:

        [B, C, D, H, W]

    Parameters
    ----------
    x : torch.Tensor
        Input seismic tensor.
    """

    # Check tensor type.
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            "Input must be a torch.Tensor."
        )

    # Require five-dimensional 3D seismic input.
    if x.ndim != 5:
        raise ValueError(
            "Input must have shape [B, C, D, H, W]. "
            f"Received shape: {tuple(x.shape)}"
        )

    # Check for NaN or infinite values.
    if not torch.isfinite(x).all():
        raise ValueError(
            "Input contains NaN or infinite values."
        )


# =========================================================
# UNCERTAINTY COMPUTATION
# =========================================================

@torch.no_grad()
def compute_uncertainty(
    model,
    x,
    num_mc_samples=NUM_MC_SAMPLES,
):
    """
    Compute aleatoric, epistemic, and predictive uncertainty.

    Monte Carlo Dropout produces multiple stochastic
    reconstruction predictions.

    Parameters
    ----------
    model : torch.nn.Module
        Trained seismic reconstruction model.

    x : torch.Tensor
        Input seismic tensor with shape:

            [B, C, D, H, W]

    num_mc_samples : int
        Number of MC Dropout forward passes.

    Returns
    -------
    dict
        Dictionary containing uncertainty quantities.
    """

    # Validate the input.
    validate_input_tensor(x)

    # Validate MC sample count.
    if num_mc_samples < 2:
        raise ValueError(
            "num_mc_samples must be at least 2."
        )

    # Ensure the input is on the correct device.
    x = x.to(DEVICE)

    # -----------------------------------------------------
    # Monte Carlo Dropout
    # -----------------------------------------------------

    # Generate stochastic predictions.
    # -----------------------------------------------------
    # Monte Carlo Dropout
    # -----------------------------------------------------

    # Create the MC Dropout estimator.
    # The number of stochastic forward passes is supplied
    # through the constructor, not through predict().
    mc_dropout = MCDropout3D(
        model=model,
        num_samples=num_mc_samples,
    )

    # Generate stochastic predictions.
    mc_results = mc_dropout.predict(
        x
    )

    # Required outputs from MCDropout3D.
    required_keys = [
        "reconstruction_mean",
        "travel_time_mean",
        "reconstruction_samples",
        "log_variance_samples",
        "reconstruction_epistemic_variance",
        "log_variance_epistemic_variance",
    ]

    # Check that all required quantities are available.
    for key in required_keys:

        if key not in mc_results:
            raise KeyError(
                f"MCDropout3D.predict() did not return "
                f"required key: '{key}'"
            )

    # Extract reconstruction mean.
    reconstruction_mean = mc_results[
        "reconstruction_mean"
    ]

    # Extract travel-time mean.
    travel_time_mean = mc_results[
        "travel_time_mean"
    ]

    # Extract MC reconstruction samples.
    reconstruction_samples = mc_results[
        "reconstruction_samples"
    ]

    # Extract MC log-variance samples.
    log_variance_samples = mc_results[
        "log_variance_samples"
    ]

    # Extract epistemic reconstruction variance.
    reconstruction_epistemic_variance = mc_results[
        "reconstruction_epistemic_variance"
    ]

    # Extract epistemic log-variance diagnostic.
    log_variance_epistemic_variance = mc_results[
        "log_variance_epistemic_variance"
    ]

    # -----------------------------------------------------
    # Shape validation
    # -----------------------------------------------------

    if reconstruction_samples.ndim != 6:
        raise ValueError(
            "Reconstruction samples must have shape "
            "[N, B, C, D, H, W]. "
            f"Received: {tuple(reconstruction_samples.shape)}"
        )

    if log_variance_samples.ndim != 6:
        raise ValueError(
            "Log-variance samples must have shape "
            "[N, B, C, D, H, W]. "
            f"Received: {tuple(log_variance_samples.shape)}"
        )

    # -----------------------------------------------------
    # Log-variance stabilization
    # -----------------------------------------------------

    # Clamp predicted log variance before exponentiation.
    clamped_log_variance = torch.clamp(
        log_variance_samples,
        min=LOG_VARIANCE_MIN,
        max=LOG_VARIANCE_MAX,
    )

    # -----------------------------------------------------
    # Aleatoric variance
    # -----------------------------------------------------

    # For MC log-variance samples s_n:
    #
    #     sigma_a^2 = mean(exp(s_n))
    #
    # Variance is therefore averaged in variance space.
    aleatoric_variance = torch.mean(
        torch.exp(clamped_log_variance),
        dim=0,
    )

    # -----------------------------------------------------
    # Epistemic variance
    # -----------------------------------------------------

    # Epistemic uncertainty is represented by the
    # variance of the stochastic reconstruction predictions.
    epistemic_variance = reconstruction_epistemic_variance

    # -----------------------------------------------------
    # Predictive variance
    # -----------------------------------------------------

    # Law of total variance:
    #
    # predictive variance
    # =
    # aleatoric variance
    # +
    # epistemic variance
    predictive_variance = (
        aleatoric_variance
        +
        epistemic_variance
    )

    # -----------------------------------------------------
    # Numerical validation
    # -----------------------------------------------------

    tensors_to_check = {
        "reconstruction_mean": reconstruction_mean,
        "travel_time_mean": travel_time_mean,
        "aleatoric_variance": aleatoric_variance,
        "epistemic_variance": epistemic_variance,
        "predictive_variance": predictive_variance,
    }

    for name, tensor in tensors_to_check.items():

        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    # -----------------------------------------------------
    # Standard deviations
    # -----------------------------------------------------

    aleatoric_std = torch.sqrt(
        torch.clamp(
            aleatoric_variance,
            min=EPSILON,
        )
    )

    epistemic_std = torch.sqrt(
        torch.clamp(
            epistemic_variance,
            min=EPSILON,
        )
    )

    predictive_std = torch.sqrt(
        torch.clamp(
            predictive_variance,
            min=EPSILON,
        )
    )

    # -----------------------------------------------------
    # Decomposition consistency check
    # -----------------------------------------------------

    decomposition_difference = torch.abs(
        predictive_variance
        -
        (
            aleatoric_variance
            +
            epistemic_variance
        )
    )

    maximum_decomposition_difference = float(
        decomposition_difference.max().item()
    )

    # -----------------------------------------------------
    # Return results
    # -----------------------------------------------------

    return {
        "reconstruction_mean":
            reconstruction_mean.detach().cpu(),

        "travel_time_mean":
            travel_time_mean.detach().cpu(),

        "reconstruction_samples":
            reconstruction_samples.detach().cpu(),

        "log_variance_samples":
            log_variance_samples.detach().cpu(),

        "aleatoric_variance":
            aleatoric_variance.detach().cpu(),

        "epistemic_variance":
            epistemic_variance.detach().cpu(),

        "predictive_variance":
            predictive_variance.detach().cpu(),

        "aleatoric_std":
            aleatoric_std.detach().cpu(),

        "epistemic_std":
            epistemic_std.detach().cpu(),

        "predictive_std":
            predictive_std.detach().cpu(),

        "reconstruction_epistemic_variance":
            reconstruction_epistemic_variance.detach().cpu(),

        "log_variance_epistemic_variance":
            log_variance_epistemic_variance.detach().cpu(),

        "maximum_decomposition_difference":
            maximum_decomposition_difference,
    }


# =========================================================
# VISUALIZATION
# =========================================================

def visualize_uncertainty(
    results,
    output_path=None,
):
    """
    Visualize reconstruction and uncertainty components
    using a central depth slice.

    Four panels are produced:

        1. Mean Reconstruction
        2. Aleatoric STD
        3. Epistemic STD
        4. Predictive STD
    """

    # Extract required tensors.
    reconstruction = results[
        "reconstruction_mean"
    ]

    aleatoric_std = results[
        "aleatoric_std"
    ]

    epistemic_std = results[
        "epistemic_std"
    ]

    predictive_std = results[
        "predictive_std"
    ]

    # Remove batch/channel dimensions.
    reconstruction = reconstruction.squeeze()

    aleatoric_std = aleatoric_std.squeeze()

    epistemic_std = epistemic_std.squeeze()

    predictive_std = predictive_std.squeeze()

    # Confirm that the remaining volume is 3D.
    if reconstruction.ndim != 3:
        raise ValueError(
            "Expected 3D reconstruction after squeezing. "
            f"Received shape: {tuple(reconstruction.shape)}"
        )

    # Central depth index.
    depth_index = reconstruction.shape[0] // 2

    # Extract central depth slice.
    reconstruction_slice = (
        reconstruction[depth_index].numpy()
    )

    aleatoric_slice = (
        aleatoric_std[depth_index].numpy()
    )

    epistemic_slice = (
        epistemic_std[depth_index].numpy()
    )

    predictive_slice = (
        predictive_std[depth_index].numpy()
    )

    # Create output directory.
    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    # Use configured output path when provided.
    if output_path is None:
        output_path = os.path.join(
            OUTPUT_DIRECTORY,
            "uncertainty_analysis.png",
        )

    # -----------------------------------------------------
    # Create figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    # -----------------------------------------------------
    # Mean reconstruction
    # -----------------------------------------------------

    axes[0, 0].imshow(
        reconstruction_slice,
        aspect="auto",
    )

    axes[0, 0].set_title(
        "Mean Reconstruction"
    )

    axes[0, 0].set_xlabel("Inline / Crossline")
    axes[0, 0].set_ylabel("Sample")

    # -----------------------------------------------------
    # Aleatoric uncertainty
    # -----------------------------------------------------

    axes[0, 1].imshow(
        aleatoric_slice,
        aspect="auto",
    )

    axes[0, 1].set_title(
        "Aleatoric Uncertainty (STD)"
    )

    axes[0, 1].set_xlabel("Inline / Crossline")
    axes[0, 1].set_ylabel("Sample")

    # -----------------------------------------------------
    # Epistemic uncertainty
    # -----------------------------------------------------

    axes[1, 0].imshow(
        epistemic_slice,
        aspect="auto",
    )

    axes[1, 0].set_title(
        "Epistemic Uncertainty (STD)"
    )

    axes[1, 0].set_xlabel("Inline / Crossline")
    axes[1, 0].set_ylabel("Sample")

    # -----------------------------------------------------
    # Predictive uncertainty
    # -----------------------------------------------------

    axes[1, 1].imshow(
        predictive_slice,
        aspect="auto",
    )

    axes[1, 1].set_title(
        "Predictive Uncertainty (STD)"
    )

    axes[1, 1].set_xlabel("Inline / Crossline")
    axes[1, 1].set_ylabel("Sample")

    # Improve spacing.
    plt.tight_layout()

    # Save figure.
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    # Close figure to prevent memory accumulation.
    plt.close(fig)

    return output_path


# =========================================================
# MAIN ANALYSIS
# =========================================================

def analyze_uncertainty():
    """
    Complete uncertainty analysis pipeline.

    Steps:

        1. Build dataset.
        2. Load trained model.
        3. Select the first representative sample.
        4. Compute uncertainty decomposition.
        5. Save uncertainty visualization.
        6. Print uncertainty summary.
    """

    print()
    print("=" * 60)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 60)

    print(
        f"Experiment : "
        f"{os.path.basename(REPORT_DIR)}"
    )

    print(
        f"Device     : {DEVICE}"
    )

    print(
        f"Checkpoint : {CHECKPOINT_PATH}"
    )

    # -----------------------------------------------------
    # Build dataset
    # -----------------------------------------------------

    print()
    print("Building dataset...")

    dataset = build_dataset()

    print(
        f"Dataset Length: {len(dataset)}"
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty."
        )

    # -----------------------------------------------------
    # Load trained model
    # -----------------------------------------------------

    print()
    print("Loading trained model...")

    model = load_model()

    print(
        "Model loaded successfully."
    )

    # -----------------------------------------------------
    # Representative sample
    # -----------------------------------------------------

    # Use the first sample for detailed uncertainty
    # analysis and visualization.
    sample = dataset[0]

    # Extract input seismic cube.
    input_cube = sample[0]

    # Convert to tensor if necessary.
    if not isinstance(
        input_cube,
        torch.Tensor,
    ):
        input_cube = torch.as_tensor(
            input_cube,
            dtype=torch.float32,
        )

    # Add batch/channel dimensions when required.
    if input_cube.ndim == 3:
        input_cube = input_cube.unsqueeze(0).unsqueeze(0)

    elif input_cube.ndim == 4:
        input_cube = input_cube.unsqueeze(0)

    elif input_cube.ndim != 5:
        raise ValueError(
            "Representative input cube must have "
            "3, 4, or 5 dimensions."
        )

    # Move input to device.
    input_cube = input_cube.to(
        DEVICE,
        dtype=torch.float32,
    )

    # -----------------------------------------------------
    # Compute uncertainty
    # -----------------------------------------------------

    print()
    print(
        "Computing uncertainty decomposition..."
    )

    results = compute_uncertainty(
        model=model,
        x=input_cube,
        num_mc_samples=NUM_MC_SAMPLES,
    )

    # -----------------------------------------------------
    # Save visualization
    # -----------------------------------------------------

    figure_path = visualize_uncertainty(
        results
    )

    print()
    print(
        "Uncertainty visualization saved:"
    )

    print(
        figure_path
    )

    # -----------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------

    aleatoric_variance_mean = float(
        results[
            "aleatoric_variance"
        ].mean().item()
    )

    epistemic_variance_mean = float(
        results[
            "epistemic_variance"
        ].mean().item()
    )

    predictive_variance_mean = float(
        results[
            "predictive_variance"
        ].mean().item()
    )

    aleatoric_std_mean = float(
        results[
            "aleatoric_std"
        ].mean().item()
    )

    epistemic_std_mean = float(
        results[
            "epistemic_std"
        ].mean().item()
    )

    predictive_std_mean = float(
        results[
            "predictive_std"
        ].mean().item()
    )

    maximum_difference = results[
        "maximum_decomposition_difference"
    ]

    # -----------------------------------------------------
    # Print final summary
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("UNCERTAINTY ANALYSIS SUMMARY")
    print("=" * 60)

    print(
        f"Aleatoric Variance Mean  : "
        f"{aleatoric_variance_mean:.8f}"
    )

    print(
        f"Epistemic Variance Mean  : "
        f"{epistemic_variance_mean:.8f}"
    )

    print(
        f"Predictive Variance Mean : "
        f"{predictive_variance_mean:.8f}"
    )

    print(
        f"Aleatoric STD Mean       : "
        f"{aleatoric_std_mean:.8f}"
    )

    print(
        f"Epistemic STD Mean       : "
        f"{epistemic_std_mean:.8f}"
    )

    print(
        f"Predictive STD Mean      : "
        f"{predictive_std_mean:.8f}"
    )

    print(
        f"Maximum Decomposition "
        f"Difference               : "
        f"{maximum_difference:.10e}"
    )

    print()
    print(
        "Output files:"
    )

    print(
        f"  Figure : {figure_path}"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Return important outputs
    # -----------------------------------------------------

    return {
        "results":
            results,

        "figure_path":
            figure_path,
    }


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    analyze_uncertainty()