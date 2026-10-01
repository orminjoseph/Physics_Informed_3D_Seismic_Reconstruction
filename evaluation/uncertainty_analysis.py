"""
======================================================================
Uncertainty Analysis
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate and visualize the predictive uncertainty produced
by the trained Physics-Informed 3D Encoder-Decoder model.

Uncertainty decomposition
--------------------------

    Predictive Variance
            |
            +---- Aleatoric Variance
            |
            +---- Epistemic Variance

Mathematically:

    sigma_predictive^2
        =
    sigma_aleatoric^2
        +
    sigma_epistemic^2

Aleatoric uncertainty
---------------------

Aleatoric uncertainty represents heteroscedastic uncertainty
predicted by the network through the log-variance output:

    sigma_aleatoric^2
        =
    mean(exp(log_variance_samples))

Epistemic uncertainty
---------------------

Epistemic uncertainty represents model uncertainty estimated
from Monte Carlo Dropout:

    sigma_epistemic^2
        =
    Var(reconstruction_samples)

Predictive uncertainty
----------------------

The total predictive uncertainty is obtained using the law
of total variance:

    sigma_predictive^2
        =
    sigma_aleatoric^2
        +
    sigma_epistemic^2

This module produces:

    1. uncertainty_analysis.png

The figure contains:

    - Mean reconstruction
    - Aleatoric uncertainty
    - Epistemic uncertainty
    - Predictive uncertainty

Important
---------

This module performs uncertainty decomposition and
visualization.

It does NOT perform:

    - formal uncertainty calibration
    - uncertainty-error correlation
    - ablation analysis
    - robustness analysis
    - statistical significance testing

Those analyses are handled by separate evaluation modules.

Author:
Ormin Joseph
======================================================================
"""


# ======================================================================
# IMPORTS
# ======================================================================

import os

import torch
import matplotlib.pyplot as plt

from models.network import Network3D

from models.mc_dropout import (
    MCDropout3D
)

from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator,
)

from dataset.build_dataset import (
    build_dataset
)

from utils.config import (
    CHECKPOINT_DIR,
    EXPERIMENT_NAME,
    LOG_VARIANCE_MIN,
    LOG_VARIANCE_MAX,
    REPORT_DIR,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    DEVICE as CONFIG_DEVICE,
    MC_DROPOUT_SAMPLES,
)


# ======================================================================
# OUTPUT PATHS
# ======================================================================

# Trained model checkpoint.
#
# The checkpoint directory is obtained from utils.config.py.
# Therefore this module does not hardcode:
#
#     outputs/synthetic_training/checkpoints
#
# directly.

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth",
)


# Directory for uncertainty-analysis outputs.
#
# REPORT_DIR is also controlled centrally through config.py.

OUTPUT_DIRECTORY = os.path.join(
    REPORT_DIR,
    "uncertainty",
)


# ======================================================================
# DEVICE RESOLUTION
# ======================================================================

def resolve_device():
    """
    Resolve the project's configured device.

    Configuration semantics
    -----------------------

    DEVICE = "cpu"
        Force CPU.

    DEVICE = "cuda"
        Require CUDA.

    DEVICE = "auto"
        Use CUDA when available; otherwise use CPU.

    Returns
    -------
    torch.device
        Resolved PyTorch device.
    """

    configured_device = str(
        CONFIG_DEVICE
    ).lower()

    # ------------------------------------------------------------------
    # Explicit CPU
    # ------------------------------------------------------------------

    if configured_device == "cpu":

        return torch.device(
            "cpu"
        )

    # ------------------------------------------------------------------
    # Explicit CUDA
    # ------------------------------------------------------------------

    if configured_device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "DEVICE='cuda' was requested, "
                "but CUDA is not available."
            )

        return torch.device(
            "cuda"
        )

    # ------------------------------------------------------------------
    # Automatic device selection
    # ------------------------------------------------------------------

    if configured_device == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # ------------------------------------------------------------------
    # Invalid configuration
    # ------------------------------------------------------------------

    raise ValueError(
        "Invalid DEVICE configuration: "
        f"{CONFIG_DEVICE!r}. "
        "Expected 'cpu', 'cuda', or 'auto'."
    )


# Resolve the actual runtime device once.
DEVICE = resolve_device()


# ======================================================================
# CONFIGURATION VALIDATION
# ======================================================================

def validate_configuration():
    """
    Validate uncertainty-analysis configuration.
    """

    # ------------------------------------------------------------------
    # MC-Dropout requires at least two stochastic forward passes.
    # ------------------------------------------------------------------

    if MC_DROPOUT_SAMPLES < 2:

        raise ValueError(
            "MC_DROPOUT_SAMPLES must be at least 2 "
            "for epistemic uncertainty estimation."
        )

    # ------------------------------------------------------------------
    # The uncertainty head must be enabled.
    # ------------------------------------------------------------------

    if not USE_UNCERTAINTY:

        raise RuntimeError(
            "USE_UNCERTAINTY=False. "
            "Predictive uncertainty analysis requires "
            "the network uncertainty head to be enabled."
        )

    # ------------------------------------------------------------------
    # Validate log-variance limits.
    # ------------------------------------------------------------------

    if LOG_VARIANCE_MIN >= LOG_VARIANCE_MAX:

        raise ValueError(
            "LOG_VARIANCE_MIN must be smaller than "
            "LOG_VARIANCE_MAX."
        )


# ======================================================================
# MODEL LOADING
# ======================================================================

def load_model():
    """
    Load the trained Physics-Informed 3D network.

    Returns
    -------
    model : torch.nn.Module
        Loaded trained network in evaluation mode.
    """

    # ------------------------------------------------------------------
    # Check checkpoint existence.
    # ------------------------------------------------------------------

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    # ------------------------------------------------------------------
    # Create the network using centralized architecture configuration.
    # ------------------------------------------------------------------

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
    )

    # ------------------------------------------------------------------
    # Load checkpoint.
    # ------------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    # ------------------------------------------------------------------
    # Identify the state dictionary.
    # ------------------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            # Support checkpoints containing the state
            # dictionary directly.
            state_dict = checkpoint

    else:

        raise TypeError(
            "Unsupported checkpoint format. "
            "Expected a dictionary containing "
            "'model_state_dict', 'state_dict', "
            "or the state dictionary itself."
        )

    # ------------------------------------------------------------------
    # Load trained parameters.
    # ------------------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    # ------------------------------------------------------------------
    # Move model to configured device.
    # ------------------------------------------------------------------

    model = model.to(
        DEVICE
    )

    # ------------------------------------------------------------------
    # Evaluation mode.
    #
    # MCDropout3D will selectively activate dropout during
    # stochastic prediction.
    # ------------------------------------------------------------------

    model.eval()

    return model


# ======================================================================
# INPUT VALIDATION
# ======================================================================

def validate_input_tensor(x):
    """
    Validate a seismic input tensor.

    Expected shape
    --------------

        [B, C, D, H, W]

    Parameters
    ----------
    x : torch.Tensor
        Seismic input tensor.
    """

    # ------------------------------------------------------------------
    # Tensor type.
    # ------------------------------------------------------------------

    if not isinstance(
        x,
        torch.Tensor,
    ):

        raise TypeError(
            "Input must be a torch.Tensor."
        )

    # ------------------------------------------------------------------
    # Tensor dimensionality.
    # ------------------------------------------------------------------

    if x.ndim != 5:

        raise ValueError(
            "Input must have shape "
            "[B, C, D, H, W]. "
            f"Received shape: {tuple(x.shape)}"
        )

    # ------------------------------------------------------------------
    # Numerical validity.
    # ------------------------------------------------------------------

    if not torch.isfinite(
        x
    ).all():

        raise ValueError(
            "Input contains NaN or infinite values."
        )


# ======================================================================
# UNCERTAINTY COMPUTATION
# ======================================================================

@torch.no_grad()
def compute_uncertainty(
    model,
    x,
    num_mc_samples=None,
):
    """
    Compute aleatoric, epistemic, and predictive uncertainty.

    Parameters
    ----------
    model : torch.nn.Module
        Trained seismic reconstruction model.

    x : torch.Tensor
        Input seismic tensor with shape:

            [B, C, D, H, W]

    num_mc_samples : int or None
        Number of MC Dropout forward passes.

        If None, the value from
        MC_DROPOUT_SAMPLES in utils.config.py
        is used.

    Returns
    -------
    dict
        Uncertainty decomposition results.
    """

    # ------------------------------------------------------------------
    # Use centralized MC-Dropout configuration.
    # ------------------------------------------------------------------

    if num_mc_samples is None:

        num_mc_samples = (
            MC_DROPOUT_SAMPLES
        )

    # ------------------------------------------------------------------
    # Validate input.
    # ------------------------------------------------------------------

    validate_input_tensor(
        x
    )

    # ------------------------------------------------------------------
    # Validate MC sample count.
    # ------------------------------------------------------------------

    if num_mc_samples < 2:

        raise ValueError(
            "num_mc_samples must be at least 2."
        )

    # ------------------------------------------------------------------
    # Move input to configured device.
    # ------------------------------------------------------------------

    x = x.to(
        DEVICE,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Create the project's MC-Dropout predictor.
    # ------------------------------------------------------------------

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=num_mc_samples,
    )

    # ------------------------------------------------------------------
    # Generate stochastic predictions.
    #
    # The authoritative MCDropout3D interface returns:
    #
    #     reconstruction_samples
    #     travel_time_samples
    #     log_variance_samples
    # ------------------------------------------------------------------

    mc_results = mc_dropout.predict(
        x
    )

    # ------------------------------------------------------------------
    # Validate returned object.
    # ------------------------------------------------------------------

    if not isinstance(
        mc_results,
        dict,
    ):

        raise TypeError(
            "MCDropout3D.predict() must return "
            "a dictionary."
        )

    # ------------------------------------------------------------------
    # Required outputs.
    # ------------------------------------------------------------------

    required_keys = {
        "reconstruction_samples",
        "travel_time_samples",
        "log_variance_samples",
    }

    missing_keys = (
        required_keys
        -
        set(mc_results.keys())
    )

    if missing_keys:

        raise KeyError(
            "MCDropout3D.predict() is missing "
            f"required outputs: {missing_keys}"
        )

    # ------------------------------------------------------------------
    # Extract MC outputs.
    # ------------------------------------------------------------------

    reconstruction_samples = (
        mc_results[
            "reconstruction_samples"
        ]
    )

    travel_time_samples = (
        mc_results[
            "travel_time_samples"
        ]
    )

    log_variance_samples = (
        mc_results[
            "log_variance_samples"
        ]
    )

    # ------------------------------------------------------------------
    # Expected MC sample shape.
    #
    # For:
    #
    #     x = [B,C,D,H,W]
    #
    # the MC tensors should be:
    #
    #     [N,B,C,D,H,W]
    #
    # where N = number of MC samples.
    # ------------------------------------------------------------------

    expected_sample_shape = (
        num_mc_samples,
        *x.shape,
    )

    # ------------------------------------------------------------------
    # Reconstruction shape validation.
    # ------------------------------------------------------------------

    if tuple(
        reconstruction_samples.shape
    ) != expected_sample_shape:

        raise ValueError(
            "Unexpected reconstruction sample shape.\n"
            f"Expected: {expected_sample_shape}\n"
            f"Received: "
            f"{tuple(reconstruction_samples.shape)}"
        )

    # ------------------------------------------------------------------
    # Log-variance shape validation.
    # ------------------------------------------------------------------

    if tuple(
        log_variance_samples.shape
    ) != expected_sample_shape:

        raise ValueError(
            "Unexpected log-variance sample shape.\n"
            f"Expected: {expected_sample_shape}\n"
            f"Received: "
            f"{tuple(log_variance_samples.shape)}"
        )

    # ------------------------------------------------------------------
    # Travel-time shape validation.
    # ------------------------------------------------------------------

    if tuple(
        travel_time_samples.shape
    ) != expected_sample_shape:

        raise ValueError(
            "Unexpected travel-time sample shape.\n"
            f"Expected: {expected_sample_shape}\n"
            f"Received: "
            f"{tuple(travel_time_samples.shape)}"
        )

    # ==================================================================
    # RECONSTRUCTION MEAN
    # ==================================================================

    reconstruction_mean = (
        reconstruction_samples.mean(
            dim=0
        )
    )

    # ==================================================================
    # PREDICTIVE UNCERTAINTY ESTIMATION
    # ==================================================================

    # IMPORTANT:
    #
    # PredictiveUncertaintyEstimator is the project's authoritative
    # implementation of the uncertainty decomposition.
    #
    # Its methods:
    #
    #     aleatoric_variance()
    #     epistemic_variance()
    #     predictive_variance()
    #
    # are INSTANCE METHODS.
    #
    # Therefore we must instantiate the estimator first.
    #
    # We deliberately do NOT calculate:
    #
    #     exp(log_variance)
    #
    # here manually.
    #
    # The estimator performs the canonical calculation, including
    # the centralized LOG_VARIANCE_MIN and LOG_VARIANCE_MAX limits.

    uncertainty_estimator = (
        PredictiveUncertaintyEstimator(
            min_log_variance=LOG_VARIANCE_MIN,
            max_log_variance=LOG_VARIANCE_MAX,
        )
    )

    # ------------------------------------------------------------------
    # Compute complete uncertainty decomposition.
    #
    # Input:
    #
    #     log_variance_samples
    #         [N,B,C,D,H,W]
    #
    #     reconstruction_samples
    #         [N,B,C,D,H,W]
    #
    # Output:
    #
    #     aleatoric_variance
    #     epistemic_variance
    #     predictive_variance
    #     aleatoric_std
    #     epistemic_std
    #     predictive_std
    # ------------------------------------------------------------------

    uncertainty_results = (
        uncertainty_estimator(
            log_variance_samples,
            reconstruction_samples,
        )
    )

    # ==================================================================
    # EXTRACT UNCERTAINTY COMPONENTS
    # ==================================================================

    aleatoric_variance = (
        uncertainty_results[
            "aleatoric_variance"
        ]
    )

    epistemic_variance = (
        uncertainty_results[
            "epistemic_variance"
        ]
    )

    predictive_variance = (
        uncertainty_results[
            "predictive_variance"
        ]
    )

    aleatoric_std = (
        uncertainty_results[
            "aleatoric_std"
        ]
    )

    epistemic_std = (
        uncertainty_results[
            "epistemic_std"
        ]
    )

    predictive_std = (
        uncertainty_results[
            "predictive_std"
        ]
    )

    # ==================================================================
    # NUMERICAL VALIDATION
    # ==================================================================

    tensors_to_check = {

        "reconstruction_mean":
            reconstruction_mean,

        "travel_time_mean":
            travel_time_samples.mean(
                dim=0
            ),

        "aleatoric_variance":
            aleatoric_variance,

        "epistemic_variance":
            epistemic_variance,

        "predictive_variance":
            predictive_variance,

        "aleatoric_std":
            aleatoric_std,

        "epistemic_std":
            epistemic_std,

        "predictive_std":
            predictive_std,
    }

    for name, tensor in (
        tensors_to_check.items()
    ):

        if not torch.isfinite(
            tensor
        ).all():

            raise ValueError(
                f"{name} contains NaN or "
                "infinite values."
            )

    # ==================================================================
    # NON-NEGATIVE VARIANCE VALIDATION
    # ==================================================================

    variance_tensors = {

        "aleatoric_variance":
            aleatoric_variance,

        "epistemic_variance":
            epistemic_variance,

        "predictive_variance":
            predictive_variance,
    }

    for name, tensor in (
        variance_tensors.items()
    ):

        if (
            tensor < 0
        ).any():

            raise ValueError(
                f"{name} contains negative values."
            )

    # ==================================================================
    # INDEPENDENT DECOMPOSITION CHECK
    # ==================================================================

    # Verify independently that:
    #
    # predictive variance
    #     =
    # aleatoric variance
    #     +
    # epistemic variance

    independent_predictive_variance = (
        aleatoric_variance
        +
        epistemic_variance
    )

    decomposition_difference = (
        torch.abs(
            predictive_variance
            -
            independent_predictive_variance
        )
    )

    maximum_decomposition_difference = float(
        decomposition_difference
        .max()
        .item()
    )

    # ==================================================================
    # RETURN RESULTS
    # ==================================================================

    return {

        "reconstruction_mean":
            reconstruction_mean
            .detach()
            .cpu(),

        "travel_time_mean":
            travel_time_samples
            .mean(dim=0)
            .detach()
            .cpu(),

        "reconstruction_samples":
            reconstruction_samples
            .detach()
            .cpu(),

        "travel_time_samples":
            travel_time_samples
            .detach()
            .cpu(),

        "log_variance_samples":
            log_variance_samples
            .detach()
            .cpu(),

        "aleatoric_variance":
            aleatoric_variance
            .detach()
            .cpu(),

        "epistemic_variance":
            epistemic_variance
            .detach()
            .cpu(),

        "predictive_variance":
            predictive_variance
            .detach()
            .cpu(),

        "aleatoric_std":
            aleatoric_std
            .detach()
            .cpu(),

        "epistemic_std":
            epistemic_std
            .detach()
            .cpu(),

        "predictive_std":
            predictive_std
            .detach()
            .cpu(),

        "maximum_decomposition_difference":
            maximum_decomposition_difference,

        "num_mc_samples":
            int(num_mc_samples),
    }


# ======================================================================
# REPRESENTATIVE SAMPLE PREPARATION
# ======================================================================

def prepare_input_cube(sample):
    """
    Extract and standardize the seismic input cube from
    one dataset sample.

    Expected final shape:

        [B, C, D, H, W]

    Parameters
    ----------
    sample
        Dataset sample.

    Returns
    -------
    torch.Tensor
        Five-dimensional input tensor.
    """

    # ------------------------------------------------------------------
    # The current dataset pipeline returns the input cube
    # as the first element of the sample.
    # ------------------------------------------------------------------

    input_cube = sample[0]

    # ------------------------------------------------------------------
    # Convert non-tensor data.
    # ------------------------------------------------------------------

    if not isinstance(
        input_cube,
        torch.Tensor,
    ):

        input_cube = torch.as_tensor(
            input_cube,
            dtype=torch.float32,
        )

    else:

        input_cube = input_cube.to(
            dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # Convert [D,H,W] to [1,1,D,H,W].
    # ------------------------------------------------------------------

    if input_cube.ndim == 3:

        input_cube = (
            input_cube
            .unsqueeze(0)
            .unsqueeze(0)
        )

    # ------------------------------------------------------------------
    # Convert [C,D,H,W] to [1,C,D,H,W].
    # ------------------------------------------------------------------

    elif input_cube.ndim == 4:

        input_cube = (
            input_cube
            .unsqueeze(0)
        )

    # ------------------------------------------------------------------
    # Already [B,C,D,H,W].
    # ------------------------------------------------------------------

    elif input_cube.ndim == 5:

        pass

    # ------------------------------------------------------------------
    # Invalid dimensionality.
    # ------------------------------------------------------------------

    else:

        raise ValueError(
            "Representative input cube must have "
            "3, 4, or 5 dimensions. "
            f"Received: {tuple(input_cube.shape)}"
        )

    # ------------------------------------------------------------------
    # Validate final tensor.
    # ------------------------------------------------------------------

    validate_input_tensor(
        input_cube
    )

    return input_cube


# ======================================================================
# VISUALIZATION
# ======================================================================

def visualize_uncertainty(
    results,
    output_path=None,
):
    """
    Visualize reconstruction and uncertainty components
    using a central depth slice.

    Four panels are produced:

        1. Mean Reconstruction
        2. Aleatoric Uncertainty
        3. Epistemic Uncertainty
        4. Predictive Uncertainty

    Parameters
    ----------
    results : dict
        Output dictionary from compute_uncertainty().

    output_path : str or None
        Optional output figure path.

    Returns
    -------
    str
        Saved figure path.
    """

    # ------------------------------------------------------------------
    # Extract tensors.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Expected shape:
    #
    #     [B,C,D,H,W]
    #
    # Select batch 0 and channel 0 explicitly.
    # ------------------------------------------------------------------

    if reconstruction.ndim != 5:

        raise ValueError(
            "Expected reconstruction to have shape "
            "[B,C,D,H,W]. "
            f"Received: {tuple(reconstruction.shape)}"
        )

    reconstruction_volume = (
        reconstruction[
            0,
            0
        ]
    )

    aleatoric_volume = (
        aleatoric_std[
            0,
            0
        ]
    )

    epistemic_volume = (
        epistemic_std[
            0,
            0
        ]
    )

    predictive_volume = (
        predictive_std[
            0,
            0
        ]
    )

    # ------------------------------------------------------------------
    # Validate 3D volumes.
    # ------------------------------------------------------------------

    if reconstruction_volume.ndim != 3:

        raise ValueError(
            "Expected 3D reconstruction volume."
        )

    # ------------------------------------------------------------------
    # Central depth slice.
    # ------------------------------------------------------------------

    depth_index = (
        reconstruction_volume.shape[0]
        // 2
    )

    reconstruction_slice = (
        reconstruction_volume[
            depth_index
        ]
        .numpy()
    )

    aleatoric_slice = (
        aleatoric_volume[
            depth_index
        ]
        .numpy()
    )

    epistemic_slice = (
        epistemic_volume[
            depth_index
        ]
        .numpy()
    )

    predictive_slice = (
        predictive_volume[
            depth_index
        ]
        .numpy()
    )

    # ------------------------------------------------------------------
    # Create output directory.
    # ------------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Default output path.
    # ------------------------------------------------------------------

    if output_path is None:

        output_path = os.path.join(
            OUTPUT_DIRECTORY,
            "uncertainty_analysis.png",
        )

    # ------------------------------------------------------------------
    # Create figure.
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    # ------------------------------------------------------------------
    # Mean reconstruction.
    # ------------------------------------------------------------------

    axes[0, 0].imshow(
        reconstruction_slice,
        aspect="auto",
    )

    axes[0, 0].set_title(
        "Mean Reconstruction"
    )

    axes[0, 0].set_xlabel(
        "Inline / Crossline"
    )

    axes[0, 0].set_ylabel(
        "Depth / Sample"
    )

    # ------------------------------------------------------------------
    # Aleatoric uncertainty.
    # ------------------------------------------------------------------

    axes[0, 1].imshow(
        aleatoric_slice,
        aspect="auto",
    )

    axes[0, 1].set_title(
        "Aleatoric Uncertainty (STD)"
    )

    axes[0, 1].set_xlabel(
        "Inline / Crossline"
    )

    axes[0, 1].set_ylabel(
        "Depth / Sample"
    )

    # ------------------------------------------------------------------
    # Epistemic uncertainty.
    # ------------------------------------------------------------------

    axes[1, 0].imshow(
        epistemic_slice,
        aspect="auto",
    )

    axes[1, 0].set_title(
        "Epistemic Uncertainty (STD)"
    )

    axes[1, 0].set_xlabel(
        "Inline / Crossline"
    )

    axes[1, 0].set_ylabel(
        "Depth / Sample"
    )

    # ------------------------------------------------------------------
    # Predictive uncertainty.
    # ------------------------------------------------------------------

    axes[1, 1].imshow(
        predictive_slice,
        aspect="auto",
    )

    axes[1, 1].set_title(
        "Predictive Uncertainty (STD)"
    )

    axes[1, 1].set_xlabel(
        "Inline / Crossline"
    )

    axes[1, 1].set_ylabel(
        "Depth / Sample"
    )

    # ------------------------------------------------------------------
    # Improve layout.
    # ------------------------------------------------------------------

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save figure.
    # ------------------------------------------------------------------

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    # ------------------------------------------------------------------
    # Close figure.
    # ------------------------------------------------------------------

    plt.close(
        fig
    )

    return output_path


# ======================================================================
# MAIN ANALYSIS
# ======================================================================

def analyze_uncertainty():
    """
    Execute the complete uncertainty-analysis pipeline.

    Steps
    -----

        1. Validate configuration.
        2. Build dataset.
        3. Load trained model.
        4. Select representative sample.
        5. Compute MC-Dropout uncertainty.
        6. Visualize uncertainty decomposition.
        7. Print summary statistics.
    """

    # ------------------------------------------------------------------
    # Validate configuration.
    # ------------------------------------------------------------------

    validate_configuration()

    # ------------------------------------------------------------------
    # Header.
    # ------------------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "UNCERTAINTY ANALYSIS"
    )

    print(
        "=" * 70
    )

    # IMPORTANT:
    #
    # Use EXPERIMENT_NAME rather than:
    #
    #     os.path.basename(REPORT_DIR)
    #
    # because REPORT_DIR ends in "reports", not the experiment name.

    print(
        f"Experiment       : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Device           : "
        f"{DEVICE}"
    )

    print(
        f"MC Samples       : "
        f"{MC_DROPOUT_SAMPLES}"
    )

    print(
        f"Checkpoint       : "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"Output Directory : "
        f"{OUTPUT_DIRECTORY}"
    )

    # ------------------------------------------------------------------
    # Build dataset.
    # ------------------------------------------------------------------

    print()

    print(
        "Building dataset..."
    )

    dataset = build_dataset()

    print(
        f"Dataset Length   : "
        f"{len(dataset)}"
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "Dataset is empty."
        )

    # ------------------------------------------------------------------
    # Load model.
    # ------------------------------------------------------------------

    print()

    print(
        "Loading trained model..."
    )

    model = load_model()

    print(
        "Model loaded successfully."
    )

    # ------------------------------------------------------------------
    # Select representative sample.
    # ------------------------------------------------------------------

    print()

    print(
        "Selecting representative sample..."
    )

    sample = dataset[0]

    input_cube = prepare_input_cube(
        sample
    )

    print(
        f"Input Shape     : "
        f"{tuple(input_cube.shape)}"
    )

    # ------------------------------------------------------------------
    # Move input to device.
    # ------------------------------------------------------------------

    input_cube = input_cube.to(
        DEVICE
    )

    # ------------------------------------------------------------------
    # Compute uncertainty.
    # ------------------------------------------------------------------

    print()

    print(
        "Computing MC-Dropout uncertainty..."
    )

    results = compute_uncertainty(
        model=model,
        x=input_cube,
        num_mc_samples=MC_DROPOUT_SAMPLES,
    )

    print(
        "Uncertainty decomposition "
        "completed successfully."
    )

    # ------------------------------------------------------------------
    # Visualization.
    # ------------------------------------------------------------------

    print()

    print(
        "Generating uncertainty visualization..."
    )

    figure_path = visualize_uncertainty(
        results
    )

    print(
        f"Saved: {figure_path}"
    )

    # ==================================================================
    # SUMMARY STATISTICS
    # ==================================================================

    aleatoric_variance_mean = float(
        results[
            "aleatoric_variance"
        ]
        .mean()
        .item()
    )

    epistemic_variance_mean = float(
        results[
            "epistemic_variance"
        ]
        .mean()
        .item()
    )

    predictive_variance_mean = float(
        results[
            "predictive_variance"
        ]
        .mean()
        .item()
    )

    aleatoric_std_mean = float(
        results[
            "aleatoric_std"
        ]
        .mean()
        .item()
    )

    epistemic_std_mean = float(
        results[
            "epistemic_std"
        ]
        .mean()
        .item()
    )

    predictive_std_mean = float(
        results[
            "predictive_std"
        ]
        .mean()
        .item()
    )

    maximum_difference = results[
        "maximum_decomposition_difference"
    ]

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print()

    print(
        "=" * 70
    )

    print(
        "UNCERTAINTY ANALYSIS SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Aleatoric Variance Mean   : "
        f"{aleatoric_variance_mean:.8f}"
    )

    print(
        f"Epistemic Variance Mean   : "
        f"{epistemic_variance_mean:.8f}"
    )

    print(
        f"Predictive Variance Mean  : "
        f"{predictive_variance_mean:.8f}"
    )

    print(
        f"Aleatoric STD Mean        : "
        f"{aleatoric_std_mean:.8f}"
    )

    print(
        f"Epistemic STD Mean        : "
        f"{epistemic_std_mean:.8f}"
    )

    print(
        f"Predictive STD Mean       : "
        f"{predictive_std_mean:.8f}"
    )

    print(
        f"MC Samples                : "
        f"{MC_DROPOUT_SAMPLES}"
    )

    print(
        f"Maximum Decomposition "
        f"Difference                : "
        f"{maximum_difference:.10e}"
    )

    print()

    print(
        "Output Files"
    )

    print(
        f"  Figure : "
        f"{figure_path}"
    )

    print(
        "=" * 70
    )

    # ------------------------------------------------------------------
    # Return analysis results.
    # ------------------------------------------------------------------

    return {
        "results":
            results,

        "figure_path":
            figure_path,
    }


# ======================================================================
# SCRIPT ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    analyze_uncertainty()