"""
======================================================================
FINAL PhD RECONSTRUCTION GALLERY
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------

Generate publication-quality reconstruction figures for the
trained Physics-Informed 3D Encoder-Decoder model.

Each figure contains five panels:

    1. Input / Corrupted Data
    2. Ground Truth
    3. Reconstructed Data
    4. Absolute Reconstruction Error
    5. Predictive Uncertainty

Predictive uncertainty is obtained from:

    Aleatoric uncertainty
        +
    Epistemic uncertainty from MC Dropout

The final predictive variance is:

    predictive variance
        =
    aleatoric variance
        +
    epistemic variance

The predictive standard deviation is:

    predictive std
        =
    sqrt(predictive variance)

Important
---------

The trained model is NOT retrained during gallery generation.

The trained checkpoint:

    best_model.pth

is loaded once and evaluated using the configured dataset.

MC Dropout is used to obtain stochastic reconstruction
predictions and stochastic log-variance predictions.

Observed seismic samples are restored exactly after
reconstruction to guarantee data consistency.

Author:
Ormin Joseph
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from dataset.build_dataset import build_dataset

from models.network import Network3D

from models.mc_dropout import MCDropout3D

from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator
)


# =====================================================================
# CONFIGURATION IMPORTS
# =====================================================================

from utils.config import (
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR,
    DEVICE,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    MC_DROPOUT_SAMPLES,
    OBSERVED_PRESERVATION_TOLERANCE,
    GALLERY_NUMBER_OF_SAMPLES,
    LOG_VARIANCE_MIN,
    LOG_VARIANCE_MAX,
)


# =====================================================================
# GLOBAL SETTINGS
# =====================================================================

# Output directory for the reconstruction gallery.
GALLERY_DIR = (
    Path(REPORT_DIR)
    / "gallery"
)

# Trained model checkpoint.
CHECKPOINT_FILE = (
    Path(CHECKPOINT_DIR)
    / "best_model.pth"
)

# Number of decimal places used in diagnostic output.
DISPLAY_DECIMALS = 6

# Figure resolution.
FIGURE_DPI = 300

# Slice axis.
#
# For the seismic tensor:
#
#     [C, D, H, W]
#
# axis 0 after selecting the channel corresponds to depth.
#
SLICE_AXIS = 0


# =====================================================================
# DEVICE RESOLUTION
# =====================================================================

def resolve_device():
    """
    Resolve the configured device.

    Configuration options:

        "auto"
        "cpu"
        "cuda"

    Returns
    -------

    torch.device
        Actual device used for inference.
    """

    configured_device = str(
        DEVICE
    ).strip().lower()

    # ---------------------------------------------------------------
    # Automatic device selection.
    # ---------------------------------------------------------------

    if configured_device == "auto":

        if torch.cuda.is_available():

            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    # ---------------------------------------------------------------
    # Explicit CPU.
    # ---------------------------------------------------------------

    if configured_device == "cpu":

        return torch.device(
            "cpu"
        )

    # ---------------------------------------------------------------
    # Explicit CUDA.
    # ---------------------------------------------------------------

    if configured_device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "DEVICE is configured as 'cuda', "
                "but CUDA is not available."
            )

        return torch.device(
            "cuda"
        )

    # ---------------------------------------------------------------
    # Unsupported device.
    # ---------------------------------------------------------------

    raise ValueError(
        f"Unsupported DEVICE configuration: {DEVICE}"
    )


# =====================================================================
# TENSOR VALIDATION
# =====================================================================

def validate_tensor(
    tensor,
    name,
    expected_ndim=None
):
    """
    Validate a tensor before reconstruction processing.

    Parameters
    ----------

    tensor : torch.Tensor
        Tensor to validate.

    name : str
        Human-readable tensor name.

    expected_ndim : int, optional
        Expected number of dimensions.
    """

    if not isinstance(
        tensor,
        torch.Tensor
    ):

        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if expected_ndim is not None:

        if tensor.ndim != expected_ndim:

            raise ValueError(
                f"{name} must have "
                f"{expected_ndim} dimensions. "
                f"Received {tuple(tensor.shape)}."
            )

    if not torch.isfinite(
        tensor
    ).all():

        raise ValueError(
            f"{name} contains NaN or Inf values."
        )


# =====================================================================
# CONVERT TENSOR TO NUMPY
# =====================================================================

def tensor_to_numpy(
    tensor
):
    """
    Convert a tensor to a NumPy array.

    Expected final form:

        [D,H,W]

    or:

        [H,W]
    """

    if isinstance(
        tensor,
        torch.Tensor
    ):

        tensor = tensor.detach().cpu()

    return tensor.numpy()


# =====================================================================
# EXTRACT DATASET SAMPLE
# =====================================================================

def extract_dataset_sample(
    sample
):
    """
    Extract corrupted input, ground truth and mask.

    Expected dataset convention:

        sample[0] = corrupted seismic input
        sample[1] = ground truth
        sample[2] = observation mask
    """

    if not isinstance(
        sample,
        (tuple, list)
    ):

        raise TypeError(
            "Dataset sample must be a tuple or list."
        )

    if len(sample) < 3:

        raise ValueError(
            "Dataset sample must contain at least "
            "input, target and mask."
        )

    corrupted = sample[0]

    target = sample[1]

    mask = sample[2]

    return (
        corrupted,
        target,
        mask
    )


# =====================================================================
# PREPARE MODEL INPUT
# =====================================================================

def prepare_model_input(
    tensor,
    device
):
    """
    Convert seismic tensor to model input format.

    Accepted input:

        [C,D,H,W]

    or:

        [B,C,D,H,W]

    Returned tensor:

        [B,C,D,H,W]
    """

    validate_tensor(
        tensor,
        "model input"
    )

    # ---------------------------------------------------------------
    # Add batch dimension when necessary.
    # ---------------------------------------------------------------

    if tensor.ndim == 4:

        tensor = tensor.unsqueeze(
            0
        )

    elif tensor.ndim != 5:

        raise ValueError(
            "Model input must have shape "
            "[C,D,H,W] or [B,C,D,H,W]. "
            f"Received {tuple(tensor.shape)}."
        )

    return tensor.to(
        device=device,
        dtype=torch.float32
    )


# =====================================================================
# PREPARE MASK
# =====================================================================

def prepare_mask(
    mask,
    device
):
    """
    Convert observation mask to:

        [B,C,D,H,W]
    """

    validate_tensor(
        mask,
        "observation mask"
    )

    # ---------------------------------------------------------------
    # Add batch dimension if required.
    # ---------------------------------------------------------------

    if mask.ndim == 4:

        mask = mask.unsqueeze(
            0
        )

    elif mask.ndim != 5:

        raise ValueError(
            "Mask must have shape "
            "[C,D,H,W] or [B,C,D,H,W]. "
            f"Received {tuple(mask.shape)}."
        )

    # ---------------------------------------------------------------
    # Convert to floating point.
    # ---------------------------------------------------------------

    mask = mask.to(
        device=device,
        dtype=torch.float32
    )

    # ---------------------------------------------------------------
    # Verify that the mask is binary.
    # ---------------------------------------------------------------

    unique_values = torch.unique(
        mask
    )

    for value in unique_values:

        if not (
            torch.isclose(
                value,
                torch.tensor(
                    0.0,
                    device=value.device
                )
            )
            or
            torch.isclose(
                value,
                torch.tensor(
                    1.0,
                    device=value.device
                )
            )
        ):

            raise ValueError(
                "Observation mask must contain only "
                "0 and 1 values."
            )

    return mask


# =====================================================================
# OBSERVED-DATA PRESERVATION
# =====================================================================

def enforce_observed_data_consistency(
    reconstruction,
    corrupted,
    mask
):
    """
    Restore observed seismic samples exactly.

    Convention:

        mask = 1 -> observed
        mask = 0 -> missing

    Therefore:

        reconstruction =
            reconstruction * (1-mask)
            +
            corrupted * mask
    """

    validate_tensor(
        reconstruction,
        "reconstruction"
    )

    validate_tensor(
        corrupted,
        "corrupted"
    )

    validate_tensor(
        mask,
        "mask"
    )

    if (
        reconstruction.shape
        !=
        corrupted.shape
    ):

        raise ValueError(
            "Reconstruction and corrupted input "
            "must have identical shapes."
        )

    if (
        reconstruction.shape
        !=
        mask.shape
    ):

        raise ValueError(
            "Reconstruction and mask "
            "must have identical shapes."
        )

    # ---------------------------------------------------------------
    # Projection onto the observed-data constraint.
    # ---------------------------------------------------------------

    projected = (
        reconstruction
        * (1.0 - mask)
        +
        corrupted
        * mask
    )

    return projected


# =====================================================================
# OBSERVED-DATA PRESERVATION ERROR
# =====================================================================

def calculate_observed_preservation_error(
    reconstruction,
    corrupted,
    mask
):
    """
    Calculate maximum absolute error on observed samples.

    A correctly projected reconstruction should produce:

        approximately 0

    """

    observed_difference = (
        torch.abs(
            reconstruction
            -
            corrupted
        )
        * mask
    )

    return float(
        observed_difference.max().item()
    )


# =====================================================================
# EXTRACT MC OUTPUTS
# =====================================================================

def extract_mc_outputs(
    mc_results
):
    """
    Extract stochastic outputs from MCDropout3D.

    Required keys:

        reconstruction_samples
        log_variance_samples
    """

    if not isinstance(
        mc_results,
        dict
    ):

        raise TypeError(
            "MC Dropout prediction must return a dictionary."
        )

    required_keys = [
        "reconstruction_samples",
        "log_variance_samples"
    ]

    for key in required_keys:

        if key not in mc_results:

            raise KeyError(
                f"MC Dropout output is missing required key: "
                f"{key}"
            )

    reconstruction_samples = (
        mc_results[
            "reconstruction_samples"
        ]
    )

    log_variance_samples = (
        mc_results[
            "log_variance_samples"
        ]
    )

    return (
        reconstruction_samples,
        log_variance_samples
    )


# =====================================================================
# MC PREDICTION
# =====================================================================

@torch.no_grad()
def run_mc_prediction(
    model,
    input_cube,
    mask,
    corrupted_cube,
    device
):
    """
    Run MC Dropout reconstruction and calculate
    predictive uncertainty.

    Returns
    -------

    dictionary containing:

        reconstruction
        predictive_std
        predictive_variance
        aleatoric_variance
        epistemic_variance
        aleatoric_std
        epistemic_std
        reconstruction_samples
        log_variance_samples
    """

    # ================================================================
    # CREATE MC DROPOUT ENGINE
    # ================================================================

    mc_predictor = MCDropout3D(
        model=model,
        num_samples=MC_DROPOUT_SAMPLES
    )

    # ================================================================
    # RUN MC DROPOUT
    # ================================================================

    mc_results = mc_predictor.predict(
        input_cube
    )

    # ================================================================
    # EXTRACT STOCHASTIC OUTPUTS
    # ================================================================

    (
        reconstruction_samples,
        log_variance_samples
    ) = extract_mc_outputs(
        mc_results
    )

    # ================================================================
    # VALIDATE MC OUTPUTS
    # ================================================================

    validate_tensor(
        reconstruction_samples,
        "reconstruction_samples",
        expected_ndim=6
    )

    validate_tensor(
        log_variance_samples,
        "log_variance_samples",
        expected_ndim=6
    )

    if (
        reconstruction_samples.shape
        !=
        log_variance_samples.shape
    ):

        raise ValueError(
            "Reconstruction samples and log-variance "
            "samples must have identical shapes."
        )

    # ================================================================
    # MEAN RECONSTRUCTION
    # ================================================================

    reconstruction_mean = (
        reconstruction_samples.mean(
            dim=0
        )
    )

    # ================================================================
    # PREDICTIVE UNCERTAINTY ESTIMATOR
    # ================================================================
    #
    # IMPORTANT:
    #
    # PredictiveUncertaintyEstimator is an nn.Module.
    #
    # Its uncertainty methods are instance methods.
    #
    # Therefore we instantiate the estimator and call it through
    # its forward() method.
    #
    # ================================================================

    uncertainty_estimator = (
        PredictiveUncertaintyEstimator(
            min_log_variance=LOG_VARIANCE_MIN,
            max_log_variance=LOG_VARIANCE_MAX
        )
    )

    uncertainty_estimator = (
        uncertainty_estimator.to(
            device
        )
    )

    # ================================================================
    # COMPLETE UNCERTAINTY DECOMPOSITION
    # ================================================================

    uncertainty_results = (
        uncertainty_estimator(
            log_variance_samples,
            reconstruction_samples
        )
    )

    # ================================================================
    # EXTRACT UNCERTAINTY COMPONENTS
    # ================================================================

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

    # ================================================================
    # ENFORCE OBSERVED-DATA CONSISTENCY
    # ================================================================

    reconstruction = (
        enforce_observed_data_consistency(
            reconstruction_mean,
            corrupted_cube,
            mask
        )
    )

    # ================================================================
    # OBSERVED-DATA PRESERVATION ERROR
    # ================================================================

    observed_error = (
        calculate_observed_preservation_error(
            reconstruction,
            corrupted_cube,
            mask
        )
    )

    # ================================================================
    # VALIDATE DATA CONSISTENCY
    # ================================================================

    if (
        observed_error
        >
        OBSERVED_PRESERVATION_TOLERANCE
    ):

        raise RuntimeError(
            "Observed-data preservation error exceeds "
            "the configured tolerance. "
            f"Error = {observed_error:.12e}, "
            f"Tolerance = "
            f"{OBSERVED_PRESERVATION_TOLERANCE:.12e}"
        )

    # ================================================================
    # RETURN COMPLETE RESULTS
    # ================================================================

    return {
        "reconstruction": reconstruction,

        "predictive_std": predictive_std,

        "predictive_variance": predictive_variance,

        "aleatoric_variance": aleatoric_variance,

        "epistemic_variance": epistemic_variance,

        "aleatoric_std": aleatoric_std,

        "epistemic_std": epistemic_std,

        "reconstruction_samples": (
            reconstruction_samples
        ),

        "log_variance_samples": (
            log_variance_samples
        ),

        "observed_preservation_error": (
            observed_error
        )
    }

# =====================================================================
# PREPARE VISUALIZATION SLICE
# =====================================================================

def prepare_visualization_slice(
    data,
    slice_index
):
    """
    Extract a 2D seismic slice from a 3D volume.

    Parameters
    ----------
    data : torch.Tensor or numpy.ndarray
        Seismic volume.

        Expected shape after removing singleton dimensions:

            [D,H,W]

    slice_index : int
        Index of the slice to extract.

    Returns
    -------
    numpy.ndarray
        2D visualization slice.
    """

    # ================================================================
    # HANDLE PYTORCH TENSOR
    # ================================================================

    if isinstance(
        data,
        torch.Tensor
    ):

        array = (
            data.detach()
            .cpu()
            .numpy()
        )

    # ================================================================
    # HANDLE NUMPY ARRAY
    # ================================================================

    elif isinstance(
        data,
        np.ndarray
    ):

        array = data

    # ================================================================
    # UNSUPPORTED TYPE
    # ================================================================

    else:

        raise TypeError(
            "Visualization data must be either "
            "torch.Tensor or numpy.ndarray. "
            f"Received: {type(data)}"
        )

    # ================================================================
    # REMOVE SINGLETON DIMENSIONS
    # ================================================================

    array = np.squeeze(
        array
    )

    # ================================================================
    # VERIFY 3D VOLUME
    # ================================================================

    if array.ndim != 3:

        raise ValueError(
            "Visualization data must represent "
            "a 3D seismic volume after squeezing. "
            f"Received shape: {array.shape}"
        )

    # ================================================================
    # EXTRACT REQUESTED SLICE
    # ================================================================

    if SLICE_AXIS == 0:

        return array[
            slice_index,
            :,
            :
        ]

    elif SLICE_AXIS == 1:

        return array[
            :,
            slice_index,
            :
        ]

    elif SLICE_AXIS == 2:

        return array[
            :,
            :,
            slice_index
        ]

    # ================================================================
    # INVALID SLICE AXIS
    # ================================================================

    raise ValueError(
        f"Unsupported SLICE_AXIS: {SLICE_AXIS}"
    )


# =====================================================================
# CREATE GALLERY FIGURE
# =====================================================================

def create_gallery_figure(
    corrupted,
    target,
    reconstruction,
    predictive_std,
    sample_index
):
    """
    Create the five-panel reconstruction gallery.

    Panels:

        1. Input / Corrupted Data
        2. Ground Truth
        3. Reconstruction
        4. Absolute Error
        5. Predictive Uncertainty
    """

    # ================================================================
    # CONVERT VOLUMES
    # ================================================================

    corrupted_array = np.squeeze(
        tensor_to_numpy(
            corrupted
        )
    )

    target_array = np.squeeze(
        tensor_to_numpy(
            target
        )
    )

    reconstruction_array = np.squeeze(
        tensor_to_numpy(
            reconstruction
        )
    )

    uncertainty_array = np.squeeze(
        tensor_to_numpy(
            predictive_std
        )
    )

    # ================================================================
    # VALIDATE VOLUME DIMENSIONS
    # ================================================================

    for name, array in [
        (
            "corrupted",
            corrupted_array
        ),
        (
            "target",
            target_array
        ),
        (
            "reconstruction",
            reconstruction_array
        ),
        (
            "predictive_std",
            uncertainty_array
        )
    ]:

        if array.ndim != 3:

            raise ValueError(
                f"{name} must be a 3D volume. "
                f"Received {array.shape}."
            )

    # ================================================================
    # CALCULATE ABSOLUTE ERROR
    # ================================================================

    absolute_error = np.abs(
        target_array
        -
        reconstruction_array
    )

    # ================================================================
    # SELECT CENTRAL SLICE
    # ================================================================

    if SLICE_AXIS == 0:

        slice_index = (
            target_array.shape[0]
            //
            2
        )

    elif SLICE_AXIS == 1:

        slice_index = (
            target_array.shape[1]
            //
            2
        )

    else:

        slice_index = (
            target_array.shape[2]
            //
            2
        )

    corrupted_slice = (
        prepare_visualization_slice(
            corrupted_array,
            slice_index
        )
    )

    target_slice = (
        prepare_visualization_slice(
            target_array,
            slice_index
        )
    )

    reconstruction_slice = (
        prepare_visualization_slice(
            reconstruction_array,
            slice_index
        )
    )

    error_slice = (
        prepare_visualization_slice(
            absolute_error,
            slice_index
        )
    )

    uncertainty_slice = (
        prepare_visualization_slice(
            uncertainty_array,
            slice_index
        )
    )

    # ================================================================
    # CREATE FIGURE
    # ================================================================

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4)
    )

    # ================================================================
    # PANEL 1 — CORRUPTED INPUT
    # ================================================================

    axes[0].imshow(
        corrupted_slice,
        cmap="gray",
        aspect="auto"
    )

    axes[0].set_title(
        "Input / Corrupted Data"
    )

    axes[0].set_xlabel(
        "Spatial axis"
    )

    axes[0].set_ylabel(
        "Depth"
    )

    # ================================================================
    # PANEL 2 — GROUND TRUTH
    # ================================================================

    axes[1].imshow(
        target_slice,
        cmap="gray",
        aspect="auto"
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[1].set_xlabel(
        "Spatial axis"
    )

    axes[1].set_ylabel(
        "Depth"
    )

    # ================================================================
    # PANEL 3 — RECONSTRUCTION
    # ================================================================

    axes[2].imshow(
        reconstruction_slice,
        cmap="gray",
        aspect="auto"
    )

    axes[2].set_title(
        "Reconstruction"
    )

    axes[2].set_xlabel(
        "Spatial axis"
    )

    axes[2].set_ylabel(
        "Depth"
    )

    # ================================================================
    # PANEL 4 — ABSOLUTE ERROR
    # ================================================================

    axes[3].imshow(
        error_slice,
        cmap="hot",
        aspect="auto"
    )

    axes[3].set_title(
        "Absolute Error"
    )

    axes[3].set_xlabel(
        "Spatial axis"
    )

    axes[3].set_ylabel(
        "Depth"
    )

    # ================================================================
    # PANEL 5 — PREDICTIVE UNCERTAINTY
    # ================================================================

    axes[4].imshow(
        uncertainty_slice,
        cmap="hot",
        aspect="auto"
    )

    axes[4].set_title(
        "Predictive Uncertainty"
    )

    axes[4].set_xlabel(
        "Spatial axis"
    )

    axes[4].set_ylabel(
        "Depth"
    )

    # ================================================================
    # MAIN FIGURE TITLE
    # ================================================================

    figure.suptitle(
        "Physics-Informed 3D Seismic Reconstruction "
        f"— Sample {sample_index + 1}",
        fontsize=14
    )

    # ================================================================
    # LAYOUT
    # ================================================================

    figure.tight_layout()

    return figure


# =====================================================================
# GENERATE GALLERY
# =====================================================================

def generate_gallery():
    """
    Generate the complete reconstruction gallery.
    """

    # ================================================================
    # HEADER
    # ================================================================

    print()
    print("=" * 70)
    print(
        "FINAL PhD RECONSTRUCTION GALLERY"
    )
    print("=" * 70)

    # ================================================================
    # RESOLVE DEVICE
    # ================================================================

    device = resolve_device()

    print()
    print(
        f"Experiment       : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Configured device: "
        f"{DEVICE}"
    )

    print(
        f"MC samples       : "
        f"{MC_DROPOUT_SAMPLES}"
    )

    print(
        f"Gallery samples  : "
        f"{GALLERY_NUMBER_OF_SAMPLES}"
    )

    print(
        f"Actual device    : "
        f"{device}"
    )

    # ================================================================
    # CHECKPOINT
    # ================================================================

    print()
    print(
        f"Checkpoint       : "
        f"{CHECKPOINT_FILE}"
    )

    if not CHECKPOINT_FILE.exists():

        raise FileNotFoundError(
            "The required checkpoint does not exist:\n"
            f"{CHECKPOINT_FILE}"
        )

    # ================================================================
    # BUILD DATASET
    # ================================================================

    print()
    print(
        "Building configured dataset..."
    )

    print()
    print(
        "============================================================"
    )

    print(
        "BUILDING DATASET"
    )

    print(
        "============================================================"
    )

    dataset = build_dataset()

    print()
    print(
        f"Dataset samples  : "
        f"{len(dataset)}"
    )

    # ================================================================
    # DETERMINE NUMBER OF GALLERY SAMPLES
    # ================================================================

    number_of_samples = min(
        int(
            GALLERY_NUMBER_OF_SAMPLES
        ),
        len(dataset)
    )

    if number_of_samples <= 0:

        raise ValueError(
            "GALLERY_NUMBER_OF_SAMPLES must be greater than zero."
        )

    # ================================================================
    # BUILD NETWORK
    # ================================================================

    print()
    print(
        "Building Network3D..."
    )

    print(
        f"Attention        : "
        f"{USE_ATTENTION}"
    )

    print(
        f"Residual         : "
        f"{USE_RESIDUAL}"
    )

    print(
        f"Uncertainty head : "
        f"{USE_UNCERTAINTY}"
    )

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY
    )

    model = model.to(
        device
    )

    # ================================================================
    # LOAD CHECKPOINT
    # ================================================================

    print()
    print(
        "Loading best_model.pth..."
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=device
    )

    # ---------------------------------------------------------------
    # Support checkpoint dictionaries containing model_state_dict.
    # ---------------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and (
        "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    print(
        "Model loaded successfully."
    )

    # ================================================================
    # CREATE GALLERY DIRECTORY
    # ================================================================

    GALLERY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        f"Gallery directory : "
        f"{GALLERY_DIR}"
    )

    # ================================================================
    # PROCESS SAMPLES
    # ================================================================

    for sample_index in range(
        number_of_samples
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"Processing sample "
            f"{sample_index + 1}/"
            f"{number_of_samples}"
        )

        print(
            "-" * 70
        )

        # ------------------------------------------------------------
        # Retrieve sample.
        # ------------------------------------------------------------

        sample = dataset[
            sample_index
        ]

        (
            corrupted,
            target,
            mask
        ) = extract_dataset_sample(
            sample
        )

        # ------------------------------------------------------------
        # Prepare tensors.
        # ------------------------------------------------------------

        corrupted_input = (
            prepare_model_input(
                corrupted,
                device
            )
        )

        target_input = (
            prepare_model_input(
                target,
                device
            )
        )

        observation_mask = (
            prepare_mask(
                mask,
                device
            )
        )

        # ------------------------------------------------------------
        # Validate shape consistency.
        # ------------------------------------------------------------

        if (
            corrupted_input.shape
            !=
            target_input.shape
        ):

            raise ValueError(
                "Corrupted input and ground truth "
                "must have identical shapes."
            )

        if (
            corrupted_input.shape
            !=
            observation_mask.shape
        ):

            raise ValueError(
                "Corrupted input and observation mask "
                "must have identical shapes."
            )

        # ------------------------------------------------------------
        # MC Dropout prediction.
        # ------------------------------------------------------------

        results = run_mc_prediction(
            model=model,
            input_cube=corrupted_input,
            mask=observation_mask,
            corrupted_cube=corrupted_input,
            device=device
        )

        # ------------------------------------------------------------
        # Extract final reconstruction.
        # ------------------------------------------------------------

        reconstruction = (
            results[
                "reconstruction"
            ]
        )

        # ------------------------------------------------------------
        # Extract predictive uncertainty.
        # ------------------------------------------------------------

        predictive_std = (
            results[
                "predictive_std"
            ]
        )

        # ------------------------------------------------------------
        # Report observed-data preservation.
        # ------------------------------------------------------------

        observed_error = (
            results[
                "observed_preservation_error"
            ]
        )

        print(
            "Observed-data preservation "
            f"error: "
            f"{observed_error:.12e}"
        )

        # ------------------------------------------------------------
        # Create figure.
        # ------------------------------------------------------------

        figure = create_gallery_figure(
            corrupted=corrupted_input,
            target=target_input,
            reconstruction=reconstruction,
            predictive_std=predictive_std,
            sample_index=sample_index
        )

        # ------------------------------------------------------------
        # Save figure.
        # ------------------------------------------------------------

        output_file = (
            GALLERY_DIR
            /
            f"sample_{sample_index:03d}.png"
        )

        figure.savefig(
            output_file,
            dpi=FIGURE_DPI,
            bbox_inches="tight"
        )

        plt.close(
            figure
        )

        print(
            f"Saved: {output_file}"
        )

    # ================================================================
    # COMPLETION
    # ================================================================

    print()
    print("=" * 70)

    print(
        "RECONSTRUCTION GALLERY COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print()
    print(
        f"Output directory:"
    )

    print(
        f"{GALLERY_DIR}"
    )

    print()


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    generate_gallery()