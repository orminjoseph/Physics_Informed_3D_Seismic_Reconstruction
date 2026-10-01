"""
====================================================================
Geological Complexity Robustness Evaluation
====================================================================

Physics-Informed 3D Encoder-Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate the robustness of the trained reconstruction model under
different levels of geological complexity.

Geological complexity levels
----------------------------
1. horizontal
2. dipping
3. faulted
4. folded
5. complex
6. highly_complex

Important methodological principles
------------------------------------
1. The trained model is evaluated without retraining.
2. The same reconstruction model is used across all geological
   complexity levels.
3. Observed seismic samples are restored exactly after inference.
4. Reconstruction metrics are obtained from the canonical
   metrics/reconstruction_metrics.py module.
5. Aleatoric uncertainty is obtained from the model's predicted
   log-variance.
6. Epistemic and predictive uncertainty are NOT fabricated during
   deterministic inference. They require MC-Dropout evaluation.
7. All configurable parameters are obtained from utils/config.py.
8. Results are saved under the configured REPORT_DIR.

====================================================================
"""

# ====================================================================
# 1. STANDARD LIBRARY IMPORTS
# ====================================================================

import json
import math
import random
from pathlib import Path

# ====================================================================
# 2. THIRD-PARTY IMPORTS
# ====================================================================

import numpy as np
import torch
from tqdm import tqdm

# ====================================================================
# 3. PROJECT IMPORTS
# ====================================================================

from utils.config import (
    DATASET_MODE,
    DEVICE,
    CHECKPOINT_DIR,
    REPORT_DIR,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    LOG_VARIANCE_MIN,
    LOG_VARIANCE_MAX,
)

from models.network import Network3D

from dataset.build_dataset import build_dataset

from metrics.reconstruction_metrics import (
    calculate_reconstruction_metrics,
    missing_mae,
    missing_rmse,
    observed_mae,
    observed_rmse,
)


# ====================================================================
# 4. REPRODUCIBILITY
# ====================================================================

SEED = 42


def set_seed(seed: int = SEED):
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====================================================================
# 5. DEVICE RESOLUTION
# ====================================================================

def resolve_device(device_setting):
    """
    Convert the configuration device setting into a valid
    torch.device.

    Supported settings
    ------------------
    cpu
    cuda
    auto

    Returns
    -------
    torch.device
    """

    if isinstance(device_setting, torch.device):
        return device_setting

    device_setting = str(device_setting).lower()

    if device_setting == "cpu":
        return torch.device("cpu")

    if device_setting == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE='cuda' was requested, but CUDA is not available."
            )

        return torch.device("cuda")

    if device_setting == "auto":

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    raise ValueError(
        f"Unsupported DEVICE setting: {device_setting}"
    )


# ====================================================================
# 6. GEOLOGICAL COMPLEXITY LEVELS
# ====================================================================

GEOLOGICAL_COMPLEXITY_LEVELS = [
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
]


# ====================================================================
# 7. CHECKPOINT
# ====================================================================

CHECKPOINT_PATH = (
    Path(CHECKPOINT_DIR)
    / "best_model.pth"
)


# ====================================================================
# 8. OUTPUT DIRECTORY
# ====================================================================

OUTPUT_DIR = (
    Path(REPORT_DIR)
    / "geological_complexity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ====================================================================
# 9. LOAD MODEL
# ====================================================================

def load_model(device):
    """
    Load the trained Physics-Informed 3D Encoder-Decoder model.

    Parameters
    ----------
    device : torch.device
        Device on which the model will operate.

    Returns
    -------
    torch.nn.Module
        Loaded model in evaluation mode.
    """

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    # ---------------------------------------------------------------
    # Construct model using centralized configuration.
    # ---------------------------------------------------------------

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
    )

    # ---------------------------------------------------------------
    # Load checkpoint.
    # ---------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    # ---------------------------------------------------------------
    # Support the standard project checkpoint format.
    # ---------------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            # Some checkpoints may contain the state dictionary
            # directly.
            state_dict = checkpoint

    else:

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    # ---------------------------------------------------------------
    # Load learned parameters.
    # ---------------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True
    )

    # ---------------------------------------------------------------
    # Move model to selected device.
    # ---------------------------------------------------------------

    model.to(device)

    # ---------------------------------------------------------------
    # Deterministic evaluation mode.
    # ---------------------------------------------------------------

    model.eval()

    return model


# ====================================================================
# 10. MODEL OUTPUT EXTRACTION
# ====================================================================

def extract_model_outputs(model_output):
    """
    Extract reconstruction, travel-time and log-variance outputs
    from the project Network3D output convention.

    Expected production convention
    -------------------------------
    reconstruction
    travel_time
    log_variance

    Parameters
    ----------
    model_output
        Output returned by Network3D.

    Returns
    -------
    reconstruction
    travel_time
    log_variance
    """

    # ---------------------------------------------------------------
    # Dictionary output.
    # ---------------------------------------------------------------

    if isinstance(model_output, dict):

        reconstruction = model_output.get(
            "reconstruction"
        )

        travel_time = model_output.get(
            "travel_time"
        )

        log_variance = model_output.get(
            "log_variance"
        )

        if reconstruction is None:

            raise RuntimeError(
                "Model output dictionary does not contain "
                "'reconstruction'."
            )

        return (
            reconstruction,
            travel_time,
            log_variance
        )

    # ---------------------------------------------------------------
    # Tuple/list output.
    # ---------------------------------------------------------------

    if isinstance(model_output, (tuple, list)):

        if len(model_output) >= 3:

            reconstruction = model_output[0]

            travel_time = model_output[1]

            log_variance = model_output[2]

            return (
                reconstruction,
                travel_time,
                log_variance
            )

        if len(model_output) == 2:

            reconstruction = model_output[0]

            log_variance = model_output[1]

            return (
                reconstruction,
                None,
                log_variance
            )

        if len(model_output) == 1:

            return (
                model_output[0],
                None,
                None
            )

    # ---------------------------------------------------------------
    # Tensor-only output.
    # ---------------------------------------------------------------

    if torch.is_tensor(model_output):

        return (
            model_output,
            None,
            None
        )

    raise RuntimeError(
        "Unsupported Network3D output format."
    )


# ====================================================================
# 11. TENSOR STANDARDIZATION
# ====================================================================

def ensure_5d(tensor, name):
    """
    Convert a seismic tensor to [B,C,D,H,W].

    Accepted input shapes
    ---------------------
    [D,H,W]
    [C,D,H,W]
    [B,C,D,H,W]

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    name : str
        Tensor name used in error messages.

    Returns
    -------
    torch.Tensor
        Tensor in [B,C,D,H,W] format.
    """

    if tensor is None:
        return None

    if tensor.ndim == 3:

        return tensor.unsqueeze(0).unsqueeze(0)

    if tensor.ndim == 4:

        return tensor.unsqueeze(0)

    if tensor.ndim == 5:

        return tensor

    raise ValueError(
        f"{name} must have 3, 4 or 5 dimensions, "
        f"but received shape {tuple(tensor.shape)}."
    )


# ====================================================================
# 12. DATA CONSISTENCY PROJECTION
# ====================================================================

def apply_data_consistency(
    reconstruction,
    observed_input,
    mask
):
    """
    Restore observed seismic samples exactly.

    mask convention
    ---------------
    1 = observed
    0 = missing

    Formula
    -------
    reconstructed = reconstruction * (1-mask)
                    + observed_input * mask

    Parameters
    ----------
    reconstruction : torch.Tensor
        Model reconstruction.

    observed_input : torch.Tensor
        Corrupted/observed seismic input.

    mask : torch.Tensor
        Observation mask.

    Returns
    -------
    torch.Tensor
        Data-consistent reconstruction.
    """

    return (
        reconstruction * (1.0 - mask)
        + observed_input * mask
    )


# ====================================================================
# 13. OBSERVED-DATA PRESERVATION ERROR
# ====================================================================

def observed_preservation_error(
    reconstruction,
    observed_input,
    mask
):
    """
    Calculate maximum absolute error on observed samples.

    This should ideally be zero after data-consistency projection.
    """

    observed = mask > 0.5

    if not torch.any(observed):

        return 0.0

    difference = torch.abs(
        reconstruction[observed]
        - observed_input[observed]
    )

    return float(
        torch.max(difference).detach().cpu()
    )


# ====================================================================
# 14. ALEATORIC UNCERTAINTY
# ====================================================================

def calculate_aleatoric_uncertainty(log_variance):
    """
    Convert predicted log-variance into aleatoric variance and
    standard deviation.

    The log-variance is clipped using the centralized configuration
    values LOG_VARIANCE_MIN and LOG_VARIANCE_MAX.

    Parameters
    ----------
    log_variance : torch.Tensor
        Predicted log variance.

    Returns
    -------
    aleatoric_variance : torch.Tensor
    aleatoric_std : torch.Tensor
    """

    if log_variance is None:

        return None, None

    # ---------------------------------------------------------------
    # Protect against numerical instability.
    # ---------------------------------------------------------------

    log_variance = torch.clamp(
        log_variance,
        min=LOG_VARIANCE_MIN,
        max=LOG_VARIANCE_MAX
    )

    # ---------------------------------------------------------------
    # Convert log variance to variance.
    # ---------------------------------------------------------------

    aleatoric_variance = torch.exp(
        log_variance
    )

    # ---------------------------------------------------------------
    # Convert variance to standard deviation.
    # ---------------------------------------------------------------

    aleatoric_std = torch.sqrt(
        aleatoric_variance
    )

    return (
        aleatoric_variance,
        aleatoric_std
    )


# ====================================================================
# 15. CASE METRICS
# ====================================================================

def calculate_case_metrics(
    prediction,
    target,
    mask
):
    """
    Calculate all canonical reconstruction metrics.

    The canonical metric implementation is:

        metrics/reconstruction_metrics.py

    Parameters
    ----------
    prediction : torch.Tensor
        Reconstructed seismic volume.

    target : torch.Tensor
        Ground-truth seismic volume.

    mask : torch.Tensor
        Observation mask.

    Returns
    -------
    dict
        Reconstruction metrics.
    """

    # ---------------------------------------------------------------
    # Standard reconstruction metrics.
    # ---------------------------------------------------------------

    metrics = calculate_reconstruction_metrics(
        prediction=prediction,
        target=target
    )

    # ---------------------------------------------------------------
    # Missing-region metrics.
    # ---------------------------------------------------------------

    metrics["Missing_MAE"] = float(
        missing_mae(
            prediction,
            target,
            mask
        )
    )

    metrics["Missing_RMSE"] = float(
        missing_rmse(
            prediction,
            target,
            mask
        )
    )

    # ---------------------------------------------------------------
    # Observed-region metrics.
    # ---------------------------------------------------------------

    metrics["Observed_MAE"] = float(
        observed_mae(
            prediction,
            target,
            mask
        )
    )

    metrics["Observed_RMSE"] = float(
        observed_rmse(
            prediction,
            target,
            mask
        )
    )

    return metrics


# ====================================================================
# 16. EXTRACT SAMPLE FROM DATASET
# ====================================================================

def extract_sample(dataset, index):
    """
    Extract a sample from the project dataset.

    The function supports the expected dictionary-based dataset
    representation as well as tuple/list samples.

    Parameters
    ----------
    dataset
        Dataset object.

    index : int
        Sample index.

    Returns
    -------
    corrupted
    target
    mask
    velocity
    """

    sample = dataset[index]

    # ---------------------------------------------------------------
    # Dictionary representation.
    # ---------------------------------------------------------------

    if isinstance(sample, dict):

        corrupted = sample.get(
            "input",
            sample.get("corrupted")
        )

        target = sample.get(
            "target"
        )

        mask = sample.get(
            "mask"
        )

        velocity = sample.get(
            "velocity"
        )

        if corrupted is None:
            raise RuntimeError(
                "Dataset sample does not contain input/corrupted data."
            )

        if target is None:
            raise RuntimeError(
                "Dataset sample does not contain target data."
            )

        if mask is None:
            raise RuntimeError(
                "Dataset sample does not contain mask data."
            )

        return (
            corrupted,
            target,
            mask,
            velocity
        )

    # ---------------------------------------------------------------
    # Tuple/list representation.
    # ---------------------------------------------------------------

    if isinstance(sample, (tuple, list)):

        if len(sample) < 3:

            raise RuntimeError(
                "Dataset sample must contain at least "
                "input, target and mask."
            )

        corrupted = sample[0]

        target = sample[1]

        mask = sample[2]

        velocity = (
            sample[3]
            if len(sample) > 3
            else None
        )

        return (
            corrupted,
            target,
            mask,
            velocity
        )

    raise RuntimeError(
        "Unsupported dataset sample format."
    )


# ====================================================================
# 17. EVALUATE ONE GEOLOGICAL COMPLEXITY LEVEL
# ====================================================================

@torch.no_grad()
def evaluate_geological_level(
    model,
    device,
    geological_mode
):
    """
    Evaluate the model for one geological complexity level.

    Parameters
    ----------
    model : torch.nn.Module
        Trained reconstruction model.

    device : torch.device
        Computation device.

    geological_mode : str
        Geological complexity level.

    Returns
    -------
    list[dict]
        Per-sample evaluation results.
    """

    set_seed(SEED)

    # ---------------------------------------------------------------
    # Build dataset.
    # ---------------------------------------------------------------
    #
    # The dataset builder remains the project's central dataset
    # selection mechanism.
    #
    # The geological_mode is passed through where supported.
    # ---------------------------------------------------------------

    try:

        dataset = build_dataset(
            geological_mode=geological_mode
        )

    except TypeError:

        # -----------------------------------------------------------
        # Some versions of build_dataset may obtain geological
        # configuration from the configuration module rather than
        # accepting it as an argument.
        #
        # We deliberately fail clearly rather than silently creating
        # a different dataset.
        # -----------------------------------------------------------

        raise TypeError(
            "build_dataset() does not accept geological_mode. "
            "Update dataset.build_dataset.build_dataset() so that "
            "geological complexity evaluation can explicitly request "
            f"'{geological_mode}'."
        )

    # ---------------------------------------------------------------
    # Number of samples.
    # ---------------------------------------------------------------

    num_samples = len(dataset)

    if num_samples == 0:

        raise RuntimeError(
            f"No samples available for geological mode "
            f"'{geological_mode}'."
        )

    results = []

    # ---------------------------------------------------------------
    # Evaluate each sample.
    # ---------------------------------------------------------------

    for sample_index in tqdm(
        range(num_samples),
        desc=f"Geology: {geological_mode}"
    ):

        corrupted, target, mask, velocity = extract_sample(
            dataset,
            sample_index
        )

        # -----------------------------------------------------------
        # Convert to tensors.
        # -----------------------------------------------------------

        if not torch.is_tensor(corrupted):
            corrupted = torch.as_tensor(
                corrupted,
                dtype=torch.float32
            )

        if not torch.is_tensor(target):
            target = torch.as_tensor(
                target,
                dtype=torch.float32
            )

        if not torch.is_tensor(mask):
            mask = torch.as_tensor(
                mask,
                dtype=torch.float32
            )

        # -----------------------------------------------------------
        # Standardize shapes to [B,C,D,H,W].
        # -----------------------------------------------------------

        corrupted = ensure_5d(
            corrupted,
            "corrupted"
        )

        target = ensure_5d(
            target,
            "target"
        )

        mask = ensure_5d(
            mask,
            "mask"
        )

        # -----------------------------------------------------------
        # Move tensors to device.
        # -----------------------------------------------------------

        corrupted = corrupted.to(device)

        target = target.to(device)

        mask = mask.to(device)

        # -----------------------------------------------------------
        # Basic validation.
        # -----------------------------------------------------------

        if corrupted.shape != target.shape:

            raise ValueError(
                "Corrupted input and target shapes differ: "
                f"{tuple(corrupted.shape)} vs "
                f"{tuple(target.shape)}"
            )

        if corrupted.shape != mask.shape:

            raise ValueError(
                "Corrupted input and mask shapes differ: "
                f"{tuple(corrupted.shape)} vs "
                f"{tuple(mask.shape)}"
            )

        if not torch.isfinite(corrupted).all():

            raise ValueError(
                "Corrupted input contains NaN or Inf."
            )

        if not torch.isfinite(target).all():

            raise ValueError(
                "Target contains NaN or Inf."
            )

        if not torch.isfinite(mask).all():

            raise ValueError(
                "Mask contains NaN or Inf."
            )

        # -----------------------------------------------------------
        # Validate mask.
        # -----------------------------------------------------------

        unique_mask_values = torch.unique(mask)

        invalid_mask = ~(
            torch.isclose(
                unique_mask_values,
                torch.tensor(
                    0.0,
                    device=device
                )
            )
            |
            torch.isclose(
                unique_mask_values,
                torch.tensor(
                    1.0,
                    device=device
                )
            )
        )

        if torch.any(invalid_mask):

            raise ValueError(
                "Mask must contain only 0 and 1 values. "
                f"Found: {unique_mask_values.detach().cpu().tolist()}"
            )

        # -----------------------------------------------------------
        # Model inference.
        # -----------------------------------------------------------

        model_output = model(
            corrupted
        )

        (
            reconstruction,
            travel_time,
            log_variance
        ) = extract_model_outputs(
            model_output
        )

        # -----------------------------------------------------------
        # Standardize reconstruction.
        # -----------------------------------------------------------

        reconstruction = ensure_5d(
            reconstruction,
            "reconstruction"
        )

        if reconstruction.shape != target.shape:

            raise ValueError(
                "Reconstruction and target shapes differ: "
                f"{tuple(reconstruction.shape)} vs "
                f"{tuple(target.shape)}"
            )

        # -----------------------------------------------------------
        # Validate reconstruction.
        # -----------------------------------------------------------

        if not torch.isfinite(reconstruction).all():

            raise ValueError(
                "Model reconstruction contains NaN or Inf."
            )

        # -----------------------------------------------------------
        # Standardize log variance if available.
        # -----------------------------------------------------------

        if log_variance is not None:

            log_variance = ensure_5d(
                log_variance,
                "log_variance"
            )

            if log_variance.shape != reconstruction.shape:

                raise ValueError(
                    "log_variance and reconstruction shapes differ."
                )

            (
                aleatoric_variance,
                aleatoric_std
            ) = calculate_aleatoric_uncertainty(
                log_variance
            )

        else:

            aleatoric_variance = None

            aleatoric_std = None

        # -----------------------------------------------------------
        # Data consistency projection.
        # -----------------------------------------------------------

        reconstruction = apply_data_consistency(
            reconstruction=reconstruction,
            observed_input=corrupted,
            mask=mask
        )

        # -----------------------------------------------------------
        # Verify observed-data preservation.
        # -----------------------------------------------------------

        preservation_error = observed_preservation_error(
            reconstruction,
            corrupted,
            mask
        )

        # -----------------------------------------------------------
        # Calculate reconstruction metrics.
        # -----------------------------------------------------------

        metrics = calculate_case_metrics(
            prediction=reconstruction,
            target=target,
            mask=mask
        )

        # -----------------------------------------------------------
        # Missing-region uncertainty.
        # -----------------------------------------------------------

        missing_region_aleatoric_std = float("nan")

        if aleatoric_std is not None:

            missing_pixels = mask < 0.5

            if torch.any(missing_pixels):

                missing_region_aleatoric_std = float(
                    aleatoric_std[missing_pixels]
                    .mean()
                    .detach()
                    .cpu()
                )

        # -----------------------------------------------------------
        # Global aleatoric uncertainty.
        # -----------------------------------------------------------

        global_aleatoric_std = float("nan")

        if aleatoric_std is not None:

            global_aleatoric_std = float(
                aleatoric_std
                .mean()
                .detach()
                .cpu()
            )

        # -----------------------------------------------------------
        # Epistemic/predictive uncertainty.
        #
        # Deterministic inference does not provide epistemic
        # uncertainty. Therefore these fields are deliberately NaN.
        # -----------------------------------------------------------

        result = {

            "Sample_ID": sample_index,

            "Geological_Mode": geological_mode,

            "MAE": metrics["MAE"],

            "MSE": metrics["MSE"],

            "RMSE": metrics["RMSE"],

            "Relative_Error": metrics["Relative_Error"],

            "PSNR": metrics["PSNR"],

            "SNR": metrics["SNR"],

            "SSIM": metrics["SSIM"],

            "Missing_MAE": metrics["Missing_MAE"],

            "Missing_RMSE": metrics["Missing_RMSE"],

            "Observed_MAE": metrics["Observed_MAE"],

            "Observed_RMSE": metrics["Observed_RMSE"],

            "Aleatoric_Variance": (
                float(
                    aleatoric_variance
                    .mean()
                    .detach()
                    .cpu()
                )
                if aleatoric_variance is not None
                else float("nan")
            ),

            "Aleatoric_Std": global_aleatoric_std,

            "Missing_Aleatoric_Std": (
                missing_region_aleatoric_std
            ),

            "Epistemic_Variance": float("nan"),

            "Epistemic_Std": float("nan"),

            "Predictive_Variance": float("nan"),

            "Predictive_Std": float("nan"),

            "Observed_Preservation_Error": (
                preservation_error
            ),

            "Missing_Rate_Actual": float(
                (mask < 0.5)
                .float()
                .mean()
                .detach()
                .cpu()
            ),

            "Epistemic_Evaluation": (
                "MC_Dropout_required"
            ),
        }

        # -----------------------------------------------------------
        # Optional physics diagnostic.
        #
        # We do not use travel time as a reconstruction-quality
        # metric. It is retained only as a physics-informed diagnostic.
        # -----------------------------------------------------------

        if travel_time is not None:

            travel_time = ensure_5d(
                travel_time,
                "travel_time"
            )

            if torch.isfinite(travel_time).all():

                result["Travel_Time_Mean"] = float(
                    travel_time
                    .mean()
                    .detach()
                    .cpu()
                )

                result["Travel_Time_Std"] = float(
                    travel_time
                    .std()
                    .detach()
                    .cpu()
                )

            else:

                result["Travel_Time_Mean"] = float("nan")

                result["Travel_Time_Std"] = float("nan")

        else:

            result["Travel_Time_Mean"] = float("nan")

            result["Travel_Time_Std"] = float("nan")

        results.append(result)

    return results


# ====================================================================
# 18. SUMMARY STATISTICS
# ====================================================================

def calculate_summary(results):
    """
    Calculate summary statistics for each geological complexity level.

    Parameters
    ----------
    results : list[dict]
        Per-sample results.

    Returns
    -------
    list[dict]
        Summary results.
    """

    summaries = []

    for geological_mode in GEOLOGICAL_COMPLEXITY_LEVELS:

        mode_results = [
            result
            for result in results
            if result["Geological_Mode"]
            == geological_mode
        ]

        if not mode_results:
            continue

        summary = {
            "Geological_Mode": geological_mode,
            "Number_of_Samples": len(mode_results),
        }

        # -----------------------------------------------------------
        # Metrics summarized by mean and standard deviation.
        # -----------------------------------------------------------

        metric_names = [
            "MAE",
            "MSE",
            "RMSE",
            "Relative_Error",
            "PSNR",
            "SNR",
            "SSIM",
            "Missing_MAE",
            "Missing_RMSE",
            "Observed_MAE",
            "Observed_RMSE",
            "Aleatoric_Variance",
            "Aleatoric_Std",
            "Missing_Aleatoric_Std",
            "Epistemic_Variance",
            "Epistemic_Std",
            "Predictive_Variance",
            "Predictive_Std",
            "Observed_Preservation_Error",
            "Missing_Rate_Actual",
            "Travel_Time_Mean",
            "Travel_Time_Std",
        ]

        for metric_name in metric_names:

            values = np.asarray(
                [
                    result[metric_name]
                    for result in mode_results
                ],
                dtype=np.float64
            )

            finite_values = values[
                np.isfinite(values)
            ]

            if finite_values.size == 0:

                summary[
                    f"Mean_{metric_name}"
                ] = float("nan")

                summary[
                    f"Std_{metric_name}"
                ] = float("nan")

            else:

                summary[
                    f"Mean_{metric_name}"
                ] = float(
                    np.mean(finite_values)
                )

                summary[
                    f"Std_{metric_name}"
                ] = float(
                    np.std(
                        finite_values,
                        ddof=0
                    )
                )

        summaries.append(summary)

    return summaries


# ====================================================================
# 19. SAVE CSV
# ====================================================================

def save_csv(results, path):
    """
    Save results as CSV.

    Parameters
    ----------
    results : list[dict]
        Results.

    path : Path
        Output CSV path.
    """

    import csv

    if not results:
        return

    # ---------------------------------------------------------------
    # Collect all field names.
    # ---------------------------------------------------------------

    fieldnames = []

    for result in results:

        for key in result.keys():

            if key not in fieldnames:

                fieldnames.append(key)

    # ---------------------------------------------------------------
    # Write CSV.
    # ---------------------------------------------------------------

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)


# ====================================================================
# 20. SAVE JSON
# ====================================================================

def save_json(data, path):
    """
    Save JSON metadata.
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            allow_nan=True
        )


# ====================================================================
# 21. MAIN EVALUATION
# ====================================================================

def main():

    print()
    print("=" * 70)
    print(
        "GEOLOGICAL COMPLEXITY ROBUSTNESS EVALUATION"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # Set reproducibility.
    # ---------------------------------------------------------------

    set_seed(SEED)

    # ---------------------------------------------------------------
    # Resolve device.
    # ---------------------------------------------------------------

    device = resolve_device(
        DEVICE
    )

    print(
        f"Dataset mode : {DATASET_MODE}"
    )

    print(
        f"Device       : {device}"
    )

    print(
        f"Checkpoint   : {CHECKPOINT_PATH}"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )

    print(
        f"Log variance : "
        f"[{LOG_VARIANCE_MIN}, {LOG_VARIANCE_MAX}]"
    )

    print()

    # ---------------------------------------------------------------
    # Load trained model once.
    #
    # IMPORTANT:
    # The same frozen model is used for every geological complexity
    # level.
    # ---------------------------------------------------------------

    model = load_model(
        device
    )

    # ---------------------------------------------------------------
    # Store all results.
    # ---------------------------------------------------------------

    all_results = []

    # ---------------------------------------------------------------
    # Evaluate every geological complexity level.
    # ---------------------------------------------------------------

    for geological_mode in GEOLOGICAL_COMPLEXITY_LEVELS:

        print()
        print("-" * 70)

        print(
            f"Evaluating geological mode: "
            f"{geological_mode}"
        )

        print("-" * 70)

        mode_results = evaluate_geological_level(
            model=model,
            device=device,
            geological_mode=geological_mode
        )

        all_results.extend(
            mode_results
        )

        print(
            f"Completed {geological_mode}: "
            f"{len(mode_results)} samples"
        )

    # ---------------------------------------------------------------
    # Calculate summaries.
    # ---------------------------------------------------------------

    summaries = calculate_summary(
        all_results
    )

    # ---------------------------------------------------------------
    # Output paths.
    # ---------------------------------------------------------------

    detailed_csv = (
        OUTPUT_DIR
        / "geological_complexity_results.csv"
    )

    summary_csv = (
        OUTPUT_DIR
        / "geological_complexity_summary.csv"
    )

    metadata_json = (
        OUTPUT_DIR
        / "geological_complexity_metadata.json"
    )

    # ---------------------------------------------------------------
    # Save detailed results.
    # ---------------------------------------------------------------

    save_csv(
        all_results,
        detailed_csv
    )

    # ---------------------------------------------------------------
    # Save summary.
    # ---------------------------------------------------------------

    save_csv(
        summaries,
        summary_csv
    )

    # ---------------------------------------------------------------
    # Metadata.
    # ---------------------------------------------------------------

    metadata = {

        "evaluation": (
            "Geological Complexity Robustness"
        ),

        "dataset_mode": DATASET_MODE,

        "device": str(device),

        "checkpoint": str(
            CHECKPOINT_PATH
        ),

        "geological_complexity_levels": (
            GEOLOGICAL_COMPLEXITY_LEVELS
        ),

        "number_of_levels": len(
            GEOLOGICAL_COMPLEXITY_LEVELS
        ),

        "seed": SEED,

        "use_attention": USE_ATTENTION,

        "use_residual": USE_RESIDUAL,

        "use_uncertainty": USE_UNCERTAINTY,

        "log_variance_min": (
            LOG_VARIANCE_MIN
        ),

        "log_variance_max": (
            LOG_VARIANCE_MAX
        ),

        "metric_source": (
            "metrics.reconstruction_metrics"
        ),

        "uncertainty_method": (
            "Deterministic aleatoric uncertainty "
            "from predicted log-variance"
        ),

        "epistemic_uncertainty": (
            "Not estimated in deterministic "
            "geological-complexity evaluation; "
            "MC-Dropout evaluation required."
        ),

        "predictive_uncertainty": (
            "Not estimated in deterministic "
            "geological-complexity evaluation; "
            "MC-Dropout evaluation required."
        ),

        "data_consistency": (
            "Observed seismic samples restored "
            "exactly after reconstruction."
        ),

        "detailed_results": str(
            detailed_csv
        ),

        "summary_results": str(
            summary_csv
        ),
    }

    save_json(
        metadata,
        metadata_json
    )

    # ---------------------------------------------------------------
    # Final validation.
    # ---------------------------------------------------------------

    expected_levels = set(
        GEOLOGICAL_COMPLEXITY_LEVELS
    )

    completed_levels = set(
        result["Geological_Mode"]
        for result in all_results
    )

    missing_levels = (
        expected_levels
        - completed_levels
    )

    if missing_levels:

        raise RuntimeError(
            "Evaluation incomplete. Missing geological "
            f"levels: {sorted(missing_levels)}"
        )

    # ---------------------------------------------------------------
    # Check observed-data preservation.
    # ---------------------------------------------------------------

    preservation_errors = [
        result[
            "Observed_Preservation_Error"
        ]
        for result in all_results
    ]

    max_preservation_error = max(
        preservation_errors
    )

    if not math.isfinite(
        max_preservation_error
    ):

        raise RuntimeError(
            "Observed-data preservation error is not finite."
        )

    # ---------------------------------------------------------------
    # Final report.
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "GEOLOGICAL COMPLEXITY EVALUATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Total samples evaluated : "
        f"{len(all_results)}"
    )

    print(
        f"Geological levels       : "
        f"{len(completed_levels)}"
    )

    print(
        f"Maximum observed-data "
        f"preservation error     : "
        f"{max_preservation_error:.6e}"
    )

    print()
    print(
        f"Detailed results : {detailed_csv}"
    )

    print(
        f"Summary results  : {summary_csv}"
    )

    print(
        f"Metadata         : {metadata_json}"
    )

    print()
    print("STATUS: PASS")
    print("=" * 70)


# ====================================================================
# 22. SCRIPT ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()