"""
====================================================================
Linear Interpolation Baseline — Controlled Experimental Matrix
====================================================================

Physics-Informed 3D Encoder–Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate the Linear Interpolation baseline using the same controlled
experimental matrix used by the other reconstruction methods.

Full controlled matrix
----------------------
    Geological modes : 6
    Missing mechanisms: 5
    Missing rates     : 5
    Random seeds      : 5

    Total = 6 × 5 × 5 × 5 = 750 cases

Smoke testing
-------------
The complete 750-case experimental design is ALWAYS constructed.

Execution can temporarily be limited through:

    CONTROLLED_MATRIX_CASE_LIMIT = 10

in utils/config.py.

For the final experiment:

    CONTROLLED_MATRIX_CASE_LIMIT = None

Important
---------
The reconstruction algorithm itself is unchanged from the original
Linear Interpolation baseline.

Input convention
----------------
    corrupted_cube : (C, D, H, W)
    mask           : (C, D, H, W)

Mask:
    1 -> observed
    0 -> missing

Observed seismic values are restored exactly after reconstruction.

Author: Ormin Joseph
====================================================================
"""

# ====================================================================
# IMPORTS
# ====================================================================

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from dataset.synthetic_dataset import SyntheticSeismicDataset

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)

from utils.config import (
    BASELINE_CUBE_SIZE,
    BASELINE_NUM_SAMPLES,
    BASELINE_SEED,
    CONTROLLED_MATRIX_CASE_LIMIT,
    OBSERVED_PRESERVATION_TOLERANCE,
    REPORT_DIR,
)


# ====================================================================
# CONTROLLED EXPERIMENTAL FACTORS
# ====================================================================

GEOLOGICAL_MODES = [
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
]

MASK_MODES = [
    "random_voxels",
    "missing_traces",
    "missing_inlines",
    "missing_crosslines",
    "missing_blocks",
]

MISSING_RATES = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

SEEDS = [
    42,
    43,
    44,
    45,
    46,
]


# ====================================================================
# OUTPUT DIRECTORY
# ====================================================================

OUTPUT_DIR = Path(REPORT_DIR) / "linear_interpolation_controlled_matrix"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RAW_RESULTS_FILE = (
    Path(REPORT_DIR)
    / "linear_interpolation_controlled_matrix.csv"
)

SUMMARY_RESULTS_FILE = (
    Path(REPORT_DIR)
    / "linear_interpolation_controlled_matrix_summary.csv"
)


# ====================================================================
# DEVICE
# ====================================================================

DEVICE = torch.device("cpu")


# ====================================================================
# REPRODUCIBILITY
# ====================================================================

def set_seed(seed):
    """
    Set all relevant random seeds.

    Parameters
    ----------
    seed : int
        Experiment seed.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def tensor_to_numpy(tensor):
    """
    Convert a tensor to a NumPy array.
    """

    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

    return np.asarray(tensor)


# --------------------------------------------------------------------

def check_finite(name, tensor):
    """
    Verify that an array/tensor contains only finite values.
    """

    if isinstance(tensor, torch.Tensor):

        valid = torch.isfinite(tensor).all().item()

    else:

        valid = np.isfinite(tensor).all()

    if not valid:

        raise ValueError(
            f"{name} contains non-finite values."
        )


# ====================================================================
# LINEAR INTERPOLATION CORE
# ====================================================================

def _interpolate_along_dimension(
        cube,
        observed_mask,
        dimension
):
    """
    Perform one-dimensional linear interpolation along one
    spatial dimension.

    Parameters
    ----------
    cube : torch.Tensor
        Seismic cube with shape (D, H, W).

    observed_mask : torch.Tensor
        Boolean observation mask with shape (D, H, W).

    dimension : int
        0 -> depth
        1 -> height
        2 -> width

    Returns
    -------
    torch.Tensor
        Interpolated cube.
    """

    result = cube.clone()

    # Move the interpolation dimension to the final axis.
    values = result.movedim(
        dimension,
        -1
    )

    mask = observed_mask.movedim(
        dimension,
        -1
    )

    original_shape = values.shape

    # Flatten all dimensions except interpolation axis.
    values = values.reshape(
        -1,
        values.shape[-1]
    )

    mask = mask.reshape(
        -1,
        mask.shape[-1]
    )

    axis = torch.arange(
        values.shape[-1],
        device=values.device,
        dtype=torch.float32
    )

    # ---------------------------------------------------------------
    # Process every 1-D profile independently.
    # ---------------------------------------------------------------

    for row in range(values.shape[0]):

        observed = mask[row]

        observed_indices = torch.nonzero(
            observed,
            as_tuple=False
        ).flatten()

        # -----------------------------------------------------------
        # CASE 1: No observed samples.
        # -----------------------------------------------------------

        if observed_indices.numel() == 0:

            continue

        # -----------------------------------------------------------
        # CASE 2: All samples observed.
        # -----------------------------------------------------------

        if (
            observed_indices.numel()
            == values.shape[1]
        ):

            continue

        # -----------------------------------------------------------
        # CASE 3: Only one observed sample.
        # -----------------------------------------------------------

        if observed_indices.numel() == 1:

            values[
                row,
                ~observed
            ] = values[
                row,
                observed_indices[0]
            ]

            continue

        # -----------------------------------------------------------
        # CASE 4: Two or more observed samples.
        # -----------------------------------------------------------

        observed_positions = observed_indices.to(
            dtype=torch.float32
        )

        observed_values = values[
            row,
            observed_indices
        ]

        missing = ~observed

        missing_positions = torch.nonzero(
            missing,
            as_tuple=False
        ).flatten()

        if missing_positions.numel() == 0:

            continue

        missing_positions_float = missing_positions.to(
            dtype=torch.float32
        )

        # -----------------------------------------------------------
        # Locate observed points surrounding each missing point.
        # -----------------------------------------------------------

        right_index = torch.searchsorted(
            observed_positions,
            missing_positions_float
        )

        right_index = torch.clamp(
            right_index,
            min=1,
            max=observed_positions.numel() - 1
        )

        left_index = right_index - 1

        x0 = observed_positions[left_index]
        x1 = observed_positions[right_index]

        y0 = observed_values[left_index]
        y1 = observed_values[right_index]

        # -----------------------------------------------------------
        # Linear interpolation.
        # -----------------------------------------------------------

        denominator = x1 - x0

        interpolated = (
            y0
            + (
                (
                    missing_positions_float - x0
                )
                / denominator
            )
            * (
                y1 - y0
            )
        )

        # -----------------------------------------------------------
        # Values before the first observed sample.
        # -----------------------------------------------------------

        before_first = (
            missing_positions_float
            < observed_positions[0]
        )

        # -----------------------------------------------------------
        # Values after the final observed sample.
        # -----------------------------------------------------------

        after_last = (
            missing_positions_float
            > observed_positions[-1]
        )

        interpolated[before_first] = (
            observed_values[0]
        )

        interpolated[after_last] = (
            observed_values[-1]
        )

        # -----------------------------------------------------------
        # Write only missing values.
        # -----------------------------------------------------------

        values[
            row,
            missing_positions
        ] = interpolated

    # Restore original shape.
    values = values.reshape(
        original_shape
    )

    return values.movedim(
        -1,
        dimension
    )


# ====================================================================
# LINEAR INTERPOLATION RECONSTRUCTION
# ====================================================================

def linear_interpolation_reconstruction(
        corrupted_cube,
        mask
):
    """
    Reconstruct missing seismic voxels using separable
    linear interpolation along the three spatial dimensions.

    Parameters
    ----------
    corrupted_cube : torch.Tensor
        Shape (C, D, H, W).

    mask : torch.Tensor
        Shape (C, D, H, W).

    Returns
    -------
    torch.Tensor
        Reconstructed cube with identical shape.
    """

    # ---------------------------------------------------------------
    # Convert input cube to tensor if necessary.
    # ---------------------------------------------------------------

    if not isinstance(
            corrupted_cube,
            torch.Tensor
    ):

        corrupted_cube = torch.as_tensor(
            corrupted_cube,
            dtype=torch.float32
        )

    # ---------------------------------------------------------------
    # Convert mask to tensor if necessary.
    # ---------------------------------------------------------------

    if not isinstance(
            mask,
            torch.Tensor
    ):

        mask = torch.as_tensor(
            mask,
            dtype=torch.float32
        )

    corrupted_cube = corrupted_cube.float()

    mask = mask.float()

    # ---------------------------------------------------------------
    # Validate dimensions.
    # ---------------------------------------------------------------

    if corrupted_cube.ndim != 4:

        raise ValueError(
            "corrupted_cube must have shape "
            "(C, D, H, W). "
            f"Received: {tuple(corrupted_cube.shape)}"
        )

    if mask.ndim != 4:

        raise ValueError(
            "mask must have shape "
            "(C, D, H, W). "
            f"Received: {tuple(mask.shape)}"
        )

    if corrupted_cube.shape != mask.shape:

        raise ValueError(
            "corrupted_cube and mask must have "
            "identical shapes. "
            f"Received "
            f"{tuple(corrupted_cube.shape)} "
            f"and "
            f"{tuple(mask.shape)}."
        )

    # ---------------------------------------------------------------
    # Numerical validation.
    # ---------------------------------------------------------------

    check_finite(
        "corrupted_cube",
        corrupted_cube
    )

    check_finite(
        "mask",
        mask
    )

    # ---------------------------------------------------------------
    # Convert mask to Boolean observed mask.
    # ---------------------------------------------------------------

    observed_mask = mask == 1

    # ---------------------------------------------------------------
    # At least one observed voxel must exist.
    # ---------------------------------------------------------------

    if not observed_mask.any():

        raise ValueError(
            "The mask contains no observed voxels. "
            "Linear interpolation cannot be performed."
        )

    # ---------------------------------------------------------------
    # Start with corrupted cube.
    # ---------------------------------------------------------------

    reconstructed = corrupted_cube.clone()

    # ---------------------------------------------------------------
    # Process each channel independently.
    # ---------------------------------------------------------------

    for channel in range(
        corrupted_cube.shape[0]
    ):

        cube = reconstructed[channel]

        channel_mask = observed_mask[channel]

        # -----------------------------------------------------------
        # Complete channel requires no reconstruction.
        # -----------------------------------------------------------

        if channel_mask.all():

            continue

        # -----------------------------------------------------------
        # Interpolate depth.
        # -----------------------------------------------------------

        cube = _interpolate_along_dimension(
            cube,
            channel_mask,
            dimension=0
        )

        # -----------------------------------------------------------
        # Interpolate height.
        # -----------------------------------------------------------

        cube = _interpolate_along_dimension(
            cube,
            channel_mask,
            dimension=1
        )

        # -----------------------------------------------------------
        # Interpolate width.
        # -----------------------------------------------------------

        cube = _interpolate_along_dimension(
            cube,
            channel_mask,
            dimension=2
        )

        # -----------------------------------------------------------
        # Restore observed values EXACTLY.
        # -----------------------------------------------------------

        cube[channel_mask] = (
            corrupted_cube[channel][channel_mask]
        )

        reconstructed[channel] = cube

    # ---------------------------------------------------------------
    # Final validation.
    # ---------------------------------------------------------------

    check_finite(
        "reconstructed",
        reconstructed
    )

    return reconstructed


# ====================================================================
# CONTROLLED CASE GENERATION
# ====================================================================

def build_controlled_cases():
    """
    Construct the complete deterministic 750-case experimental matrix.

    Returns
    -------
    list
        List containing all controlled experimental cases.
    """

    cases = []

    case_id = 1

    for geological_mode in GEOLOGICAL_MODES:

        for mask_mode in MASK_MODES:

            for missing_rate in MISSING_RATES:

                for seed in SEEDS:

                    cases.append(
                        {
                            "Case_ID": case_id,
                            "geological_mode": geological_mode,
                            "mask_mode": mask_mode,
                            "missing_rate": missing_rate,
                            "seed": seed,
                        }
                    )

                    case_id += 1

    return cases


# ====================================================================
# CASE SELECTION
# ====================================================================

def select_cases_for_execution(
        all_cases
):
    """
    Apply the configured smoke-test limit.

    The full 750-case matrix remains intact.

    Parameters
    ----------
    all_cases : list
        Complete controlled experimental matrix.

    Returns
    -------
    list
        Cases selected for current execution.
    """

    total_cases = len(all_cases)

    limit = CONTROLLED_MATRIX_CASE_LIMIT

    if limit is None:

        return all_cases

    if not isinstance(
            limit,
            int
    ):

        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT must "
            "be an integer or None."
        )

    if limit <= 0:

        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT must "
            "be greater than zero or None."
        )

    if limit > total_cases:

        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT "
            f"({limit}) exceeds the total "
            f"number of controlled cases "
            f"({total_cases})."
        )

    return all_cases[:limit]


# ====================================================================
# SINGLE CONTROLLED EXPERIMENT
# ====================================================================

def run_single_experiment(
        case
):
    """
    Execute one controlled Linear Interpolation experiment.
    """

    case_id = case["Case_ID"]

    geological_mode = case[
        "geological_mode"
    ]

    mask_mode = case[
        "mask_mode"
    ]

    missing_rate = case[
        "missing_rate"
    ]

    seed = case["seed"]

    set_seed(seed)

    start_time = time.perf_counter()

    # ---------------------------------------------------------------
    # Construct deterministic synthetic dataset.
    # ---------------------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=BASELINE_NUM_SAMPLES,
        cube_size=BASELINE_CUBE_SIZE,
        missing_probability=missing_rate,
        geological_mode=geological_mode,
        mask_mode=mask_mode,
        seed=seed,
    )

    # ---------------------------------------------------------------
    # One controlled case uses the first sample.
    # ---------------------------------------------------------------

    (
        corrupted,
        target,
        mask,
        velocity,
        actual_mask_mode,
        actual_geological_mode,
    ) = dataset[0]

    # ---------------------------------------------------------------
    # Convert everything to tensors.
    # ---------------------------------------------------------------

    corrupted = torch.as_tensor(
        corrupted,
        dtype=torch.float32
    )

    target = torch.as_tensor(
        target,
        dtype=torch.float32
    )

    mask = torch.as_tensor(
        mask,
        dtype=torch.float32
    )

    velocity = torch.as_tensor(
        velocity,
        dtype=torch.float32
    )

    # ---------------------------------------------------------------
    # Validate metadata.
    # ---------------------------------------------------------------

    if actual_mask_mode != mask_mode:

        raise ValueError(
            "Dataset returned a different mask mode. "
            f"Requested={mask_mode}, "
            f"Actual={actual_mask_mode}"
        )

    if (
        actual_geological_mode
        != geological_mode
    ):

        raise ValueError(
            "Dataset returned a different "
            "geological mode. "
            f"Requested={geological_mode}, "
            f"Actual={actual_geological_mode}"
        )

    # ---------------------------------------------------------------
    # Validate shapes.
    # ---------------------------------------------------------------

    expected_shape = (
        1,
        *BASELINE_CUBE_SIZE
    )

    if tuple(corrupted.shape) != expected_shape:

        raise ValueError(
            "Unexpected corrupted cube shape. "
            f"Expected={expected_shape}, "
            f"Actual={tuple(corrupted.shape)}"
        )

    if target.shape != corrupted.shape:

        raise ValueError(
            "target and corrupted cube shapes "
            "do not match."
        )

    if mask.shape != corrupted.shape:

        raise ValueError(
            "mask and corrupted cube shapes "
            "do not match."
        )

    if velocity.shape != corrupted.shape:

        raise ValueError(
            "velocity and corrupted cube shapes "
            "do not match."
        )

    # ---------------------------------------------------------------
    # Numerical validation.
    # ---------------------------------------------------------------

    check_finite(
        "corrupted",
        corrupted
    )

    check_finite(
        "target",
        target
    )

    check_finite(
        "mask",
        mask
    )

    check_finite(
        "velocity",
        velocity
    )

    # ---------------------------------------------------------------
    # Calculate actual missing rate.
    # ---------------------------------------------------------------

    observed_mask = mask == 1

    actual_missing_rate = (
        1.0
        - observed_mask.float().mean().item()
    )

    # ---------------------------------------------------------------
    # Run Linear Interpolation.
    # ---------------------------------------------------------------

    reconstruction = (
        linear_interpolation_reconstruction(
            corrupted,
            mask
        )
    )

    # ---------------------------------------------------------------
    # Data-consistency projection.
    #
    # This guarantees that observed seismic samples remain exactly
    # equal to the original corrupted input.
    # ---------------------------------------------------------------

    reconstruction[observed_mask] = (
        corrupted[observed_mask]
    )

    # ---------------------------------------------------------------
    # Verify observed-data preservation.
    # ---------------------------------------------------------------

    preservation_error = torch.max(
        torch.abs(
            reconstruction[observed_mask]
            - corrupted[observed_mask]
        )
    ).item()

    if (
        preservation_error
        > OBSERVED_PRESERVATION_TOLERANCE
    ):

        raise RuntimeError(
            "Observed-data preservation failed. "
            f"Maximum error="
            f"{preservation_error:.6e}"
        )

    # ---------------------------------------------------------------
    # Missing-region mask.
    # ---------------------------------------------------------------

    missing_mask = mask == 0

    if missing_mask.any():

        # -----------------------------------------------------------
        # Extract only the missing voxels.
        #
        # Boolean indexing produces a 1-D vector. Therefore these
        # regional metrics are calculated directly rather than being
        # passed to the canonical 5-D metric interface.
        # -----------------------------------------------------------

        missing_error = (
                reconstruction[missing_mask]
                - target[missing_mask]
        )

        # -----------------------------------------------------------
        # Missing-region MAE.
        # -----------------------------------------------------------

        missing_mae = torch.mean(
            torch.abs(missing_error)
        ).item()

        # -----------------------------------------------------------
        # Missing-region RMSE.
        # -----------------------------------------------------------

        missing_rmse = torch.sqrt(
            torch.mean(
                missing_error ** 2
            )
        ).item()

    else:

        missing_mae = 0.0

        missing_rmse = 0.0


    # ---------------------------------------------------------------
    # Prepare tensors for the canonical reconstruction metrics.
    #
    # The baseline reconstruction uses:
    #
    #     (C, D, H, W)
    #
    # The canonical metrics require:
    #
    #     (B, C, D, H, W)
    #
    # Therefore add the batch dimension here.
    # ---------------------------------------------------------------

    reconstruction_metric = reconstruction.unsqueeze(0)

    target_metric = target.unsqueeze(0)

    mask_metric = mask.unsqueeze(0)

    # ---------------------------------------------------------------
    # Validate metric tensor dimensions.
    # ---------------------------------------------------------------

    expected_metric_shape = (
        1,
        reconstruction.shape[0],
        reconstruction.shape[1],
        reconstruction.shape[2],
        reconstruction.shape[3],
    )

    if tuple(reconstruction_metric.shape) != expected_metric_shape:
        raise ValueError(
            "Unexpected reconstruction metric shape. "
            f"Expected={expected_metric_shape}, "
            f"Actual={tuple(reconstruction_metric.shape)}"
        )

    if tuple(target_metric.shape) != expected_metric_shape:
        raise ValueError(
            "Unexpected target metric shape. "
            f"Expected={expected_metric_shape}, "
            f"Actual={tuple(target_metric.shape)}"
        )

    # ---------------------------------------------------------------
    # Global reconstruction metrics.
    #
    # These calls now use the canonical [B,C,D,H,W] convention.
    # ---------------------------------------------------------------

    global_mae = mae(
        reconstruction_metric,
        target_metric
    )

    global_rmse = rmse(
        reconstruction_metric,
        target_metric
    )

    global_psnr = psnr(
        reconstruction_metric,
        target_metric
    )

    global_snr = snr(
        reconstruction_metric,
        target_metric
    )

    global_ssim = ssim(
        reconstruction_metric,
        target_metric
    )

    # ---------------------------------------------------------------
    # Runtime.
    # ---------------------------------------------------------------

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    # ---------------------------------------------------------------
    # Final finite validation.
    # ---------------------------------------------------------------

    check_finite(
        "reconstruction",
        reconstruction
    )

    # ---------------------------------------------------------------
    # Return complete result record.
    # ---------------------------------------------------------------

    return {
        "Case_ID": case_id,
        "Method": "Linear Interpolation",
        "geological_mode": geological_mode,
        "mask_mode": mask_mode,
        "requested_missing_rate": missing_rate,
        "actual_missing_rate": actual_missing_rate,
        "seed": seed,
        "channels": corrupted.shape[0],
        "depth": corrupted.shape[1],
        "height": corrupted.shape[2],
        "width": corrupted.shape[3],
        "MAE": float(global_mae),
        "RMSE": float(global_rmse),
        "PSNR": float(global_psnr),
        "SNR": float(global_snr),
        "SSIM": float(global_ssim),
        "Missing_MAE": float(missing_mae),
        "Missing_RMSE": float(missing_rmse),
        "Observed_Preservation_Error": float(
            preservation_error
        ),
        "Runtime_Seconds": float(
            runtime_seconds
        ),
        "Status": "PASS",
        "Error": "",
    }


# ====================================================================
# SUMMARY CALCULATION
# ====================================================================

def calculate_summary(
        records
):
    """
    Calculate grouped summary statistics.

    Results are grouped by:

        geological_mode
        mask_mode
        requested_missing_rate
    """

    groups = {}

    for record in records:

        if record["Status"] != "PASS":

            continue

        key = (
            record["geological_mode"],
            record["mask_mode"],
            record["requested_missing_rate"],
        )

        groups.setdefault(
            key,
            []
        ).append(record)

    summary = []

    for (
        geological_mode,
        mask_mode,
        missing_rate,
    ), group in groups.items():

        def values(field):

            return np.asarray(
                [
                    float(row[field])
                    for row in group
                ],
                dtype=np.float64
            )

        def mean(field):

            return float(
                np.mean(values(field))
            )

        def std(field):

            data = values(field)

            if len(data) <= 1:

                return 0.0

            return float(
                np.std(
                    data,
                    ddof=1
                )
            )

        summary.append(
            {
                "Method": "Linear Interpolation",
                "geological_mode": geological_mode,
                "mask_mode": mask_mode,
                "requested_missing_rate": missing_rate,
                "n_seeds": len(group),

                "Mean_MAE": mean("MAE"),
                "Std_MAE": std("MAE"),

                "Mean_RMSE": mean("RMSE"),
                "Std_RMSE": std("RMSE"),

                "Mean_PSNR": mean("PSNR"),
                "Std_PSNR": std("PSNR"),

                "Mean_SNR": mean("SNR"),
                "Std_SNR": std("SNR"),

                "Mean_SSIM": mean("SSIM"),
                "Std_SSIM": std("SSIM"),

                "Mean_Missing_MAE":
                    mean("Missing_MAE"),

                "Std_Missing_MAE":
                    std("Missing_MAE"),

                "Mean_Missing_RMSE":
                    mean("Missing_RMSE"),

                "Std_Missing_RMSE":
                    std("Missing_RMSE"),

                "Mean_Runtime_Seconds":
                    mean("Runtime_Seconds"),

                "Mean_Observed_Preservation_Error":
                    mean(
                        "Observed_Preservation_Error"
                    ),

                "Max_Observed_Preservation_Error":
                    max(
                        values(
                            "Observed_Preservation_Error"
                        )
                    ),
            }
        )

    return summary


# ====================================================================
# CSV WRITER
# ====================================================================

def write_csv(
        filepath,
        records
):
    """
    Write dictionaries to CSV.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not records:

        return

    fieldnames = list(
        records[0].keys()
    )

    with filepath.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(records)


# ====================================================================
# MAIN
# ====================================================================

def main():
    """
    Execute the controlled Linear Interpolation experiment.
    """

    print()
    print("=" * 70)
    print(
        "LINEAR INTERPOLATION — "
        "CONTROLLED EXPERIMENTAL MATRIX"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # Construct COMPLETE experimental matrix.
    # ---------------------------------------------------------------

    all_cases = build_controlled_cases()

    expected_cases = len(
        all_cases
    )

    # ---------------------------------------------------------------
    # Apply smoke-test execution limit.
    # ---------------------------------------------------------------

    cases_to_run = (
        select_cases_for_execution(
            all_cases
        )
    )

    cases_to_execute = len(
        cases_to_run
    )

    print(
        f"Full controlled matrix : "
        f"{expected_cases} cases"
    )

    print(
        f"Cases to execute       : "
        f"{cases_to_execute} cases"
    )

    if (
        CONTROLLED_MATRIX_CASE_LIMIT
        is None
    ):

        print(
            "Execution mode         : "
            "FULL 750-CASE EXPERIMENT"
        )

    else:

        print(
            "Execution mode         : "
            "SMOKE TEST"
        )

    print(
        f"Cube size              : "
        f"{BASELINE_CUBE_SIZE}"
    )

    print(
        "Method                 : "
        "Linear Interpolation"
    )

    print("=" * 70)

    # ---------------------------------------------------------------
    # Execute cases.
    # ---------------------------------------------------------------

    records = []

    successful_cases = 0

    failed_cases = 0

    for index, case in enumerate(
        cases_to_run,
        start=1
    ):

        print(
            f"\nCase {index}/{cases_to_execute} "
            f"| Matrix Case ID = "
            f"{case['Case_ID']}"
        )

        print(
            f"  Geology      : "
            f"{case['geological_mode']}"
        )

        print(
            f"  Mask         : "
            f"{case['mask_mode']}"
        )

        print(
            f"  Missing rate : "
            f"{case['missing_rate']:.0%}"
        )

        print(
            f"  Seed         : "
            f"{case['seed']}"
        )

        try:

            result = run_single_experiment(
                case
            )

            records.append(
                result
            )

            successful_cases += 1

            print(
                "  Status       : PASS"
            )

            print(
                f"  SSIM         : "
                f"{result['SSIM']:.6f}"
            )

            print(
                f"  MAE          : "
                f"{result['MAE']:.6f}"
            )

            print(
                f"  Missing MAE  : "
                f"{result['Missing_MAE']:.6f}"
            )

            print(
                f"  Runtime      : "
                f"{result['Runtime_Seconds']:.4f} s"
            )

        except Exception as error:

            failed_cases += 1

            error_record = {
                "Case_ID": case["Case_ID"],
                "Method": "Linear Interpolation",
                "geological_mode":
                    case["geological_mode"],
                "mask_mode":
                    case["mask_mode"],
                "requested_missing_rate":
                    case["missing_rate"],
                "actual_missing_rate": "",
                "seed": case["seed"],
                "channels": "",
                "depth": "",
                "height": "",
                "width": "",
                "MAE": "",
                "RMSE": "",
                "PSNR": "",
                "SNR": "",
                "SSIM": "",
                "Missing_MAE": "",
                "Missing_RMSE": "",
                "Observed_Preservation_Error":
                    "",
                "Runtime_Seconds": "",
                "Status": "FAIL",
                "Error": str(error),
            }

            records.append(
                error_record
            )

            print(
                "  Status       : FAIL"
            )

            print(
                f"  Error        : {error}"
            )

    # ---------------------------------------------------------------
    # Write raw case-level results.
    # ---------------------------------------------------------------

    write_csv(
        RAW_RESULTS_FILE,
        records
    )

    # ---------------------------------------------------------------
    # Calculate and save summary.
    # ---------------------------------------------------------------

    successful_records = [
        record
        for record in records
        if record["Status"] == "PASS"
    ]

    summary = calculate_summary(
        successful_records
    )

    write_csv(
        SUMMARY_RESULTS_FILE,
        summary
    )

    # ---------------------------------------------------------------
    # Final report.
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LINEAR INTERPOLATION "
        "CONTROLLED MATRIX COMPLETE"
    )
    print("=" * 70)

    print(
        f"Full matrix cases      : "
        f"{expected_cases}"
    )

    print(
        f"Cases executed         : "
        f"{cases_to_execute}"
    )

    print(
        f"Successful cases       : "
        f"{successful_cases}"
    )

    print(
        f"Failed cases           : "
        f"{failed_cases}"
    )

    print(
        f"Raw results            : "
        f"{RAW_RESULTS_FILE}"
    )

    print(
        f"Summary results        : "
        f"{SUMMARY_RESULTS_FILE}"
    )

    # ---------------------------------------------------------------
    # PASS/FAIL criterion.
    #
    # For smoke test:
    #     successful_cases must equal cases_to_execute.
    #
    # For final experiment:
    #     cases_to_execute = 750.
    # ---------------------------------------------------------------

    if (
        successful_cases
        == cases_to_execute
        and failed_cases == 0
    ):

        print()
        print(
            "OVERALL STATUS: PASS"
        )

    else:

        print()
        print(
            "OVERALL STATUS: FAIL"
        )

        raise RuntimeError(
            "One or more controlled "
            "Linear Interpolation cases failed."
        )

    print("=" * 70)


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()