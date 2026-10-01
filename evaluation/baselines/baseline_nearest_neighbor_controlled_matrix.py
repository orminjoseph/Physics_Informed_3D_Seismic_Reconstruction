"""
====================================================================
Nearest Neighbor Reconstruction Baseline — Controlled Matrix
====================================================================

Physics-Informed 3D Encoder–Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate the Nearest Neighbor reconstruction baseline using the
same controlled experimental matrix used by all seven reconstruction
methods.

Full controlled experimental matrix
------------------------------------
    Geological modes       : 6
    Missing-data mechanisms: 5
    Missing rates          : 5
    Random seeds           : 5

    Total = 6 × 5 × 5 × 5 = 750 cases

Smoke testing
-------------
The complete 750-case matrix is always constructed.

The number of cases executed can be temporarily controlled through:

    CONTROLLED_MATRIX_CASE_LIMIT

in utils/config.py.

For smoke testing:

    CONTROLLED_MATRIX_CASE_LIMIT = 10

For the final experiment:

    CONTROLLED_MATRIX_CASE_LIMIT = None

Important
---------
The nearest-neighbor reconstruction algorithm is kept independent
of the controlled experimental framework.

For each missing voxel, the spatially nearest observed voxel is
used to reconstruct its value.

Observed seismic samples are restored exactly after reconstruction.

Input convention
----------------
    corrupted_cube : (C, D, H, W)
    mask           : (C, D, H, W)

Mask:
    1 -> observed
    0 -> missing

Metric convention
-----------------
The reconstruction algorithm operates on:

    (C, D, H, W)

The canonical reconstruction metrics operate on:

    (B, C, D, H, W)

Therefore, a batch dimension is added only when calculating the
global reconstruction metrics.

Missing-region MAE and RMSE are calculated directly from the
missing-region error values because boolean indexing produces
a flattened tensor.

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

from scipy.ndimage import distance_transform_edt

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
# OUTPUT PATHS
# ====================================================================

RAW_RESULTS_FILE = (
    Path(REPORT_DIR)
    / "nearest_neighbor_controlled_matrix.csv"
)


SUMMARY_RESULTS_FILE = (
    Path(REPORT_DIR)
    / "nearest_neighbor_controlled_matrix_summary.csv"
)


# Ensure report directory exists.
Path(REPORT_DIR).mkdir(
    parents=True,
    exist_ok=True,
)


# ====================================================================
# DEVICE
# ====================================================================

# The SciPy distance transform is CPU/NumPy based.
#
# The reconstruction therefore performs the distance calculation
# on CPU and moves the resulting indices back to the seismic tensor
# device when necessary.
DEVICE = torch.device("cpu")


# ====================================================================
# REPRODUCIBILITY
# ====================================================================

def set_seed(seed):
    """
    Set all relevant random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed for the experiment.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ====================================================================
# NUMERICAL VALIDATION
# ====================================================================

def check_finite(name, data):
    """
    Verify that a tensor or NumPy array contains only finite values.

    Parameters
    ----------
    name : str
        Name of the object being checked.

    data : torch.Tensor or numpy.ndarray
        Data to validate.
    """

    if isinstance(data, torch.Tensor):

        valid = torch.isfinite(data).all().item()

    else:

        valid = np.isfinite(data).all()

    if not valid:

        raise ValueError(
            f"{name} contains non-finite values."
        )


# ====================================================================
# NEAREST NEIGHBOR RECONSTRUCTION
# ====================================================================

def nearest_neighbor_reconstruction(
        corrupted_cube,
        mask
):
    """
    Reconstruct missing seismic voxels using spatial nearest
    neighbor interpolation.

    Parameters
    ----------
    corrupted_cube : torch.Tensor
        Corrupted seismic cube with shape:

            (C, D, H, W)

    mask : torch.Tensor
        Observation mask with the same shape.

            1 -> observed
            0 -> missing

    Returns
    -------
    torch.Tensor
        Reconstructed seismic cube with the same shape.

    Notes
    -----
    For every missing voxel, the value of the spatially nearest
    observed voxel is used.

    SciPy's distance_transform_edt is used to determine the
    nearest observed location.

    Observed voxels are preserved exactly.
    """

    # ---------------------------------------------------------------
    # Convert input cube to tensor when necessary.
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
    # Convert mask to tensor when necessary.
    # ---------------------------------------------------------------

    if not isinstance(
            mask,
            torch.Tensor
    ):

        mask = torch.as_tensor(
            mask,
            dtype=torch.float32
        )

    # ---------------------------------------------------------------
    # Preserve original device.
    # ---------------------------------------------------------------

    original_device = corrupted_cube.device

    # ---------------------------------------------------------------
    # Convert to floating-point representation.
    # ---------------------------------------------------------------

    corrupted_cube = corrupted_cube.float()

    mask = mask.float()

    # ---------------------------------------------------------------
    # Validate dimensions.
    # ---------------------------------------------------------------

    if corrupted_cube.ndim != 4:

        raise ValueError(
            "corrupted_cube must have shape "
            "(C, D, H, W). "
            f"Received shape: "
            f"{tuple(corrupted_cube.shape)}"
        )

    if mask.ndim != 4:

        raise ValueError(
            "mask must have shape "
            "(C, D, H, W). "
            f"Received shape: "
            f"{tuple(mask.shape)}"
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
    # Validate numerical values.
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
    # Create reconstruction from the original corrupted cube.
    # ---------------------------------------------------------------

    reconstructed = corrupted_cube.clone()

    channels, depth, height, width = (
        corrupted_cube.shape
    )

    # ---------------------------------------------------------------
    # Process each channel independently.
    # ---------------------------------------------------------------

    for channel in range(channels):

        cube = corrupted_cube[channel]

        channel_mask = mask[channel]

        # -----------------------------------------------------------
        # Convert mask to NumPy because SciPy operates on CPU arrays.
        # -----------------------------------------------------------

        mask_numpy = (
            channel_mask
            .detach()
            .cpu()
            .numpy()
        )

        # -----------------------------------------------------------
        # Define observed and missing locations.
        # -----------------------------------------------------------

        observed = mask_numpy == 1

        missing = mask_numpy == 0

        # -----------------------------------------------------------
        # If there are no missing voxels, nothing needs to be
        # reconstructed.
        # -----------------------------------------------------------

        if not missing.any():

            continue

        # -----------------------------------------------------------
        # If there are no observed voxels, nearest-neighbor
        # reconstruction is mathematically impossible.
        # -----------------------------------------------------------

        if not observed.any():

            raise ValueError(
                f"Channel {channel} contains no observed "
                "voxels. Nearest-neighbor reconstruction "
                "cannot be performed."
            )

        # -----------------------------------------------------------
        # Distance transform.
        #
        # SciPy treats zero-valued locations as the reference
        # locations.
        #
        # Therefore:
        #
        #     observed = True  -> 0
        #     missing  = True  -> 1
        #
        # The returned indices identify the nearest observed
        # voxel for every location.
        # -----------------------------------------------------------

        _, nearest_indices = (
            distance_transform_edt(
                missing,
                return_distances=True,
                return_indices=True
            )
        )

        # -----------------------------------------------------------
        # Convert nearest-neighbor indices to a tensor.
        #
        # Shape:
        #
        #     (3, D, H, W)
        #
        # corresponding to:
        #
        #     depth, height, width
        # -----------------------------------------------------------

        nearest_positions = torch.from_numpy(
            nearest_indices
        ).long()

        # -----------------------------------------------------------
        # Move indices to the same device as the seismic cube.
        # -----------------------------------------------------------

        nearest_positions = (
            nearest_positions.to(
                device=cube.device
            )
        )

        # -----------------------------------------------------------
        # Obtain the value of the nearest observed voxel for every
        # position in the cube.
        # -----------------------------------------------------------

        nearest_values = cube[
            nearest_positions[0],
            nearest_positions[1],
            nearest_positions[2]
        ]

        # -----------------------------------------------------------
        # Replace ONLY missing voxels.
        # -----------------------------------------------------------

        reconstructed[channel][
            channel_mask == 0
        ] = nearest_values[
            channel_mask == 0
        ]

    # ---------------------------------------------------------------
    # Explicitly restore observed values.
    #
    # This guarantees exact data consistency.
    # ---------------------------------------------------------------

    observed_mask = mask == 1

    reconstructed[observed_mask] = (
        corrupted_cube[observed_mask]
    )

    # ---------------------------------------------------------------
    # Final numerical validation.
    # ---------------------------------------------------------------

    check_finite(
        "reconstructed",
        reconstructed
    )

    # ---------------------------------------------------------------
    # Return to the original device.
    # ---------------------------------------------------------------

    return reconstructed.to(
        original_device
    )


# ====================================================================
# CONTROLLED MATRIX CONSTRUCTION
# ====================================================================

def build_controlled_cases():
    """
    Construct the complete deterministic controlled matrix.

    The resulting matrix contains:

        6 geological modes
        × 5 mask modes
        × 5 missing rates
        × 5 seeds

        = 750 cases.

    Returns
    -------
    list
        Complete list of controlled experimental cases.
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
                            "geological_mode":
                                geological_mode,
                            "mask_mode":
                                mask_mode,
                            "missing_rate":
                                missing_rate,
                            "seed":
                                seed,
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
    Select cases for the current execution.

    The full 750-case matrix remains intact.

    CONTROLLED_MATRIX_CASE_LIMIT controls only how many cases
    are executed during the current run.

    Parameters
    ----------
    all_cases : list
        Complete controlled experimental matrix.

    Returns
    -------
    list
        Cases selected for execution.
    """

    total_cases = len(all_cases)

    limit = CONTROLLED_MATRIX_CASE_LIMIT

    # ---------------------------------------------------------------
    # None means execute all cases.
    # ---------------------------------------------------------------

    if limit is None:

        return all_cases

    # ---------------------------------------------------------------
    # Validate limit type.
    # ---------------------------------------------------------------

    if not isinstance(
            limit,
            int
    ):

        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT must "
            "be an integer or None."
        )

    # ---------------------------------------------------------------
    # Validate positive limit.
    # ---------------------------------------------------------------

    if limit <= 0:

        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT must "
            "be greater than zero or None."
        )

    # ---------------------------------------------------------------
    # Prevent an invalid limit.
    # ---------------------------------------------------------------

    if limit > total_cases:

        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT "
            f"({limit}) exceeds the total "
            f"number of controlled cases "
            f"({total_cases})."
        )

    # ---------------------------------------------------------------
    # Select deterministic first N cases.
    # ---------------------------------------------------------------

    return all_cases[:limit]


# ====================================================================
# SINGLE CONTROLLED EXPERIMENT
# ====================================================================

def run_single_experiment(
        case
):
    """
    Execute one controlled nearest-neighbor experiment.

    Parameters
    ----------
    case : dict
        One controlled experimental case.

    Returns
    -------
    dict
        Case-level evaluation results.
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

    # ---------------------------------------------------------------
    # Set deterministic random seed.
    # ---------------------------------------------------------------

    set_seed(seed)

    # ---------------------------------------------------------------
    # Start runtime measurement.
    # ---------------------------------------------------------------

    start_time = time.perf_counter()

    # ---------------------------------------------------------------
    # Generate controlled synthetic dataset.
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
    # Retrieve the first sample.
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
    # Convert all arrays to tensors.
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
    # Validate returned metadata.
    # ---------------------------------------------------------------

    if actual_mask_mode != mask_mode:

        raise ValueError(
            "Dataset returned a different mask mode. "
            f"Requested={mask_mode}, "
            f"Actual={actual_mask_mode}"
        )

    if actual_geological_mode != geological_mode:

        raise ValueError(
            "Dataset returned a different geological "
            "mode. "
            f"Requested={geological_mode}, "
            f"Actual={actual_geological_mode}"
        )

    # ---------------------------------------------------------------
    # Validate expected tensor shape.
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
    # Validate numerical values.
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

    missing_mask = mask == 0

    actual_missing_rate = (
        missing_mask.float()
        .mean()
        .item()
    )

    # ---------------------------------------------------------------
    # Perform nearest-neighbor reconstruction.
    # ---------------------------------------------------------------

    reconstruction = (
        nearest_neighbor_reconstruction(
            corrupted,
            mask
        )
    )

    # ---------------------------------------------------------------
    # Enforce exact data consistency.
    #
    # Observed seismic values are copied back from the corrupted
    # input so that no measured sample can be modified.
    # ---------------------------------------------------------------

    reconstruction[observed_mask] = (
        corrupted[observed_mask]
    )

    # ---------------------------------------------------------------
    # Calculate observed-data preservation error.
    # ---------------------------------------------------------------

    if observed_mask.any():

        preservation_error = torch.max(
            torch.abs(
                reconstruction[observed_mask]
                - corrupted[observed_mask]
            )
        ).item()

    else:

        preservation_error = float("nan")

    # ---------------------------------------------------------------
    # Validate observed-data preservation.
    # ---------------------------------------------------------------

    if (
        preservation_error
        > OBSERVED_PRESERVATION_TOLERANCE
    ):

        raise RuntimeError(
            "Observed-data preservation failed. "
            f"Maximum error="
            f"{preservation_error:.6e}"
        )

    # ===============================================================
    # PREPARE TENSORS FOR CANONICAL METRICS
    # ===============================================================
    #
    # The controlled reconstruction has:
    #
    #     reconstruction : [C,D,H,W]
    #     target         : [C,D,H,W]
    #
    # The canonical reconstruction metrics require:
    #
    #     [B,C,D,H,W]
    #
    # Therefore, add exactly one batch dimension.
    # ===============================================================

    reconstruction_metric = (
        reconstruction.unsqueeze(0)
    )

    target_metric = (
        target.unsqueeze(0)
    )

    mask_metric = (
        mask.unsqueeze(0)
    )

    # ---------------------------------------------------------------
    # Validate metric tensor dimensions.
    # ---------------------------------------------------------------

    if reconstruction_metric.ndim != 5:

        raise RuntimeError(
            "Reconstruction metric tensor must "
            "have shape [B,C,D,H,W]. "
            f"Got {reconstruction_metric.ndim} "
            "dimensions."
        )

    if target_metric.ndim != 5:

        raise RuntimeError(
            "Target metric tensor must "
            "have shape [B,C,D,H,W]. "
            f"Got {target_metric.ndim} "
            "dimensions."
        )

    if mask_metric.ndim != 5:

        raise RuntimeError(
            "Mask metric tensor must "
            "have shape [B,C,D,H,W]. "
            f"Got {mask_metric.ndim} "
            "dimensions."
        )

    # ---------------------------------------------------------------
    # Validate metric tensor shapes.
    # ---------------------------------------------------------------

    if (
        reconstruction_metric.shape
        != target_metric.shape
    ):

        raise RuntimeError(
            "Reconstruction and target metric "
            "shapes do not match. "
            f"Reconstruction="
            f"{tuple(reconstruction_metric.shape)}, "
            f"Target="
            f"{tuple(target_metric.shape)}"
        )

    # ===============================================================
    # GLOBAL RECONSTRUCTION METRICS
    # ===============================================================

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

    # ===============================================================
    # MISSING-REGION METRICS
    # ===============================================================
    #
    # Boolean indexing:
    #
    #     reconstruction[missing_mask]
    #
    # produces a flattened 1-D tensor.
    #
    # The canonical 3-D reconstruction metrics require
    # [B,C,D,H,W], so regional MAE and RMSE are calculated
    # directly from the missing-region error.
    # ===============================================================

    if missing_mask.any():

        missing_error = (
            reconstruction[missing_mask]
            - target[missing_mask]
        )

        missing_mae = torch.mean(
            torch.abs(
                missing_error
            )
        ).item()

        missing_rmse = torch.sqrt(
            torch.mean(
                missing_error ** 2
            )
        ).item()

    else:

        missing_mae = 0.0

        missing_rmse = 0.0

    # ---------------------------------------------------------------
    # Calculate runtime.
    # ---------------------------------------------------------------

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    # ---------------------------------------------------------------
    # Final numerical validation.
    # ---------------------------------------------------------------

    check_finite(
        "reconstruction",
        reconstruction
    )

    # ---------------------------------------------------------------
    # Validate metric outputs.
    # ---------------------------------------------------------------

    metric_values = {
        "MAE": global_mae,
        "RMSE": global_rmse,
        "PSNR": global_psnr,
        "SNR": global_snr,
        "SSIM": global_ssim,
        "Missing_MAE": missing_mae,
        "Missing_RMSE": missing_rmse,
    }

    for metric_name, metric_value in (
        metric_values.items()
    ):

        if not np.isfinite(
            float(metric_value)
        ):

            raise ValueError(
                f"{metric_name} produced a "
                "non-finite value."
            )

    # ---------------------------------------------------------------
    # Return complete case-level record.
    # ---------------------------------------------------------------

    return {
        "Case_ID":
            case_id,

        "Method":
            "Nearest Neighbor",

        "geological_mode":
            geological_mode,

        "mask_mode":
            mask_mode,

        "requested_missing_rate":
            missing_rate,

        "actual_missing_rate":
            actual_missing_rate,

        "seed":
            seed,

        "channels":
            corrupted.shape[0],

        "depth":
            corrupted.shape[1],

        "height":
            corrupted.shape[2],

        "width":
            corrupted.shape[3],

        "MAE":
            float(global_mae),

        "RMSE":
            float(global_rmse),

        "PSNR":
            float(global_psnr),

        "SNR":
            float(global_snr),

        "SSIM":
            float(global_ssim),

        "Missing_MAE":
            float(missing_mae),

        "Missing_RMSE":
            float(missing_rmse),

        "Observed_Preservation_Error":
            float(preservation_error),

        "Runtime_Seconds":
            float(runtime_seconds),

        "Status":
            "PASS",

        "Error":
            "",
    }


# ====================================================================
# SUMMARY CALCULATION
# ====================================================================

def calculate_summary(records):
    """
    Calculate grouped summary statistics.

    Grouping factors:

        geological_mode
        mask_mode
        requested_missing_rate

    The five seeds are summarized within each group.
    """

    groups = {}

    # ---------------------------------------------------------------
    # Group successful records.
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Calculate summary statistics.
    # ---------------------------------------------------------------

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
                np.mean(
                    values(field)
                )
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
                "Method":
                    "Nearest Neighbor",

                "geological_mode":
                    geological_mode,

                "mask_mode":
                    mask_mode,

                "requested_missing_rate":
                    missing_rate,

                "n_seeds":
                    len(group),

                "Mean_MAE":
                    mean("MAE"),

                "Std_MAE":
                    std("MAE"),

                "Mean_RMSE":
                    mean("RMSE"),

                "Std_RMSE":
                    std("RMSE"),

                "Mean_PSNR":
                    mean("PSNR"),

                "Std_PSNR":
                    std("PSNR"),

                "Mean_SNR":
                    mean("SNR"),

                "Std_SNR":
                    std("SNR"),

                "Mean_SSIM":
                    mean("SSIM"),

                "Std_SSIM":
                    std("SSIM"),

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
    Write a list of dictionaries to CSV.
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
    Run the controlled nearest-neighbor experiment.
    """

    print()
    print("=" * 72)

    print(
        "NEAREST NEIGHBOR — "
        "CONTROLLED EXPERIMENTAL MATRIX"
    )

    print("=" * 72)

    # ---------------------------------------------------------------
    # Build the COMPLETE 750-case matrix first.
    # ---------------------------------------------------------------

    all_cases = build_controlled_cases()

    expected_cases = len(
        all_cases
    )

    # ---------------------------------------------------------------
    # Apply the configured execution limit.
    # ---------------------------------------------------------------

    cases_to_run = (
        select_cases_for_execution(
            all_cases
        )
    )

    cases_to_execute = len(
        cases_to_run
    )

    # ---------------------------------------------------------------
    # Display experiment configuration.
    # ---------------------------------------------------------------

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
        "Nearest Neighbor"
    )

    print("=" * 72)

    # ---------------------------------------------------------------
    # Execute selected cases.
    # ---------------------------------------------------------------

    records = []

    successful_cases = 0

    failed_cases = 0

    for index, case in enumerate(
        cases_to_run,
        start=1
    ):

        print()
        print(
            f"Case {index}/{cases_to_execute} "
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
                f"  MAE          : "
                f"{result['MAE']:.6f}"
            )

            print(
                f"  RMSE         : "
                f"{result['RMSE']:.6f}"
            )

            print(
                f"  SSIM         : "
                f"{result['SSIM']:.6f}"
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
                "Case_ID":
                    case["Case_ID"],

                "Method":
                    "Nearest Neighbor",

                "geological_mode":
                    case["geological_mode"],

                "mask_mode":
                    case["mask_mode"],

                "requested_missing_rate":
                    case["missing_rate"],

                "actual_missing_rate":
                    "",

                "seed":
                    case["seed"],

                "channels":
                    "",

                "depth":
                    "",

                "height":
                    "",

                "width":
                    "",

                "MAE":
                    "",

                "RMSE":
                    "",

                "PSNR":
                    "",

                "SNR":
                    "",

                "SSIM":
                    "",

                "Missing_MAE":
                    "",

                "Missing_RMSE":
                    "",

                "Observed_Preservation_Error":
                    "",

                "Runtime_Seconds":
                    "",

                "Status":
                    "FAIL",

                "Error":
                    str(error),
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
    # Save raw case-level results.
    # ---------------------------------------------------------------

    write_csv(
        RAW_RESULTS_FILE,
        records
    )

    # ---------------------------------------------------------------
    # Calculate summary using successful cases.
    # ---------------------------------------------------------------

    successful_records = [
        record
        for record in records
        if record["Status"] == "PASS"
    ]

    summary = calculate_summary(
        successful_records
    )

    # ---------------------------------------------------------------
    # Save summary results.
    # ---------------------------------------------------------------

    write_csv(
        SUMMARY_RESULTS_FILE,
        summary
    )

    # ---------------------------------------------------------------
    # Final experiment report.
    # ---------------------------------------------------------------

    print()
    print("=" * 72)

    print(
        "NEAREST NEIGHBOR "
        "CONTROLLED MATRIX COMPLETE"
    )

    print("=" * 72)

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
    # Determine overall status.
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
            "Nearest Neighbor cases failed."
        )

    print("=" * 72)


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()