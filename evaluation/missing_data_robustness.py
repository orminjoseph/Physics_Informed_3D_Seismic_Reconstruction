"""
======================================================================
MISSING-DATA ROBUSTNESS EVALUATION
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate the robustness of the trained reconstruction framework when
the percentage of missing seismic data is systematically increased.

Controlled missing-data levels:
    10%
    20%
    30%
    40%
    50%

For every missing-data level:

    1. Obtain clean target seismic patches.
    2. Generate a reproducible random missing-data mask.
    3. Create the corrupted input by zero-filling missing voxels.
    4. Reconstruct using the trained Physics-Informed 3D network.
    5. Apply the production Evaluator.
    6. Calculate reconstruction quality metrics.
    7. Calculate missing-region reconstruction metrics.
    8. Calculate aleatoric uncertainty.
    9. Calculate epistemic uncertainty.
   10. Calculate predictive uncertainty.
   11. Verify observed-data preservation.
   12. Record the actual measured missing-data fraction.

Important
---------
This is a CONTROLLED MISSING-DATA ROBUSTNESS experiment.

The clean target is used only to construct the controlled
corrupted input and remains the reference for evaluation.

The clean target is NEVER supplied to the reconstruction model
as an input.

The same dataset samples are evaluated at every missing-data
level so that the missing-data percentage is the principal
experimental variable.

Author: Ormin Joseph
======================================================================
"""


# ======================================================================
# 1. STANDARD-LIBRARY IMPORTS
# ======================================================================

from pathlib import Path
import json
import random


# ======================================================================
# 2. SCIENTIFIC-COMPUTING IMPORTS
# ======================================================================

import numpy as np
import pandas as pd
import torch


# ======================================================================
# 3. PROJECT IMPORTS
# ======================================================================

from dataset.build_dataset import build_dataset

from models.network import Network3D

from inference.predictor import Predictor

from evaluation.evaluator import Evaluator


# ======================================================================
# 4. PROJECT CONFIGURATION
# ======================================================================

from utils.config import (
    DATASET_MODE,
    EXPERIMENT_NAME,

    CHECKPOINT_DIR,
    REPORT_DIR,

    BATCH_SIZE,

    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,

    MC_DROPOUT_SAMPLES,

    DEVICE,

    MASK_OBSERVED_VALUE,
    MASK_MISSING_VALUE,

    OBSERVED_PRESERVATION_TOLERANCE,

    MISSING_DATA_ROBUSTNESS_LEVELS,
    MISSING_DATA_ROBUSTNESS_NUM_SAMPLES,
    MISSING_DATA_ROBUSTNESS_SEED,
)


# ======================================================================
# 5. OUTPUT PATHS
# ======================================================================

# Convert the configured report directory to a Path object.
REPORT_PATH = Path(
    REPORT_DIR
)

# Create the report directory if necessary.
REPORT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ----------------------------------------------------------------------
# Main patch-level results.
# ----------------------------------------------------------------------

PATCH_RESULTS_FILE = (
    REPORT_PATH
    /
    "missing_data_robustness_samples.csv"
)


# ----------------------------------------------------------------------
# Aggregated results by missing-data level.
# ----------------------------------------------------------------------

SUMMARY_RESULTS_FILE = (
    REPORT_PATH
    /
    "missing_data_robustness.csv"
)


# ----------------------------------------------------------------------
# Experiment metadata.
# ----------------------------------------------------------------------

METADATA_FILE = (
    REPORT_PATH
    /
    "missing_data_robustness_metadata.json"
)


# ======================================================================
# 6. SINGLE-SAMPLE DATASET ADAPTER
# ======================================================================

class SingleSampleDataset(
    torch.utils.data.Dataset
):
    """
    Adapt one dataset sample to the dictionary format expected by
    evaluation.Evaluator.
    """

    def __init__(
        self,
        sample
    ):
        """
        Store one sample.

        The current dataset structure is expected to contain:

            input
            target
            mask
            velocity
            mask_type
            geological_mode
        """

        self.sample = sample


    def __len__(
        self
    ):
        """
        Return the number of samples.

        This adapter contains exactly one sample.
        """

        return 1


    def __getitem__(
        self,
        index
    ):
        """
        Return the sample in Evaluator-compatible dictionary format.
        """

        if index != 0:

            raise IndexError(
                "SingleSampleDataset contains "
                "only one sample."
            )


        (
            input_cube,
            target_cube,
            mask,
            velocity,
            mask_type,
            geological_mode,
        ) = self.sample


        return {
            "input": input_cube,
            "target": target_cube,
            "mask": mask,
            "velocity": velocity,
            "mask_type": mask_type,
            "geological_mode": geological_mode,
        }


# ======================================================================
# 7. NUMERICAL CONVERSION HELPER
# ======================================================================

def to_float(
    value
):
    """
    Convert a scalar tensor or numerical value to Python float.
    """

    if torch.is_tensor(
        value
    ):

        return float(
            value.detach()
            .cpu()
            .item()
        )


    return float(
        value
    )


# ======================================================================
# 8. RANDOM MASK GENERATION
# ======================================================================

def create_missing_mask(
    target,
    missing_probability,
    seed
):
    """
    Create a reproducible random missing-data mask.

    Parameters
    ----------
    target : torch.Tensor
        Clean seismic target with shape [C,D,H,W].

    missing_probability : float
        Fraction of voxels to remove.

    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        Binary mask:

            1 = observed
            0 = missing
    """

    # --------------------------------------------------------------
    # Validate missing probability.
    # --------------------------------------------------------------

    if not (
        0.0
        <=
        missing_probability
        <
        1.0
    ):

        raise ValueError(
            "Missing probability must satisfy "
            "0.0 <= probability < 1.0. "
            f"Received: {missing_probability}"
        )


    # --------------------------------------------------------------
    # Use a CPU generator so that mask generation is reproducible
    # independently of whether inference runs on CPU or GPU.
    # --------------------------------------------------------------

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        int(seed)
    )


    # --------------------------------------------------------------
    # Generate random values on CPU.
    # --------------------------------------------------------------

    random_values = torch.rand(
        target.shape,
        generator=generator,
        device="cpu",
    )


    # --------------------------------------------------------------
    # Move the random values to the target device.
    # --------------------------------------------------------------

    random_values = random_values.to(
        target.device
    )


    # --------------------------------------------------------------
    # Observed voxels:
    #
    # random value >= missing probability
    #
    # Missing voxels:
    #
    # random value < missing probability
    # --------------------------------------------------------------

    mask = (
        random_values
        >=
        missing_probability
    ).to(
        dtype=target.dtype
    )


    return mask


# ======================================================================
# 9. CREATE CONTROLLED CORRUPTED INPUT
# ======================================================================

def create_corrupted_input(
    target,
    mask
):
    """
    Create a zero-filled corrupted seismic input.

    Observed voxels retain their original amplitudes.

    Missing voxels are replaced with zero.
    """

    corrupted = (
        target
        *
        mask
    )


    return corrupted


# ======================================================================
# 10. VALIDATE TENSOR STRUCTURE
# ======================================================================

def validate_sample_tensors(
    corrupted,
    target,
    mask
):
    """
    Validate input, target and mask tensors.
    """

    if not (
        torch.is_tensor(corrupted)
        and
        torch.is_tensor(target)
        and
        torch.is_tensor(mask)
    ):

        raise TypeError(
            "Input, target and mask must all "
            "be PyTorch tensors."
        )


    # --------------------------------------------------------------
    # Individual dataset samples must be [C,D,H,W].
    # --------------------------------------------------------------

    if not (
        corrupted.ndim == 4
        and
        target.ndim == 4
        and
        mask.ndim == 4
    ):

        raise ValueError(
            "Input, target and mask must have "
            "shape [C,D,H,W]. "
            f"Received input={tuple(corrupted.shape)}, "
            f"target={tuple(target.shape)}, "
            f"mask={tuple(mask.shape)}."
        )


    # --------------------------------------------------------------
    # Shapes must agree.
    # --------------------------------------------------------------

    if not (
        corrupted.shape
        ==
        target.shape
        ==
        mask.shape
    ):

        raise ValueError(
            "Input, target and mask must have "
            "identical shapes."
        )


    # --------------------------------------------------------------
    # Validate mask values.
    # --------------------------------------------------------------

    unique_values = torch.unique(
        mask
    )


    allowed = (
        torch.isclose(
            unique_values,
            torch.tensor(
                MASK_OBSERVED_VALUE,
                device=unique_values.device,
                dtype=unique_values.dtype,
            )
        )
        |
        torch.isclose(
            unique_values,
            torch.tensor(
                MASK_MISSING_VALUE,
                device=unique_values.device,
                dtype=unique_values.dtype,
            )
        )
    )


    if not torch.all(
        allowed
    ):

        raise ValueError(
            "Generated mask contains values "
            "other than the configured observed "
            "and missing values."
        )


# ======================================================================
# 11. SET GLOBAL REPRODUCIBILITY
# ======================================================================

def set_global_seed(
    seed
):
    """
    Set the major random-number generators.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


# ======================================================================
# 12. MAIN EVALUATION
# ======================================================================

def main():

    print()
    print("=" * 80)
    print("MISSING-DATA ROBUSTNESS EVALUATION")
    print("=" * 80)


    # ==================================================================
    # STEP 1: DISPLAY CONFIGURATION
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 1: CONFIGURATION")
    print("-" * 80)

    print(
        f"Dataset mode                         : "
        f"{DATASET_MODE}"
    )

    print(
        f"Experiment                           : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Device                               : "
        f"{DEVICE}"
    )

    print(
        f"MC-Dropout samples                   : "
        f"{MC_DROPOUT_SAMPLES}"
    )

    print(
        f"Attention                            : "
        f"{USE_ATTENTION}"
    )

    print(
        f"Residual connections                 : "
        f"{USE_RESIDUAL}"
    )

    print(
        f"Uncertainty                          : "
        f"{USE_UNCERTAINTY}"
    )

    print(
        f"Missing-data levels                  : "
        f"{MISSING_DATA_ROBUSTNESS_LEVELS}"
    )

    print(
        f"Maximum samples per level            : "
        f"{MISSING_DATA_ROBUSTNESS_NUM_SAMPLES}"
    )

    print(
        f"Random seed                          : "
        f"{MISSING_DATA_ROBUSTNESS_SEED}"
    )

    print(
        f"Observed-data tolerance              : "
        f"{OBSERVED_PRESERVATION_TOLERANCE}"
    )


    # ==================================================================
    # STEP 2: SET REPRODUCIBILITY
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 2: REPRODUCIBILITY")
    print("-" * 80)

    set_global_seed(
        MISSING_DATA_ROBUSTNESS_SEED
    )

    print(
        "Global random seeds initialized."
    )


    # ==================================================================
    # STEP 3: VALIDATE MISSING-DATA LEVELS
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 3: VALIDATE MISSING-DATA LEVELS")
    print("-" * 80)

    for level in (
        MISSING_DATA_ROBUSTNESS_LEVELS
    ):

        if not (
            0.0
            <=
            float(level)
            <
            1.0
        ):

            raise ValueError(
                "Every missing-data level must "
                "satisfy 0.0 <= level < 1.0. "
                f"Invalid value: {level}"
            )


    print(
        "All missing-data levels are valid."
    )


    # ==================================================================
    # STEP 4: LOAD DATASET
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 4: BUILD DATASET")
    print("-" * 80)

    dataset = build_dataset()


    total_dataset_samples = len(
        dataset
    )


    if total_dataset_samples <= 0:

        raise RuntimeError(
            "The configured dataset contains "
            "no samples."
        )


    number_of_samples = min(
        MISSING_DATA_ROBUSTNESS_NUM_SAMPLES,
        total_dataset_samples,
    )


    print(
        f"Available dataset samples           : "
        f"{total_dataset_samples}"
    )

    print(
        f"Samples evaluated per level         : "
        f"{number_of_samples}"
    )


    # ==================================================================
    # STEP 5: LOAD CHECKPOINT
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 5: LOAD TRAINED MODEL")
    print("-" * 80)

    checkpoint_path = (
        Path(
            CHECKPOINT_DIR
        )
        /
        "best_model.pth"
    )


    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Trained model checkpoint was not found:\n"
            f"{checkpoint_path}\n\n"
            "Ensure that best_model.pth exists in "
            "the configured checkpoint directory."
        )


    print(
        f"Checkpoint                          : "
        f"{checkpoint_path}"
    )


    # ==================================================================
    # STEP 6: CREATE PRODUCTION MODEL
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 6: CREATE PRODUCTION MODEL")
    print("-" * 80)

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
    )


    print(
        "Network3D created successfully."
    )


    # ==================================================================
    # STEP 7: LOAD CHECKPOINT THROUGH PREDICTOR
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 7: LOAD CHECKPOINT")
    print("-" * 80)

    predictor = Predictor(
        model=model,
        checkpoint_path=str(
            checkpoint_path
        ),
        device=DEVICE,
    )


    loaded_model = predictor.model


    print(
        "Checkpoint loaded successfully."
    )


    # ==================================================================
    # STEP 8: CREATE PRODUCTION EVALUATOR
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 8: CREATE PRODUCTION EVALUATOR")
    print("-" * 80)

    evaluator = Evaluator(
        model=loaded_model,
        device=DEVICE,
        mc_samples=MC_DROPOUT_SAMPLES,
    )


    print(
        "Evaluator created successfully."
    )


    # ==================================================================
    # STEP 9: STORAGE
    # ==================================================================

    patch_results = []


    # ==================================================================
    # STEP 10: EVALUATE EACH MISSING-DATA LEVEL
    # ==================================================================

    for level_index, missing_level in enumerate(
        MISSING_DATA_ROBUSTNESS_LEVELS
    ):

        print()
        print("=" * 80)

        print(
            f"MISSING-DATA LEVEL: "
            f"{missing_level * 100:.0f}%"
        )

        print("=" * 80)


        # --------------------------------------------------------------
        # Evaluate exactly the same dataset sample IDs at every level.
        # --------------------------------------------------------------

        for sample_index in range(
            number_of_samples
        ):

            print()
            print(
                f"Sample "
                f"{sample_index + 1}/"
                f"{number_of_samples}"
            )


            # ----------------------------------------------------------
            # Obtain clean dataset sample.
            # ----------------------------------------------------------

            sample = dataset[
                sample_index
            ]


            # ----------------------------------------------------------
            # Validate expected tuple structure.
            # ----------------------------------------------------------

            if len(sample) < 6:

                raise ValueError(
                    "The configured dataset sample does not "
                    "match the expected six-element structure."
                )


            (
                _original_input,
                target,
                _original_mask,
                velocity,
                mask_type,
                geological_mode,
            ) = sample


            # ----------------------------------------------------------
            # Validate target.
            # ----------------------------------------------------------

            if not torch.is_tensor(
                target
            ):

                raise TypeError(
                    "Dataset target must be a PyTorch tensor."
                )


            # ----------------------------------------------------------
            # Ensure target has [C,D,H,W] structure.
            # ----------------------------------------------------------

            if target.ndim != 4:

                raise ValueError(
                    "Target must have shape [C,D,H,W]. "
                    f"Received: {tuple(target.shape)}"
                )


            # ----------------------------------------------------------
            # Create deterministic seed for this exact:
            #
            #     missing level + sample
            #
            # combination.
            # ----------------------------------------------------------

            mask_seed = (
                int(
                    MISSING_DATA_ROBUSTNESS_SEED
                )
                +
                (
                    level_index
                    *
                    100000
                )
                +
                sample_index
            )


            # ----------------------------------------------------------
            # Generate controlled missing-data mask.
            # ----------------------------------------------------------

            mask = create_missing_mask(
                target=target,
                missing_probability=float(
                    missing_level
                ),
                seed=mask_seed,
            )


            # ----------------------------------------------------------
            # Generate corrupted input.
            # ----------------------------------------------------------

            corrupted = create_corrupted_input(
                target=target,
                mask=mask,
            )


            # ----------------------------------------------------------
            # Validate generated tensors.
            # ----------------------------------------------------------

            validate_sample_tensors(
                corrupted=corrupted,
                target=target,
                mask=mask,
            )


            # ----------------------------------------------------------
            # Calculate actual missing fraction.
            # ----------------------------------------------------------

            actual_missing_fraction = float(
                torch.mean(
                    (
                        mask
                        ==
                        MASK_MISSING_VALUE
                    ).to(
                        torch.float32
                    )
                ).item()
            )


            # ----------------------------------------------------------
            # Construct evaluator sample.
            # ----------------------------------------------------------

            evaluator_sample = (
                corrupted,
                target,
                mask,
                velocity,
                mask_type,
                geological_mode,
            )


            # ----------------------------------------------------------
            # Create one-sample Dataset.
            # ----------------------------------------------------------

            single_dataset = (
                SingleSampleDataset(
                    evaluator_sample
                )
            )


            # ----------------------------------------------------------
            # Create DataLoader.
            # ----------------------------------------------------------

            dataloader = torch.utils.data.DataLoader(
                single_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )


            # ----------------------------------------------------------
            # Run the production evaluator.
            # ----------------------------------------------------------

            evaluation_result = (
                evaluator.evaluate(
                    dataloader
                )
            )


            # ----------------------------------------------------------
            # Extract reconstruction metrics.
            # ----------------------------------------------------------

            mae = to_float(
                evaluation_result[
                    "mae"
                ]
            )

            rmse = to_float(
                evaluation_result[
                    "rmse"
                ]
            )

            psnr = to_float(
                evaluation_result[
                    "psnr"
                ]
            )

            snr = to_float(
                evaluation_result[
                    "snr"
                ]
            )

            ssim = to_float(
                evaluation_result[
                    "ssim"
                ]
            )


            # ----------------------------------------------------------
            # Extract missing-region metrics.
            # ----------------------------------------------------------

            missing_mae = to_float(
                evaluation_result[
                    "missing_mae"
                ]
            )

            missing_rmse = to_float(
                evaluation_result[
                    "missing_rmse"
                ]
            )


            # ----------------------------------------------------------
            # Extract observed-region metrics.
            # ----------------------------------------------------------

            observed_mae = to_float(
                evaluation_result[
                    "observed_mae"
                ]
            )

            observed_rmse = to_float(
                evaluation_result[
                    "observed_rmse"
                ]
            )


            # ----------------------------------------------------------
            # Extract uncertainty decomposition.
            # ----------------------------------------------------------

            aleatoric_variance = to_float(
                evaluation_result[
                    "aleatoric_variance"
                ]
            )

            epistemic_variance = to_float(
                evaluation_result[
                    "epistemic_variance"
                ]
            )

            predictive_variance = to_float(
                evaluation_result[
                    "predictive_variance"
                ]
            )

            predictive_std = to_float(
                evaluation_result[
                    "predictive_std"
                ]
            )


            # ----------------------------------------------------------
            # Extract quality-control statistics.
            # ----------------------------------------------------------

            observed_preservation_error = to_float(
                evaluation_result[
                    "observed_preservation_error"
                ]
            )

            measured_missing_rate = to_float(
                evaluation_result[
                    "measured_missing_rate"
                ]
            )


            # ----------------------------------------------------------
            # Verify observed-data preservation.
            # ----------------------------------------------------------

            if (
                observed_preservation_error
                >
                OBSERVED_PRESERVATION_TOLERANCE
            ):

                raise RuntimeError(
                    "Observed-data preservation tolerance "
                    "was exceeded.\n"
                    f"Missing level = "
                    f"{missing_level:.2f}\n"
                    f"Sample = "
                    f"{sample_index}\n"
                    f"Observed preservation error = "
                    f"{observed_preservation_error:.6e}\n"
                    f"Allowed tolerance = "
                    f"{OBSERVED_PRESERVATION_TOLERANCE:.6e}"
                )


            # ----------------------------------------------------------
            # Verify uncertainty values.
            # ----------------------------------------------------------

            uncertainty_values = (
                aleatoric_variance,
                epistemic_variance,
                predictive_variance,
                predictive_std,
            )


            if not all(
                np.isfinite(
                    value
                )
                for value
                in uncertainty_values
            ):

                raise RuntimeError(
                    "Non-finite uncertainty value detected."
                )


            if any(
                value < 0.0
                for value
                in (
                    aleatoric_variance,
                    epistemic_variance,
                    predictive_variance,
                )
            ):

                raise RuntimeError(
                    "Negative uncertainty variance detected."
                )


            # ----------------------------------------------------------
            # Verify reconstruction metrics.
            # ----------------------------------------------------------

            metric_values = (
                mae,
                rmse,
                psnr,
                snr,
                ssim,
                missing_mae,
                missing_rmse,
                observed_mae,
                observed_rmse,
            )


            if not all(
                np.isfinite(
                    value
                )
                for value
                in metric_values
            ):

                raise RuntimeError(
                    "Non-finite reconstruction metric detected."
                )


            # ----------------------------------------------------------
            # Store patch-level result.
            # ----------------------------------------------------------

            patch_results.append(
                {
                    "Missing_Level": float(
                        missing_level
                    ),

                    "Missing_Level_Percent": (
                        float(
                            missing_level
                        )
                        *
                        100.0
                    ),

                    "Sample_Index": int(
                        sample_index
                    ),

                    "Mask_Seed": int(
                        mask_seed
                    ),

                    "MAE": mae,

                    "RMSE": rmse,

                    "PSNR": psnr,

                    "SNR": snr,

                    "SSIM": ssim,

                    "Missing_MAE": missing_mae,

                    "Missing_RMSE": missing_rmse,

                    "Observed_MAE": observed_mae,

                    "Observed_RMSE": observed_rmse,

                    "Aleatoric_Variance":
                        aleatoric_variance,

                    "Epistemic_Variance":
                        epistemic_variance,

                    "Predictive_Variance":
                        predictive_variance,

                    "Predictive_Std":
                        predictive_std,

                    "Requested_Missing_Rate":
                        float(
                            missing_level
                        ),

                    "Measured_Missing_Rate":
                        measured_missing_rate,

                    "Observed_Preservation_Error":
                        observed_preservation_error,

                    "MC_Samples": int(
                        MC_DROPOUT_SAMPLES
                    ),
                }
            )


            # ----------------------------------------------------------
            # Display result.
            # ----------------------------------------------------------

            print(
                f"  MAE                  : "
                f"{mae:.6f}"
            )

            print(
                f"  Missing MAE          : "
                f"{missing_mae:.6f}"
            )

            print(
                f"  Missing RMSE         : "
                f"{missing_rmse:.6f}"
            )

            print(
                f"  PSNR                 : "
                f"{psnr:.6f}"
            )

            print(
                f"  SSIM                 : "
                f"{ssim:.6f}"
            )

            print(
                f"  Predictive Std       : "
                f"{predictive_std:.6f}"
            )

            print(
                f"  Measured missing    : "
                f"{measured_missing_rate:.6f}"
            )


    # ==================================================================
    # STEP 11: VALIDATE PATCH RESULTS
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 11: VALIDATE RESULTS")
    print("-" * 80)


    if not patch_results:

        raise RuntimeError(
            "No missing-data robustness results "
            "were generated."
        )


    patch_dataframe = pd.DataFrame(
        patch_results
    )


    # --------------------------------------------------------------
    # Check all numeric columns for finite values.
    # --------------------------------------------------------------

    numeric_columns = (
        patch_dataframe
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )


    for column in numeric_columns:

        values = (
            patch_dataframe[
                column
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        if not np.all(
            np.isfinite(
                values
            )
        ):

            raise RuntimeError(
                "Non-finite values detected in "
                f"column: {column}"
            )


    print(
        "Patch-level results passed "
        "finite-value validation."
    )


    # ==================================================================
    # STEP 12: AGGREGATE RESULTS BY MISSING LEVEL
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 12: AGGREGATE RESULTS")
    print("-" * 80)


    summary_dataframe = (
        patch_dataframe
        .groupby(
            "Missing_Level",
            as_index=False
        )
        .agg(
            Number_of_Samples=(
                "Sample_Index",
                "count"
            ),

            Mean_MAE=(
                "MAE",
                "mean"
            ),

            Std_MAE=(
                "MAE",
                "std"
            ),

            Mean_RMSE=(
                "RMSE",
                "mean"
            ),

            Std_RMSE=(
                "RMSE",
                "std"
            ),

            Mean_PSNR=(
                "PSNR",
                "mean"
            ),

            Std_PSNR=(
                "PSNR",
                "std"
            ),

            Mean_SNR=(
                "SNR",
                "mean"
            ),

            Std_SNR=(
                "SNR",
                "std"
            ),

            Mean_SSIM=(
                "SSIM",
                "mean"
            ),

            Std_SSIM=(
                "SSIM",
                "std"
            ),

            Mean_Missing_MAE=(
                "Missing_MAE",
                "mean"
            ),

            Std_Missing_MAE=(
                "Missing_MAE",
                "std"
            ),

            Mean_Missing_RMSE=(
                "Missing_RMSE",
                "mean"
            ),

            Std_Missing_RMSE=(
                "Missing_RMSE",
                "std"
            ),

            Mean_Observed_MAE=(
                "Observed_MAE",
                "mean"
            ),

            Mean_Observed_RMSE=(
                "Observed_RMSE",
                "mean"
            ),

            Mean_Aleatoric_Variance=(
                "Aleatoric_Variance",
                "mean"
            ),

            Mean_Epistemic_Variance=(
                "Epistemic_Variance",
                "mean"
            ),

            Mean_Predictive_Variance=(
                "Predictive_Variance",
                "mean"
            ),

            Mean_Predictive_Std=(
                "Predictive_Std",
                "mean"
            ),

            Mean_Measured_Missing_Rate=(
                "Measured_Missing_Rate",
                "mean"
            ),

            Max_Observed_Preservation_Error=(
                "Observed_Preservation_Error",
                "max"
            ),
        )
    )


    # --------------------------------------------------------------
    # A standard deviation is undefined for a single observation.
    # Replace such NaN values with zero.
    # --------------------------------------------------------------

    summary_dataframe = (
        summary_dataframe
        .fillna(0.0)
    )


    # ==================================================================
    # STEP 13: SAVE PATCH-LEVEL RESULTS
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 13: SAVE PATCH-LEVEL RESULTS")
    print("-" * 80)


    patch_dataframe.to_csv(
        PATCH_RESULTS_FILE,
        index=False
    )


    print(
        f"Patch-level results saved:\n"
        f"{PATCH_RESULTS_FILE}"
    )


    # ==================================================================
    # STEP 14: SAVE AGGREGATED RESULTS
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 14: SAVE AGGREGATED RESULTS")
    print("-" * 80)


    summary_dataframe.to_csv(
        SUMMARY_RESULTS_FILE,
        index=False
    )


    print(
        f"Summary results saved:\n"
        f"{SUMMARY_RESULTS_FILE}"
    )


    # ==================================================================
    # STEP 15: SAVE EXPERIMENT METADATA
    # ==================================================================

    print()
    print("-" * 80)
    print("STEP 15: SAVE EXPERIMENT METADATA")
    print("-" * 80)


    metadata = {

        "experiment":
            "Missing-Data Robustness Evaluation",

        "dataset_mode":
            str(
                DATASET_MODE
            ),

        "experiment_name":
            str(
                EXPERIMENT_NAME
            ),

        "checkpoint":
            str(
                checkpoint_path
            ),

        "device":
            str(
                DEVICE
            ),

        "attention":
            bool(
                USE_ATTENTION
            ),

        "residual":
            bool(
                USE_RESIDUAL
            ),

        "uncertainty":
            bool(
                USE_UNCERTAINTY
            ),

        "mc_dropout_samples":
            int(
                MC_DROPOUT_SAMPLES
            ),

        "missing_data_levels":
            [
                float(level)
                for level
                in MISSING_DATA_ROBUSTNESS_LEVELS
            ],

        "maximum_samples_per_level":
            int(
                MISSING_DATA_ROBUSTNESS_NUM_SAMPLES
            ),

        "actual_samples_per_level":
            int(
                number_of_samples
            ),

        "random_seed":
            int(
                MISSING_DATA_ROBUSTNESS_SEED
            ),

        "observed_preservation_tolerance":
            float(
                OBSERVED_PRESERVATION_TOLERANCE
            ),

        "mask_observed_value":
            float(
                MASK_OBSERVED_VALUE
            ),

        "mask_missing_value":
            float(
                MASK_MISSING_VALUE
            ),

        "method":
            "Controlled random voxel missingness",

        "missing_voxel_fill_value":
            0.0,

        "same_sample_ids_across_levels":
            True,

        "target_used_only_for_corruption_and_evaluation":
            True,
    }


    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as metadata_file:

        json.dump(
            metadata,
            metadata_file,
            indent=4
        )


    print(
        f"Metadata saved:\n"
        f"{METADATA_FILE}"
    )


    # ==================================================================
    # STEP 16: FINAL SUMMARY
    # ==================================================================

    print()
    print("=" * 80)
    print("MISSING-DATA ROBUSTNESS SUMMARY")
    print("=" * 80)

    print()

    print(
        summary_dataframe[
            [
                "Missing_Level",
                "Number_of_Samples",
                "Mean_MAE",
                "Mean_RMSE",
                "Mean_PSNR",
                "Mean_SNR",
                "Mean_SSIM",
                "Mean_Missing_MAE",
                "Mean_Missing_RMSE",
                "Mean_Predictive_Std",
                "Mean_Measured_Missing_Rate",
            ]
        ].to_string(
            index=False
        )
    )


    print()
    print("=" * 80)

    print(
        "MISSING-DATA ROBUSTNESS EVALUATION COMPLETE"
    )

    print("=" * 80)

    print()

    print(
        "Output files:"
    )

    print(
        f"1. {PATCH_RESULTS_FILE}"
    )

    print(
        f"2. {SUMMARY_RESULTS_FILE}"
    )

    print(
        f"3. {METADATA_FILE}"
    )

    print()


# ======================================================================
# 13. SCRIPT ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()