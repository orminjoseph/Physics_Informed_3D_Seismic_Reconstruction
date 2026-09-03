"""
=========================================================
Physics-Informed 3D Encoder-Decoder Network Test
=========================================================

Tests the complete Network3D including:

    1. Model construction
    2. Input validation
    3. Output shapes
    4. Output data types
    5. Finite output values
    6. Reconstruction output
    7. Travel-time output
    8. Travel-time positivity
    9. Travel-time scaling
   10. Log-variance output
   11. Uncertainty-enabled mode
   12. Uncertainty-disabled mode
   13. Batch processing
   14. Gradient propagation
   15. Invalid input dimensions
   16. NaN/Inf input rejection

Network outputs:

    reconstructed_cube
        [B, C, D, H, W]

    travel_time
        [B, C, D, H, W]

    log_variance
        [B, C, D, H, W]

Tensor convention:

    Input:
        [B, C, D, H, W]

Author: Ormin Joseph
=========================================================
"""

import torch

from models.network import Network3D
from utils.config import TRAVEL_TIME_SCALE


# =====================================================
# CONFIGURATION
# =====================================================

BATCH_SIZE = 2

IN_CHANNELS = 1

OUT_CHANNELS = 1

CUBE_SIZE = (16, 32, 32)


# =====================================================
# HELPER FUNCTION
# =====================================================

def create_input(
    batch_size=BATCH_SIZE
):
    """
    Create a synthetic seismic input tensor.

    Returns
    -------

    torch.Tensor

        Tensor with shape:

            [B, C, D, H, W]
    """

    return torch.randn(
        batch_size,
        IN_CHANNELS,
        *CUBE_SIZE,
        dtype=torch.float32
    )


# =====================================================
# TEST MODEL CONSTRUCTION
# =====================================================

def test_model_construction():

    print()
    print("Testing Network Construction")

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        use_uncertainty=True,
        use_residual=True,
        use_attention=True
    )

    assert model is not None

    print(
        "Network Construction Test: PASSED"
    )


# =====================================================
# TEST INPUT VALIDATION
# =====================================================

def test_input_validation():

    print()
    print("Testing Input Validation")

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS
    )

    valid_input = create_input()

    # -------------------------------------------------
    # Valid input should be accepted.
    # -------------------------------------------------

    model._validate_input(
        valid_input
    )

    print(
        "Valid input accepted."
    )

    # -------------------------------------------------
    # Invalid dimensionality.
    # -------------------------------------------------

    invalid_input = torch.randn(
        1,
        16,
        32,
        32
    )

    try:

        model._validate_input(
            invalid_input
        )

        raise AssertionError(
            "Network accepted an invalid "
            "4D input tensor."
        )

    except ValueError:

        print(
            "Invalid dimensionality correctly rejected."
        )

    print(
        "Input Validation Test: PASSED"
    )


# =====================================================
# TEST OUTPUT SHAPES
# =====================================================

def test_output_shapes():

    print()
    print("Testing Network Output Shapes")

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS
    )

    model.eval()

    x = create_input()

    with torch.no_grad():

        (
            reconstruction,
            travel_time,
            log_variance
        ) = model(x)

    expected_shape = torch.Size(
        (
            BATCH_SIZE,
            OUT_CHANNELS,
            *CUBE_SIZE
        )
    )

    print(
        f"Input Shape         : {x.shape}"
    )

    print(
        f"Reconstruction Shape: "
        f"{reconstruction.shape}"
    )

    print(
        f"Travel-Time Shape   : "
        f"{travel_time.shape}"
    )

    print(
        f"Log-Variance Shape  : "
        f"{log_variance.shape}"
    )

    assert reconstruction.shape == (
        expected_shape
    )

    assert travel_time.shape == (
        expected_shape
    )

    assert log_variance.shape == (
        expected_shape
    )

    print(
        "Output Shape Test: PASSED"
    )


# =====================================================
# TEST OUTPUT DATA TYPES
# =====================================================

def test_output_dtypes():

    print()
    print("Testing Network Output Data Types")

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS
    )

    model.eval()

    x = create_input()

    with torch.no_grad():

        (
            reconstruction,
            travel_time,
            log_variance
        ) = model(x)

    print(
        f"Reconstruction dtype: "
        f"{reconstruction.dtype}"
    )

    print(
        f"Travel-Time dtype   : "
        f"{travel_time.dtype}"
    )

    print(
        f"Log-Variance dtype  : "
        f"{log_variance.dtype}"
    )

    assert reconstruction.dtype == torch.float32

    assert travel_time.dtype == torch.float32

    assert log_variance.dtype == torch.float32

    print(
        "Output Data Type Test: PASSED"
    )


# =====================================================
# TEST FINITE OUTPUTS
# =====================================================

def test_finite_outputs():

    print()
    print("Testing Finite Network Outputs")

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS
    )

    model.eval()

    x = create_input()

    with torch.no_grad():

        (
            reconstruction,
            travel_time,
            log_variance
        ) = model(x)

    assert torch.isfinite(
        reconstruction
    ).all()

    assert torch.isfinite(
        travel_time
    ).all()

    assert torch.isfinite(
        log_variance
    ).all()

    print(
        "All network outputs contain "
        "finite values."
    )

    print(
        "Finite Output Test: PASSED"
    )


# =====================================================
# TEST RECONSTRUCTION OUTPUT
# =====================================================

def test_reconstruction_output():

    print()
    print("Testing Reconstruction Output")

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS
    )

    model.eval()

    x = create_input()

    with torch.no_grad():

        reconstruction, _, _ = model(x)

    print(
        f"Reconstruction Minimum: "
        f"{reconstruction.min().item():.6f}"
    )

    print(
        f"Reconstruction Maximum: "
        f"{reconstruction.max().item():.6f}"
    )

    # -------------------------------------------------
    # Reconstruction must remain unrestricted because
    # seismic amplitudes may be positive or negative.
    # -------------------------------------------------

    assert torch.isfinite(
        reconstruction
    ).all()

    print(
        "Reconstruction Output Test: PASSED"
    )


# =====================================================
# TEST TRAVEL-TIME OUTPUT
# =====================================================

def test_travel_time_output():

    print()
    print("Testing Travel-Time Output")

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS
    )

    model.eval()

    x = create_input()

    with torch.no_grad():

        _, travel_time, _ = model(x)

    minimum_travel_time = (
        travel_time.min().item()
    )

    maximum_travel_time = (
        travel_time.max().item()
    )

    print(
        f"Travel-Time Minimum: "
        f"{minimum_travel_time:.6e}"
    )

    print(
        f"Travel-Time Maximum: "
        f"{maximum_travel_time:.6e}"
    )

    # -------------------------------------------------
    # Softplus guarantees non-negative travel time.
    # -------------------------------------------------

    assert torch.all(
        travel_time >= 0.0
    )

    assert torch.isfinite(
        travel_time
    ).all()

    print(
        "Travel-Time Output Test: PASSED"
    )


# =====================================================
# TEST TRAVEL-TIME SCALING
# =====================================================

def test_travel_time_scaling():

    print()
    print("Testing Travel-Time Scaling")

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS
    )

    model.eval()

    x = create_input()

    with torch.no_grad():

        (
            _,
            travel_time,
            _
        ) = model(x)

    # -------------------------------------------------
    # Since:
    #
    # travel_time =
    #     TRAVEL_TIME_SCALE *
    #     Softplus(raw_travel_time)
    #
    # and Softplus(raw_travel_time) > 0,
    # the output should not exceed the configured
    # scale for this implementation only if the
    # normalized interpretation is bounded elsewhere.
    #
    # Therefore, we verify the configured scale is
    # positive rather than imposing an incorrect
    # upper-bound assumption.
    # -------------------------------------------------

    print(
        f"TRAVEL_TIME_SCALE: "
        f"{TRAVEL_TIME_SCALE}"
    )

    assert (
        TRAVEL_TIME_SCALE > 0
    )

    assert torch.all(
        travel_time >= 0.0
    )

    print(
        "Travel-Time Scaling Test: PASSED"
    )


# =====================================================
# TEST LOG-VARIANCE OUTPUT
# =====================================================

def test_log_variance_output():

    print()
    print("Testing Log-Variance Output")

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        use_uncertainty=True
    )

    model.eval()

    x = create_input()

    with torch.no_grad():

        _, _, log_variance = model(x)

    print(
        f"Log-Variance Minimum: "
        f"{log_variance.min().item():.6f}"
    )

    print(
        f"Log-Variance Maximum: "
        f"{log_variance.max().item():.6f}"
    )

    assert torch.isfinite(
        log_variance
    ).all()

    # -------------------------------------------------
    # Log variance is intentionally unrestricted.
    #
    # Therefore we DO NOT assert:
    #
    #     log_variance >= 0
    #
    # because negative log variance is mathematically
    # valid and corresponds to variance < 1.
    # -------------------------------------------------

    print(
        "Log-Variance Output Test: PASSED"
    )


# =====================================================
# TEST UNCERTAINTY-ENABLED MODE
# =====================================================

def test_uncertainty_enabled():

    print()
    print(
        "Testing Uncertainty-Enabled Mode"
    )

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        use_uncertainty=True
    )

    assert hasattr(
        model,
        "uncertainty_head"
    )

    x = create_input()

    model.eval()

    with torch.no_grad():

        _, _, log_variance = model(x)

    assert torch.isfinite(
        log_variance
    ).all()

    assert not torch.allclose(
        log_variance,
        torch.zeros_like(log_variance)
    )

    print(
        "Uncertainty head is active."
    )

    print(
        "Uncertainty-Enabled Test: PASSED"
    )


# =====================================================
# TEST UNCERTAINTY-DISABLED MODE
# =====================================================

def test_uncertainty_disabled():

    print()
    print(
        "Testing Uncertainty-Disabled Mode"
    )

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        use_uncertainty=False
    )

    # -------------------------------------------------
    # The uncertainty head should not exist when
    # uncertainty estimation is disabled.
    # -------------------------------------------------

    assert not hasattr(
        model,
        "uncertainty_head"
    )

    x = create_input()

    model.eval()

    with torch.no_grad():

        _, _, log_variance = model(x)

    # -------------------------------------------------
    # Network3D returns zeros when uncertainty is
    # disabled.
    # -------------------------------------------------

    assert torch.allclose(
        log_variance,
        torch.zeros_like(log_variance)
    )

    print(
        "Uncertainty-disabled mode correctly "
        "returns zero log variance."
    )

    print(
        "Uncertainty-Disabled Test: PASSED"
    )


# =====================================================
# TEST BATCH PROCESSING
# =====================================================

def test_batch_processing():

    print()
    print("Testing Batch Processing")

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS
    )

    model.eval()

    batch_sizes = [1, 2]

    for batch_size in batch_sizes:

        x = create_input(
            batch_size=batch_size
        )

        with torch.no_grad():

            (
                reconstruction,
                travel_time,
                log_variance
            ) = model(x)

        expected_shape = torch.Size(
            (
                batch_size,
                OUT_CHANNELS,
                *CUBE_SIZE
            )
        )

        assert reconstruction.shape == (
            expected_shape
        )

        assert travel_time.shape == (
            expected_shape
        )

        assert log_variance.shape == (
            expected_shape
        )

        print(
            f"Batch Size {batch_size}: "
            f"PASSED"
        )

    print(
        "Batch Processing Test: PASSED"
    )


# =====================================================
# TEST GRADIENT PROPAGATION
# =====================================================

def test_gradient_propagation():

    print()
    print(
        "Testing Gradient Propagation"
    )

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        use_uncertainty=True
    )

    model.train()

    x = create_input(
        batch_size=1
    )

    (
        reconstruction,
        travel_time,
        log_variance
    ) = model(x)

    # -------------------------------------------------
    # Construct a simple scalar objective involving
    # all three network outputs.
    #
    # This is NOT the final training loss.
    #
    # It only verifies that gradients can propagate
    # through all output branches.
    # -------------------------------------------------

    test_loss = (
        reconstruction.mean()
        +
        travel_time.mean()
        +
        log_variance.mean()
    )

    test_loss.backward()

    # -------------------------------------------------
    # Verify that at least one parameter from each
    # output head receives a gradient.
    # -------------------------------------------------

    reconstruction_gradient = (
        model.reconstruction_head.weight.grad
    )

    travel_time_gradient = (
        model.travel_time_head.weight.grad
    )

    uncertainty_gradient = (
        model.uncertainty_head.weight.grad
    )

    assert reconstruction_gradient is not None

    assert travel_time_gradient is not None

    assert uncertainty_gradient is not None

    assert torch.isfinite(
        reconstruction_gradient
    ).all()

    assert torch.isfinite(
        travel_time_gradient
    ).all()

    assert torch.isfinite(
        uncertainty_gradient
    ).all()

    print(
        "Reconstruction head gradient: PASSED"
    )

    print(
        "Travel-time head gradient: PASSED"
    )

    print(
        "Uncertainty head gradient: PASSED"
    )

    print(
        "Gradient Propagation Test: PASSED"
    )


# =====================================================
# TEST NAN / INF INPUT REJECTION
# =====================================================

def test_nan_inf_input_rejection():

    print()
    print(
        "Testing NaN/Inf Input Rejection"
    )

    model = Network3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS
    )

    # -------------------------------------------------
    # NaN input
    # -------------------------------------------------

    nan_input = create_input(
        batch_size=1
    )

    nan_input[0, 0, 0, 0, 0] = float(
        "nan"
    )

    try:

        model(nan_input)

        raise AssertionError(
            "Network accepted NaN input."
        )

    except ValueError:

        print(
            "NaN input correctly rejected."
        )

    # -------------------------------------------------
    # Inf input
    # -------------------------------------------------

    inf_input = create_input(
        batch_size=1
    )

    inf_input[0, 0, 0, 0, 0] = float(
        "inf"
    )

    try:

        model(inf_input)

        raise AssertionError(
            "Network accepted Inf input."
        )

    except ValueError:

        print(
            "Inf input correctly rejected."
        )

    print(
        "NaN/Inf Input Rejection Test: PASSED"
    )


# =====================================================
# MAIN TEST
# =====================================================

def main():

    print()
    print("=" * 60)

    print(
        "TESTING PHYSICS-INFORMED 3D "
        "ENCODER-DECODER NETWORK"
    )

    print("=" * 60)

    test_model_construction()

    test_input_validation()

    test_output_shapes()

    test_output_dtypes()

    test_finite_outputs()

    test_reconstruction_output()

    test_travel_time_output()

    test_travel_time_scaling()

    test_log_variance_output()

    test_uncertainty_enabled()

    test_uncertainty_disabled()

    test_batch_processing()

    test_gradient_propagation()

    test_nan_inf_input_rejection()

    print()
    print("=" * 60)

    print(
        "NETWORK TEST: PASSED"
    )

    print("=" * 60)


# =====================================================
# EXECUTION
# =====================================================

if __name__ == "__main__":

    main()