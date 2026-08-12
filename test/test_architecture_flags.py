import torch

from models.network import Network3D


def test_model(
        name,
        use_attention,
        use_residual,
        use_uncertainty
):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    model = Network3D(
        use_attention=use_attention,
        use_residual=use_residual,
        use_uncertainty=use_uncertainty
    )

    x = torch.randn(
        1,
        1,
        64,
        128,
        128
    )

    reconstruction, log_variance = model(x)

    print(
        "Reconstruction:",
        reconstruction.shape
    )

    print(
        "Log Variance:",
        log_variance.shape
    )


def main():

    test_model(
        "FULL MODEL",
        True,
        True,
        True
    )

    test_model(
        "NO ATTENTION",
        False,
        True,
        True
    )

    test_model(
        "NO RESIDUAL",
        True,
        False,
        True
    )

    test_model(
        "NO UNCERTAINTY",
        True,
        True,
        False
    )

    test_model(
        "PLAIN U-NET",
        False,
        False,
        False
    )


if __name__ == "__main__":
    main()