import test.setup_path

from test.test_factory import (
    create_trainer,
    create_dataloader
)


def test_fit():

    print()
    print("=" * 60)
    print("TEST - COMPLETE TRAINING PROCEDURE")
    print("=" * 60)

    # -------------------------------------------------------
    # Create Trainer and DataLoader
    # -------------------------------------------------------

    trainer = create_trainer()

    train_loader, validation_loader = (
        create_dataloader()
    )

    # -------------------------------------------------------
    # Train model
    # -------------------------------------------------------

    trainer.fit(
        train_loader,
        validation_loader,
        epochs=3,
        resume=False
    )

    # -------------------------------------------------------
    # Check training history
    # -------------------------------------------------------

    assert hasattr(
        trainer,
        "history"
    ), (
        "Trainer does not contain "
        "'history' attribute."
    )

    assert len(
        trainer.history["total"]
    ) > 0, (
        "Training history is empty."
    )

    # -------------------------------------------------------
    # Check validation history
    # -------------------------------------------------------

    assert hasattr(
        trainer,
        "validation_history"
    ), (
        "Trainer does not contain "
        "'validation_history' attribute."
    )

    assert len(
        trainer.validation_history["total"]
    ) > 0, (
        "Validation history is empty."
    )

    # -------------------------------------------------------
    # Display training history
    # -------------------------------------------------------

    print()
    print("=" * 60)
    print("Training History")
    print("=" * 60)

    for key, values in trainer.history.items():

        print(
            f"{key:<15}: {values}"
        )

    print("=" * 60)

    # -------------------------------------------------------
    # Display validation history
    # -------------------------------------------------------

    print()
    print("=" * 60)
    print("Validation History")
    print("=" * 60)

    for key, values in (
        trainer.validation_history.items()
    ):

        print(
            f"{key:<15}: {values}"
        )

    print("=" * 60)

    print()
    print("=" * 60)
    print("FIT TEST: PASSED")
    print("=" * 60)