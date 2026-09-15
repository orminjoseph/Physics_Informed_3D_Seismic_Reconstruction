"""
=========================================================
Geologically Conditioned 3D Velocity Model Generator
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Generate physically plausible synthetic 3D seismic velocity
models for:

    1. Synthetic training
    2. Physics-informed training
    3. Controlled validation
    4. Geological-complexity experiments
    5. Eikonal physics-loss evaluation

The velocity model is spatially varying and is conditioned
on the geological structure used to generate the corresponding
synthetic seismic volume.

Supported geological modes
--------------------------
    horizontal
    gradient
    dipping
    folded
    faulted
    complex
    highly_complex
    random

Tensor convention
-----------------
Velocity output:

    [C, D, H, W]

where:

    C = velocity channel
    D = depth
    H = crossline
    W = inline

Training adds the batch dimension:

    [B, C, D, H, W]

Physical units
--------------
Velocity : m/s
Travel time : s

Eikonal relation
----------------
For isotropic acoustic propagation:

    V^2 |grad(T)|^2 - 1 = 0

or equivalently:

    |grad(T)|^2 = 1 / V^2

Important
---------
The velocity model is supplied to the physics-informed
loss. It is NOT predicted by the neural network.

The final velocity model is always constrained to:

    min_velocity <= V <= max_velocity

Author
------
Ormin Joseph
=========================================================
"""

import math
import random

import torch


class VelocityGenerator:
    """
    Generate geologically conditioned synthetic 3D
    seismic velocity models.

    Parameters
    ----------
    cube_size : tuple
        Velocity cube dimensions:

            (D, H, W)

    min_velocity : float
        Minimum physically allowed velocity in m/s.

    max_velocity : float
        Maximum physically allowed velocity in m/s.

    seed : int or None
        Random seed for reproducibility.
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        cube_size=(64, 128, 128),
        min_velocity=1500.0,
        max_velocity=5000.0,
        seed=None,
    ):
        """Initialize the velocity model generator."""

        # -------------------------------------------------
        # Validate cube size.
        # -------------------------------------------------

        if (
            not isinstance(cube_size, tuple)
            or len(cube_size) != 3
        ):
            raise ValueError(
                "cube_size must be a tuple "
                "(depth, height, width)."
            )

        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in cube_size
        ):
            raise ValueError(
                "All cube dimensions must be "
                "positive integers."
            )

        # -------------------------------------------------
        # Validate velocity limits.
        # -------------------------------------------------

        if not isinstance(
            min_velocity,
            (int, float),
        ):
            raise TypeError(
                "min_velocity must be a real number."
            )

        if not isinstance(
            max_velocity,
            (int, float),
        ):
            raise TypeError(
                "max_velocity must be a real number."
            )

        if min_velocity <= 0:
            raise ValueError(
                "min_velocity must be greater than zero."
            )

        if max_velocity <= min_velocity:
            raise ValueError(
                "max_velocity must be greater than "
                "min_velocity."
            )

        # -------------------------------------------------
        # Store configuration.
        # -------------------------------------------------

        self.cube_size = cube_size

        self.depth = cube_size[0]
        self.height = cube_size[1]
        self.width = cube_size[2]

        self.min_velocity = float(min_velocity)
        self.max_velocity = float(max_velocity)

        self.seed = seed

        # -------------------------------------------------
        # Independent random-number generator.
        #
        # This does not modify Python's global RNG.
        # -------------------------------------------------

        self.rng = random.Random(seed)

    # =====================================================
    # VALIDATE GEOLOGICAL MODE
    # =====================================================

    @staticmethod
    def _validate_mode(mode):
        """Validate the requested geological mode."""

        valid_modes = {
            "horizontal",
            "gradient",
            "dipping",
            "folded",
            "faulted",
            "complex",
            "highly_complex",
            "random",
        }

        if mode not in valid_modes:
            raise ValueError(
                f"Invalid velocity mode '{mode}'. "
                f"Valid modes are: {sorted(valid_modes)}"
            )

    # =====================================================
    # VALIDATE NUMBER OF LAYERS
    # =====================================================

    def _validate_number_of_layers(
        self,
        number_of_layers,
    ):
        """Validate the number of velocity layers."""

        if (
            not isinstance(number_of_layers, int)
            or isinstance(number_of_layers, bool)
        ):
            raise TypeError(
                "number_of_layers must be an integer."
            )

        if number_of_layers < 1:
            raise ValueError(
                "number_of_layers must be at least 1."
            )

        if number_of_layers > self.depth:
            raise ValueError(
                "number_of_layers cannot exceed "
                "the depth of the velocity cube."
            )

    # =====================================================
    # FINAL PHYSICAL CLAMP
    # =====================================================

    def _enforce_physical_bounds(
        self,
        velocity,
    ):
        """
        Enforce the configured physical velocity range.

        This function MUST be applied after all geological
        perturbations have been introduced.

        Returns
        -------
        torch.Tensor
            Physically bounded velocity cube.
        """

        velocity = torch.clamp(
            velocity,
            min=self.min_velocity,
            max=self.max_velocity,
        )

        return velocity.contiguous()

    # =====================================================
    # CREATE NORMALIZED COORDINATES
    # =====================================================

    def _coordinates(self):
        """
        Create normalized 3-D coordinates.

        Returns
        -------
        z : Tensor
            Depth coordinate [D,1,1]

        y : Tensor
            Crossline coordinate [1,H,1]

        x : Tensor
            Inline coordinate [1,1,W]
        """

        z = torch.linspace(
            0.0,
            1.0,
            self.depth,
            dtype=torch.float32,
        ).view(
            self.depth,
            1,
            1,
        )

        y = torch.linspace(
            0.0,
            1.0,
            self.height,
            dtype=torch.float32,
        ).view(
            1,
            self.height,
            1,
        )

        x = torch.linspace(
            0.0,
            1.0,
            self.width,
            dtype=torch.float32,
        ).view(
            1,
            1,
            self.width,
        )

        return z, y, x

    # =====================================================
    # EMPTY VELOCITY CUBE
    # =====================================================

    def _empty_velocity_cube(self):
        """
        Create an empty velocity cube.

        Returns
        -------
        torch.Tensor
            Shape [1,D,H,W].
        """

        return torch.empty(
            (
                1,
                self.depth,
                self.height,
                self.width,
            ),
            dtype=torch.float32,
        )

    # =====================================================
    # LAYER BOUNDARIES
    # =====================================================

    def _generate_layer_boundaries(
        self,
        number_of_layers,
    ):
        """
        Generate valid horizontal layer boundaries.

        Every layer receives at least one depth sample.
        """

        self._validate_number_of_layers(
            number_of_layers
        )

        if number_of_layers == 1:
            return [0, self.depth]

        internal_boundaries = self.rng.sample(
            range(1, self.depth),
            number_of_layers - 1,
        )

        internal_boundaries.sort()

        return (
            [0]
            + internal_boundaries
            + [self.depth]
        )

    # =====================================================
    # HORIZONTAL LAYERED MODEL
    # =====================================================

    def generate_layered_model(
        self,
        number_of_layers=5,
    ):
        """
        Generate horizontally layered velocity.

        Velocity increases with geological depth.
        """

        boundaries = (
            self._generate_layer_boundaries(
                number_of_layers
            )
        )

        velocity = (
            self._empty_velocity_cube()
        )

        if number_of_layers == 1:

            velocity.fill_(
                self.min_velocity
            )

            return velocity

        velocity_increment = (
            self.max_velocity
            - self.min_velocity
        ) / (
            number_of_layers - 1
        )

        for layer_index in range(
            number_of_layers
        ):

            top = boundaries[layer_index]

            bottom = boundaries[
                layer_index + 1
            ]

            layer_velocity = (
                self.min_velocity
                + layer_index
                * velocity_increment
            )

            velocity[
                :,
                top:bottom,
                :,
                :,
            ] = layer_velocity

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # LINEAR GRADIENT MODEL
    # =====================================================

    def generate_gradient_model(self):
        """
        Generate a continuous velocity gradient:

            V(z) = Vmin + (Vmax - Vmin) z
        """

        z, _, _ = self._coordinates()

        velocity_profile = (
            self.min_velocity
            + (
                self.max_velocity
                - self.min_velocity
            )
            * z
        )

        velocity = (
            velocity_profile
            .expand(
                self.depth,
                self.height,
                self.width,
            )
            .unsqueeze(0)
        )

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # DIPPING LAYERED MODEL
    # =====================================================

    def generate_dipping_model(
        self,
        number_of_layers=5,
        dip=0.20,
    ):
        """
        Generate genuinely layered dipping velocity
        structures.

        The horizontal interfaces are warped laterally
        according to the normalized dip parameter.

        Parameters
        ----------
        number_of_layers : int
            Number of velocity layers.

        dip : float
            Normalized interface displacement.
        """

        self._validate_number_of_layers(
            number_of_layers
        )

        if dip < 0:
            raise ValueError(
                "dip must be non-negative."
            )

        z, _, x = self._coordinates()

        # -------------------------------------------------
        # Generate geological layer boundaries.
        # -------------------------------------------------

        boundaries = (
            self._generate_layer_boundaries(
                number_of_layers
            )
        )

        # -------------------------------------------------
        # Convert boundaries into normalized depth.
        # -------------------------------------------------

        normalized_boundaries = [
            boundary / self.depth
            for boundary in boundaries
        ]

        # -------------------------------------------------
        # Dipping structural coordinate.
        # -------------------------------------------------

        structural_z = (
            z
            - dip * (x - 0.5)
        )

        # -------------------------------------------------
        # Allocate velocity.
        # -------------------------------------------------

        velocity = torch.empty(
            (
                self.depth,
                self.height,
                self.width,
            ),
            dtype=torch.float32,
        )

        # -------------------------------------------------
        # Assign velocity to each dipping layer.
        # -------------------------------------------------

        for layer_index in range(
            number_of_layers
        ):

            top = normalized_boundaries[
                layer_index
            ]

            bottom = normalized_boundaries[
                layer_index + 1
            ]

            mask = (
                (structural_z >= top)
                & (structural_z < bottom)
            )

            layer_velocity = (
                self.min_velocity
                + (
                    layer_index
                    / max(
                        number_of_layers - 1,
                        1,
                    )
                )
                * (
                    self.max_velocity
                    - self.min_velocity
                )
            )

            velocity = torch.where(
                mask,
                torch.as_tensor(
                    layer_velocity,
                    dtype=torch.float32,
                ),
                velocity,
            )

        # -------------------------------------------------
        # Ensure the deepest layer is assigned.
        # -------------------------------------------------

        deepest_layer_velocity = (
            self.max_velocity
        )

        unassigned = torch.isnan(
            velocity
        )

        velocity = torch.where(
            unassigned,
            torch.as_tensor(
                deepest_layer_velocity,
                dtype=torch.float32,
            ),
            velocity,
        )

        velocity = velocity.unsqueeze(0)

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # FOLDED MODEL
    # =====================================================

    def generate_folded_model(
        self,
        fold_amplitude=0.10,
        fold_frequency=2.0,
    ):
        """
        Generate a folded velocity structure.

            z' = z - A sin(2 pi f x)
        """

        if fold_amplitude < 0:
            raise ValueError(
                "fold_amplitude must be non-negative."
            )

        if fold_frequency <= 0:
            raise ValueError(
                "fold_frequency must be positive."
            )

        z, _, x = self._coordinates()

        fold = (
            fold_amplitude
            * torch.sin(
                2.0
                * math.pi
                * fold_frequency
                * x
            )
        )

        structural_z = (
            z - fold
        )

        structural_z = torch.clamp(
            structural_z,
            0.0,
            1.0,
        )

        velocity = (
            self.min_velocity
            + (
                self.max_velocity
                - self.min_velocity
            )
            * structural_z
        )

        velocity = velocity.expand(
            self.depth,
            self.height,
            self.width,
        )

        velocity = velocity.unsqueeze(0)

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # FAULTED MODEL
    # =====================================================

    def generate_faulted_model(
        self,
        fault_position=0.50,
        fault_throw=0.12,
        dip=0.10,
    ):
        """
        Generate a faulted velocity structure.
        """

        if not 0.0 <= fault_position <= 1.0:
            raise ValueError(
                "fault_position must lie between 0 and 1."
            )

        if fault_throw < 0:
            raise ValueError(
                "fault_throw must be non-negative."
            )

        if dip < 0:
            raise ValueError(
                "dip must be non-negative."
            )

        z, _, x = self._coordinates()

        # -------------------------------------------------
        # Background dip.
        # -------------------------------------------------

        structural_z = (
            z
            - dip * (x - 0.5)
        )

        # -------------------------------------------------
        # Fault displacement.
        # -------------------------------------------------

        fault_mask = (
            x >= fault_position
        )

        structural_z = torch.where(
            fault_mask,
            structural_z + fault_throw,
            structural_z,
        )

        structural_z = torch.clamp(
            structural_z,
            0.0,
            1.0,
        )

        velocity = (
            self.min_velocity
            + (
                self.max_velocity
                - self.min_velocity
            )
            * structural_z
        )

        velocity = velocity.expand(
            self.depth,
            self.height,
            self.width,
        )

        velocity = velocity.unsqueeze(0)

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # LATERAL HETEROGENEITY
    # =====================================================

    def _add_lateral_heterogeneity(
        self,
        velocity,
        amplitude=0.05,
    ):
        """
        Add smooth geological heterogeneity.

        This represents broad-scale lateral variation
        rather than independent voxel noise.
        """

        if amplitude < 0:
            raise ValueError(
                "amplitude must be non-negative."
            )

        _, y, x = self._coordinates()

        heterogeneity = (
            torch.sin(
                2.0
                * math.pi
                * x
            )
            * torch.cos(
                2.0
                * math.pi
                * y
            )
        )

        velocity_range = (
            self.max_velocity
            - self.min_velocity
        )

        perturbation = (
            amplitude
            * velocity_range
            * heterogeneity
        )

        velocity = (
            velocity
            + perturbation.unsqueeze(0)
        )

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # COMPLEX MODEL
    # =====================================================

    def generate_complex_model(self):
        """
        Generate a complex 3-D geological velocity model.

        Components:

            - dipping structure
            - folding
            - multiple faults
            - smooth lateral heterogeneity
        """

        z, _, x = self._coordinates()

        # -------------------------------------------------
        # Background dipping structure.
        # -------------------------------------------------

        structural_z = (
            z
            - 0.10 * (x - 0.5)
        )

        # -------------------------------------------------
        # Folding.
        # -------------------------------------------------

        fold = (
            0.08
            * torch.sin(
                4.0
                * math.pi
                * x
            )
        )

        structural_z = (
            structural_z
            - fold
        )

        # -------------------------------------------------
        # Fault 1.
        # -------------------------------------------------

        fault_1 = (
            x >= 0.32
        )

        structural_z = torch.where(
            fault_1,
            structural_z + 0.08,
            structural_z,
        )

        # -------------------------------------------------
        # Fault 2.
        # -------------------------------------------------

        fault_2 = (
            x >= 0.68
        )

        structural_z = torch.where(
            fault_2,
            structural_z - 0.10,
            structural_z,
        )

        structural_z = torch.clamp(
            structural_z,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Convert structural geometry to velocity.
        # -------------------------------------------------

        velocity = (
            self.min_velocity
            + (
                self.max_velocity
                - self.min_velocity
            )
            * structural_z
        )

        velocity = velocity.expand(
            self.depth,
            self.height,
            self.width,
        )

        velocity = velocity.unsqueeze(0)

        # -------------------------------------------------
        # Add geological heterogeneity.
        # -------------------------------------------------

        velocity = (
            self._add_lateral_heterogeneity(
                velocity,
                amplitude=0.04,
            )
        )

        # -------------------------------------------------
        # FINAL PHYSICAL BOUND.
        #
        # This is important because multiple geological
        # perturbations can push values outside the
        # configured physical range.
        # -------------------------------------------------

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # HIGHLY COMPLEX MODEL
    # =====================================================

    def generate_highly_complex_model(self):
        """
        Generate a highly complex 3-D geological velocity
        model.

        Components:

            - dipping structure
            - strong folding
            - multiple faults
            - smooth lateral heterogeneity
            - salt-like high-velocity body
        """

        z, y, x = self._coordinates()

        # -------------------------------------------------
        # Background dipping structure.
        # -------------------------------------------------

        structural_z = (
            z
            - 0.12 * (x - 0.5)
        )

        # -------------------------------------------------
        # Strong folding.
        # -------------------------------------------------

        fold = (
            0.12
            * torch.sin(
                4.0
                * math.pi
                * x
            )
        )

        structural_z = (
            structural_z
            - fold
        )

        # -------------------------------------------------
        # Fault 1.
        # -------------------------------------------------

        fault_1 = (
            x >= 0.30
        )

        structural_z = torch.where(
            fault_1,
            structural_z + 0.10,
            structural_z,
        )

        # -------------------------------------------------
        # Fault 2.
        # -------------------------------------------------

        fault_2 = (
            x >= 0.65
        )

        structural_z = torch.where(
            fault_2,
            structural_z - 0.14,
            structural_z,
        )

        structural_z = torch.clamp(
            structural_z,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Background velocity.
        # -------------------------------------------------

        velocity = (
            self.min_velocity
            + (
                self.max_velocity
                - self.min_velocity
            )
            * structural_z
        )

        velocity = velocity.expand(
            self.depth,
            self.height,
            self.width,
        )

        velocity = velocity.unsqueeze(0)

        # -------------------------------------------------
        # Smooth lateral heterogeneity.
        # -------------------------------------------------

        velocity = (
            self._add_lateral_heterogeneity(
                velocity,
                amplitude=0.05,
            )
        )

        # -------------------------------------------------
        # Salt-like high-velocity body.
        # -------------------------------------------------

        salt_center_x = 0.50
        salt_center_y = 0.50
        salt_center_z = 0.55

        sigma_x = 0.12
        sigma_y = 0.15
        sigma_z = 0.18

        salt = torch.exp(
            -(
                (
                    (x - salt_center_x)
                    / sigma_x
                ) ** 2
                +
                (
                    (y - salt_center_y)
                    / sigma_y
                ) ** 2
                +
                (
                    (z - salt_center_z)
                    / sigma_z
                ) ** 2
            )
        )

        velocity_range = (
            self.max_velocity
            - self.min_velocity
        )

        salt_strength = (
            0.20
            * velocity_range
        )

        velocity = (
            velocity
            + salt_strength
            * salt.unsqueeze(0)
        )

        # -------------------------------------------------
        # FINAL PHYSICAL BOUND.
        # -------------------------------------------------

        velocity = (
            self._enforce_physical_bounds(
                velocity
            )
        )

        return velocity

    # =====================================================
    # RANDOM MODEL
    # =====================================================

    def generate_random_model(
        self,
        number_of_layers=5,
    ):
        """
        Randomly select a geological velocity model.

        The selection is reproducible when a seed is supplied.
        """

        modes = [
            "horizontal",
            "gradient",
            "dipping",
            "folded",
            "faulted",
            "complex",
            "highly_complex",
        ]

        mode = self.rng.choice(
            modes
        )

        return self.generate(
            mode=mode,
            number_of_layers=number_of_layers,
        )

    # =====================================================
    # MAIN GENERATOR
    # =====================================================

    def generate(
        self,
        mode="random",
        number_of_layers=5,
    ):
        """
        Generate a velocity model.

        Parameters
        ----------
        mode : str
            Geological velocity mode.

        number_of_layers : int
            Number of layers for applicable models.

        Returns
        -------
        torch.Tensor
            Shape [1,D,H,W]

        Units
        -----
        m/s
        """

        self._validate_mode(
            mode
        )

        if mode == "random":

            return self.generate_random_model(
                number_of_layers=number_of_layers
            )

        if mode == "horizontal":

            return self.generate_layered_model(
                number_of_layers=number_of_layers
            )

        if mode == "gradient":

            return self.generate_gradient_model()

        if mode == "dipping":

            return self.generate_dipping_model(
                number_of_layers=number_of_layers
            )

        if mode == "folded":

            return self.generate_folded_model()

        if mode == "faulted":

            return self.generate_faulted_model()

        if mode == "complex":

            return self.generate_complex_model()

        if mode == "highly_complex":

            return self.generate_highly_complex_model()

        raise RuntimeError(
            "Unhandled velocity generation mode."
        )

    # =====================================================
    # VELOCITY VALIDATION
    # =====================================================

    @staticmethod
    def validate_velocity(
        velocity,
        min_velocity=None,
        max_velocity=None,
    ):
        """
        Validate a generated velocity model.

        Expected shape:

            [C,D,H,W]
        """

        if not isinstance(
            velocity,
            torch.Tensor,
        ):
            raise TypeError(
                "velocity must be a torch.Tensor."
            )

        if velocity.ndim != 4:
            raise ValueError(
                "velocity must have shape "
                "[C,D,H,W]. "
                f"Received: {tuple(velocity.shape)}."
            )

        if velocity.shape[0] != 1:
            raise ValueError(
                "velocity must contain exactly "
                "one channel. "
                f"Received {velocity.shape[0]}."
            )

        if not torch.isfinite(
            velocity
        ).all():
            raise ValueError(
                "Velocity model contains NaN or Inf."
            )

        if torch.any(
            velocity <= 0
        ):
            raise ValueError(
                "Velocity values must be strictly "
                "greater than zero."
            )

        if (
            min_velocity is not None
            and torch.any(
                velocity < min_velocity
            )
        ):
            raise ValueError(
                "Velocity contains values below "
                "the specified minimum velocity."
            )

        if (
            max_velocity is not None
            and torch.any(
                velocity > max_velocity
            )
        ):
            raise ValueError(
                "Velocity contains values above "
                "the specified maximum velocity."
            )

        return True

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self):
        """Return a concise generator description."""

        return {
            "cube_size": self.cube_size,
            "min_velocity_mps": self.min_velocity,
            "max_velocity_mps": self.max_velocity,
            "seed": self.seed,
            "output_shape": (
                1,
                self.depth,
                self.height,
                self.width,
            ),
            "supported_modes": [
                "horizontal",
                "gradient",
                "dipping",
                "folded",
                "faulted",
                "complex",
                "highly_complex",
                "random",
            ],
        }


# =========================================================
# STANDALONE TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("GEOLOGICALLY CONDITIONED VELOCITY GENERATOR TEST")
    print("=" * 70)

    generator = VelocityGenerator(
        cube_size=(64, 128, 128),
        min_velocity=1500.0,
        max_velocity=5000.0,
        seed=42,
    )

    print("\nGenerator summary:")
    print(generator.summary())

    modes = [
        "horizontal",
        "gradient",
        "dipping",
        "folded",
        "faulted",
        "complex",
        "highly_complex",
    ]

    for mode in modes:

        velocity = generator.generate(
            mode=mode
        )

        VelocityGenerator.validate_velocity(
            velocity,
            min_velocity=1500.0,
            max_velocity=5000.0,
        )

        print(
            f"\nMode: {mode}"
        )

        print(
            f"Shape: {tuple(velocity.shape)}"
        )

        print(
            f"Minimum velocity: "
            f"{velocity.min().item():.2f} m/s"
        )

        print(
            f"Maximum velocity: "
            f"{velocity.max().item():.2f} m/s"
        )

        print(
            f"Mean velocity: "
            f"{velocity.mean().item():.2f} m/s"
        )

        print(
            "Validation: PASSED"
        )

    print("\n" + "=" * 70)
    print("ALL VELOCITY MODEL TESTS PASSED")
    print("=" * 70)