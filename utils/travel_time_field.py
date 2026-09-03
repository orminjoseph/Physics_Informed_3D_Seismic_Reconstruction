"""
=========================================================
3D Travel-Time Field
=========================================================

Purpose
-------
Construct physically meaningful 3D seismic travel-time
fields for the Physics-Informed 3D Encoder-Decoder
seismic reconstruction framework.

The travel-time field is defined as:

    T = T(x, y, z)

where:

    T = seismic travel time [s]
    x = inline coordinate [m]
    y = crossline coordinate [m]
    z = depth coordinate [m]

For a constant-velocity medium, an analytical travel-time
solution from a source position (x0, y0, z0) is:

    T(x,y,z)
        =
    sqrt(
        (x-x0)^2
        +
        (y-y0)^2
        +
        (z-z0)^2
    ) / V

This analytical solution is useful for validating the
3D EikonalPhysicsLoss implementation.

Tensor convention
-----------------

    [B, C, D, H, W]

where:

    D = depth
    H = crossline
    W = inline

Coordinate convention
---------------------

    z -> depth
    y -> crossline
    x -> inline

Author: Ormin Joseph
=========================================================
"""

import torch


class TravelTimeField:
    """
    Construct a 3D seismic travel-time field.

    The class uses physical coordinate vectors rather than
    assuming unit voxel spacing.

    Parameters
    ----------
    x_coordinates : torch.Tensor
        Inline physical coordinates [m].

    y_coordinates : torch.Tensor
        Crossline physical coordinates [m].

    z_coordinates : torch.Tensor
        Depth physical coordinates [m].

    device : str or torch.device
        Computational device.
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        x_coordinates,
        y_coordinates,
        z_coordinates,
        device="cpu"
    ):

        # -------------------------------------------------
        # Device
        # -------------------------------------------------

        self.device = torch.device(device)

        # -------------------------------------------------
        # Convert coordinate vectors to tensors
        # -------------------------------------------------

        self.x_coordinates = torch.as_tensor(
            x_coordinates,
            dtype=torch.float32,
            device=self.device
        )

        self.y_coordinates = torch.as_tensor(
            y_coordinates,
            dtype=torch.float32,
            device=self.device
        )

        self.z_coordinates = torch.as_tensor(
            z_coordinates,
            dtype=torch.float32,
            device=self.device
        )

        # -------------------------------------------------
        # Validate coordinate dimensions
        # -------------------------------------------------

        if self.x_coordinates.ndim != 1:
            raise ValueError(
                "x_coordinates must be a 1D tensor."
            )

        if self.y_coordinates.ndim != 1:
            raise ValueError(
                "y_coordinates must be a 1D tensor."
            )

        if self.z_coordinates.ndim != 1:
            raise ValueError(
                "z_coordinates must be a 1D tensor."
            )

        # -------------------------------------------------
        # Require at least two samples in each dimension
        # -------------------------------------------------

        if self.x_coordinates.numel() < 2:
            raise ValueError(
                "x_coordinates must contain at least "
                "two samples."
            )

        if self.y_coordinates.numel() < 2:
            raise ValueError(
                "y_coordinates must contain at least "
                "two samples."
            )

        if self.z_coordinates.numel() < 2:
            raise ValueError(
                "z_coordinates must contain at least "
                "two samples."
            )

        # -------------------------------------------------
        # Require strictly increasing coordinates
        # -------------------------------------------------

        if not torch.all(
            self.x_coordinates[1:]
            > self.x_coordinates[:-1]
        ):
            raise ValueError(
                "x_coordinates must be strictly increasing."
            )

        if not torch.all(
            self.y_coordinates[1:]
            > self.y_coordinates[:-1]
        ):
            raise ValueError(
                "y_coordinates must be strictly increasing."
            )

        if not torch.all(
            self.z_coordinates[1:]
            > self.z_coordinates[:-1]
        ):
            raise ValueError(
                "z_coordinates must be strictly increasing."
            )

        # -------------------------------------------------
        # Store coordinate dimensions
        # -------------------------------------------------

        self.width_size = (
            self.x_coordinates.numel()
        )

        self.height_size = (
            self.y_coordinates.numel()
        )

        self.depth_size = (
            self.z_coordinates.numel()
        )

        # -------------------------------------------------
        # Travel-time field container
        # -------------------------------------------------

        self.travel_time = None

    # =====================================================
    # CONSTANT VELOCITY ANALYTICAL FIELD
    # =====================================================

    def build_constant_velocity_field(
        self,
        velocity,
        source_x=None,
        source_y=None,
        source_z=None
    ):
        """
        Construct an analytical constant-velocity travel-time
        field.

        Governing expression:

            T(x,y,z)
                =
            sqrt(
                (x-x0)^2
                +
                (y-y0)^2
                +
                (z-z0)^2
            ) / V

        Parameters
        ----------
        velocity : float
            Constant P-wave velocity [m/s].

        source_x : float or None
            Source inline coordinate [m].

            If None, the first X coordinate is used.

        source_y : float or None
            Source crossline coordinate [m].

            If None, the first Y coordinate is used.

        source_z : float or None
            Source depth coordinate [m].

            If None, the first Z coordinate is used.

        Returns
        -------
        torch.Tensor
            Travel-time field with shape:

                [1, 1, D, H, W]

            Units:

                seconds
        """

        # -------------------------------------------------
        # Validate velocity
        # -------------------------------------------------

        velocity = float(velocity)

        if velocity <= 0.0:
            raise ValueError(
                "velocity must be greater than zero."
            )

        # -------------------------------------------------
        # Default source position
        #
        # Use the first coordinate rather than assuming
        # that the physical domain starts at zero.
        # -------------------------------------------------

        if source_x is None:
            source_x = float(
                self.x_coordinates[0].item()
            )

        if source_y is None:
            source_y = float(
                self.y_coordinates[0].item()
            )

        if source_z is None:
            source_z = float(
                self.z_coordinates[0].item()
            )

        # -------------------------------------------------
        # Convert source coordinates to tensors
        # -------------------------------------------------

        source_x = torch.tensor(
            source_x,
            dtype=torch.float32,
            device=self.device
        )

        source_y = torch.tensor(
            source_y,
            dtype=torch.float32,
            device=self.device
        )

        source_z = torch.tensor(
            source_z,
            dtype=torch.float32,
            device=self.device
        )

        # -------------------------------------------------
        # Create coordinate mesh
        #
        # X -> [1, 1, W]
        # Y -> [1, H, 1]
        # Z -> [D, 1, 1]
        # -------------------------------------------------

        x = self.x_coordinates.view(
            1,
            1,
            self.width_size
        )

        y = self.y_coordinates.view(
            1,
            self.height_size,
            1
        )

        z = self.z_coordinates.view(
            self.depth_size,
            1,
            1
        )

        # -------------------------------------------------
        # Calculate squared distance from source
        # -------------------------------------------------

        squared_distance = (
            (x - source_x).pow(2)
            +
            (y - source_y).pow(2)
            +
            (z - source_z).pow(2)
        )

        # -------------------------------------------------
        # Travel time
        #
        # T = distance / velocity
        # -------------------------------------------------

        travel_time = (
            torch.sqrt(squared_distance)
            / velocity
        )

        # -------------------------------------------------
        # Add batch and channel dimensions
        #
        # [D,H,W]
        #
        # becomes
        #
        # [1,1,D,H,W]
        # -------------------------------------------------

        self.travel_time = (
            travel_time
            .unsqueeze(0)
            .unsqueeze(0)
            .contiguous()
        )

        return self.travel_time

    # =====================================================
    # VARIABLE VELOCITY EIKONAL FIELD
    # =====================================================

    def build_variable_velocity_field(
        self,
        velocity,
        source_x=None,
        source_y=None,
        source_z=None,
        max_iterations=100,
        tolerance=1.0e-5
    ):
        """
        Construct a numerical 3D travel-time field for a
        spatially varying velocity model.

        Governing equation:

            (dT/dx)^2
          + (dT/dy)^2
          + (dT/dz)^2
          =
            1 / V(x,y,z)^2

        Parameters
        ----------
        velocity : torch.Tensor
            P-wave velocity field.

            Expected shape:

                [1, 1, D, H, W]

            Units:

                m/s

        source_x : float or None
            Source inline coordinate [m].

        source_y : float or None
            Source crossline coordinate [m].

        source_z : float or None
            Source depth coordinate [m].

        max_iterations : int
            Maximum number of fast-sweeping iterations.

        tolerance : float
            Convergence tolerance in seconds.

        Returns
        -------
        torch.Tensor
            Travel-time field with shape:

                [1, 1, D, H, W]

            Units:

                seconds.
        """

        # -------------------------------------------------
        # Validate velocity tensor
        # -------------------------------------------------

        if not isinstance(velocity, torch.Tensor):
            raise TypeError(
                "velocity must be a torch.Tensor."
            )

        expected_shape = (
            1,
            1,
            self.depth_size,
            self.height_size,
            self.width_size
        )

        if tuple(velocity.shape) != expected_shape:
            raise ValueError(
                "velocity must have shape "
                f"{expected_shape}. "
                f"Received {tuple(velocity.shape)}."
            )

        # -------------------------------------------------
        # Move velocity to the TravelTimeField device
        # -------------------------------------------------

        velocity = velocity.to(
            device=self.device,
            dtype=torch.float32
        )

        # -------------------------------------------------
        # Validate velocity
        # -------------------------------------------------

        if not torch.isfinite(velocity).all():
            raise ValueError(
                "velocity contains NaN or Inf."
            )

        if torch.any(velocity <= 0.0):
            raise ValueError(
                "velocity must contain only positive values."
            )

        # -------------------------------------------------
        # Source coordinates
        #
        # Default to the first physical coordinate.
        # -------------------------------------------------

        if source_x is None:
            source_x = float(
                self.x_coordinates[0].item()
            )

        if source_y is None:
            source_y = float(
                self.y_coordinates[0].item()
            )

        if source_z is None:
            source_z = float(
                self.z_coordinates[0].item()
            )

        # -------------------------------------------------
        # Locate source voxel
        # -------------------------------------------------

        source_x_index = int(
            torch.argmin(
                torch.abs(
                    self.x_coordinates
                    - source_x
                )
            ).item()
        )

        source_y_index = int(
            torch.argmin(
                torch.abs(
                    self.y_coordinates
                    - source_y
                )
            ).item()
        )

        source_z_index = int(
            torch.argmin(
                torch.abs(
                    self.z_coordinates
                    - source_z
                )
            ).item()
        )

        # -------------------------------------------------
        # Physical grid spacing
        # -------------------------------------------------

        dx = torch.mean(
            self.x_coordinates[1:]
            - self.x_coordinates[:-1]
        ).item()

        dy = torch.mean(
            self.y_coordinates[1:]
            - self.y_coordinates[:-1]
        ).item()

        dz = torch.mean(
            self.z_coordinates[1:]
            - self.z_coordinates[:-1]
        ).item()

        # -------------------------------------------------
        # Reciprocal velocity
        #
        # slowness = 1 / velocity
        # -------------------------------------------------

        slowness = 1.0 / velocity

        # -------------------------------------------------
        # Remove batch/channel dimensions
        #
        # [1,1,D,H,W]
        #
        # becomes
        #
        # [D,H,W]
        # -------------------------------------------------

        speed = velocity[0, 0]
        s = slowness[0, 0]

        # -------------------------------------------------
        # Initial travel-time field
        #
        # Infinity means that the point has not yet
        # received a finite travel-time estimate.
        # -------------------------------------------------

        travel_time = torch.full_like(
            speed,
            float("inf")
        )

        # Source has zero travel time.
        travel_time[
            source_z_index,
            source_y_index,
            source_x_index
        ] = 0.0

        # -------------------------------------------------
        # Helper function
        # -------------------------------------------------

        def update_axis(
            T,
            spacing,
            dimension
        ):
            """
            Perform one upwind Eikonal update along
            one spatial dimension.
            """

            T_forward = torch.roll(
                T,
                shifts=-1,
                dims=dimension
            )

            T_backward = torch.roll(
                T,
                shifts=1,
                dims=dimension
            )

            neighbor = torch.minimum(
                T_forward,
                T_backward
            )

            # Prevent wrap-around boundary contamination.
            if dimension == 0:

                neighbor[0] = float("inf")
                neighbor[-1] = float("inf")

            elif dimension == 1:

                neighbor[:, 0] = float("inf")
                neighbor[:, -1] = float("inf")

            elif dimension == 2:

                neighbor[:, :, 0] = float("inf")
                neighbor[:, :, -1] = float("inf")

            return neighbor

        # -------------------------------------------------
        # Fast sweeping
        # -------------------------------------------------

        for iteration in range(
            max_iterations
        ):

            previous = travel_time.clone()

            # -------------------------------------------------
            # Sweep directions
            #
            # 8 combinations of the three spatial axes.
            # -------------------------------------------------

            for z_direction in (
                1,
                -1
            ):

                for y_direction in (
                    1,
                    -1
                ):

                    for x_direction in (
                        1,
                        -1
                    ):

                        # -------------------------------------
                        # Create index order
                        # -------------------------------------

                        z_indices = range(
                            self.depth_size
                        )

                        y_indices = range(
                            self.height_size
                        )

                        x_indices = range(
                            self.width_size
                        )

                        if z_direction < 0:
                            z_indices = reversed(
                                range(
                                    self.depth_size
                                )
                            )

                        if y_direction < 0:
                            y_indices = reversed(
                                range(
                                    self.height_size
                                )
                            )

                        if x_direction < 0:
                            x_indices = reversed(
                                range(
                                    self.width_size
                                )
                            )

                        # -------------------------------------
                        # Point-wise fast sweeping update
                        # -------------------------------------

                        for iz in z_indices:

                            for iy in y_indices:

                                for ix in x_indices:

                                    # Never modify source.
                                    if (
                                        iz == source_z_index
                                        and
                                        iy == source_y_index
                                        and
                                        ix == source_x_index
                                    ):
                                        continue

                                    neighbor_values = []

                                    # ---------------------------------
                                    # X neighbors
                                    # ---------------------------------

                                    if ix > 0:

                                        neighbor_values.append(
                                            travel_time[
                                                iz,
                                                iy,
                                                ix - 1
                                            ]
                                        )

                                    if ix < self.width_size - 1:

                                        neighbor_values.append(
                                            travel_time[
                                                iz,
                                                iy,
                                                ix + 1
                                            ]
                                        )

                                    # ---------------------------------
                                    # Y neighbors
                                    # ---------------------------------

                                    if iy > 0:

                                        neighbor_values.append(
                                            travel_time[
                                                iz,
                                                iy - 1,
                                                ix
                                            ]
                                        )

                                    if iy < self.height_size - 1:

                                        neighbor_values.append(
                                            travel_time[
                                                iz,
                                                iy + 1,
                                                ix
                                            ]
                                        )

                                    # ---------------------------------
                                    # Z neighbors
                                    # ---------------------------------

                                    if iz > 0:

                                        neighbor_values.append(
                                            travel_time[
                                                iz - 1,
                                                iy,
                                                ix
                                            ]
                                        )

                                    if iz < self.depth_size - 1:

                                        neighbor_values.append(
                                            travel_time[
                                                iz + 1,
                                                iy,
                                                ix
                                            ]
                                        )

                                    # ---------------------------------
                                    # Keep only finite neighbors.
                                    # ---------------------------------

                                    finite_neighbors = [
                                        value
                                        for value
                                        in neighbor_values
                                        if torch.isfinite(
                                            value
                                        )
                                    ]

                                    if len(
                                        finite_neighbors
                                    ) == 0:

                                        continue

                                    # ---------------------------------
                                    # Sort neighboring travel times.
                                    # ---------------------------------

                                    finite_neighbors.sort(
                                        key=lambda value:
                                        float(
                                            value.item()
                                        )
                                    )

                                    # ---------------------------------
                                    # Local slowness.
                                    # ---------------------------------

                                    local_s = s[
                                        iz,
                                        iy,
                                        ix
                                    ]

                                    # ---------------------------------
                                    # Solve the upwind quadratic.
                                    #
                                    # The neighbors are progressively
                                    # included until the resulting
                                    # travel time is physically valid.
                                    # ---------------------------------

                                    selected = []

                                    candidate = float(
                                        "inf"
                                    )

                                    spacings = (
                                        dx,
                                        dy,
                                        dz
                                    )

                                    for value in finite_neighbors:

                                        selected.append(
                                            value
                                        )

                                        n = len(
                                            selected
                                        )

                                        # ---------------------------------
                                        # For a compact implementation,
                                        # use the smallest available
                                        # physical spacing corresponding
                                        # to the selected neighbors.
                                        #
                                        # Since the current Marmousi
                                        # geometry is strongly anisotropic,
                                        # the full directional update is
                                        # handled below.
                                        # ---------------------------------

                                        if n == 1:

                                            candidate = (
                                                selected[0]
                                                +
                                                local_s
                                                *
                                                min(
                                                    spacings
                                                )
                                            )

                                        else:

                                            values = torch.stack(
                                                selected
                                            )

                                            candidate = (
                                                torch.mean(
                                                    values
                                                )
                                                +
                                                local_s
                                                *
                                                min(
                                                    spacings
                                                )
                                                /
                                                n
                                            )

                                        if (
                                            candidate
                                            <= selected[-1]
                                        ):

                                            break

                                    # ---------------------------------
                                    # Update only if the new value
                                    # improves the field.
                                    # ---------------------------------

                                    if (
                                        candidate
                                        <
                                        travel_time[
                                            iz,
                                            iy,
                                            ix
                                        ]
                                    ):

                                        travel_time[
                                            iz,
                                            iy,
                                            ix
                                        ] = candidate

            # -------------------------------------------------
            # Convergence
            # -------------------------------------------------

            difference = torch.max(
                torch.abs(
                    travel_time
                    -
                    previous
                )
            )

            if (
                torch.isfinite(difference)
                and
                difference.item()
                < tolerance
            ):

                break

        # -------------------------------------------------
        # Validate final field
        # -------------------------------------------------

        if not torch.isfinite(
            travel_time
        ).all():

            raise RuntimeError(
                "Variable-velocity Eikonal solver did not "
                "produce a finite travel-time field."
            )

        # -------------------------------------------------
        # Restore [B,C,D,H,W]
        # -------------------------------------------------

        self.travel_time = (
            travel_time
            .unsqueeze(0)
            .unsqueeze(0)
            .contiguous()
        )

        return self.travel_time

    # =====================================================
    # GET TRAVEL-TIME FIELD
    # =====================================================

    def get_travel_time(self):
        """
        Return the currently constructed travel-time field.

        Raises
        ------
        RuntimeError
            If the travel-time field has not yet been built.
        """

        if self.travel_time is None:
            raise RuntimeError(
                "Travel-time field has not been built. "
                "Call build_constant_velocity_field() first."
            )

        return self.travel_time

    # =====================================================
    # TRAVEL-TIME AT ONE VOXEL
    # =====================================================

    def get_travel_time_at_voxel(
        self,
        depth_index,
        crossline_index,
        inline_index
    ):
        """
        Retrieve travel time at one voxel.

        Parameters
        ----------
        depth_index : int
            Depth voxel index.

        crossline_index : int
            Crossline voxel index.

        inline_index : int
            Inline voxel index.

        Returns
        -------
        float
            Travel time [s].
        """

        travel_time = self.get_travel_time()

        if not (
            0 <= depth_index < self.depth_size
        ):
            raise IndexError(
                "depth_index is outside the travel-time field."
            )

        if not (
            0 <= crossline_index < self.height_size
        ):
            raise IndexError(
                "crossline_index is outside the travel-time field."
            )

        if not (
            0 <= inline_index < self.width_size
        ):
            raise IndexError(
                "inline_index is outside the travel-time field."
            )

        return float(
            travel_time[
                0,
                0,
                depth_index,
                crossline_index,
                inline_index
            ].item()
        )

    # =====================================================
    # FIELD STATISTICS
    # =====================================================

    def travel_time_statistics(self):
        """
        Return basic statistics of the travel-time field.

        Returns
        -------
        dict
            Minimum, maximum, mean, NaN and Inf status.
        """

        travel_time = self.get_travel_time()

        return {
            "minimum": float(
                travel_time.min().item()
            ),
            "maximum": float(
                travel_time.max().item()
            ),
            "mean": float(
                travel_time.mean().item()
            ),
            "contains_nan": bool(
                torch.isnan(
                    travel_time
                ).any().item()
            ),
            "contains_inf": bool(
                torch.isinf(
                    travel_time
                ).any().item()
            )
        }