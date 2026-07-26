"""
=========================================================
Synthetic Seismic Generator
=========================================================

Generates realistic synthetic 3D seismic cubes for the
Physics-Informed 3D Encoder–Decoder Framework with
Predictive Uncertainty.

Stage 1 Features
----------------
✓ Empty reflectivity cube
✓ Horizontal reflectors
✓ Random reflector amplitudes
✓ Random reflector spacing
✓ Random polarity

Future Extensions
-----------------
• Dipping reflectors
• Folded structures
• Faults
• Channels
• Salt bodies
• Unconformities
• Realistic acquisition effects

Author: Ormin Joseph
=========================================================
"""

import os
import numpy as np


class SyntheticGenerator:
    """
    SyntheticGenerator

    Creates realistic 3D synthetic seismic data for
    training and evaluating the Physics-Informed
    3D Encoder–Decoder Framework.

    The generator first constructs a 3D reflectivity
    model consisting of geological interfaces. The
    reflectivity model is subsequently converted into
    synthetic seismic data through wavelet convolution.

    Current Stage
    -------------
    • Empty reflectivity cube
    • Random horizontal reflectors
    • Random reflector amplitudes
    • Random reflector spacing
    • Random reflector polarity

    Future Stages
    -------------
    • Dipping reflectors
    • Folded structures
    • Faults
    • Channels
    • Salt bodies
    • Acquisition effects
    """

    def __init__(
            self,
            cube_size=(64, 64, 64),

            # Geological parameters
            number_of_horizontal_reflectors=10,
            number_of_dipping_reflectors=6,
            reflector_spacing=(4, 8),
            amplitude_range=(0.5, 1.5),
            starting_depth=(3, 6),
            maximum_dip=0.30,

            # Fold parameters
            number_of_folded_reflectors=5,
            fold_amplitude=(3, 8),
            fold_wavelength=(30, 60),

            # ------------------------------------------
            # Fault parameters
            # ------------------------------------------

            number_of_faults=3,

            fault_throw=(4, 12),

            fault_position=(15, 50),

            # ------------------------------------------
            # Channel parameters
            # ------------------------------------------

            number_of_channels=3,

            channel_depth=(20, 45),

            channel_width=(6, 15),

            channel_amplitude=(-1.2, -0.4),

            # Salt body parameters

            number_of_salt_bodies=2,

            salt_radius=(6, 15),

            salt_depth=(15, 45),

            salt_amplitude=(1.8, 2.8),

            # Layer heterogeneity

            heterogeneity_strength=0.15,

            # Wavelet parameters
            sampling_interval=0.004,
            wavelet_frequency=25.0,
            wavelet_length=0.128,

            # Noise parameters
            noise_std=0.02,

            # ------------------------------------------
            # Missing trace parameters
            # ------------------------------------------

            missing_trace_percentage=0.30,

            mask_type="random",

            trace_interval=3,

            # Receiver-line masking
            receiver_line_interval=4,
            shot_line_interval=4,

            random_seed=None


    ):
        """
        Initialize the synthetic seismic generator.

        Parameters
        ----------
        cube_size : tuple
            Dimensions of the synthetic seismic cube in the form
            (Depth, Inline, Crossline).

        number_of_horizontal_reflectors : int
            Number of horizontal geological reflectors.

        number_of_dipping_reflectors : int
            Number of dipping geological reflectors.
            (Reserved for future implementation.)

        reflector_spacing : tuple
            Minimum and maximum spacing (samples)
            between successive reflectors.

        amplitude_range : tuple
            Minimum and maximum reflector amplitudes.

        starting_depth : tuple
            Minimum and maximum depth where the first
            reflector may be placed.

        maximum_dip : float
            Maximum reflector dip (samples per inline).
            Reserved for future dipping reflector generation.

        sampling_interval : float
            Temporal sampling interval (seconds).
        wavelet_frequency : float
            Dominant Ricker wavelet frequency (Hz).

        wavelet_length : float
            Duration of the Ricker wavelet (seconds).

        noise_std : float
            Standard deviation of additive Gaussian noise.

        random_seed : int or None
            Random seed for reproducibility.
        """


        # ------------------------------------------
        # Cube dimensions
        # ------------------------------------------

        self.depth = cube_size[0]

        self.inline = cube_size[1]

        self.crossline = cube_size[2]

        # ------------------------------------------
        # Geological parameters
        # ------------------------------------------

        self.number_of_horizontal_reflectors = (
            number_of_horizontal_reflectors
        )

        self.number_of_dipping_reflectors = (
            number_of_dipping_reflectors
        )

        self.reflector_spacing = reflector_spacing

        self.amplitude_range = amplitude_range

        self.starting_depth = starting_depth

        self.maximum_dip = maximum_dip

        # ------------------------------------------
        # Fold parameters
        # ------------------------------------------

        self.number_of_folded_reflectors = (
            number_of_folded_reflectors
        )

        self.fold_amplitude = fold_amplitude

        self.fold_wavelength = fold_wavelength

        # ------------------------------------------
        # Fault parameters
        # ------------------------------------------

        self.number_of_faults = number_of_faults

        self.fault_throw = fault_throw

        self.fault_position = fault_position

        # ------------------------------------------
        # Channel parameters
        # ------------------------------------------

        self.number_of_channels = (
            number_of_channels
        )

        self.channel_depth = channel_depth

        self.channel_width = channel_width

        self.channel_amplitude = channel_amplitude



        # ------------------------------------------
        # Salt parameters
        # ------------------------------------------

        self.number_of_salt_bodies = (
            number_of_salt_bodies
        )

        self.salt_radius = salt_radius

        self.salt_depth = salt_depth

        self.salt_amplitude = salt_amplitude

        # ------------------------------------------
        # Layer heterogeneity
        # ------------------------------------------

        self.heterogeneity_strength = heterogeneity_strength

        # ------------------------------------------
        # Wavelet parameters
        # ------------------------------------------

        self.sampling_interval = sampling_interval

        self.wavelet_frequency = wavelet_frequency

        self.wavelet_length = wavelet_length


        # ------------------------------------------
        # Noise parameters
        # ------------------------------------------

        self.noise_std = noise_std

        # ------------------------------------------
        # Missing trace parameters
        # ------------------------------------------

        self.missing_trace_percentage = (
            missing_trace_percentage
        )
        self.mask_type = mask_type

        self.trace_interval = trace_interval

        self.receiver_line_interval = receiver_line_interval

        self.shot_line_interval = shot_line_interval

        # ------------------------------------------
        # Random seed
        # ------------------------------------------

        if random_seed is not None:
            np.random.seed(random_seed)


    # --------------------------------------------------
    # Empty cube
    # --------------------------------------------------

    def create_empty_cube(self):
        """
        Create an empty 3D reflectivity cube.

        Returns
        -------
        ndarray

            Shape

            (Depth,
             Inline,
             Crossline)
        """

        cube = np.zeros(

            (
                self.depth,
                self.inline,
                self.crossline
            ),

            dtype=np.float32

        )

        return cube


    # --------------------------------------------------
    # Horizontal Reflectors
    # --------------------------------------------------

    def generate_horizontal_reflectors(
            self,
            cube
    ):

        """
        Generate horizontal geological reflectors.

        Each reflector is assigned

        • a random amplitude,
        • a random polarity,
        • a random spacing from the previous reflector,

        to simulate impedance contrasts between
        subsurface geological layers.
        """

        # Randomly choose the depth of the first reflector.
        current_depth = np.random.randint(
            self.starting_depth[0],
            self.starting_depth[1]
        )

        for _ in range(
                self.number_of_horizontal_reflectors
        ):

            if current_depth >= (self.depth - 2):
                break



            amplitude = np.random.uniform(

                self.amplitude_range[0],

                self.amplitude_range[1]

            )

            polarity = np.random.choice(

                [-1.0, 1.0]

            )
            # Assign the reflector amplitude across the
            # entire inline and crossline section.


            cube[
                current_depth,
                :,
                :
            ] = amplitude * polarity

            spacing = np.random.randint(

                self.reflector_spacing[0],

                self.reflector_spacing[1] + 1

            )

            current_depth += spacing

        return cube

    # --------------------------------------------------
    # Dipping Reflectors
    # --------------------------------------------------

    def generate_dipping_reflectors(
            self,
            cube
    ):
        """
        Generate dipping geological reflectors.

        Each reflector is assigned

        • a random starting depth,
        • a random dip,
        • a random amplitude,
        • a random polarity,

        producing planar geological layers that dip across
        the inline direction.
        """

        for _ in range(self.number_of_dipping_reflectors):

            # ------------------------------------------
            # Random reflector properties
            # ------------------------------------------

            start_depth = np.random.randint(
                self.starting_depth[0],
                self.starting_depth[1]
            )

            dip = np.random.uniform(
                -self.maximum_dip,
                self.maximum_dip
            )

            amplitude = np.random.uniform(
                self.amplitude_range[0],
                self.amplitude_range[1]
            )

            polarity = np.random.choice(
                [-1.0, 1.0]
            )

            # ------------------------------------------
            # Draw reflector
            # ------------------------------------------

            for inline in range(self.inline):

                depth = int(
                    start_depth +
                    dip * inline
                )

                if 0 <= depth < self.depth:
                    cube[
                        depth,
                        inline,
                        :
                    ] = amplitude * polarity

        return cube

    # --------------------------------------------------
    # Folded Reflectors
    # --------------------------------------------------

    def generate_folded_reflectors(
            self,
            cube
    ):
        """
        Generate folded geological reflectors.

        Reflectors are generated using a sinusoidal
        deformation that simulates anticlines and
        synclines commonly observed in sedimentary
        basins.

        Returns
        -------
        ndarray
            Updated reflectivity cube.
        """

        # ------------------------------------------
        # Generate folded reflectors
        # ------------------------------------------

        for _ in range(self.number_of_folded_reflectors):
            # Starting depth
            start_depth = np.random.randint(
                self.starting_depth[0],
                self.starting_depth[1]
            )

            # Fold amplitude (samples)
            fold_amplitude = np.random.uniform(
                self.fold_amplitude[0],
                self.fold_amplitude[1]
            )

            # Fold wavelength (samples)
            fold_wavelength = np.random.uniform(
                self.fold_wavelength[0],
                self.fold_wavelength[1]
            )

            # Random reflector amplitude
            amplitude = np.random.uniform(
                self.amplitude_range[0],
                self.amplitude_range[1]
            )

            # Random polarity
            polarity = np.random.choice(
                [-1.0, 1.0]
            )
            # ------------------------------------------
            # Draw folded reflector
            # ------------------------------------------

            for inline in range(self.inline):

                # Sinusoidal fold
                folded_depth = int(

                    start_depth +

                    fold_amplitude *

                    np.sin(

                        2.0 *

                        np.pi *

                        inline /

                        fold_wavelength

                    )

                )

                # Ensure reflector stays inside cube
                if 0 <= folded_depth < self.depth:
                    cube[
                        folded_depth,
                        inline,
                        :
                    ] = amplitude * polarity

        return cube

    # --------------------------------------------------
    # Faults
    # --------------------------------------------------

    def generate_faults(
            self,
            cube
    ):
        """
        Generate geological faults by vertically shifting
        one side of the seismic cube.

        Faults are simulated as normal faults where
        reflectors on one side of the fault plane are
        displaced by a random throw.

        Returns
        -------
        ndarray
            Updated reflectivity cube.
        """
        # ------------------------------------------
        # Generate faults
        # ------------------------------------------

        for _ in range(self.number_of_faults):

            # Random inline location of the fault plane
            fault_inline = np.random.randint(
                self.fault_position[0],
                self.fault_position[1]
            )

            # Random fault throw (vertical displacement)
            throw = np.random.randint(
                self.fault_throw[0],
                self.fault_throw[1] + 1
            )

            # ------------------------------------------
            # Shift one side of the fault
            # ------------------------------------------

            shifted = cube[
                :-throw,
                fault_inline:,
                :
            ].copy()

            cube[
                throw:,
                fault_inline:,
                :
            ] = shifted

            # ------------------------------------------
            # Clear the gap created by the fault
            # ------------------------------------------

            cube[
                :throw,
                fault_inline:,
                :
            ] = 0.0

        return cube



    # --------------------------------------------------
    # Channels
    # --------------------------------------------------

    def generate_channels(
            self,
            cube
    ):
        """
        Generate fluvial channels using a parabolic
        geometry.

        Channels are represented as erosional features
        that cut through existing geological reflectors.

        Returns
        -------
        ndarray
            Updated reflectivity cube.
        """
        # ------------------------------------------
        # Generate channels
        # ------------------------------------------

        for _ in range(self.number_of_channels):
            # Channel centre depth
            channel_depth = np.random.randint(
                self.channel_depth[0],
                self.channel_depth[1]
            )

            # Channel width
            channel_width = np.random.randint(
                self.channel_width[0],
                self.channel_width[1] + 1
            )

            # Channel amplitude
            channel_amplitude = np.random.uniform(
                self.channel_amplitude[0],
                self.channel_amplitude[1]
            )
            # ------------------------------------------
            # Draw channel
            # ------------------------------------------

            for inline in range(self.inline):
                # Centre of the channel
                centre = self.inline // 2
                distance = inline - centre
                offset = int(

                    (distance ** 2)

                    / channel_width

                )
                # Depth of the channel at this inline
                depth = channel_depth + offset
                # Ensure channel remains inside cube
                for inline in range(self.inline):

                    ...

                    if depth >= self.depth:
                        continue

                    cube[
                        depth:,
                        inline,
                        :
                    ] = channel_amplitude


        return cube


    # --------------------------------------------------
    # Layer Heterogeneity
    # --------------------------------------------------

    def apply_layer_heterogeneity(
            self,
            cube
    ):
        """
        Apply lateral amplitude variations to the
        reflectivity cube.

        This simulates natural changes in rock
        properties across the seismic survey.
        """
        # Random amplitude variation

        variation = np.random.normal(

            loc=1.0,

            scale=self.heterogeneity_strength,

            size=cube.shape

        ).astype(np.float32)
        cube *= variation

        return cube

    # --------------------------------------------------
    # Salt Bodies
    # --------------------------------------------------

    def generate_salt_bodies(
            self,
            cube
    ):
        """
        Generate salt bodies.

        Salt bodies are modelled as approximately
        cylindrical structures extending vertically
        through the seismic volume.

        Returns
        -------
        ndarray
            Updated reflectivity cube.
        """
        # ------------------------------------------
        # Generate salt bodies
        # ------------------------------------------

        for _ in range(self.number_of_salt_bodies):
            # Salt centre

            centre_inline = np.random.randint(
                10,
                self.inline - 10
            )

            centre_crossline = np.random.randint(
                10,
                self.crossline - 10
            )

            centre_depth = np.random.randint(
                self.salt_depth[0],
                self.salt_depth[1]
            )

            radius = np.random.randint(
                self.salt_radius[0],
                self.salt_radius[1] + 1
            )

            amplitude = np.random.uniform(
                self.salt_amplitude[0],
                self.salt_amplitude[1]
            )
            # ------------------------------------------
            # Draw cylindrical salt body
            # ------------------------------------------

            for inline in range(self.inline):

                for crossline in range(self.crossline):

                    # Horizontal distance from salt centre
                    distance = np.sqrt(

                        (inline - centre_inline) ** 2 +

                        (crossline - centre_crossline) ** 2

                    )

                    # Inside salt radius
                    if distance <= radius:

                        for depth in range(
                                centre_depth,
                                self.depth
                        ):
                            cube[
                                depth,
                                inline,
                                crossline
                            ] = amplitude
        return cube

    # --------------------------------------------------
    # Complete Reflectivity Model
    # --------------------------------------------------

    def generate_reflectivity_model(self):
        """
        Generate a complete 3D reflectivity model.

        The reflectivity model combines all currently
        implemented geological structures into a single
        impedance model.

        Current implementation includes
        • Horizontal reflectors
        • Dipping reflectors
        • Folded reflectors
        • Faults

        Future versions will include
        • Channels
        • Salt bodies
        • Unconformities




        Returns
        -------
        ndarray
            Reflectivity cube with shape

            (Depth, Inline, Crossline)
        """

        # ------------------------------------------
        # Empty reflectivity cube
        # ------------------------------------------

        cube = self.create_empty_cube()

        # ------------------------------------------
        # Horizontal reflectors
        # ------------------------------------------

        cube = self.generate_horizontal_reflectors(
            cube
        )

        # ------------------------------------------
        # Dipping reflectors
        # ------------------------------------------

        cube = self.generate_dipping_reflectors(
            cube
        )

        # ------------------------------------------
        # Folded reflectors
        # ------------------------------------------

        cube = self.generate_folded_reflectors(
            cube
        )

        # ------------------------------------------
        # Faults
        # ------------------------------------------

        cube = self.generate_faults(
            cube
        )

        # ------------------------------------------
        # Channels
        # ------------------------------------------

        cube = self.generate_channels(
            cube
        )

        # ------------------------------------------
        # Salt bodies
        # ------------------------------------------

        cube = self.generate_salt_bodies(
            cube
        )

        # ------------------------------------------
        # Layer heterogeneity
        # ------------------------------------------

        cube = self.apply_layer_heterogeneity(
            cube
        )
        return cube

    # --------------------------------------------------
    # Acquisition Mask
    # --------------------------------------------------

    def generate_acquisition_mask(self):
        """
        Generate a binary acquisition mask that simulates
        missing seismic traces.

        Returns
        -------
        ndarray
            Binary mask with values

            1 = observed trace
            0 = missing trace
        """

        mask = np.ones(

            (
                self.depth,
                self.inline,
                self.crossline
            ),

            dtype=np.float32

        )

        # ------------------------------------------
        # Random missing traces
        # ------------------------------------------

        total_traces = self.inline * self.crossline

        number_missing = int(

            self.missing_trace_percentage *

            total_traces

        )

        indices = np.random.choice(

            total_traces,

            number_missing,

            replace=False

        )

        for index in indices:
            inline = index // self.crossline

            crossline = index % self.crossline

            mask[
                :,
                inline,
                crossline
            ] = 0.0

        return mask

    # --------------------------------------------------
    # Ricker Wavelet
    # --------------------------------------------------

    def generate_ricker_wavelet(self):
        """
        Generate a zero-phase Ricker wavelet.

        Returns
        -------
        ndarray
            One-dimensional Ricker wavelet.
        """

        # Sampling interval (seconds)
        dt = self.sampling_interval

        # Time axis centred at zero
        time = np.arange(
            -self.wavelet_length / 2,
            self.wavelet_length / 2 + dt,
            dt,
            dtype=np.float32
        )

        f = self.wavelet_frequency

        wavelet = (
                          1.0
                          - 2.0 * (np.pi * f * time) ** 2
                  ) * np.exp(
            -(np.pi * f * time) ** 2
        )

        # Normalize amplitude
        wavelet /= np.max(np.abs(wavelet))

        return wavelet.astype(np.float32)

    # --------------------------------------------------
    # Wavelet Convolution
    # --------------------------------------------------

    def convolve_wavelet(
            self,
            reflectivity_cube
    ):

        """
        Convolve every seismic trace with the Ricker wavelet.

        Parameters
        ----------
        reflectivity_cube : ndarray
            3D reflectivity model
            (Depth, Inline, Crossline)

        Returns
        -------
        ndarray
            Synthetic seismic cube.
        """

        wavelet = self.generate_ricker_wavelet()

        seismic_cube = np.zeros_like(
            reflectivity_cube,
            dtype=np.float32
        )

        for inline in range(self.inline):

            for crossline in range(self.crossline):
                trace = reflectivity_cube[
                    :,
                    inline,
                    crossline
                ]

                seismic_cube[
                    :,
                    inline,
                    crossline
                ] = np.convolve(
                    trace,
                    wavelet,
                    mode="same"
                )

        return seismic_cube

    # --------------------------------------------------
    # Add Gaussian Noise
    # --------------------------------------------------

    def add_noise(
            self,
            seismic_cube
    ):
        """
        Add Gaussian noise to the synthetic seismic cube.

        Parameters
        ----------
        seismic_cube : ndarray
            Clean synthetic seismic cube.

        Returns
        -------
        ndarray
            Noisy synthetic seismic cube.
        """

        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_std,
            size=seismic_cube.shape
        ).astype(np.float32)

        noisy_cube = seismic_cube + noise

        return noisy_cube


    # --------------------------------------------------

    def generate_missing_trace_mask(
            self,
            seismic_cube
    ):
        """
        Generate a random missing-trace mask.

        Parameters
        ----------
        seismic_cube : ndarray
            Complete seismic cube.

        Returns
        -------
        ndarray
            Binary mask with

            1 = trace available

            0 = trace missing.
        """

        # ------------------------------------------
        # Start with all traces available
        # ------------------------------------------

        mask = np.ones_like(
            seismic_cube,
            dtype=np.float32
        )
        # ------------------------------------------
        # Total number of seismic traces
        # ------------------------------------------

        number_of_traces = (
                self.inline *
                self.crossline
        )
        # ------------------------------------------
        # Number of traces to remove
        # ------------------------------------------

        number_missing = int(
            number_of_traces *
            self.missing_trace_percentage
        )
        if self.mask_type == "random":

            missing_indices = np.random.choice(
                number_of_traces,
                number_missing,
                replace=False
            )

            for index in missing_indices:
                inline = index // self.crossline

                crossline = index % self.crossline

                mask[:, inline, crossline] = 0.0

        elif self.mask_type == "regular":

            for inline in range(self.inline):

                for crossline in range(self.crossline):

                    trace_number = (
                            inline * self.crossline +
                            crossline
                    )

                    if (
                            trace_number %
                            self.trace_interval
                    ) == 0:
                        mask[
                            :,
                            inline,
                            crossline
                        ] = 0.0

        elif self.mask_type == "receiver_lines":

            for inline in range(self.inline):

                if (
                        inline %
                        self.receiver_line_interval
                ) == 0:
                    mask[
                        :,
                        inline,
                        :
                    ] = 0.0
        elif self.mask_type == "shot_lines":

            for crossline in range(self.crossline):

                if (
                        crossline %
                        self.shot_line_interval
                ) == 0:
                    mask[
                        :,
                        :,
                        crossline
                    ] = 0.0
        return mask

    # --------------------------------------------------
    # Apply Missing Trace Mask
    # --------------------------------------------------

    def apply_missing_trace_mask(
            self,
            seismic_cube,
            mask
    ):
        """
        Apply the missing-trace mask to the seismic cube.

        Parameters
        ----------
        seismic_cube : ndarray
            Complete seismic cube.

        mask : ndarray
            Binary trace mask.

        Returns
        -------
        ndarray
            Corrupted seismic cube.
        """

        corrupted_cube = seismic_cube * mask

        return corrupted_cube

    # --------------------------------------------------
    # Complete Synthetic Seismic Cube
    # --------------------------------------------------

    def generate_seismic_cube(self):
        """
        Generate a complete synthetic seismic cube.

        Workflow
        --------
        1. Generate reflectivity model.
        2. Convolve with a Ricker wavelet.
        3. Add Gaussian noise.

        Returns
        -------
        ndarray
            Final synthetic seismic cube.
        """

        # ------------------------------------------
        # Reflectivity model
        # ------------------------------------------

        reflectivity = self.generate_reflectivity_model()

        # ------------------------------------------
        # Wavelet convolution
        # ------------------------------------------

        seismic = self.convolve_wavelet(
            reflectivity
        )

        # ------------------------------------------
        # Add acquisition noise
        # ------------------------------------------

        seismic = self.add_noise(
            seismic
        )

        maximum = np.max(np.abs(seismic))

        if maximum > 0:
            seismic = seismic / maximum
            
        return seismic

    def generate_training_sample(
            self
    ):
        """
        Generate one complete training sample.

        Returns
        -------
        tuple
            (
                ground_truth,
                corrupted,
                mask
            )
        """

        # ------------------------------------------
        # Generate clean seismic cube
        # ------------------------------------------

        ground_truth = self.generate_seismic_cube()

        # ------------------------------------------
        # Generate binary sampling mask
        # ------------------------------------------

        mask = self.generate_missing_trace_mask(
            ground_truth
        )

        # ------------------------------------------
        # Apply missing-trace mask
        # ------------------------------------------

        corrupted = self.apply_missing_trace_mask(
            ground_truth,
            mask
        )

        # ------------------------------------------
        # Return training sample
        # ------------------------------------------

        return (
            ground_truth,
            corrupted,
            mask
        )


