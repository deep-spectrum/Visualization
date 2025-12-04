
# Data class and helper functions for representing an entire SigMF file.
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

import jax
import jax.numpy as jnp
import numpy as np
from sigmf import SigMFFile


@dataclass
class SigMFSampleSequence:
    ####################
    # Class Parameters #
    ####################
    name: str
    sample_rate: float             # Hz
    samples: jnp.ndarray           # 1-D complex64 JAX array of IQ samples

    # Optional RF center frequency from SigMF metadata (Hz).
    # If present, fft_freqs will be baseband bins + center_freq.
    center_freq: float | None = None

    # Index of the first loaded sample in the original recording.
    # This lets segment_times / indices stay "absolute" even if we only
    # load a window of the file.
    base_sample_index: int = 0

    # FFT / segmentation results, populated by compute_segment_ffts(...)
    fft_n: int | None = None
    fft_complex: jnp.ndarray | None = None   # shape (num_segments, n_fft)
    fft_mag: jnp.ndarray | None = None       # shape (num_segments, n_fft)
    fft_phase: jnp.ndarray | None = None     # shape (num_segments, n_fft)
    fft_freqs: jnp.ndarray | None = None     # shape (n_fft,) in Hz

    # One index and one time per segment: start of each FFT window.
    segment_sample_indices: jnp.ndarray | None = None  # shape (num_segments,)
    segment_times: jnp.ndarray | None = None           # shape (num_segments,)

    ##################
    # Segmented FFTs #
    ##################
    # Compute segment-by-segment FFTs for phase and spectrum visualization.
    # n_fft = segment length
    # window_type = hanning, hamming, or rectangular FFT window
    # debug = flag for whether to display debugging information.
    def compute_segment_ffts(
        self,
        n_fft: int,
        window_type: str = "hann",
        debug: bool = False,
    ) -> "SigMFSampleSequence":
        # Precondition
        if n_fft <= 0:
            raise ValueError("n_fft must be a positive integer")

        # Generate the window using NumPy (JAX jnp compatible).
        wt = window_type.lower()
        if wt in ("hann", "hanning"):
            window_np = np.hanning(n_fft).astype(np.float32)
        elif wt == "hamming":
            window_np = np.hamming(n_fft).astype(np.float32)
        elif wt in ("rect", "rectangular", "boxcar"):
            window_np = np.ones(n_fft, dtype=np.float32)
        else:
            raise ValueError(
                f"Unsupported window_type {window_type!r}. "
                "Supported: 'hann', 'hamming', 'rect', 'boxcar'."
            )
        window = jnp.array(window_np)  # JAX array

        # Determine the number of segments and trim out any IQ samples that don't
        # fit within an integer number of segments.
        # WARNING: Any 'excess' samples WILL be dropped by this process.
        total_samples = int(self.samples.shape[0])
        num_segments = total_samples // n_fft
        used_samples = num_segments * n_fft

        if num_segments == 0:
            raise ValueError(
                f"Not enough samples ({total_samples}) to form a single "
                f"{n_fft}-point segment."
            )

        trimmed = self.samples[:used_samples]

        # Use JAX JIT to efficiently compute the FFTs (and segments) themselves.
        (
            fft_complex,
            fft_mag,
            fft_phase,
            segment_start_indices,
            segment_start_times,
        ) = _compute_segment_ffts_jax(
            samples=trimmed,
            sample_rate=float(self.sample_rate),
            window=window,
            base_sample_index=int(self.base_sample_index),
        )

        # Store results within the object.
        self.fft_n = n_fft
        self.fft_complex = fft_complex
        self.fft_mag = fft_mag
        self.fft_phase = fft_phase
        self.segment_sample_indices = segment_start_indices
        self.segment_times = segment_start_times

        # Make frequency adjustments to go from DFT-domain to regular frequency
        # domain.
        # Baseband FFT frequency bins in Hz, unshifted:
        # e.g. [0, fs/N, 2fs/N, ..., -fs/2, ..., -fs/N]
        fft_freqs = jnp.fft.fftfreq(n_fft, d=1.0 / float(self.sample_rate))

        # Shift to center zero frequency: [-fs/2, ..., 0, ..., +fs/2)
        fft_freqs_centered = jnp.fft.fftshift(fft_freqs)
        #print("FFT freqs: ", fft_freqs_centered[0], " ", fft_freqs_centered[-1])

        # If we have a center frequency from SigMF, shift to RF
        if self.center_freq is not None:
            fc = float(self.center_freq)
        else:
            fc = 0.0

        self.fft_freqs = (fft_freqs_centered + fc).astype(jnp.float32)
        #print("Final FFT freqs: ", self.fft_freqs[0], " ", self.fft_freqs[-1])

        # Apply fftshift to all FFT-related outputs so zero frequency is centered
        self.fft_complex = jnp.fft.fftshift(self.fft_complex, axes=-1)
        self.fft_mag = jnp.fft.fftshift(self.fft_mag, axes=-1)
        self.fft_phase = jnp.fft.fftshift(self.fft_phase, axes=-1)

        return self




########################################
# Loading from the original SigMF file #
########################################
# Load the samples from t_start to t_end in the SigMF file pair with path specified in path_like.
def load_sigmf_sequence(
        path_like: str,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
) -> SigMFSampleSequence:

    meta_file = path_like + ".sigmf-meta"
    data_file = path_like + ".sigmf-data"

    # Step 1: Read metadata
    try:
        with open(meta_file, "r") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Metadata file not found: {meta_file}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON in metadata file: {meta_file}")

    # Extract relevant metadata
    try:
        raw_datatype = json_data["global"]["core:datatype"]
        raw_sampling_rate = json_data["global"]["core:sample_rate"]
    except KeyError as e:
        raise ValueError(f"Missing required key in metadata: {e}")

    datatype_np = to_np_datatype(raw_datatype)
    sampling_rate = float(raw_sampling_rate)
    center_freq = _extract_center_freq(json_data["global"], json_data["captures"][0])

    # Step 2: Determine the number of samples
    try:
        datafile_size = os.path.getsize(data_file)
    except FileNotFoundError:
        raise ValueError(f"Data file not found: {data_file}")

    samples_available = math.floor(datafile_size / np.dtype(datatype_np).itemsize)

    # Step 3: Compute start and end indices
    start_index = _validate_time_index(t_start, sampling_rate, samples_available, is_start=True)
    end_index = _validate_time_index(t_end, sampling_rate, samples_available, is_start=False)

    # Step 4: Memory mapping and segment extraction
    try:
        memmapped_recording = np.memmap(data_file, mode="r", dtype=datatype_np)
    except ValueError as e:
        raise ValueError(f"Error mapping data file: {e}")

    # Extract the segment of interest
    segment_of_interest = memmapped_recording[start_index:end_index]
    samples_jax = jnp.array(segment_of_interest, dtype=datatype_np)

    name = _base_name_from_path(path_like)

    return SigMFSampleSequence(
        name=name,
        sample_rate=sampling_rate,
        samples=samples_jax,
        center_freq=center_freq,
        base_sample_index=int(start_index),
    )

# Validate the requested time period against the boundaries of the file from which the data is requested.
def _validate_time_index(
        time: Optional[float],
        sampling_rate: float,
        samples_available: int,
        is_start: bool
) -> int:
    if time is None:
        return 0 if is_start else samples_available

    if time < 0:
        raise ValueError(f"{'Start' if is_start else 'End'} time must be non-negative")

    index = int(np.floor(time * sampling_rate)) if is_start else int(np.ceil(time * sampling_rate))

    if index < 0 or index > samples_available:
        raise ValueError(f"{'Start' if is_start else 'End'} index out of bounds: {index}")

    return max(0, min(index, samples_available))


# Convert a SigMF datatype to the 'equivalent' numpy datatype, where possible.
# Note that the names generally do not match directly between the two.
def to_np_datatype(raw_datatype: str) -> np.dtype:
    if raw_datatype == "cf32_le":
        return np.complex64
    # Add more as needed for other types...
    raise ValueError(f"Unsupported datatype: {raw_datatype}")


# Strip any SigMF-related extensions that were included in a path specification to get the plain path.
def _base_name_from_path(path_like: Union[str, Path]) -> str:
    p = Path(path_like)
    name = p.name
    for suffix in (".sigmf-meta", ".sigmf-data", ".sigmf"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


# Extract the center frequency from the JSON SigMF metadata; the SigMF standard permits that parameter to be stored in
# multiple locations within the metadata file.
def _extract_center_freq(global_info: dict, capture0) -> Optional[float]:

    center = None

    # Try to use a SigMFFile constant if present
    for attr in ("FREQUENCY_KEY", "FREQ_KEY", "CENTER_FREQUENCY_KEY"):
        if hasattr(SigMFFile, attr):
            key_name = getattr(SigMFFile, attr)
            center = global_info.get(key_name)
            if center is not None:
                break

    # Fallback to common keys
    if center is None:
        center = global_info.get("core:frequency")
    if center is None:
        center = capture0.get("core:frequency")

    if center is None:
        return None

    try:
        return float(center)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid center frequency value: {center}")


# JAX JIT-accelerated computation of the segment-by-segment FFTs.
@jax.jit
def _compute_segment_ffts_jax(
    samples: jnp.ndarray,
    sample_rate: float,
    window: jnp.ndarray,
    base_sample_index: int,
):
    # Parameters.
    n_fft = window.shape[0]
    total_samples = samples.shape[0]
    num_segments = total_samples // n_fft

    # Reshape to (num_segments, n_fft)
    segments = samples.reshape((num_segments, n_fft))

    # Apply the window (broadcast across segments)
    windowed = segments * window

    # Batched FFT along the last axis (one FFT per segment)
    fft_complex = jnp.fft.fft(windowed, n=n_fft, axis=-1).astype(jnp.complex64)
    fft_mag = jnp.abs(fft_complex).astype(jnp.float32)
    fft_phase = jnp.angle(fft_complex).astype(jnp.float32)

    # Each segment starts at index base_sample_index + k*n_fft in the
    # original stream.
    base = jnp.int64(base_sample_index)
    segment_indices_local = jnp.arange(num_segments, dtype=jnp.int64) * n_fft
    segment_start_indices = base + segment_indices_local

    # Start times (in seconds) for each FFT segment, absolute w.r.t recording
    sr = jnp.float32(sample_rate)
    segment_start_times = segment_start_indices.astype(jnp.float32) / sr

    return (
        fft_complex,
        fft_mag,
        fft_phase,
        segment_start_indices,
        segment_start_times,
    )
