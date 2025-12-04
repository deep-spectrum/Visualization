import numpy as np
import matplotlib.pyplot as plt
from typing import Any

from Plotting.Format_Axes import set_axis_decimal_format

# Utility function to allow the selection of the correct scaling factor (from seconds) and string description for the
# time-axis from a shorter string unit-code.
def _time_scale_and_label(unit: str) -> tuple[float, str]:
    unit = unit.lower()
    if unit == "s":
        return 1.0, "Relative Time (s)"
    if unit == "ms":
        return 1e3, "Relative Time (ms)"
    if unit in ("us", "µs"):
        return 1e6, "Relative Time (µs)"
    raise ValueError(f"Unsupported time_unit {unit!r}. Use 's', 'ms', or 'us'.")


# Utility function to allow the selection of the correct scaling factor (from Hertz) and string description for the
# frequency-axis from a short string unit-code.
def _freq_scale_and_label(unit: str) -> tuple[float, str]:
    unit = unit.lower()
    if unit == "hz":
        return 1.0, "Frequency (Hz)"
    if unit == "khz":
        return 1e-3, "Frequency (kHz)"
    if unit == "mhz":
        return 1e-6, "Frequency (MHz)"
    if unit == "ghz":
        return 1e-9, "Frequency (GHz)"
    raise ValueError(f"Unsupported freq_unit {unit!r}. Use 'Hz', 'kHz', or 'MHz'.")

# Plot a spectrogram visualization of the SigMF sequence seq on the provided plt.Axes ax.
def plot_spectrogram_on_axis(
    ax: plt.Axes,
    seq,
    # Display boundaries.
    t_start: float | None = None,
    t_end: float | None = None,
    f_min: float | None = None,
    f_max: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,

    # Display controls.
    cmap: str = "viridis",
    time_unit: str = "s",
    freq_unit: str = "Hz",
    n_freq_ticks: int = 8,
    n_time_ticks: int = 8,

    # Control flags.
    show_colorbar: bool = True,
    show_xlabel = False,
    show_ylabel = False,
    show_title = False,
    title = "Untitled Spectrogram"
) -> Any:
    # Check that the required FFTs have been computed.
    if seq.fft_mag is None or seq.fft_freqs is None or seq.segment_times is None:
        raise ValueError("Sequence missing FFT magnitude data. Call compute_segment_ffts(...) first.")

    # Extract the required FFT fields from the sequence.
    mag = np.asarray(seq.fft_mag)          # (num_segments, n_fft)
    freqs = np.asarray(seq.fft_freqs)      # (n_fft,)
    segment_starts = np.asarray(seq.segment_times, dtype=float)
    seg_duration = float(seq.fft_n) / float(seq.sample_rate)
    seg_ends = segment_starts + seg_duration

    # Constrain the times of the delayed data with the global time bounds, if relevant.
    global_t_start = segment_starts[0]
    global_t_end = seg_ends[-1]
    if t_start is None:
        t_start = global_t_start
    if t_end is None:
        t_end = global_t_end
    if t_end <= t_start:
        raise ValueError(f"Empty time window: t_start={t_start}, t_end={t_end}")

    # Select segments that have any overlap (start or end) with the desired window of times to plot.
    mask = (
            ((segment_starts >= t_start) & (segment_starts <= t_end)) |
            ((seg_ends   >= t_start) & (seg_ends   <= t_end))
    )
    if not np.any(mask):
        raise ValueError(f"No segments have start or end times within [{t_start}, {t_end}] s")

    mag = mag[mask, :]
    seg_starts_sel = segment_starts[mask]
    seg_ends_sel = seg_ends[mask]

    # Use the segment midpoints for the "time" location of each column
    time_axis = (seg_starts_sel + seg_ends_sel) / 2  # absolute times

    # Produce a time-relative-to-start axis for display purposes
    # Shift so that the earliest included segment is at time 0
    base_time = time_axis[0]
    relative_time = time_axis - base_time


    # Also display only the frequencies that are within the desired range.
    if f_min is not None or f_max is not None:
        if f_min is None:
            f_min = float(freqs.min())
        if f_max is None:
            f_max = float(freqs.max())
        f_mask = (freqs >= f_min) & (freqs <= f_max)
        if not np.any(f_mask):
            raise ValueError(f"No frequency bins in [{f_min}, {f_max}] Hz")
        freqs = freqs[f_mask]
        mag = mag[:, f_mask]

    # Convert magnitude to dB
    eps = 1e-12
    spec_db = 20.0 * np.log10(mag + eps)
    if vmin is None:
        vmin = float(spec_db.min())
    if vmax is None:
        vmax = float(spec_db.max())

    # Scaling for labels
    t_scale, t_label = _time_scale_and_label(time_unit)
    f_scale, f_label = _freq_scale_and_label(freq_unit)
    time_plot = relative_time * t_scale
    freq_plot = freqs * f_scale

    # Extent: x from 0 to max relative time
    extent = [0, (t_end - t_start) * t_scale, freq_plot[0], freq_plot[-1]]

    im = ax.imshow(
        spec_db.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        rasterized=True,
    )

    # Labels/titles (toggle)
    if show_xlabel:
        ax.set_xlabel(t_label)
        t_ticks = np.linspace(0, (t_end - t_start) * t_scale, n_time_ticks)
        ax.set_xticks(t_ticks)
        ax.set_xticklabels([f"{tick:.3g}" for tick in t_ticks])
    else:
        ax.set_xticks([])

    if show_ylabel:
        ax.set_ylabel(f_label)

    if show_title:
        ax.set_title(title)

    # Ticks
    f_ticks = np.linspace(freq_plot.min(), freq_plot.max(), n_freq_ticks)
    ax.set_yticks(f_ticks)
    ax.set_yticklabels([f"{tick:.3g}" for tick in f_ticks])

    # Numeric formatting (you can tweak decimals here if you like)
    set_axis_decimal_format(ax, x_decimals=0, y_decimals=1)

    # Colorbar (toggle)
    if show_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label("Power (dB)")


