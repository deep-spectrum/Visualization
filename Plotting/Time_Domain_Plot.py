import numpy as np
import matplotlib.pyplot as plt
from typing import Any

from Plotting.Format_Axes import set_axis_decimal_format

def _time_scale_and_label(unit):
    unit = unit.lower()
    if unit == "s":
        return 1.0, "Relative Time (s)"
    if unit == "ms":
        return 1e3, "Relative Time (ms)"
    if unit in ("us", "µs"):
        return 1e6, "Relative Time (µs)"
    raise ValueError(f"Unsupported time_unit {unit!r}. Use 's', 'ms', or 'us'.")

def plot_waveform_on_axis(
    ax: plt.Axes,
    seq,
    max_seconds = None,
    t_start = None,
    t_end = None,
    time_unit = "s",
    show_title = True,
    show_xlabel = True,
    show_ylabel = True,
):
    # Ensure that the received samples and their metadata is properly formatted for processing.
    samples = np.asarray(seq.samples)
    fs = float(seq.sample_rate)
    num_samples = samples.shape[0]
    total_duration = num_samples / fs

    # Set up the start and end times of the time-domain plot and their corresponding indices in the samples array.
    if t_start is None:
        t_start = 0.0
    if t_end is None:
        t_end = t_start + max_seconds if max_seconds is not None else total_duration

    t_start = max(0.0, t_start)
    t_end = min(t_end, total_duration)
    if t_end <= t_start:
        raise ValueError(f"Empty time window: t_start={t_start}, t_end={t_end}")

    start_idx = int(np.floor(t_start * fs))
    end_idx = int(np.ceil(t_end * fs))
    start_idx = max(0, min(start_idx, num_samples))
    end_idx = max(0, min(end_idx, num_samples))

    # Extract only the samples of interest.
    samples_win = samples[start_idx:end_idx]


    # Set up the time axis.
    t = np.arange(start_idx, end_idx, dtype=np.float64) / fs
    t_begin = t[0]
    t = t - t_begin
    scale, label = _time_scale_and_label(time_unit)
    t_scaled = t * scale

    # Plot the data. This could be adjusted to plot I and Q, absolute value, phase, or another quantity.
    #ax.plot(t_scaled, samples_win.real, label="I")
    #ax.plot(t_scaled, samples_win.imag, label="Q", alpha=0.7)
    ax.plot(t_scaled, np.angle(samples_win))


    # Control axis labels and limits.
    ax.set_xlim(0 * scale, (t_end - t_begin) * scale)

    if show_xlabel:
        ax.set_xlabel(label)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_ylabel("Phase (rad)")

    if show_title:
        ax.set_title("Time-Domain Waveform")
    else:
        ax.set_title("")



    ax.legend(loc="upper right", fontsize="x-small")

    set_axis_decimal_format(ax, x_decimals=0, y_decimals=2)
