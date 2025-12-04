import numpy as np
import matplotlib.pyplot as plt
from typing import Any
import jax.numpy as jnp

from Plotting.Format_Axes import set_axis_decimal_format

def _time_scale_and_label(unit: str) -> tuple[float, str]:
    unit = unit.lower()
    if unit == "s":
        return 1.0, "Relative Time (s)"
    if unit == "ms":
        return 1e3, "Relative Time (ms)"
    if unit in ("us", "µs"):
        return 1e6, "Relative Time (µs)"
    raise ValueError(f"Unsupported time_unit {unit!r}. Use 's', 'ms', or 'us'.")

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

def plot_phasegram_on_axis(
    ax: plt.Axes,
    seq,
    t_start: float | None = None,
    t_end: float | None = None,
    f_min: float | None = None,
    f_max: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    time_unit: str = "s",
    freq_unit: str = "Hz",
    n_freq_ticks: int = 8,
    n_time_ticks: int = 8,
    # NEW toggles:
    show_colorbar: bool = True,
    show_title: bool = True,
    show_xlabel: bool = True,
    show_ylabel = False,
) -> Any:

    if seq.fft_phase is None or seq.fft_freqs is None or seq.segment_times is None:
        raise ValueError("Sequence missing FFT phase data. Call compute_segment_ffts(...) first.")

    phase = np.asarray(seq.fft_phase)      # (num_segments, n_fft)
    freqs = np.asarray(seq.fft_freqs)      # (n_fft,)
    seg_starts = np.asarray(seq.segment_times, dtype=float)

    num_segments = phase.shape[0]
    if num_segments == 0:
        raise ValueError("No FFT segments available. Did compute_segment_ffts(...) succeed?")
    if seq.fft_n is None:
        raise ValueError("seq.fft_n is None; compute_segment_ffts(...) must set it.")

    seg_duration = float(seq.fft_n) / float(seq.sample_rate)
    seg_ends = seg_starts + seg_duration

    global_t_start = seg_starts[0]
    global_t_end = seg_ends[-1]
    if t_start is None: t_start = global_t_start
    if t_end   is None: t_end   = global_t_end
    if t_end <= t_start:
        raise ValueError(f"Empty time window: t_start={t_start}, t_end={t_end}")

    mask = (
        ((seg_starts >= t_start) & (seg_starts <= t_end)) |
        ((seg_ends   >= t_start) & (seg_ends   <= t_end))
    )
    if not np.any(mask):
        raise ValueError(f"No segments have start or end times within [{t_start}, {t_end}] s")

    phase = phase[mask, :]
    seg_starts_sel = seg_starts[mask]
    seg_ends_sel = seg_ends[mask]
    time_axis = (seg_starts_sel + seg_ends_sel) / 2  # absolute times

    if f_min is not None or f_max is not None:
        if f_min is None: f_min = float(freqs.min())
        if f_max is None: f_max = float(freqs.max())
        f_mask = (freqs >= f_min) & (freqs <= f_max)
        if not np.any(f_mask):
            raise ValueError(f"No frequency bins in [{f_min}, {f_max}] Hz")
        freqs = freqs[f_mask]
        phase = phase[:, f_mask]

    if vmin is None: vmin = float(phase.min())
    if vmax is None: vmax = float(phase.max())

    t0 = time_axis[0]
    relative_time = time_axis - t0


    t_scale, t_label = _time_scale_and_label(time_unit)
    f_scale, f_label = _freq_scale_and_label(freq_unit)
    time_plot = relative_time * t_scale
    freq_plot = freqs * f_scale
    extent = [0, (t_end - t_start) * t_scale, freq_plot[0], freq_plot[-1]]



    phasegram_data = jnp.stack(phase, axis=0)

    im = ax.imshow(
        phasegram_data.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        rasterized=True,
    )


    if show_title:
        ax.set_title("Phase")
    else:
        ax.set_title("")

    f_ticks = np.linspace(freq_plot.min(), freq_plot.max(), n_freq_ticks)
    if show_ylabel:
        ax.set_ylabel(f_label)
        ax.set_yticks(f_ticks)
        ax.set_yticklabels([f"{tick:.3g}" for tick in f_ticks])
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])




    if show_xlabel:
        ax.set_xlabel(t_label)

        t_ticks = np.linspace(0, (t_end - t_start) * t_scale, n_time_ticks)
        ax.set_xticks(t_ticks)
        ax.set_xticklabels([f"{tick:.3g}" for tick in t_ticks])
    else:
        ax.set_xticklabels([])




    set_axis_decimal_format(ax, x_decimals=0, y_decimals=1)

    if show_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label("Phase (rad)")

