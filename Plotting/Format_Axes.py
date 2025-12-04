from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def set_axis_decimal_format(ax: plt.Axes, x_decimals: int = 4, y_decimals: int = 1) -> None:
    """
    Force both x and y axes on `ax` to use fixed-point tick labels
    with the given number of decimal places (default: x 4, y 2).
    """
    x_fmt = f"%.{x_decimals}f"
    ax.xaxis.set_major_formatter(FormatStrFormatter(x_fmt))
    y_fmt = f"%.{y_decimals}f"

    ax.yaxis.set_major_formatter(FormatStrFormatter(y_fmt))