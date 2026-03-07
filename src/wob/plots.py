"""Plotting utilities for WOB Project."""
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def parallel_coordinates_polars(
    df: pl.DataFrame,
    dependent_var: str,
    columns: list[str] | None = None,
    cmap: str = "viridis",
    cmap_min: float | None = None,
    cmap_max: float | None = None,
    figsize=(12, 6),
    alpha: float = 0.05,
    linewidth: float = 1.0,
):
    """
    Parallel coordinates plot for a Polars DataFrame using matplotlib.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe

    dependent_var : str
        Continuous variable used to color lines

    columns : list[str], optional
        Columns to include (default: all numeric except dependent_var)

    cmap : str
        Matplotlib colormap name

    cmap_min : float, optional
        Minimum value for colormap scaling

    cmap_max : float, optional
        Maximum value for colormap scaling

    figsize : tuple
        Figure size

    alpha : float
        Line transparency (density smoothing)

    linewidth : float
        Line width

    Notes
    -----
    Any axis with strictly positive values spanning more than two orders of
    magnitude (max/min > 100) is log10-transformed before normalization.
    """

    # Select numeric columns automatically
    if columns is None:
        columns = [
            c for c, dt in zip(df.columns, df.dtypes)
            if dt.is_numeric() and c != dependent_var
        ]

    if dependent_var in columns:
        columns.remove(dependent_var)

    plot_cols = columns + [dependent_var]

    data = df.select(plot_cols).to_numpy()
    X = data[:, :-1]
    y = data[:, -1]

    _, n_dims = X.shape

    # Axis scaling
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    # Use log scaling for axes spanning > 2 orders of magnitude when values are positive.
    log_scaled = np.zeros(n_dims, dtype=bool)
    X_for_scaling = X.astype(float).copy()
    mins_for_scaling = mins.astype(float).copy()
    maxs_for_scaling = maxs.astype(float).copy()

    for i in range(n_dims):
        col = X[:, i]
        col_min = mins[i]
        col_max = maxs[i]
        if col_min > 0 and (col_max / col_min) > 100:
            log_scaled[i] = True
            X_for_scaling[:, i] = np.log10(col)
            mins_for_scaling[i] = np.log10(col_min)
            maxs_for_scaling[i] = np.log10(col_max)

    X_scaled = (X_for_scaling - mins_for_scaling) / (
        maxs_for_scaling - mins_for_scaling + 1e-12
    )

    x = np.arange(n_dims)

    fig, ax = plt.subplots(figsize=figsize)

    # Determine color scale bounds
    if cmap_min is None:
        cmap_min = float(y.min())

    if cmap_max is None:
        cmap_max = float(y.max())

    norm = Normalize(vmin=cmap_min, vmax=cmap_max)
    cmap_obj = cm.get_cmap(cmap)

    # Draw lines
    for row, val in zip(X_scaled, y):
        ax.plot(
            x,
            row,
            color=cmap_obj(norm(val)),
            alpha=alpha,
            linewidth=linewidth,
        )

    # Draw vertical axes
    for i in range(n_dims):
        ax.vlines(x[i], 0, 1, color="black", linewidth=1)

        ax.text(
            x[i]+0.2,
            0.03,
            f"{mins[i]:.2f}",
            ha="center",
            va="top",
            fontsize=9,
        )

        ax.text(
            x[i]+0.2,
            0.97,
            f"{maxs[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Axis formatting
    ax.set_xticks(x)
    axis_labels = [f"{col} (log10)" if log_scaled[i] else col for i, col in enumerate(columns)]
    ax.set_xticklabels(axis_labels)

    ax.set_xlim(x.min() - 0.1, x.max() + 0.1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(dependent_var)

    ax.set_title("Parallel Coordinates Plot")

    plt.tight_layout()
