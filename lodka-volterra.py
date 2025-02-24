import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
from numba import njit
from pymc.ode import DifferentialEquation
from pytensor.compile.ops import as_op
from scipy.integrate import odeint
from scipy.optimize import least_squares

print(f"Running on PyMC v{pm.__version__}")

# ------------- The Hudson Bay Company data
data = pd.DataFrame(dict(
    year=np.arange(1900., 1921., 1),
    lynx=np.array([4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
                   8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6]),
    hare=np.array([30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
                   27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7])))
print(data.head())


# plot data function for reuse later
def plot_data(ax, data, save_as=None, lw=2, title="Hudson's Bay Company Data"):
    ax.plot(data.year, data.lynx, color="b", lw=lw, marker="o", markersize=12, label="Lynx (Data)")
    ax.plot(data.year, data.hare, color="g", lw=lw, marker="+", markersize=14, label="Hare (Data)")
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlim([1900, 1920])
    ax.set_ylim(0)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Pelts (Thousands)", fontsize=14)
    ax.set_xticks(data.year.astype(int))
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    ax.set_title(title, fontsize=16)

    if save_as:
        plt.savefig(save_as, bbox_inches='tight')  # Save the figure
        plt.close()  # Close the figure to free up memory

    return ax


# Example usage:
fig, ax = plt.subplots(figsize=(10, 5))
plot_data(ax, data, save_as='plot-lodka-volterra/hudsons_bay_plot.png')  # Specify the filename to save
plt.show()


# ------------------- Estimate params of the dynamical system from such data, using odeint

@njit
def rhs(X, t, theta):
    # unpack parameters
    x, y = X
    alpha, beta, gamma, delta, xt0, yt0 = theta
    # equations
    dx_dt = alpha * x - beta * x * y
    dy_dt = -gamma * y + delta * x * y
    return [dx_dt, dy_dt]


# plot model function
def plot_model(
        ax,
        x_y,
        time=np.arange(1900, 1921, 0.01),
        alpha=1,
        lw=3,
        title="Hudson's Bay Company Data and\nExample Model Run",
        save_as=None
):
    ax.plot(time, x_y[:, 1], color="b", alpha=alpha, lw=lw, label="Lynx (Model)")
    ax.plot(time, x_y[:, 0], color="g", alpha=alpha, lw=lw, label="Hare (Model)")
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)

    if save_as:
        plt.savefig(save_as, bbox_inches='tight')  # Save the figure
        plt.close()  # Close the figure to free up memory

    return ax


# Parameters
theta = np.array([0.52, 0.026, 0.84, 0.026, 34.0, 5.9])
time = np.arange(1900, 1921, 0.01)

# Initial conditions
initial_conditions = theta[-2:]

# Call Scipy's odeint function
x_y = odeint(func=rhs, y0=initial_conditions, t=time, args=(theta,))

# Plot
_, ax = plt.subplots(figsize=(12, 4))
plot_data(ax, data, save_as='plot-lodka-volterra/scipy-data.png', lw=0)
plot_model(ax, x_y, save_as='plot-lodka-volterra/model-plot.png')  # Specify filename to save the model plot

plt.show()  # Show the plot if needed
