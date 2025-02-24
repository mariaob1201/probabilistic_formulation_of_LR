# Maria Oros
#------------------------------------------------ Bayesina GLM in PyMC
# import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import xarray as xr

# import pymc as pm

# from pymc import HalfCauchy, Model, Normal, sample

# print(f"Running on PyMC v{pm.__version__}")

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

size = 200
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
# y = a + b*x
true_regression_line = true_intercept + true_slope * x
# add noise
y = true_regression_line + rng.normal(scale=0.5, size=size)

data = pd.DataFrame({"x": x, "y": y})

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, xlabel="x", ylabel="y", title="Generated data and underlying model")
ax.plot(x, y, "x", label="sampled data")
ax.plot(x, true_regression_line, label="true regression line", lw=2.0)
plt.legend(loc=0);

plt.savefig("plots/sample_data.png", dpi=300, bbox_inches="tight")

###-------------------------- Probabilistic model using bambi https://bambinos.github.io/bambi/
'''
Bayesian inference does not give us only one line of best fit (as maximum likelihood does) 
but rather a whole posterior distribution of plausible parameters. 
Lets plot the posterior distribution of our parameters and the individual samples we drew.
'''
import bambi as bmb
import arviz as az  # <-- Import arviz
import matplotlib.pyplot as plt

# Define the model
model = bmb.Model("y ~ x", data)
idata = model.fit(draws=3000)

# Plot and save trace
az.plot_trace(idata, figsize=(10, 7))  # Remove 'ax' argument

# Save the figure
plt.savefig("plots/trace_plot.png", dpi=300, bbox_inches="tight")

#The left side shows our marginal posterior – for each parameter value on the x-axis we get a probability on the y-axis that tells us how likely that parameter value is.

# --------------------- Posterior predictive lines
import arviz as az
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

# Compute y_model correctly
idata_pp = idata['posterior']
idata_pp["y_model"] = (
    idata_pp["Intercept"] +
    idata_pp["x"] * xr.DataArray(
        x,
        dims=["obs_id"],
        coords={"obs_id": np.arange(len(x))}
    )
)

# Create visualization
plt.figure(figsize=(10, 6))

# Plot original data points
plt.plot(x, y, 'o', color='black', alpha=0.5, label='Observed data')

# Plot true regression line
true_regression_line = true_intercept + true_slope * x
plt.plot(x, true_regression_line, 'r-', label='True regression line', linewidth=2)

# Plot posterior predictive lines
# Sample 100 random draws from the posterior
n_lines = 100
random_draws = np.random.randint(0, idata_pp.dims["draw"], n_lines)
for draw in random_draws:
    plt.plot(x, idata_pp["y_model"].sel(chain=0, draw=draw),
             'b-', alpha=0.1)

# Plot mean prediction
mean_prediction = idata_pp["y_model"].mean(dim=["chain", "draw"])
plt.plot(x, mean_prediction, 'b-', label='Mean prediction', linewidth=2)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Posterior Predictive Check')
plt.legend()

# Save the figure
plt.savefig("plots/posterior_predictive.png", dpi=300, bbox_inches="tight")
plt.close()

# Calculate and print summary statistics
summary = az.summary(idata, var_names=['Intercept', 'x'])
print("\nParameter Summary:")
print(summary)

# The estimated regression lines are very similar to the true regression line. But since we only have limited data we have uncertainty in our estimates, here expressed by the variability of the lines.