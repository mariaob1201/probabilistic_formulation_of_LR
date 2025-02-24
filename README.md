# Probabilistic formulation of a GLM using BAMBI

I use https://bambinos.github.io/bambi/

This is an example f a probabilistic re formulation of a Generalized Linear Model (GLM) – linear regression.

That is, from a frequentist perspective, linear regression is typically expressed as:
 `Y = X * B + e`
where:
- X represents the predictor variables,
- Y is the dependent variable,
- B denotes the coefficients, and
- e is the error term, assumed to follow a normal distribution.

The probabilistic formulation of this model can be written as:
` Y ~ N(X * B, sigma^2)`
where Y is view as a random variable (or random vector) of which each element (data point) is distributed according to a Normal distribution. 

Some advantages of a probabilistic approach include:
1. **Priors**: We can incorporate prior knowledge by assigning probability distributions to parameters.
    For instance, if we expect B to be small, we can use a prior that places more probability mass on lower values.
2. **Quantifying Uncertainty**: Instead of obtaining a single point estimate for B, we derive a full posterior distribution.
    This allows us to assess the likelihood of different values for B. When data is sparse, our uncertainty in B is high,
    leading to wider posterior distributions.

Maria, Data Scientist
