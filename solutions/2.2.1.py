#| Because in the first case we were using grid points (uniform in prior), whilst now we are using samples (uniform in posterior). This difference in measure means that we no longer need to introduce pdf terms into the cdf, since samples drawn from the distribution will populate it's CDF uniformly.