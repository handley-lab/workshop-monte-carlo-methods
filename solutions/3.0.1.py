samples = planck.prior.rvs(size=(10000,6))
logL = planck.loglikelihood(samples)
print(np.max(logL))
