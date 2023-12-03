samples = planck.prior.rvs(size=(10000,d))
logL = planck.loglikelihood(samples)
print(np.max(logL))
