Q = scipy.stats.multivariate_normal(planck.mean, planck.cov)
nsamples = 10000
x = Q.rvs(nsamples)
logP = planck.loglikelihood(x)
logpi = planck.prior.logpdf(x).sum(axis=-1)
logQ = Q.logpdf(x)
w = logP + logpi - logQ
print(scipy.special.logsumexp(w) - np.log(nsamples))
