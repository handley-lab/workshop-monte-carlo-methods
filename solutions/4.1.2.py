mean = scipy.stats.multivariate_normal(planck.mean, planck.cov).rvs()
Q = scipy.stats.multivariate_normal(mean, planck.cov)

nsamples = 10000
x = Q.rvs(nsamples)
logP = planck.loglikelihood(x)
logpi = planck.prior.logpdf(x).sum(axis=-1)
logQ = Q.logpdf(x)
w = logP + logpi - logQ

logZ = scipy.special.logsumexp(w) - np.log(nsamples)
print(f'logZ = {logZ}')

neff = np.exp(2*scipy.special.logsumexp(w) - scipy.special.logsumexp(2*w))
print(f'effective number of samples = {neff}')
print(f'efficiency = {neff/nsamples}')

