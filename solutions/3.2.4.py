from tqdm import trange

def generate_point(num_repeats=30):

    cov = 0.1*np.cov(live_points.T)
    x0 = np.random.default_rng().choice(live_points)
    logL0 = planck.loglikelihood(x0)
    for _ in range(num_repeats):
        x = scipy.stats.multivariate_normal(x0, cov).rvs()
        logL = planck.loglikelihood(x)
        if logL > live_logLs.min() and planck.prior.pdf(x).prod() > 0:
            x0 = x.copy()
            logL0 = logL

    return x0, logL0

nlive = 50
live_points = planck.prior.rvs((nlive, 6))
live_logLs = planck.loglikelihood(live_points)
live_logL_births = -np.inf * np.ones(nlive)
dead_points = []
dead_logLs = []
dead_logL_births = []

for i in trange(nlive*40):
    x, logL = generate_point()
    dead_points.append(live_points[live_logLs.argmin()].copy())
    dead_logLs.append(live_logLs.min())
    dead_logL_births.append(live_logL_births[live_logLs.argmin()])
    live_points[live_logLs.argmin()] = x
    live_logL_births[live_logLs.argmin()] = live_logLs.min()
    live_logLs[live_logLs.argmin()] = logL

dead_points += live_points.tolist()
dead_logLs += live_logLs.tolist()
dead_logL_births += live_logL_births.tolist()

from anesthetic import NestedSamples
samples = NestedSamples(dead_points, logL=dead_logLs, logL_birth=dead_logL_births)
plt.plot(*samples[[0,1]].to_numpy().T, '.')
