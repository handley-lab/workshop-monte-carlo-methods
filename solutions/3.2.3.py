from tqdm import trange
nlive = 50
live_points = planck.prior.rvs((nlive, 6))
live_logLs = planck.loglikelihood(live_points)
live_logL_births = -np.inf * np.ones(nlive)
dead_points = []
dead_logLs = []
dead_logL_births = []
expansion = 1.1

for i in trange(nlive*40):
    print(i)
    while True:
        mn = live_points.min(axis=0)
        mx = live_points.max(axis=0)
        box = (mx-mn)
        mn -= box*(expansion-1)/2
        mx += box*(expansion-1)/2 
        dist = scipy.stats.uniform(mn, mx-mn)

        x = dist.rvs()
        logL = planck.loglikelihood(x)
        if logL > live_logLs.min():
            break
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


