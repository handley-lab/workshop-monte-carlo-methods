from tqdm import trange
nlive = 50
live_points = planck.prior.rvs((nlive, 6))
live_logLs = planck.loglikelihood(live_points)
expansion = 1.1

for i in trange(nlive*11):
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
    live_points[live_logLs.argmin()] = x
    live_logLs[live_logLs.argmin()] = logL

