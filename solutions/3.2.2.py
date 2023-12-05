from tqdm import trange

def generate_point(expansion = 1.1):

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
    return x, logL

nlive = 50
live_points = planck.prior.rvs((nlive, 6))
live_logLs = planck.loglikelihood(live_points)

for i in trange(nlive*11):
    x, logL = generate_point()
    live_points[live_logLs.argmin()] = x
    live_logLs[live_logLs.argmin()] = logL
