nlive = 50
live_points = planck.prior.rvs((nlive, 6))
live_logLs = planck.loglikelihood(live_points)

for i in range(nlive*11):
    print(i)
    while True:
        x = planck.prior.rvs()
        logL = planck.loglikelihood(x)
        if logL > live_logLs.min():
            break
    live_points[live_logLs.argmin()] = x
    live_logLs[live_logLs.argmin()] = logL
