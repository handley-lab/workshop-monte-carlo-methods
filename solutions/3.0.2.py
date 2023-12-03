def f(x):
    return -planck.loglikelihood(x)

while True:
    x0 = planck.prior.rvs()
    sol = scipy.optimize.minimize(f, x0)
    if sol.success:
        break

print(sol.x)
print(planck.loglikelihood(sol.x))
