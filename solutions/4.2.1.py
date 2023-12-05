X0 = np.diff(planck.bounds).prod()
logL_max = max(dead_logLs)
Z = 0
for logL in dead_logLs:
    X1 = X0 * nlive/(nlive+1)
    Z += np.exp(logL-logL_max) * (X0-X1)
    X0 = X1

print(f'logZ = {logL_max+np.log(Z)}')
