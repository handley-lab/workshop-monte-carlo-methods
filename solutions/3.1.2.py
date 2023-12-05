N = 10000

def Q(x0):
    dist = scipy.stats.multivariate_normal(x0, planck.cov)
    return dist.rvs()

for _ in range(3):
    x0 = planck.mean
    samples = []
    for i in range(N):
        x_ = Q(x0)
        logalpha = planck.loglikelihood(x_) - planck.loglikelihood(x0)
        alpha = np.exp(logalpha)
        if np.random.rand() < alpha:
            x0 = x_
        samples.append(x0[:])

    samples = np.array(samples)
    plt.plot(*samples[:,[0,1]].T)

plt.plot(*planck.mean[:2], 'x', markersize=10, color='black');
