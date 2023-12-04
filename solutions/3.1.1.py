N = 10000
for _ in range(3):
    x0 = planck.prior.rvs()
    samples = []
    for i in range(N):
        x_ = x0 + np.random.randn(len(x0))*0.01
        logalpha = planck.loglikelihood(x_) - planck.loglikelihood(x0)
        alpha = np.exp(logalpha)
        if np.random.rand() < alpha:
            x0 = x_
        samples.append(x0[:])

    samples = np.array(samples)
    plt.plot(*samples[:,[0,1]].T)

plt.plot(*planck.mean, 'x', markersize=10, color='black');
