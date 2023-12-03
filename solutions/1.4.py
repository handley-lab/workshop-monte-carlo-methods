x = scipy.stats.norm().rvs(10000)

def f(x):
    return 2**x

plt.hist(f(x), bins=50, density=True)
x_ = np.linspace(0, f(x).max(), 1000)
s = np.log(2)
plt.plot(x_, scipy.stats.lognorm(s).pdf(x_))
