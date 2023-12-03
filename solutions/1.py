
for kappa, mu in [(1,np.pi), (0.5,0), (2,np.pi/2)]:
    dist = scipy.stats.vonmises(kappa, mu)
    x = np.linspace(0, 2*np.pi,1000)
    p = dist.pdf(x)
    plt.plot(x, p);
