import numpy as np
from scipy.stats import multivariate_normal, uniform

# A planck gaussian
class PlanckGaussian(object):
    def __init__(self):
        self.cov = np.array([
            [2.12e-08, -9.03e-08, 1.76e-08, 2.96e-07, 4.97e-07, 2.38e-07],
            [-9.03e-08, 1.39e-06, -1.26e-07, -3.41e-06, -4.15e-06, -3.28e-06],
            [1.76e-08, -1.26e-07, 9.71e-08, 4.30e-07, 7.41e-07, 4.13e-07],
            [2.96e-07, -3.41e-06, 4.30e-07, 5.33e-05, 9.53e-05, 1.05e-05],
            [4.97e-07, -4.15e-06, 7.41e-07, 9.53e-05, 2.00e-04, 1.35e-05],
            [2.38e-07, -3.28e-06, 4.13e-07, 1.05e-05, 1.35e-05, 1.73e-05]])

        self.mean = np.array([0.02237, 0.1200, 1.04092, 0.0544, 3.044, 0.9649])

        self.bounds = np.array([
            [5.00e-03, 1.00e-01],
            [1.00e-03, 9.90e-01],
            [5.00e-01, 1.00e+01],
            [1.00e-02, 8.00e-01],
            [1.61e+00, 3.91e+00],
            [8.00e-01, 1.20e+00]])

        logL_mean = -1400.35
        d = len(self.mean)
        logLmax = logL_mean + d/2
        self.dist = multivariate_normal(self.mean, self.cov)
        self.offset = self.dist.logpdf(self.mean) + logLmax
        self.prior = uniform(self.bounds[:,0], self.bounds[:,1]-self.bounds[:,0])
        self.columns = ['omegabh2', 'omegach2', 'theta', 'tau', 'logA', 'ns']
        self.labels = [r'$\Omega_b h^2$', r'$\Omega_c h^2$', r'$100\theta_{MC}$',
                       r'$\tau$', r'${\rm{ln}}(10^{10} A_s)$', r'$n_s$']


    def loglikelihood(self, x):
        return self.dist.logpdf(x) + self.offset

planck = PlanckGaussian()
