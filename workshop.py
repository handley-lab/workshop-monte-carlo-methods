#| # Monte Carlo Methods
#| This is a 2 hour workshop exploring the fundamentals of monte carlo methods as used in particle physics and astronomy.
#| The workshop is designed to be run in a Jupyter notebook
#| 
#| By the end of this workshop you will have explored the core concepts of:
#| 1. Probability distributions
#| 2. Samples
#| 3. Sampling
#| 4. Monte Carlo integration
#| 
#| and the fundamental concepts of the core tools we use to do this.

#| ## 0. Setup
#| I will presume that none of these commands are unfamiliar

import numpy as np
import matplotlib.pyplot as plt
import scipy

#-
#| ## 1. Probability distributions
#| 
#|
#| __Recap:__ a probability distribution $P$ on a variable $x$ is defined such that 
#|  $$P(a<x<b) = \int_a^b P(x) dx,$$
#| or equivalently that $P(x)dx$ represents the probability that $x$ lies in the interval $[x,x+dx]$.
#|
#| The [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) module is full of a whole variety of distributions, which forms the basis of a lot of other software.
#| Let's create an example with a [von Mises distribution](https://en.wikipedia.org/wiki/Von_Mises_distribution) (an example you may not have seen before), rather than starting with a boring normal distribution
#| $$P(x) = \frac{e^{\kappa \cos(x-\mu)}}{2\pi I_0(\kappa)}.$$

kappa = 1
mu = np.pi
dist = scipy.stats.vonmises(kappa, mu)
x = np.linspace(0, 2*np.pi,1000)
p = dist.pdf(x)
plt.plot(x, p);

#| __Exercise 1:__ produce a plot showing different choices of `kappa` and `mu`?
#| - __Question 1.1:__ What happens to the plot if you set `kappa` very large?
#| - __Question 1.2:__ Try out a few other distribution

# Answer 1
# Write your answer here into this code block

#| uncomment and execute the below to see the solution (only after you've had a go yourself).
# %load solutions/1.py

#| ### Two dimensional distributions
#| These concepts are extended relatively straightforwardly to two-dimensional distributions $P(x,y)
#|
#| For a two-dimensional [von mises Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) we can plot it as a contour plot:

# Define the parameters of von-mises in terms of concentration kappa and mean direction (phi0, theta0)

def vmf_dist(kappa, phi, theta):
    mu = np.array([np.cos(phi)*np.sin(theta),
                   np.sin(phi)*np.sin(theta),
                   np.cos(theta)])
    return scipy.stats.vonmises_fisher(mu, kappa)

kappa = 1
phi0 = np.pi
theta0 = np.pi/2
dist = vmf_dist(kappa, phi0, theta0)

# Compute a meshgrid 
phi, theta = np.linspace(0, 2*np.pi, 100), np.arccos(np.linspace(1, -1, 100))
phi, theta = np.meshgrid(phi, theta)
x, y, z  = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)

# Compute the pdf on the meshgrid with numpy broadcasting
x = np.stack([x, y, z], axis=2)
pdf = dist.pdf(x)
plt.contourf(phi, theta, pdf)

#| __Exercise 2:__ Copy and paste the above code and adjust it to get an understanding of the effect of  `kappa` and `mu`?
#| - __Question 2.1:__ What do the colours represent?
#| - __Question 2.2:__ What is the relevance of the `arccos` in the above?

# Answer 2
# Write your answer here into this code block

#| The colours above correspond to a matplotlib default colour scheme mapped to the values of the probability density.
#|
#| It is more usual to plot 'one sigma' and 'two sigma' contours.

pdf = vmf_dist(10, np.pi, np.pi/4).pdf(x)
pdf = pdf.reshape(-1)
i = np.argsort(pdf)
cdf = pdf[i].cumsum()
cdf /= cdf[-1]
sigma = np.empty_like(pdf)
sigma[i] = np.sqrt(2) * scipy.special.erfinv(1-cdf)
sigma = sigma.reshape(phi.shape)
plt.contourf(phi, theta, sigma, levels=[0, 1, 2], colors=['black', 'gray'])

#| - __Question 3.1:__ What exactly do "one sigma" and "two sigma" contours mean, and how does the above code compute them?
#| - __Question 3.2:__ How should one choose the colours for the contours? (have a look at the [anesthetic](https://github.com/handley-lab/anesthetic#another-posterior-plotting-tool) README for one option for doing this.
#|
#| __Beware:__ [corner.py](https://corner.readthedocs.io/en/latest/pages/sigmas/) uses a different definition of sigma.

#| ### Higher dimensional distributions
#| In most cases we have far more than two variables, which makes plotting the full distribution difficult.
#| 
#| The best we can do visually is to plot the _marginal distributions_, i.e. the distributions with each other variable integrated out. Since 1D and 2D distributions are plottable, we typically do this by considering the 1D marginals individually:
#| $$ P(x_i) = \int P(x) \prod\limits_{k\ne i} dx_k$$
#| and pairwise marginals
#| $$ P(x_i, x_j) = \int P(x) \prod\limits_{k\ne i, k\ne j} dx_k$$
#| and arrange these into a "corner" or "triangle" plot.

#| This example takes a few seconds to generate and plot -- don't worry about the details, we will cover them later:

from anesthetic.examples.perfect_ns import planck_gaussian
params = ['omegabh2', 'omegach2', 'theta', 'tau', 'logA', 'ns']
s = planck_gaussian()[params].plot_2d(kind='kde')


#|The next section builds up to how we go about producing these plots in practice.


#-
#| ## 1. Why do sampling
#| The core concept in numerical inference is that of *samples*.
#|
#| Given some a-priori unknown probability distribution $P(x)$

#| Some example code
import numpy as np

samples = np.random.rand(5)
print(samples)

#| uncomment the below to see the solution
# %load solutions/1.py

