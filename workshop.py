#| # Monte Carlo Methods
#| This is a 2 hour workshop exploring the fundamentals of Monte Carlo methods as used in particle physics and astronomy.
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
#| __Recap:__ a probability distribution $P$ on a variable $x$ is defined such that 
#|  $$P(a<x<b) = \int_a^b P(x) dx,$$
#| or equivalently that $P(x)dx$ represents the probability that $x$ lies in the interval $[x,x+dx]$.
#|
#| Probability distributions are the building blocks of a lot of code, for example a cross section in particle physics:
#| $$ \sigma = \int |\mathcal{M}|^2 d\Omega$$
#| the matrix element $|\mathcal{M}|^2$ is a distribution over collision events.
#|
#| In Cosmology, these will be posterior distributions $\mathcal{P}(\theta|D)$ generated from cosmological likelihoods $\mathcal{L}(D|\theta)$.
#|
#| The [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) module is full of a whole variety of distributions, which forms the basis of a lot of other software.
#| Let's create a [von Mises distribution](https://en.wikipedia.org/wiki/Von_Mises_distribution) (an example you may not have seen before), rather than starting with a boring normal distribution
#| $$P(x) = \frac{e^{\kappa \cos(x-\mu)}}{2\pi I_0(\kappa)}.$$

kappa = 1
mu = 0
dist = scipy.stats.vonmises(kappa, mu)
x = np.linspace(-np.pi, np.pi, 1000)
p = dist.pdf(x)
plt.plot(x, p);

#| __Exercise 1.1:__ produce a plot showing different choices of `kappa` and `mu`?

# Answer
# Write your answer here into this code block

#| uncomment and execute the below to see the solution (only after you've had a go yourself).
# %load solutions/1.1.py
#| - __Question 1.1.1:__ What happens to the plot if you set `kappa` very large?
#|   - __Answer 1.1.1:__ _insert_
#| - __Question 1.1.2:__ [extension] Try out a few other distribution
#|   - __Answer 1.1.2:__ _insert_

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
plt.colorbar();

#| __Exercise 1.2:__ Copy and paste the above code and adjust it to get an understanding of the effect of  `kappa` and `mu`.

# Answer
# Write your answer here into this code block
#-
# %load solutions/1.2.py
#| - __Question 1.2.1:__ What do the colours represent?
#|   - __Answer 1.2.1:__ _insert_
#| - __Question 1.2.2:__ What is the relevance of the `arccos` in the above?
#|   - __Answer 1.2.2:__ _insert_


#| It is more usual to plot 'one sigma' and 'two sigma' contours.

pdf = dist.pdf(x)
pdf = pdf.reshape(-1)
i = np.argsort(pdf)
cdf = pdf[i].cumsum()
cdf /= cdf[-1]
sigma = np.empty_like(pdf)
sigma[i] = np.sqrt(2) * scipy.special.erfinv(1-cdf)
sigma = sigma.reshape(phi.shape)
plt.contourf(phi, theta, sigma, levels=[0, 1, 2], colors=['black', 'gray'])
plt.colorbar();

#| - __Question 1.3.1:__ What exactly do "one sigma" and "two sigma" contours mean, and how does the above code compute them?
#|   - __Answer 1.3.1:__ _insert_
#| - __Question 1.3.2:__ How should one choose the colours for the contours? (have a look at the [anesthetic](https://github.com/handley-lab/anesthetic#another-posterior-plotting-tool) README for one option for doing this.
#|   - __Answer 1.3.2:__ _insert_
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

#| This example takes a few seconds to generate and plot -- don't worry about the code details, we will cover them later:

from anesthetic.examples.perfect_ns import planck_gaussian
params = ['omegabh2', 'omegach2', 'theta', 'tau', 'logA', 'ns']
planck_samples = planck_gaussian()[params].compress()
planck_samples.plot_2d(kind='kde');


#|The next section builds up to how we go about producing these plots in practice.


#-
#| ## 2. Samples: why do sampling?
#| 
#| The core concept in numerical inference is that of _samples_. The premise is straightforward -- given some density $P(x)$, generate random numbers whose density in the large-number limit is equal to $P(x)$.
#|
#| `scipy.stats` functions have the 'random variables' `.rvs()` method built in, which does exactly this.

kappa = 1
mu = 0
dist = scipy.stats.vonmises(kappa, mu)
x = np.linspace(-np.pi, np.pi, 1000)
p = dist.pdf(x)
plt.plot(x, p);
samples = dist.rvs(10000)
plt.hist(samples, density=True, bins=50);

#| - __Question 2.1.1:__ what is the relevance of the `density=True` and `bins=50` arguments to `plt.hist`?
#|   - __Answer 2.1.1:__ _insert_

#| We can also sample from the 2D distribution we defined above:

kappa = 4
phi0 = 0
theta0 = np.pi/2
dist = vmf_dist(kappa, phi0, theta0)
N = 10000
samples = dist.rvs(N)
theta, phi = np.arccos(samples[:,2]), np.arctan2(samples[:,1], samples[:,0])
plt.plot(phi, theta, '.', markersize=1);

#| One of the lesser-know functionalities of matplotlib is that the samples above can also be used to plot contours with the triangulation functionality provided by the `tricontourf` method:

pdf = dist.pdf(samples)
i = np.argsort(pdf)
cdf = np.arange(1,N+1)/(N+1)
sigma = np.empty_like(pdf)
sigma[i] = np.sqrt(2) * scipy.special.erfinv(1-cdf)
plt.tricontourf(phi, theta, sigma, levels=[0, 1, 2], colors=['black', 'gray'], alpha=0.8)
plt.plot(phi, theta, '.', markersize=1)
plt.colorbar();


#| - __Question 2.2.1:__  Why is the sigma calculation different in comparison to the previous example? [hard]
#|   - __Answer 2.2.1:__ _insert_

#| One way to see how powerful the above approach of using samples to make plots is, is to consider how we might go about making the triangle plot using the meshgrid approach. To do this, we would have to solve two problems:
#| - __Question 2.3.1:__ How would you use a meshgrid to compute marginal distributions?
#|   - __Answer 2.3.1:__ _insert_
#| - __Question 2.3.2:__ How much more expensive would this be in 6 dimensions?
#|   - __Answer 2.3.2:__ _insert_

#| Sampling solves both of these:
#| - marginal samples are found just by ignoring columns
#| - drawing samples from higher-dimensional distributions is not exponentially harder
#| To see this, let's take a look at the planck_samples, which when printed show an anesthetic (pandas extension) array

planck_samples

#| We can plot samples from the marginal distribution by

plt.plot(*planck_samples[['logA', 'tau']].to_numpy().T, '.');

#| To plot the marginal contours, we have to use some form of low-dimensional density estimation. Kernel density estimation (KDE) is the standard (basically putting a small gaussian on each sample and adding these together) but in principle one could use neural density estimators or histograms for more/less advanced versions. Density estimation is an acceptable approximation in one and two dimensions but becomes very innacurate in higher dimensions. 

#| There are many packages that implement all of this for you, and I would encourage you to resist the temptation to write your own! (irony noted)
#| - [getdist](https://getdist.readthedocs.io/en/latest/)
#|   - state-of-the-art in KDE edge correction
#|   - industry standard since 2010
#|   - difficult to extend/use
#| - [corner.py](https://corner.readthedocs.io/en/latest/)
#|   - histogram-based foreman-mackay software
#| - [chainconsumer](https://samreay.github.io/ChainConsumer/)
#|   - increasingly popular python tool for MCMC samples
#| - [anesthetic](https://anesthetic.readthedocs.io/en/latest/)
#|   - specialised for nested sampling, but can also do MCMC
#|   - explicitly builds on the numpy/scipy/pandas stack.
#| 
#| You can see here that the python package anesthetic by default uses the whitespace above the diagonal to plot samples, in addition to estimates of the 1D and 2D marginals:

planck_samples[['logA', 'tau']].plot_2d();

#| Samples are an extremely powerful tool for performing numerical inference. In particular the following property holds
#| $$ x\sim P(x)dx \quad\Rightarrow\quad f(x) \sim P(f)df$$
#| namely, if you have a set of samples in a variable $x$, and you want to know how $y=f(x)$ is distributed, you just assume the answer is $x_i$ and compute $y_i = f(x_i)$ for each sample. Sampling turns uncertainty quantification into repeated forward models.

#| __Exercise 2.4:__ if $x$ is normally distributed, plot the distribution of $2^xx$. Bonus question: prove mathematically that in this case x is log-normally distributed, find its scale parameter, and plot this on your plot.

# Answer
#-
# %load solutions/2.4.py

#| One can compute averages by summing over samples:
#| $$ \langle f \rangle_P \quad=\quad \int f(x) P(x) dx \quad\approx\quad \frac{1}{N} \sum\limits_{i=1}^N f(x_i)$$
#| Note that the (generally unknown) density __does not__ appear in the final expression, which shows that sampling is our primary tool for avoiding high-dimensional density estimation.

#| To confuse you on this last point, most methods for generating samples will often create _weighted samples_, namely each sample $x_i$ has an associated weight $w_i$, and the average is computed as:
#| $$ \langle f \rangle_P \quad\approx\quad \tfrac{1}{\sum_i w_i}\sum_i w_i f(x_i)$$
#| This weight is __not__ generally the probability density.
#|
#| `anesthetic` is a pandas extension which computes weighted statistics.

from anesthetic import Samples

x = np.random.rand(100,3)
w = np.random.rand(100)
samples = Samples(x, weights=w)
samples
#-
samples.mean()
#-
samples.std()



#| Exercise 2.5: confirm that the the weighted mean and standard deviation are not the same as numpy's default.

# Answer
#-
# %load solutions/2.5.py

#| Getting weighted means and standard deviations correct is a faff, and a common source of bugs in numerical inference code, so it's worth being aware of this/using anesthetic's extensive functionality.



#| ### Summary
#| Samples are the fundamental building block of numerical inference. You should view them as "souped-up error bars", summarising uncertainty in your analysis. If you have an analysis that you could do if you knew all the inputs $x$, and you have samples from the distribution $x$ belongs to, you just do the analysis $N$ times for each sample, get $N$ answers, and the distribution of these answers quantifies your uncertainty.
#|
#| The golden rule of Numerical inference is therefore to __stay in samples__ until the end. You should know by now that in general
#| $$f(\langle x \rangle) \ne \langle f(x) \rangle $$
#| so taking an average/summary before the end can bias your inference.
#|
#| You can then use samples to compute averages, and extract marginal distributions by ignoring columns, from which you can compute 1D and 2D contours. This is the basis of the triangle plot.
#|
#| __Question 2.6:__ [hard] what statistical operation can you not get easily from samples?

#| We now move on to techniques as to how one can generate samples from an arbitrary distribution.


#| ## 3. Sampling
#| We will assume that we can evaluate the probability density function $P(x)$ (or at least the unnormalised density $P^*(x)\propto P(x)$) in finite time (i.e. seconds). For initiates, it might be surprising that access to the exact $P(x)$ is not sufficient to generate samples, but it is not.
#|
#|(N.B. The frontier of inference at the moment is simulation based inference, which develops methods for inference when you only have a simulator $f(x)$, but this is beyond the scope of this workshop).

#| ### 3.0 Random sampling
#| Let's first see why random sampling is not sufficient.
#|

from handleymcmethods.examples import planck

#| __Exercise 3.0.1:__ Generate 10000 samples from planck.prior, and find the largest planck.loglikelihood value in the samples. Repeat this process a few times. What do you notice?

# Answer
#-
# %load solutions/3.0.1.py

#| The issue is that the prior is much wider than the likelihood/posterior, so random samples are very unlikely to be close to the peak of the likelihood. This same problem would occur for meshgrid sampling.
#|
#| One solution for finding a good region is of course a gradient descent
#| __Exercise 3.0.2:__ Use `scipy.optimize.minimize` to find the maximum of the loglikelihood. How does this compare to the maximum of the other results

# Answer
#-
# %load solutions/3.0.2.py

#| Whilst this approach does find the maximum probability point, this does not generate samples, which is what we need for our error bars.

#| ### 3.1 Metropolis Hastings
#|
#| The first approach that can successfully generate samples from a distribution is the [Metropolis-Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm).
#| The simplest version of this algorithm is as follows:
#| - start at some point $x_0$
#| - at iteration $i$: 
#|   - propose a new point $x'$ a random step away from $x_i$
#|   - accept the point with probability
#|   $$ \alpha = \frac{P(x')}{P(x_i)}$$
#|   - if the point is accepted, $x_{i+1} = x'$, otherwise $x_{i+1} = x_i$
#|   - stop when you have enough samples

#| __Exercise 3.1.1:__ Implement the Metropolis-Hastings algorithm for the planck likelihood. 
#| - hint: if it's not working, try plotting the the set of points you're generating

# Answer
#-
# %load solutions/3.1.1.py

#| Things can be made a lot better by choosing a better method for proposing new points.
#| For example, if you use the true posterior coviariance matrix, the solution converges much better
#| - hint: you can get the covariance matrix from `planck.cov` and the mean from `planck.mean`

# Answer
#-
# %load solutions/3.1.2.py

#| Of coures in practice one doesn't know the answer going in, and therefore typically it has to be learned, either by knowing something about the physics, using another algorithm first, or gradually updating the proposal distribution.

#| - __Question 3.1.3:__ Where does the metropolis hastings algorithm fail?
#|   - __Answer 3.1.3:__ _insert_
#| - __Question 3.1.4:__ How well does this algorithm parallelise?
#|   - __Answer 3.1.4:__ _insert_

#| The full algorithm is as follows:
#| - start at some point $x_0$
#| - propose a new point $x'$ from some proposal distribution $Q(x'|x)$
#| - accept the point with probability
#| $$ \alpha = \min\left(1, \frac{P(x')Q(x|x')}{P(x)Q(x'|x)}\right)$$
#| - repeat until convergence criterion is reached
#|
#| Note this includes the generalisation to asymmetric proposal distributions, which is necessary for the algorithm to converge, and more carefully acounts for the fact that the probability shouldn't be greater than 1. Convergence criteria usually include a variation of the Gelman-Rubin statistic.
#| 
#| Example implementations of metropolis hastings include
#| - [PyMC](https://www.pymc.io/welcome.html)
#| - [Cobaya](https://cobaya.readthedocs.io/en/latest/)
#| - [CosmoSIS](https://cosmosis.readthedocs.io/en/latest/)
#| - [MontePython](https://monte-python.readthedocs.io/en/latest/)
#|
#| Though in practice, since a metropolis sampler is so easy to write, and aspects of the proposal distribution are so system-specific, many people just choose to write their own!
#|
#| More modern work is exploring the use of neural networks to learn the proposal distribution, which can be much more efficient than the above.
#| - [FlowMC](https://arxiv.org/abs/2211.06397)
#| - [MCMC-diffusion](https://arxiv.org/abs/2309.01454)
#|
#| Further extensions to this approach include ensemble sampling ([emcee](https://emcee.readthedocs.io)), slice sampling ([zeus](https://zeus-mcmc.readthedocs.io/en/latest/)) and many others.


#| ### 3.2 Nested sampling
#|
#| As discussed in the [talk](https://github.com/williamjameshandley/talks/raw/unam_2023/will_handley_unam_2023.pdf), the nested sampling algorithm can be summarised as:
#| - generate nlive samples from the prior
#| - at each iteration, replace the lowest likelihood sample with a new sample from the prior at greater likelihood
#| - stop when the live points have sufficiently compressed

#| __Exercise 3.2.1:__ Implement the nested sampling algorithm for the planck likelihood with 50 live points, using a brute-force prior-sample+rejection approach. How many iterations do you get through? (put in a print statement to see the slow down)

# Answer
#-
# %load solutions/3.2.1.py
    
#| You should find you get about to about 500 iterations before you run out of patience!

#| __Exercise 3.2.2:__ This time, implement a more efficient approach by using a box around the live points to generate samples from the prior. To be correct, you should make the box slightly larger than this! Run the algorithm for 
#| - __Question 3.2.2:__ What are the failure modes of this method?

# Answer
#-
# %load solutions/3.2.2.py
    
#| __Exercise 3.2.3:__ Adjust your algorithm so that it records the dead points, as well as the 'birth contour'. Plot the dead points. Pass these into the anesthetic gui

# Answer
#-
# %load solutions/3.2.3.py

#| __Exercise 3.2.4:__ Write a non-rejection based sampling algorithm (e.g. metropolis hastings using the covariance of the live points to build a proposal distribution) and compare the speed.

# Answer
#-
# %load solutions/3.2.4.py

#| ## Interlude
#|
#| The goal of the above 'my first metropolis algorithm' and 'my first nested sampler' exercises is to give you a feel for how at these algorithms work at the base level, and how they can be implemented, so that you can assess the efficiacy of new ideas. In practice you should use one of the more established libraries for doing this, which have been battle-tested and optimised for speed. In particular, some of the things which cause ~O(10) lines of code to expand to ~O(1000) lines are:
#| - robust live point generation
#| - parallelisation
#| - mode identification & handling
#|
#| Most packages focus on the first of these, many (but not all) provide the second, with very few implementing the substantial bookkeeping required for the third.
#|
#| As discussed in the talk, samplers 
#|
#| #### Rejection samplers (or 'region samplers')
#| - [MultiNest](https://github.com/farhanferoz/MultiNest)
#|   - ellipsoidal decomposition, parallelised, clustered
#|   - the original nested sampler
#|   - Incorporated into a wide variety of packages (GAMBIT, CosmoSIS, MontePython, Cobaya, etc)
#|   - Johannes Buchner implemented a [python wrapper](https://johannesbuchner.github.io/PyMultiNest/)
#| - [UltraNest](https://johannesbuchner.github.io/UltraNest/)
#|   - kernel-based method for generating live points, paralellised
#|   - Johannes Buchner's new python nested sampler
#|   - incorporates new advances in nested sampling not in MultiNest
#|   - python only (so for very fast likelihoods may be worth compiling multinest)
#|   - also implements many other strategies (including path sampling)
#| - [nessai](https://nessai.readthedocs.io/en/latest/)
#|   - normalising flows to generate live points
#|   - popular in gravitational waves community
#|
#| #### Chain-based samplers (or 'path samplers')
#| - [PolyChord](https://github.com/PolyChord/PolyChordLite)
#|   - slice sampling, parallelised, clustered
#|   - the original path sampler
#| - [pymatnext](https://libatoms.github.io/pymatnest/intro.html)
#|   - specific for materials science
#| 
#| #### Other samplers
#| - [dynesty](https://dynesty.readthedocs.io/en/latest/)
#|   - python re-implementation of many of the above
#|   - very popular
#|   - easy to install
#|   - defaults not well chosen for path-based samplers
#|   - although 'dy' is in the name, most nested samplers (polychord, ultranest etc) are now dynamic nested samplers as well
#|
#| __Will Handley's recommendations on choosing samplers:__
#| - if your problem is low-dimensional ~O(10) parameters, use UltraNest
#|   - easier for newcomers to install than multinest
#|   - self-tunes its efficiency parameters, so gives an honest scaling with dimensionality
#| - if your problem is high-dimensional, use PolyChord
#|   - whilst other samplers (dynesty, ultranest) implement slice sampling, neither implement clustering, which is necessary for multi-modal problems
#|   - in particular dynesty is very poorly parallelised (since it is optimised for notebook rather than HPC work).
#|   - the dynesty defaults for slice sampling are too generous (and therefore often give wrong, but faster answers)
#|   - it is slightly harder to install due to the legacy fortran component.
#|
#| ### Practical advice for using nested sampling
#| Nested sampling run-time scales as:
#| $$ T \propto  f_\mathrm{sampler} \times n_\mathrm{live}/n_\mathrm{cores}$$
#| providing $n_\mathrm{cores} \le n_\mathrm{live}$.
#| - rejection samplers: the $f_\mathrm{sampler} = e^{d/d_0}$ where $d_0$ is problem dependent$.
#| - chain-based samplers: the $f_\mathrm{sampler} = \mathcal{O}(3) \times n_\mathrm{repeat}$
#|
#| (The proportionality constant is $\mathcal{D}_\mathrm{KL}(\mathcal{P}||\pi) T_\mathrm{like}$)
#|
#| This scaling is very useful for planning big runs, all you need to do is determine the proportionality constant, which in practice you get from a preliminary run.
#|
#| The strategy is therefore:
#| 1. run a preliminary run with a small number of live points, debug code, and determine $T_0$.
#|    - for rejection samplers, you need a minimum number of samples to train the region proxy (ellipsoidal decomposition, normalising flow), usually ~O(500)
#|    - for path samplers this can be set much lower (as low as 5!)
#|    - Since nested sampling is parallelised up to the number of cores, setting nlive ~ ncores is a good starting point (e.g. on modern HPC machines ~O(50), on your laptop ~O(10)).
#| 2. scale up to a production run (nlive ~ 1000)
#|   - If you had $n_\mathrm{live}^0$ and $n_\mathrm{cores}^0$ giving you $T_0$, the full runtime will be
#|    $$T = T_0 \times \frac{n_\mathrm{live}}{n_\mathrm{live}^0} \times \frac{n_\mathrm{cores}^0}{n_\mathrm{cores}}$$
#|   - With arbitrary HPC, you can scale the cores with the live points and retain the same walltime $T_0$.
#| 3. check that you ran with high enough $f_sampler$ by halving/doubling it and checking the results are consistent.





#| General advice on usage

#| ## References:
#| - [Nature review article](https://arxiv.org/abs/2205.15570): Nested sampling for physical sscientists
#|   - pedagogical introduction by the community to nested sampling
#| - [Technical review article](https://arxiv.org/abs/2101.09675): Nested sampling methods
#|   - technical review by Johannes Buchner (author of UltraNest)
#|   - detailed and complete reference list of the entire nested sampling literature up to early 2023.
#| - [John Skillings original paper](https://projecteuclid.org/journals/bayesian-analysis/volume-1/issue-4/Nested-sampling-for-general-Bayesian-computation/10.1214/06-BA127.full)
#|   - original paper on nested sampling
#|   - a goldmine of insight in John's unique style




#| ## 4. Integration
#| 
#| If your probability distribution is not normalised, the integration constant is of critical importance, either as a cross section (particle physics), a Bayesian evidence (cosmology) or a partition function (statistical mechanics).
#|
#| Fundamentally we want to compute:
#| $$ Z = \int P^*(x) dx$$
#| The traditional discussion begins by pointing out that over the domain of $P^*(x)$, the region where $P^*(x)$ is of non-zero probability is very small, which we don't know a-priori where it is.  On the face of it, this doesn't look like a deal-breaker -- we have been developing ever more sophisticated ways of generating samples $x\sim P$, so we have plenty of points with non-zero $P^*(x)$
#|
#| However, there is another portion of the integral, namely $dx$, which posterior samples __do not__ give us. The challenge is therefore not finding the "typical set", or generating points within it, it is measuring its volume.

#| ### 4.1 Importance sampling
#| The go-to method in particle physics for doing this is importance sampling.
#| The premise here is to find a normalised distribution $Q(x)$ which easy to sample from (for example a scipy distribution), and which is similar to $P^*(x)$ in the region where $P^*(x)$ is non-zero.
#| 
#| One then uses the (almost trivial) result:
#| $$ \int P^*(x) dx = \int \frac{P^*(x)}{Q(x)} Q(x) dx = \left\langle \frac{P^*(x)}{Q(x)} \right\rangle_Q$$
#| I.e. one generates samples from $x\sim Q(x)$ and computes the average of $P^*(x)/Q(x)$.
#|
#| You can think of this intuitively as 'trimming off' the regions where $P^*(x)$ is zero, and then computing the average of the remaining regions -- a more sophisticated way of picking a narrower prior
#|
#| If you choose a poor $Q$, then this will be very inefficient, with very few samples contributing to the integral. These weighted samples are in a definite sense exact

#| __Exercise 4.1.1:__ Take $P^*(x)$ to be the `planck.loglikelihood` added to the `planck.prior.logpdf`, and Q(x) to be a scipy.stats_multivariate_normal with `planck.mean`, `planck.cov`. Compute the integral using importance sampling. You should get `logZ=-1431.403883060199`. You may find it useful to use `scipy.special.logsumexp` to compute the average.

# Answer
#-
# %load solutions/4.1.1.py


#| __Exercise 4.1.2:__ Now compute the integral with a slightly misspecified proposal $Q$ by choosing a mean offset by a random amount. Compute the effective number of samples using the formula $$ n = \frac{(\sum_i w_i)^2 }{ \sum_i w_i^2} $$ where $w_i$ are the importance weights we are averaging over

# Answer
#-
# %load solutions/4.1.2.py

#| __Exercise 4.1.2:__ Try other distributions (e.g. multivariate_t, adjusting the covariance matrix using a wishart distribution, etc). How does the efficiency change?

#| What the above exercise shows is that if you have a good proposal $Q$, then this can be a very efficient way of computing an integral. However, the efficiency drops for even slightly misspecified $Q$, and this problem exponentially worsens in high dimensions

#| ### 4.2 Nested sampling
#|
#| Nested sampling provides a more sophisticated way to achieve the above. We begin by making the standard [Lebesgue integral](https://en.wikipedia.org/wiki/Lebesgue_integration) manipulation. First define the volume (/prior volume/measure/CDF) as:
#| $$ X(P^*) = \int_{P^*(x)>P^*} dx$$
#| this is the volume of the space contained inside each contour of $P^*$.  With this definition, we can say
#| $$ Z = \int P^* dx = \int P^* dX \approx \sum_i P^*_i \Delta X_i $$
#| which has transformed the multidimensional integral into a one-dimensional one which can be tractably numerically integrated.

#| The power of nested sampling is that we can estimate $X_i$ from the nested sampling compression procedure:
#| $$X_i \approx \frac{n}{n+1} X_{i-1} $$

#| __Exercise 4.2.1:__ Compute the nested sampling evidence estimate, and compare to the value you found with importance sampling. 

# Answer
#-
# %load solutions/4.2.1.py

#| It's a little more involved to quantify the $\approx$ using samples from the distribution of $X_i$, but fortunately anesthetic has implemented all of this for you
samples = planck_gaussian()
samples.logZ(1000).hist()

#| and much more besides
samples.nlive.plot()

#| and much more besides
samples.stats()

#| and much more besides
samples.stats(1000).plot_2d()
