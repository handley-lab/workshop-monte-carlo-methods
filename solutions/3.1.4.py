#| Not well! You could run multiple chains, but in practice most of a metropolis-hastings algorithm is spent 'burning in' to the final distribution, which is a constant cost independent of how many chains you are running. Once all the chains have converged to the distribution, nchains generate points n times faster (embaressingly parallel), but this is not where most of the cost of the algorithm is.
