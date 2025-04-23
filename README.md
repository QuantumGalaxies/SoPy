# Sums Of Product
## SoPy

## Representation and Decomposition with Sums of Product for Operations in separated dimensions

### Conceptual

Let multidimensional distributions be handled in the new-old fashion way... Methods as old as the census and modernized by Beylkin and Mohlenkamp 2005 for physics. Wherein is a suite of code to hold and decompose SoP vectors. We engage with the word *decomposition* not as a dimensional reduction, but as a canonical-rank reducer. See, data already is in SoP form, why write it in dense hyper dimensions?

Since 2018, we have been aware that Coulomb and other functions can be written in SoP ways, but thats the published secret sauce.
We simply are publishing our best understanding of how the SoP vector should be decomposed.  Including some tricks which have not seen the light of day before that fundamentally improve the process, see Fibonacci.

Expect a paper to be published when time can be found to do so.

### How to install

`pip install sopy-quantum`

`import sopy as sp`

### Functions

First set a lattice, 
	
 	lattices = 2*[np.linspace(-10,10,100)]

2D gaussian at (2,6) with sigmas (1,1), and polynominal 0,0

	u = sp.vector().gaussian(a = 1,positions = [2,6],sigmas = [1,1],ls = [0,0], lattices = lattices)

2D gaussian at (0.1,-0.6) with sigmas (1,1), and polynominal 0,0

	k = sp.vector().gaussian(a = 1,positions = [0.1,-0.6],sigmas = [1,1],ls = [0,0], lattices = lattices)

2D gaussian at (-1,-2) with sigmas (1,1), and polynominal 1,1

	k = k.gaussian(a = 2,positions = [-1,-2],sigmas = [1,1],ls = [1,1], lattices = lattices)

2D gaussian at (-2,-5) with sigmas (1,1), and polynominal 1,0

	v = k.copy().gaussian(a = 2,positions = [-2,-5],sigmas = [1,1],ls = [1,0], lattices = lattices)

linear dependence factor...

	alpha = 0

take v and remove k from it, and decompose into vector u ; outputing to vector q

	q = u.learn(v-k,  alpha = alpha, iterate = 1)

Get the Euclidean distance from vector v-k and q
 	
  	q.dist(v-k)

Reduce v with Fibonacci procedure

	[ v.fibonacci( partition = partition, iterate = 10, total_iterate = 10).dist(v) for partition in range(1,len(v))]

Compare with standard approaches

	[ v.decompose( partition = partition, iterate = 10, total_iterate = 10).dist(v) for partition in range(1,len(v))]

Use boost

	[ v.boost().fibonacci(  partition = partition, iterate = 10 ,alpha = 1e-2).unboost().dist(v) for partition in range(1,len(v))]

### How to Contribute
* Write to disk/database/json
* Develop amplitude/component to various non-local resources
  * Engage with Quantum Galaxies deploying matrices in separated dimensions

### Contact Info

[ SoPy Website ](https://sopy.quantumgalaxies.org)

[ Quantum Galaxies Articles ](https://www.quantumgalaxies.org/articles)

[ Quantum Galaxies Corporation ](https://www.quantumgalaxies.com)
