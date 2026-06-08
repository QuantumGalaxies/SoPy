import numpy as np
import pytest
import sopy as sp

g1 = sp.Vector().gaussian(a = 1, \
      positions=[0., 0.], \
      sigmas = [ 1., 1.  ], \
      ls = [0,0], \
      lattices = 2*[ np.linspace( - 20, 20, 1000)])


np.testing.assert_allclose((g1+g1).decompose(1).dist(g1), 1, atol=1e-7)


g2 = sp.Vector().gaussian(a = 1, \
      positions=[0., 0.], \
      sigmas = [ 0.1, 0.1  ], \
      ls = [1,0], \
      lattices = 2*[ np.linspace( - 20, 20, 1000)])


np.testing.assert_allclose((g1+g2+g2.mul(-1)).decompose(1).dist(g1), 0, atol=1e-7)
np.testing.assert_allclose((g1+g2+g2.mul(-1)+g2+g2.mul(-1)+g2+g2.mul(-1)+g2+g2.mul(-1)).fibonacci(1).dist(g1), 0, atol=1e-7)

