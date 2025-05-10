####################  ##########################
################  SoPy        ##################
####################  ########################## 

################################################
###          by Quantum Galaxies Corp        ###
##           (2023,2024,2025)                 ##
################################################

import tensorflow as tf
from . import component as c

class momentum(c.component):
    def __init__(self, lattice : [float] , contents = [[]], transform = [[]] ):
        super().__init__(lattice, contents, transform)
        zero = tf.constant(0., tf.float64)
        one  = tf.constant(1., tf.float64)
        self.complexity = tf.constant(tf.complex(one,zero), dtype = tf.complex128)
        self.spacing = abs(lattice[1] - lattice[0])
        self.p = tf.convert_to_tensor( [ [ -( -1 )**(i-j)/(i-j) if i != j else 0 for j in range(len(lattice)) ] for i in range(len(lattice))] , dtype = tf.float64)
        pass


    def copy(self, norm_ = True, threshold = 0):
        return momentum( lattice = self.lattice, contents = self.contents, transform = self.transform)
        


    ##document changes
    ##(A)reduce band-pass to pi/spacing
    ##
    
    def P(self, compl = True):
        zero = tf.constant(0., tf.float64)
        one  = tf.constant(1., tf.float64)
        new = self.copy()
        new.contents = tf.matmul(self.contents ,self.p)/self.spacing
        if compl :
            new.complexity *= tf.constant(tf.complex(zero,-one))
        return new

    def g(self, k, P = False):
        zero = tf.constant(0., tf.float64)
        one  = tf.constant(1., tf.float64)
        k_re = tf.constant(k, dtype = tf.float64)

        i_m  = tf.complex(zero,-one)
        coef = ( (-one if k_re >= 0 else one)  )*(self.spacing/c.pi2)
        
        if ( abs(k_re) > c.pi2 / self.spacing /2):##A
            mul = tf.convert_to_tensor([ tf.complex(zero,zero) for position in tf.constant(self.lattice, dtype = tf.float64) ])
        else:
            mul = tf.convert_to_tensor([ 
                tf.complex(coef*tf.math.cos( k_re * position ), 
                           -coef*tf.math.sin(k_re * position ) ) for position in tf.constant(self.lattice, dtype = tf.float64) ] )

        if P:
            re = momentum( self.lattice, tf.convert_to_tensor([ tf.math.real(mul * self.complexity*i_m) * value for value in self.contents ], dtype = tf.float64),self.transform).P(False)
            im = momentum( self.lattice, tf.convert_to_tensor([ tf.math.imag(mul * self.complexity*i_m) * value for value in self.contents ], dtype = tf.float64),self.transform).P(False)
        else:
            re = momentum( self.lattice, tf.convert_to_tensor([ tf.math.real(mul * self.complexity) * value for value in self.contents ], dtype = tf.float64),self.transform)
            im = momentum( self.lattice, tf.convert_to_tensor([ tf.math.imag(mul * self.complexity) * value for value in self.contents ], dtype = tf.float64),self.transform)             
        return re,im
    
    def h(self, k, poly = False):
        zero = tf.constant(0., tf.float64)
        one  = tf.constant(1., tf.float64)
        k_re = tf.constant(k, dtype = tf.float64)
        coef = ( (one if k_re >= 0 else one) if not poly else -tf.math.abs(k_re)*self.spacing/c.pi2 )

        if ( abs(k_re) > c.pi2 / self.spacing /2 ):##A
            mul = tf.convert_to_tensor([ tf.complex(zero,zero) for position in tf.constant(self.lattice, dtype = tf.float64) ])
        else:
            mul = tf.convert_to_tensor([ 
                tf.complex(coef*tf.math.cos( k_re * position ), -coef*tf.math.sin(k_re * position ) ) for position in tf.constant(self.lattice, dtype = tf.float64) ] )

        re = momentum( self.lattice, tf.convert_to_tensor([ tf.math.real(mul * self.complexity) * value for value in self.contents ], dtype = tf.float64),self.transform)
        im = momentum( self.lattice, tf.convert_to_tensor([ tf.math.imag(mul * self.complexity) * value for value in self.contents ], dtype = tf.float64),self.transform) 
        return re,im
        