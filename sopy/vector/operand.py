####################  ##########################
################  SoPy        ##################
####################  ########################## 

################################################
###          by Quantum Galaxies Corp        ###
##           (2023,2024,2025)                 ##
################################################

from .  import Vector
import tensorflow as tf
import itertools

class Operand():
    def __init__(self, re, im):
        assert isinstance( re , Vector)
        assert isinstance( im , Vector)
        self.re = re.copy(True)
        self.im = im.copy(True)
    
    def copy(self, norm_ = True, threshold = 0.):
        return Operand(self.re.copy(norm_ = norm_, threshold = threshold), self.im.copy(norm_=norm_, threshold = threshold))
        
        
    def dict_lattices(self):
        re = self.re
        return { space  : re.contents[0][space].lattice for space in re.dims(True) }
        
        
    def complex1(self,  ctl1, ext_i, mask = [] , dict_lattices = None):
        """
        one rank
        2**D terms max
        """
        dims = len(ctl1)
        if dict_lattices is None:
            dict_lattices = self.dict_lattices()
        
        assert dims == len(self.re.dims(False))
        re1 = Vector()
        im1 = Vector()
        dim_list = ([[0,1]]*dims)
        for seq in itertools.product( * dim_list):
            dict_ar = {}
            for space, link in enumerate( seq ):
                dict_ar[space] = tf.cast([ tf.math.imag( ctl1[space][0] ) if (link == 1) and ( space not in mask ) else tf.math.real( ctl1[space][0] ) ], dtype = tf.float64)

            if ((sum(seq)+ext_i) % 4) == 0:
                re1 += Vector().transpose( dict_ar, dict_lattices ) 
            if ((sum(seq)+ext_i) % 4) == 1:
                im1 += Vector().transpose( dict_ar, dict_lattices ) 
            if ((sum(seq)+ext_i) % 4 ) == 2:
                dict_ar[0] *= -1.0
                re1 += Vector().transpose( dict_ar, dict_lattices ) 
            if ((sum(seq)+ext_i)  % 4) == 3:
                dict_ar[0] *= -1.0
                im1 += Vector().transpose( dict_ar, dict_lattices ) 
        return ( re1, im1 )


    def transform(self, dict_lattices, tss, partition_re, partition_im):
        """
        tss = [ {1:op1_1, 2:op2_1},{1:op1_2, 2:op2_2},{1:op1_3, 2:op2_3},...]
        
        """
        ctl1 = {}
        re = Vector()
        im = Vector()
        if len( self.re ) > 0 :
          for rank in self.re.set(partition_re):
            mask = [0]
            for ts in tss:
                ctl1[0] = tf.convert_to_tensor( [rank[0][0]] ,dtype=tf.complex128)
                for d,space in enumerate(self.re.dims(True)):
                    if space in ts:
                       ctl1[space] =([tf.linalg.matvec(ts[space], tf.cast(rank[space][0], tf.complex128), adjoint_a=True)])
                    else:
                       ctl1[space] = tf.convert_to_tensor([rank[space][0]], dtype= tf.complex128)
                       mask += [space]

            re1, im1 = self.complex1( ctl1, ext_i = 0, dict_lattices = dict_lattices, mask = mask)
            re += re1
            im += im1
        if len( self.im ) > 0 :
          for rank in self.im.set(partition_im):
            mask = [0]
            for ts in tss:
                ctl1[0] = tf.convert_to_tensor( [rank[0][0]] ,dtype=tf.complex128)
                for d,space in enumerate(self.im.dims(True)):
                    if space in ts:
                       ctl1[space] = ([tf.linalg.matvec(ts[space], tf.cast(rank[space][0], tf.complex128), adjoint_a=True)])
                    else:
                       ctl1[space] = tf.convert_to_tensor([rank[space][0]], dtype = tf.float128)
                       mask += [space]

            re1, im1 = self.im.complex1( ctl1, ext_i = 1,dict_lattices = dict_lattices,mask= mask)
            re += re1
            im += im1
        return Operand( re, im)

    def exp_i(self, ks):
        """  
        cascade operators across dimensions in direct product
         exp( - ks[dim]*x )| x'>
        """
        partition_re = len(self.re)
        partition_im = len(self.im)
        ctl1 = {}
        re = Vector()
        im = Vector()
        if len( self.re ) > 0 :
          for rank in self.re.set(partition_re):
            mask = [0]
            list_ops = [[1,2,3,4]]*len( self.re.dims(True) ) 
            for tss in itertools.product( *list_ops ) :
                ctl1[0] = tf.convert_to_tensor( [rank[0][0]] ,dtype=tf.complex128)
                for d,space in enumerate(self.re.dims(True)):
                    ts = tss[d]
                    if ts == 1:
                       re_momentum, im_momentum = rank.contents[0][space].g(ks[d],P=True)
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 2:
                       re_momentum, im_momentum = rank.contents[0][space].h(ks[d],poly=False)
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 3:
                       re_momentum, im_momentum = rank.contents[0][space].h(ks[d],poly=True)
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 4:
                       re_momentum, im_momentum = rank.contents[0][space].P().g(ks[d])
                       ctl1[space] = [-1.0*(tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128))]

                re1, im1 = self.complex1(ctl1, ext_i = 0, mask = mask)
                re += re1
                im += im1

        if len( self.im ) > 0 :
          for rank in self.im.set(partition_im):
            mask = [0]
            list_ops = [[1,2,3,4]]*len( self.im.dims(True) ) 
            for tss in itertools.product( *list_ops ) :
                ctl1[0] = tf.convert_to_tensor( [rank[0][0]] ,dtype=tf.complex128)
                for d,space in enumerate(self.im.dims(True)):
                    ts = tss[d]
                    if ts == 1:
                       re_momentum, im_momentum = rank.contents[0][space].g(ks[d],P=True)
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 2:
                       re_momentum, im_momentum = rank.contents[0][space].h(ks[d],poly=False)
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 3:
                       re_momentum, im_momentum = rank.contents[0][space].h(ks[d],poly=True)
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 4:
                       re_momentum, im_momentum = rank.contents[0][space].P().g(ks[d])
                       ctl1[space] = [-1.0*(tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128))]

                re1, im1 = self.complex1(ctl1, ext_i = 1, mask = mask)
                re += re1
                im += im1
        return Operand( re, im)


    def exp2(self, alphas, positions):
        """  
        cascade operators across dimensions in direct product
         exp( - 0.5 alphas[dim] (x-positions[dim])^2 ) | x'>
        """
        partition_re = len(self.re)
        partition_im = len(self.im)
        ctl1 = {}
        re = Vector()
        im = Vector()
        if len( self.re ) > 0 :
          for rank in self.re.set(partition_re):
            mask = [0]
            list_ops = [[1,2,3]]*len( self.re.dims(True) ) 
            for tss in itertools.product( *list_ops ) :
                ctl1[0] = tf.convert_to_tensor( [rank[0][0]] ,dtype=tf.complex128)
                for d,space in enumerate(self.re.dims(True)):
                    ts = tss[d]
                    if ts == 1:
                       re_momentum, im_momentum = rank.contents[0][space].g2(alphas[d], positions[d], P=True)
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 2:
                       re_momentum, im_momentum = rank.contents[0][space].h2(alphas[d], positions[d])
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 3:
                       re_momentum, im_momentum = rank.contents[0][space].P().g2(alphas[d], positions[d])
                       ctl1[space] = [-(tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128))]

                re1, im1 = self.complex1( ctl1, ext_i = 0, mask = mask)
                re += re1
                im += im1

        if len( self.im ) > 0 :
          for rank in self.im.set(partition_im):
            mask = [0]
            list_ops = [[1,2,3]]*len( self.im.dims(True) ) 
            for tss in itertools.product( *list_ops ) :
                ctl1[0] = tf.convert_to_tensor( [rank[0][0]] ,dtype=tf.complex128)
                for d,space in enumerate(self.im.dims(True)):
                    ts = tss[d]
                    if ts == 1:
                       re_momentum, im_momentum = rank.contents[0][space].g2(alphas[d], positions[d], P=True)
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 2:
                       re_momentum, im_momentum = rank.contents[0][space].h2(alphas[d], positions[d])
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 3:
                       re_momentum, im_momentum = rank.contents[0][space].P().g2(alphas[d], positions[d])
                       ctl1[space] = [-(tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128))]

                re1, im1 = self.complex1( ctl1, ext_i = 1, mask = mask)
                re += re1
                im += im1
        return Operand( re, im)

    def trace(self, lattices):
        "integrate!"
        f = Operand( Vector().flat(lattices) , Vector().flat(lattices))
        return self.cdot(f)

    def cot(self, other):
        """complex outer product"""
        return self.cdot(other)

    def cdot(self, other):
        """complex dot product"""
        return complex( self.re.dot(other.re) + self.im.dot(other.im) , self.im.dot(other.re) - self.re.dot(other.im) ) 

