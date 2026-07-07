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
        
    def __len__(self):
        return (len(self.re)+len(self.im))
        
    def complex1(self,  ctl1, ext_i, mask = [] , dict_lattices = None):
        """
        one rank
        2**D terms max
        """
        len_dims = len(ctl1)
        if dict_lattices is None:
            dict_lattices = self.re.dict_lattices()
        
        assert len_dims == len(self.re.dims(False))
        re1 = Vector()
        im1 = Vector()
        dim_list = []
        for space in range(len_dims):
            if space in mask:
                dim_list += [[0]]
            elif -space in mask:
                dim_list += [[1]]
            else:
                dim_list += [[0,1]]
        for seq in itertools.product( * dim_list):
            dict_ar = {}
            for space, link in enumerate( seq ):
                dict_ar[space] = tf.cast([ tf.math.imag( ctl1[space][0] ) if (link == 1) else tf.math.real( ctl1[space][0] ) ], dtype = tf.float64)
            if ((sum(seq)+ext_i) % 4) == 0:
                re1 += Vector().transpose( dict_ar, dict_lattices ) 
            elif ((sum(seq)+ext_i) % 4) == 1:
                im1 += Vector().transpose( dict_ar, dict_lattices ) 
            elif ((sum(seq)+ext_i) % 4 ) == 2:
                dict_ar[0] *= -1.0
                re1 += Vector().transpose( dict_ar, dict_lattices ) 
            elif ((sum(seq)+ext_i)  % 4) == 3:
                dict_ar[0] *= -1.0
                im1 += Vector().transpose( dict_ar, dict_lattices ) 
        return ( re1, im1 )


    def transform(self, dict_lattices, tss, threshold:float=1e-6, partition_re = None, partition_im = None ):
        """
        tss = [ {1:op1_1, 2:op2_1},{1:op1_2, 2:op2_2},{1:op1_3, 2:op2_3},...]
        
        """
        if partition_re is None:
           partition_re = len(self.re)
        if partition_im is None:
           partition_im = len(self.im)
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

            re1, im1 = self.complex1( ctl1, ext_i=0, dict_lattices=dict_lattices, mask=mask)
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

            re1, im1 = self.complex1( ctl1, ext_i=1, dict_lattices=dict_lattices, mask=mask)
            re += re1
            im += im1
        return Operand( re.balance(threshold), im.balance(threshold))

    def exp_i(self, ks, threshold:float=1e-6, partition_re = None, partition_im = None, test_expedite = False):
        """  
        cascade operators across dimensions in direct product
         exp( - ks[dim]*x )| x'>
        """
        if partition_re is None:
           partition_re = len(self.re)
        if partition_im is None:
           partition_im = len(self.im)
        ctl1 = {}
        re = Vector()
        im = Vector()
        if len( self.re ) > 0 :
          for rank in self.re.set(partition_re):
            mask = [0]
            if test_expedite:
                list_ops = [[2,3,3]]*len( self.re.dims(True) ) 
            else:
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
                       #mask += [space]
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 3:
                       re_momentum, im_momentum = rank.contents[0][space].h(ks[d],poly=True)
                       #mask += [space]
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 4:
                       re_momentum, im_momentum = rank.contents[0][space].P().g(ks[d])
                       ctl1[space] = [-1.0*(tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128))]

                re1, im1 = self.complex1( ctl1, ext_i=0, mask=mask)
                re += re1
                im += im1

        if len( self.im ) > 0 :
          for rank in self.im.set(partition_im):
            mask = []
            if test_expedite:
                list_ops = [[2,3,3]]*len( self.re.dims(True) ) 
            else:
                list_ops = [[1,2,3,4]]*len( self.re.dims(True) ) 
            for tss in itertools.product( *list_ops ) :
                ctl1[0] = tf.convert_to_tensor( [rank[0][0]] ,dtype=tf.complex128)
                for d,space in enumerate(self.im.dims(True)):
                    ts = tss[d]
                    if ts == 1:
                       re_momentum, im_momentum = rank.contents[0][space].g(ks[d],P=True)
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 2:
                       re_momentum, im_momentum = rank.contents[0][space].h(ks[d],poly=False)
                       #mask += [space]
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 3:
                       re_momentum, im_momentum = rank.contents[0][space].h(ks[d],poly=True)
                       #mask += [space]
                       ctl1[space] = [tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128)]
                    if ts == 4:
                       re_momentum, im_momentum = rank.contents[0][space].P().g(ks[d])
                       ctl1[space] = [-1.0*(tf.cast(re_momentum.contents[0], dtype = tf.complex128) + 1.0j*tf.cast( im_momentum.contents[0] , dtype = tf.complex128))]

                re1, im1 = self.complex1( ctl1, ext_i=1, mask=mask)
                re += re1
                im += im1
        return Operand( re.balance(threshold), im.balance(threshold))


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

    def trace(self, lattices  = None):
        "integrate!"
        if lattices is None:
            lattices = self.re.lattices()
            
        f = Operand( Vector().flat(lattices) , Vector().flat(lattices))
        return self.cdot(f)

    def dot(self, other):
        """complex outer product"""
        return self.cdot(other)

    def cdot(self, other):
        """complex dot product"""
        return complex( self.re.dot(other.re) + self.im.dot(other.im) , self.im.dot(other.re) - self.re.dot(other.im) ) 

    def __imul__(self, c):
        if abs(c) == 0:
            return Operand( Vector(), Vector())
        if abs( tf.math.imag(c)) < 1e-6 :
           re = tf.math.real(c)
           im = None
        elif abs( tf.math.real(c)) < 1e-6 :
           im = tf.math.imag(c)
           re = None
        else:
           re = tf.math.real(c)
           im = tf.math.imag(c)

        if re is not None:
            self.re *= re
            self.im *= re
            return self
        if im is not None:
            self.re *= im
            self.im *= -im
            return self.swap()
        return Operand( self.re.mul(re) - self.im.mul(im), self.re.mul(im) + self.im.mul(re))


    def swap(self):
       self.re, self.im = self.im, self.re
       return self
    
    def __iadd__(self, other):
        self.re += other.re
        self.im += other.im
        return self
    
    def Fibonacci(self, canon=None, ambiguity_rate = 0.1,  level = 0, iterate=10, total_iterate=3, alpha=1e-9, total_alpha=1e-9, tune_rate=0.01, max_allowed_distance=2.0):
        return Operand( self.re.Fibonacci(canon, ambiguity_rate, level, iterate, total_iterate, alpha, total_alpha, tune_rate),
                        self.im.Fibonacci(canon, ambiguity_rate, level, iterate, total_iterate, alpha, total_alpha, tune_rate)
        )
    
    def mul(self, re):
       self *= re
       return self

    def n(self):
       return tf.math.real(tf.math.sqrt( self.dot(self) ) )
    
    def __sub__(self, spc):
       self.re -= spc.re
       self.im -= spc.im
       return self