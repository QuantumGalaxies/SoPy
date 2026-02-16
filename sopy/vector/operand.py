####################  ##########################
################  SoPy        ##################
####################  ########################## 

################################################
###          by Quantum Galaxies Corp        ###
##           (2023,2024,2025)                 ##
################################################

from .  import Vector
import tensorflow as tf

class Operand():
    def __init__(self, re, im):
        assert isinstance( re , Vector)
        assert isinstance( im , Vector)
        self.re = re.copy(True)
        self.im = im.copy(True)
    
    def copy(self, norm_ = True, threshold = 0.):
        return Operand(self.re.copy(norm_ = norm_, threshold = threshold), self.im.copy(norm_=norm_, threshold = threshold))
        
    def load(self, real_other, imag_other, partition, threshold):
        assert isinstance( real_other , Vector)
        assert isinstance( imag_other , Vector)
        for rank in real_other.set(partition):
            n   = rank.n()
            if n > threshold:
                self.re += rank.copy(norm_=True, threshold = threshold)
    
        for rank in imag_other.set(partition):
            n   = rank.n()
            if n > threshold:
                self.im += rank.copy(norm_=True, threshold = threshold)
        return self

    def transform(self, lattices, tss):
        """
        tss = [ {1:op1_1, 2:op2_1},{1:op1_2, 2:op2_2},{1:op1_3, 2:op2_3},...]
        
        """
        new_re = Vector()
        new_im = Vector()
        if len( self.re ) > 0 :
          for rank in self.re.set(len(self.re)):
            amp   = rank.n()
            for ts in tss:
                contents_r = {0:rank[0].numpy()}
                contents_i = {0:rank[0].numpy()}
                for d,space in enumerate(self.re.dims(True)):
                    if space in ts:
                       contents_r[space] = [tf.linalg.matvec(tf.cast(tf.math.real(ts[space]), tf.float64), rank[space][0], adjoint_a=True)]
                       contents_i[space] = [tf.linalg.matvec(tf.cast(tf.math.imag(ts[space]), tf.float64), rank[space][0], adjoint_a=True)]
                    else:
                       contents_r[space] = rank[space][0]
                       contents_i[space] = rank[space][0]

                new_re += Vector().transpose( contents_r, { space: lattices[d] for d,space in enumerate(self.re.dims(True))})
                new_im += Vector().transpose( contents_i, { space: lattices[d] for d,space  in enumerate(self.re.dims(True))})
        if len( self.im ) > 0 :
          for rank in self.im.set(len(self.im)):
            for ts in tss:
                contents_r = {0:rank[0].numpy()}
                contents_i = {0:rank[0].numpy()}
                for d,space in enumerate(self.im.dims(True)):
                    if space in ts:
                        contents_r[space] = [-tf.linalg.matvec(tf.cast(tf.math.imag(ts[space]), tf.float64), rank[space][0], adjoint_a=True)]
                        contents_i[space] = [ tf.linalg.matvec(tf.cast(tf.math.real(ts[space]), tf.float64), rank[space][0], adjoint_a=True)]
                    else:
                       contents_r[space] = rank[space][0]
                       contents_i[space] = rank[space][0]

                new_re += Vector().transpose( contents_r, { space: lattices[d] for d,space  in enumerate(self.im.dims(True))})
                new_im += Vector().transpose( contents_i, { space: lattices[d] for d,space  in enumerate(self.im.dims(True))})
                
        return Operand( new_re, new_im )
        
        
    def exp_i(self, ks):
        re__ = Vector()
        im__ = Vector()

        signage = 1.0
        #PG.self
        if True:
            for rank in self.re.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [signage*channel]
                        new_rank_im += [signage*channel]
                    else:
                        re,im = channel.g(ks[i-1],P=True)
                        new_rank_re += [re]
                        new_rank_im += [im]
    
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)  
            
            for rank in self.im.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [-1*signage*channel]
                        new_rank_im += [signage*channel]
                    else:
                        im,re_m = channel.g(ks[i-1],P=True)
                        new_rank_re += [re_m]
                        new_rank_im += [im]
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)  


        ## H.self
        if True:
            for rank in self.re.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [channel]
                        new_rank_im += [channel]
                    else:
                        re,im = channel.h(ks[i-1],poly=False)
                        new_rank_re += [re]
                        new_rank_im += [im]
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)     
                 
            for rank in self.im.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [-1*channel]
                        new_rank_im += [channel]
                    else:
                        im, re_m = channel.h(ks[i-1],poly=False)
                        new_rank_re += [re_m]
                        new_rank_im += [im]
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)     


        ## kH.self
        if True:
            for rank in self.re.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [channel]
                        new_rank_im += [channel]
                    else:
                        re,im = channel.h(ks[i-1],poly=True)
                        new_rank_re += [re]
                        new_rank_im += [im]
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)  
    
            for rank in self.im.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [-1*channel]
                        new_rank_im += [channel]
                    else:
                        im, re_m = channel.h(ks[i-1],poly=True)
                        new_rank_re += [re_m]
                        new_rank_im += [im]
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)     

            
        ##GP.self
        if True:
            signage *= -1
            for rank in self.re.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [signage*channel]
                        new_rank_im += [signage*channel]
                    else:
                        re,im = channel.P().g(ks[i-1])
                        new_rank_re += [re]
                        new_rank_im += [im]
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)     
    
            for rank in self.im.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [-1*signage*channel]
                        new_rank_im += [   signage*channel]
                    else:
                        im, re_m = channel.P().g(ks[i-1])
                        new_rank_re += [re_m]
                        new_rank_im += [im]
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)     

        return Operand(re__,im__)

    def exp2(self, alphas, positions):
        """PI_dim < x | compute exp( - 0.5 alphas[dim] (x-positions[dim])^2 ) | x'>"""
        re__ = Vector()
        im__ = Vector()

        signage = 1.0
        #PG2.self
        if True:
            for rank in self.re.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [signage*channel]
                        new_rank_im += [signage*channel]
                    else:
                        re,im = channel.g2(alphas[i-1], positions[i-1], P=True)
                        new_rank_re += [re]
                        new_rank_im += [im]
    
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)  
            
            for rank in self.im.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [-1*signage*channel]
                        new_rank_im += [signage*channel]
                    else:
                        im,re_m = channel.g2(alphas[i-1], positions[i-1], P=True)
                        new_rank_re += [re_m]
                        new_rank_im += [im]
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)  


        ## H2.self
        if True:
            for rank in self.re.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [channel]
                        new_rank_im += [channel]
                    else:
                        re,im = channel.h2(alphas[i-1], positions[i-1])
                        new_rank_re += [re]
                        new_rank_im += [im]
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)     
                 
            for rank in self.im.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [-1*channel]
                        new_rank_im += [channel]
                    else:
                        im, re_m = channel.h2(alphas[i-1], positions[i-1])
                        new_rank_re += [re_m]
                        new_rank_im += [im]
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)     


        ##G2P.self
        if True:
            signage *= -1
            for rank in self.re.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [signage*channel]
                        new_rank_im += [signage*channel]
                    else:
                        re,im = channel.P().g2(alphas[i-1], positions[i-1])
                        new_rank_re += [re]
                        new_rank_im += [im]
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)     
    
            for rank in self.im.contents:
                new_rank_re = []
                new_rank_im = []
                for i,channel in enumerate(rank):
                    if i == 0:
                        new_rank_re += [-1*signage*channel]
                        new_rank_im += [   signage*channel]
                    else:
                        im, re_m = channel.P().g2(alphas[i-1], positions[i-1])
                        new_rank_re += [re_m]
                        new_rank_im += [im]
                re__ += Vector().load([new_rank_re]).copy(True)
                im__ += Vector().load([new_rank_im]).copy(True)     

        return Operand(re__,im__)



    def cot(self, other):
        """complex outer product"""
        return self.cdot(other)

    def cdot(self, other):
        """complex dot product"""
        return complex( self.re.dot(other.re) + self.im.dot(other.im) , self.im.dot(other.re) - self.re.dot(other.im) ) 

