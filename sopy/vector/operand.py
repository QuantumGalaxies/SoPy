####################  ##########################
################  SoPy        ##################
####################  ########################## 

################################################
###          by Quantum Galaxies Corp        ###
##           (2023,2024,2025)                 ##
################################################

from .  import vector as v


class operand():
    def __init__(self, re, im):
        assert isinstance( re , v.vector)
        assert isinstance( im , v.vector)
        self.re = re.copy(True)
        self.im = im.copy(True)
    
    def copy(self, norm_ = True, threshold = 0.):
        return operand(self.re.copy(norm_ = norm_, threshold = threshold), self.im.copy(norm_=norm_, threshold = threshold))
        
    def load(self, real_other, imag_other, partition, threshold):
        assert isinstance( real_other , v.vector)
        assert isinstance( imag_other , v.vector)
        for rank in real_other.set(partition):
            n   = rank.n()
            if n > threshold:
                self.re += rank.copy(norm_=True, threshold = threshold)
    
        for rank in imag_other.set(partition):
            n   = rank.n()
            if n > threshold:
                self.im += rank.copy(norm_=True, threshold = threshold)
        return self

    def exp_i(self, ks):
        re__ = v.vector()
        im__ = v.vector()

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
    
                re__ += v.vector().load([new_rank_re]).copy(True)
                im__ += v.vector().load([new_rank_im]).copy(True)  
            
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
                re__ += v.vector().load([new_rank_re]).copy(True)
                im__ += v.vector().load([new_rank_im]).copy(True)  


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
                re__ += v.vector().load([new_rank_re]).copy(True)
                im__ += v.vector().load([new_rank_im]).copy(True)     
                 
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
                re__ += v.vector().load([new_rank_re]).copy(True)
                im__ += v.vector().load([new_rank_im]).copy(True)     


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
                re__ += v.vector().load([new_rank_re]).copy(True)
                im__ += v.vector().load([new_rank_im]).copy(True)  
    
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
                re__ += v.vector().load([new_rank_re]).copy(True)
                im__ += v.vector().load([new_rank_im]).copy(True)     

            
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
                re__ += v.vector().load([new_rank_re]).copy(True)
                im__ += v.vector().load([new_rank_im]).copy(True)     
    
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
                re__ += v.vector().load([new_rank_re]).copy(True)
                im__ += v.vector().load([new_rank_im]).copy(True)     

        return operand(re__,im__)


    def cot(self, other):
        return complex( self.re.dot(other.re) + self.im.dot(other.im) , self.im.dot(other.re) - self.re.dot(other.im) ) 

