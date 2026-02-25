####################  ##########################
################  SoPy        ##################
####################  ########################## 

################################################
###          by Quantum Galaxies Corp        ###
##           (2023,2024,2025)                 ##
################################################

from .. import Amplitude
from .. import Momentum
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
class Vector :

    def __init__(self):
        self.contents = []
    
    def __len__(self):
        ranks = len(self.contents) 
        return ranks

    def dist( self, other):
        return tf.math.sqrt(tf.math.abs(self.dot(self) + other.dot(other) - self.dot(other) - other.dot(self)))

    def ld1(self):
        return tf.math.reduce_sum(tf.math.abs(self[0]))

    def ld2(self):
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.abs(self[0])**2))
    
    def n(self):
        return tf.math.sqrt(tf.math.abs(self.dot(self)))

    def boost(self):
        transforms =[[]]+ [Momentum ( contents = self[d] ,lattice = self.contents[0][d].lattice ).set_boost().transform for d in self.dims(True)]
        new = Vector()
        for r in range(len(self)):
            new.contents += [ [self.contents[r][d].copy().set_boost(transform = transforms[d]).boost() for d in self.dims(False)] ]
        return new
        
    def unboost(self):
        new = Vector()
        for r in range(len(self)):
            new.contents += [ [self.contents[r][d].copy().unboost() for d in self.dims(False)] ]
        return new    
    
    def dot(self, other, another = None, norm_ = False, exclude_dim : int = -1 , sum_ = True):
        """
        inner product Euclidiean between v and u in [R]**N, same as (canonRanks)* N*[R]

        norm_=True will not include dim=0
        """
        assert isinstance(other, Vector)
        if (len(self)>0) and (len(other)>0):
            def innert(vector1, vector2):
                uv = []
                for dim in vector1.dims(norm_):
                    if dim != exclude_dim:
                      uv += [tf.matmul(vector1[dim],vector2[dim], transpose_b = True)]
                return tf.math.reduce_prod(tf.convert_to_tensor(uv),axis=0)

            def innert2(vector1, vector2, vector3):
                uv = []
                for dim in vector1.dims(norm_):
                    if dim != exclude_dim:
                      uv += [tf.matmul(vector1[dim],[ np.reshape(np.outer(v2, v3),(-1)) for v2 in vector2[dim] for v3 in vector3[dim] ],  transpose_b = True)]
                return tf.math.reduce_prod(tf.convert_to_tensor(uv),axis=0)
            if another == None:
                uv = innert(self,other)
            else:
                uv = innert2(self,other,another)
            if sum_:
                return tf.math.reduce_sum(uv)
            else:
                return tf.convert_to_tensor(uv)
        else:
            return tf.constant(0., dtype = tf.float64)

    def ref(self ) :
        other = Vector()
        other.contents = self.contents
        return other

    
    def copy(self, norm_ = True, threshold = 0):
        other = Vector()
        for rank in self.set(partition = len(self)):
            contents = []
            if norm_:
                n = rank.n()
                if n > threshold:
                        contents = [Amplitude(a = n)]
                else:
                    continue
            else:
                contents = [Amplitude(contents = rank[0])]

            
            for d in self.dims(True):
                if norm_:
                    if (rank[0][0][0] > 0) or d > 1:
                        contents += [rank.contents[0][d].normalize().copy()]
                    else:
                        contents += [rank.contents[0][d].normalize(anti = True).copy()]
                else:
                    contents += [rank.contents[0][d].copy()]
            if contents != []:
                other.contents += [contents]
        return other
        
    def mul(self, m , norm_ = False):
        other = self.copy()
        for r in range(len(self)):
            other.contents[r][0] *= m
        return other

    def learn(self, other , iterate = 0, alpha = 1e-9):
        assert isinstance(other, Vector)
        if (len(other) == 0) or ( len(self) == 0 ) :
            return Vector()
        u = self##train
        v = other##origin
        eye = tf.linalg.eye(len(u),dtype = tf.float64)            
        q = Vector()

        if self.dims(True) == [1]:
            ##raw sum
            q.contents = [ [ Amplitude(a=1), Momentum(contents = [tf.math.reduce_sum(v[0]*v[1],axis=0)], lattice = self.contents[0][1].lattice) ] ]
            
            return q.copy(True)
        comps = [[]]+[ Momentum(contents =tf.linalg.matmul(  
            tf.linalg.inv( u.dot(u,norm_ = True, exclude_dim = target_dim, sum_ = False) + alpha*eye),
            tf.linalg.matmul(u.dot(v,norm_ = True, exclude_dim = target_dim, sum_ = False),
            tf.multiply(v[0],v[target_dim]), transpose_b = False) )
         , lattice = u.contents[0][target_dim].lattice, transform = self.contents[0][target_dim].transform
        ) for target_dim in u.dims(True) ]
        amps = Amplitude(contents = 1./len(u.dims(True))*tf.math.reduce_sum([comps[d].amplitude() for d in u.dims(True) ],axis=0))
        q.contents = [[ amps[r] ] + [ comps[d][r].normalize() for d in u.dims(True) ] for r in range(len(u)) ]
        if iterate == 0:
            return q
        else:
            return q.learn(other, iterate - 1, alpha = alpha )  

    def decompose(self, partition , iterate = 0 , alpha = 1e-9):
        if len(self) < partition:
            return self
        
        new = self.max(partition)
        return new.learn( self, iterate = iterate, alpha = alpha)

    def fibonacci(self, partition, iterate=0, total_iterate=0, alpha=1e-9, total_alpha=1e-9):
        #written with help from gemini to form recursion, 3x tries makes perfect
        #copying original Andromeda codes intent
        # 1. THE BASE CASE (Bottom of the tree)
        # If we only want 1 partition (or the data is too small to split), stop dividing.
        if partition <= 1 or len(self) <= 1:
            Y = Vector()
            # Decompose this specific chunk of data
            reduced_ranks = self.decompose(partition=1, alpha=alpha, iterate=iterate)
            Y += reduced_ranks
            
            return Y

        # 2. THE RECURSIVE CASE (The "Doubling" step)
        Y = Vector()
        T = Vector()
        
        # We always split into exactly 2 at this level.
        # This creates the 2 -> 4 -> 8 -> 16 doubling effect as it recurses.
        for like_ranks in self.set(partition=2):
            
            # Ask the subset to handle half of the remaining target partitions
            reduced_ranks = like_ranks.fibonacci(
                partition=partition // 2, 
                iterate=iterate, 
                total_iterate=total_iterate, 
                alpha=alpha, 
                total_alpha=total_alpha
            )
            
            # Combine the results from the two branches
            Y += reduced_ranks
            T += like_ranks
            
        # 3. MERGE & LEARN
        # As the branches merge back together, Y learns the combined T
        Y.learn(T, iterate=total_iterate, alpha=total_alpha)
        
        return Y        
            
    def dims(self, norm = True):
        """
        an iterator for N 
    
        norm == False will loop over Weights as well...
        """
        if self.contents == []:
            return list(range(norm==True,1))
        else:
            return list(range(norm==True, len(self.contents[0])))

    def __getitem__(self, dim):
        if len(self)>0:
            return tf.concat([ (self.contents)[r][dim].values() for r in range(len(self)) ],0)
        else:
            return tf.convert_to_tensor([[]], dtype = tf.float64)
    def __imul__(self, m):
        for r in range(len(self)):
            self.contents[r][0] *= m
        return self

    def __add__(self, other):
        new = self.copy()
        new.contents += other.contents
        return new

    def __iadd__(self,other):
        self.contents += other.contents
        return self

    def __isub__(self,other):
        return self-other

    def __sub__(self,other):
        kmeans = KMeans(n_clusters=min(len(self), len(other)+1), random_state=42, n_init="auto")
        M = (self+other).dot(other, sum_ = False)
        kmeans.fit( M)
        new = Vector()
        for i in range(len(M)):
            if kmeans.labels_[i] not in kmeans.labels_[len(self):] :
                new.contents += [ self.contents[i] ] 
        return new

    def max(self, num = 1):
        """
        max
                
        Returns the maximum absolute value Momentum up to 'num' elements.
        """


        copy = self.copy(True)
        new = Vector()
        if len(copy)==0:
            return new

        def modify_tensor(x, i):
            """Modifies q2[0][i][0] using tensor_scatter_nd_update."""
        
            indices = [[i, 0]]
            updates = [0.0]
        
            x = tf.tensor_scatter_nd_update(x, indices, updates)
            return x
    
            
        args =(tf.math.abs(copy[0]))
        for n in range(min(len(copy),num)):
            i = tf.math.argmax(args)[0]
            if args[i] >0:
                new.contents += [copy.contents[i]]
                args = modify_tensor(args, i )
        return new

    def min(self, num = 1):
        new = Vector()
        def modify_tensor(x, i):
            """Modifies q2[0][i][0] using tensor_scatter_nd_update."""
        
            indices = [[i, 0]]
            updates = [tf.math.max(x)]
        
            x = tf.tensor_scatter_nd_update(x, indices, updates)
            return x
        args =(tf.math.abs(self[0]))
        for n in range(min(len(self),num)):
            i = tf.math.argmin(args)[0]
            new.contents += [self.contents[i]]
            args = modify_tensor(args, i )
        return new

    
    def gaussian(self, a , positions  , sigmas ,ls , lattices):
        lens = [ len(x) for x in [ls,positions,sigmas,lattices]]
        assert min(lens) == max(lens)
        v =  [ Amplitude(a) ]
        for d,(l,position, sigma, lattice) in enumerate(zip( ls, positions, sigmas ,lattices)):
             v +=[ Momentum(lattice = lattice).gaussian(position = position,sigma = sigma, l = l)]
        self.contents += [v]
        return self

    def set(self,partition):
        self.partition = partition
        return self
    
    def __iter__(self):
        try:
            self.partition
        except:
            self.partition = len(self)

        if self.partition == len(self):
            new = self.ref()
            new.index = 0
            new.labels = range(len(self))
            return new

        else:
            kmeans = KMeans(n_clusters=self.partition, random_state=42, n_init="auto")
            kmeans.fit(self.dot(self,  sum_ = False))
            labels = (kmeans.labels_)
            new = self.copy()
            
            new.index = 0
            new.labels = labels
            return new

    def __next__(self):
        # SAFER TERMINATION: Stop when we've checked all partitions
        # (Assuming self.partition was carried over to the 'new' object in __iter__)
        if hasattr(self, 'partition') and self.index >= self.partition:
            raise StopIteration
            
        # If partition isn't set, default to stopping when index exceeds the number of labels
        elif self.index > max(self.labels, default=-1): 
            raise StopIteration
    
        new = Vector()
        for index in range(len(self)):
            if self.labels[index] == self.index:
                new.contents += [self.contents[index]]
        self.index += 1
        return new
    
    def delta(self, a , positions  , spacings, lattices  ):
        lens = [ len(x) for x in [positions,spacings, lattices]]
        assert min(lens) == max(lens)
        v =  [ Amplitude(a) ]
        for d,(position, spacing, lattice) in enumerate(zip( positions, spacings, lattices)):
             v +=[ Momentum(lattice = lattice).delta(position = position,spacing = spacing)]
        self.contents += [v]
        return self
    
    def resample(self, partition, lattices2 ):
        ve = Vector()
        for rank in self.set(partition):
            rank2 = {0:rank[0]}
            dict_lattices2 = {}
            for space, lattice2 in zip(self.dims(True), lattices2):
                rank2[space] = rank.contents[0][space].resample( lattice2 ).contents
                dict_lattices2[space] = lattice2
            ve += Vector().transpose( rank2 , dict_lattices2 )
        return ve
    
    def flat(self, lattices ):
        v =  [ Amplitude(1) ]

        for lattice in lattices:
             v +=[ Momentum(lattice = lattice).flat()]
        self.contents += [v]
        return self

    def sample(self, num_samples ):
        sample_ranks = Amplitude( contents = self[0] ).sample( num_samples ) 
        return tf.convert_to_tensor([ [ self.contents[r][d].sample(sample_rank=0,num_samples=1) for d in self.dims() ] for r in sample_ranks ])


    def transpose(self, tl, lattices_dict):
        """
        meant to input a dictionary with integer keys, which includes 0 as a amplitude

        lattices_dict is a dictionary in same key of lattices        
        """
        
        comps = [ Amplitude( contents = tl[0]) ]+ [ Momentum(contents=tl[key], lattice=lattices_dict[key]) for key in tl if key != 0 ]   
        other = Vector()
        other.contents = [ [ comps[d][r] for d in range(len(comps)) ] for r in range(len(comps[0])) ]
        return other


    ##### ADDED

    def load( self, contents ):
        self.contents = contents
        return self

    def trace( self, tr_dims ):
        """
        trace over dimensions specified
        """
        tl = {}
        for space in self.dims(False):
            if space not in tr_dims:
                tl[space] = self[space]
        return self.transpose(tl)

    def render1( self ):
        """
        report 1d 
        """
        assert self.dims() == [1]
        d =  self.decompose(1)
        return (d[1] * d[0])[0]
