####################  ##########################
################  SoPy        ##################
####################  ########################## 

################################################
###          by Quantum Galaxies Corp        ###
##           (2023,2024,2025)                 ##
################################################

from .. import amplitude
from .. import component
import tensorflow as tf
from sklearn.cluster import KMeans
class vector :

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
        return tf.math.sqrt(self.dot(self))

    def boost(self):
        transforms =[[]]+ [component ( contents = self[d] ,lattice = self.contents[0][d].lattice ).set_boost().transform for d in self.dims(True)]
        new = vector()
        for r in range(len(self)):
            new.contents += [ [self.contents[r][d].copy().set_boost(transform = transforms[d]).boost() for d in self.dims(False)] ]
        return new
        
    def unboost(self):
        new = vector()
        for r in range(len(self)):
            new.contents += [ [self.contents[r][d].copy().unboost() for d in self.dims(False)] ]
        return new    
    
    def dot(self, other, norm_ = False, exclude_dim : int = -1 , sum_ = True):
        """
        inner product Euclidiean between v and u in [R]**N, same as (canonRanks)* N*[R]

        norm_=True will not include dim=0
        """
        assert isinstance(other, vector)
        if (len(self)>0) and (len(other)>0):
            def innert(vector1, vector2):
                uv = []
                for dim in vector1.dims(norm_):
                    if dim != exclude_dim:
                      uv += [tf.matmul(vector1[dim],vector2[dim], transpose_b = True)]
                return tf.math.reduce_prod(tf.convert_to_tensor(uv),axis=0)
            uv = innert(self,other)
            if sum_:
                return tf.math.reduce_sum(uv)
            else:
                return tf.convert_to_tensor(uv)
        else:
            return tf.constant(0., dtype = tf.float64)

    def ref(self ) :
        other = vector()
        other.contents = self.contents
        return other

    
    def copy(self, norm_ = True, threshold = 0):
        other = vector()
        for rank in self.set(partition = len(self)):
            contents = []
            if norm_:
                n = rank.n()
                if n > threshold:
                        contents = [amplitude(a = n)]
                else:
                    continue
            else:
                contents = [amplitude(contents = rank[0])]

            
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
        assert isinstance(other, vector)
        if (len(other) == 0) or ( len(self) == 0 ) :
            return vector()
        u = self##train
        v = other##origin
        eye = tf.linalg.eye(len(u),dtype = tf.float64)            
        q = vector()

        if self.dims(True) == [1]:
            ##raw sum
            q.contents = [ [ amplitude(a=1), component(contents = [tf.math.reduce_sum(v[0]*v[1],axis=0)], lattice = self.contents[0][1].lattice) ] ]
            
            return q.copy(True)
        comps = [[]]+[ component(contents =tf.linalg.matmul(  
            tf.linalg.inv( u.dot(u,norm_ = True, exclude_dim = target_dim, sum_ = False) + alpha*eye),
            tf.linalg.matmul(u.dot(v,norm_ = True, exclude_dim = target_dim, sum_ = False),
            tf.multiply(v[0],v[target_dim]), transpose_b = False) )
         , lattice = u.contents[0][target_dim].lattice, transform = self.contents[0][target_dim].transform
        ) for target_dim in u.dims(True) ]
        amps = amplitude(contents = 1./len(u.dims(True))*tf.math.reduce_sum([comps[d].amplitude() for d in u.dims(True) ],axis=0))
        q.contents = [[ amps[r] ] + [ comps[d][r].normalize() for d in u.dims(True) ] for r in range(len(u)) ]
        if iterate == 0:
            return q
        else:
            return q.learn(other, iterate - 1, alpha = alpha )  

    def decompose(self, partition , iterate = 0 , alpha = 1e-9):
        new = self.max(partition)
        return new.learn( self, iterate = iterate, alpha = alpha)

    def fibonacci(self, partition, iterate = 0 , total_iterate = 0, alpha = 1e-9, total_alpha = 1e-9):
        Y = vector()
        for like_ranks in self.set(partition=partition):
            Y += like_ranks.decompose(partition = 1, alpha = alpha , iterate = iterate)
        return Y.learn( self, iterate = total_iterate, alpha = total_alpha )
        
    
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
        new = vector()
        for i in range(len(M)):
            if kmeans.labels_[i] not in kmeans.labels_[len(self):] :
                new.contents += [ self.contents[i] ] 
        return new

    def max(self, num = 1):
        copy = self.copy(True)
        new = vector()
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
        new = vector()
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

    
    def gaussian(self, a , positions  , sigmas ,ls , lattices , wavenumbers , phis):
        lens = [ len(x) for x in [ls,positions,sigmas,lattices,wavenumbers, phis]]
        assert min(lens) == max(lens)
        v =  [ amplitude(a) ]
        for d,(l,position, sigma,lattice,wavenumber,phi) in enumerate(zip( ls, positions, sigmas ,lattices,wavenumbers, phis)):
             v +=[ component(lattice = lattice).gaussian(position = position,sigma = sigma, l = l, wavenumber = wavenumber, phi = phi)]
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
        new = vector()
        for index in range(len(self)):
            if self.labels[index] == self.index:
                new.contents += [self.contents[index]]
        self.index += 1
        if len(new) == 0:
            raise StopIteration
        return new
    
    def delta(self, a , positions  , spacings, lattices , wavenumbers,phis ):
        lens = [ len(x) for x in [positions,spacings, lattices, wavenumbers,phis]]
        assert min(lens) == max(lens)
        v =  [ amplitude(a) ] 
        for d,(position, spacing,wavenumber,phi, lattice) in enumerate(zip( positions, spacings, wavenumbers,phis, lattices)):
             v +=[ component(lattice = lattice).delta(position = position,spacing = spacing, wavenumber = wavenumber, phi = phi)]
        self.contents += [v]
        return self

    def sample(self, num_samples ):
        sample_ranks = amplitude( contents = self[0] ).sample( num_samples ) 
        return tf.convert_to_tensor([ [ self.contents[r][d].sample(sample_rank=0,num_samples=1) for d in self.dims() ] for r in sample_ranks ])


    def transpose(self,tl):
        """
        meant to input a dictionary with integer keys, which includes 0 as a amplitude
        """
        
        comps = [ amplitude( contents = tl[0]) ]+ [ component( contents = tl[key], lattice = range(len(tl[key]))) for key in tl if key != 0 ]   
        other = vector()
        other.contents = [ [ comps[d][r] for d in range(len(comps)) ] for r in range(len(comps[0])) ]
        return other


    ##### ADDED

    def load( self, contents ):
        self.contents = contents
        return self

