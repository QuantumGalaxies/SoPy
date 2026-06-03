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
        self.components = []
    
    def dict_lattices(self):
        return { space  : self.components[space].lattice for space in self.dims(True) }

    def lattices(self):
        return [ self.components[space].lattice for space in self.dims(True) ]

    def __len__(self):
        """
        Returns the number of ranks (rows).
        Triggered when you call len(my_vector).
        """
        if not self.components:
            return 0
        # The number of ranks is simply the length of the first component
        return len(self.components[0])

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
        new.components = [self.components[d].set_boost(transform = transforms[d]).boost() for d in self.dims(False)]
        return new
        
    def unboost(self):
        new = Vector()
        new.components += [self.components[d].unboost() for d in self.dims(False)] 
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
        other.components = self.components
        return other

    def copy(self, norm_:bool =False, threshold:float = 0.):
        new = Vector()
        for rank in self:
            print(f"Copying rank with amplitude {rank[0]}")
            other = Vector()
            components = [rank.components[0].copy()]
            for d in self.dims(True):
                components += [rank.components[d].copy()]
            other.components = components
            new += other
        if norm_:
           return other.balance(threshold)
        return new
        
    def balance(self, threshold = 0.):
        ranks = Vector()
        for rank in self:
            amp = rank.n()
            if tf.math.abs(amp) < threshold:
                continue
            else:
                content = [Amplitude(a = amp)] 
            for d in self.dims(True):
                content += [rank.components[d].copy().normalize()]
            new = Vector()
            new.components = [content]
            if new.dot(rank) < 0.:
                new *= -1.0
            ranks += new
        return ranks

    def iterate(self, other, alpha:float, target_dim:int, norm_component:bool, signage_test:bool = False, threshold:float = 0.):
        u = self##train
        v = other##origin
        eye = tf.linalg.eye(len(u),dtype = tf.float64)            
        target_dim = (target_dim % len(self.dims(True)))+1
        vec = tf.linalg.matmul(  
               tf.linalg.inv( u.dot(u,norm_ = True, exclude_dim = target_dim, sum_ = False) + alpha*eye),
                tf.linalg.matmul(u.dot(v,norm_ = True, exclude_dim = target_dim, sum_ = False),
                 tf.multiply(v[0],v[target_dim]), transpose_b = False))
        amplitude_target = []
        component_target = []
        indices_to_remove = []
        for l in range(len(self)):
            component_target_l = (tf.transpose(tf.linalg.normalize(tf.transpose(vec[l]),axis=0)[0]))
            inner = (tf.linalg.matmul([u[target_dim][l]], [component_target_l], transpose_a=False, transpose_b=True))[0][0]
            if signage_test:
                amplitude_target_l = u[0][l] * tf.cast( (1.0 if inner>=0 else -1.0) if norm_component else inner , dtype=tf.float64)
            else:
                amplitude_target_l = u[0][l] * tf.cast( 1.0 if norm_component else inner , dtype=tf.float64)
            if tf.math.abs(amplitude_target_l) < threshold:
                indices_to_remove += [l]
            else:
                amplitude_target += [amplitude_target_l]
                component_target += [component_target_l]
        for space in self.dims(False):
            if space == 0 :
                self.components[0] = Amplitude(contents = amplitude_target)
            elif space != target_dim:
                if indices_to_remove:
                    self.components[space] = Momentum( lattice = self.components[space].lattice, contents = tf.gather(self.components[space].values(), [i for i in range(len(self)) if i not in indices_to_remove], axis=0))
            else:
                self.components[target_dim] = Momentum(lattice = self.components[target_dim].lattice, contents = component_target)
        return self
    
    def mul(self, m , norm_ = False):
        other = self.copy()
        other.components[0] *= m
        return other

    def learn(self, other , iterate = 0, alpha = 1e-9, threshold=0., ):
        assert isinstance(other, Vector)
        if (len(other) == 0) or ( len(self) == 0 ) :
            return Vector()
        q = Vector()

        if self.dims(True) == [1]:
            ##raw sum
            q.components = [ Amplitude(a=1.0), Momentum(contents = [tf.math.reduce_sum(other[0]*other[1],axis=0)], lattice = self.components[1].lattice) ]
            return q
        train = self#.copy(False,threshold)
        for target_dim in range( iterate * len(train.dims(True)), 0, -1):
           train.iterate(other, alpha, target_dim, target_dim != 1, signage_test=True, threshold=threshold)
        return train

    def decompose(self, partition , iterate = 0 , alpha = 1e-9, threshold:float = 0.):        
        new = self.max(min(len(self),partition))
        return new.learn( self, iterate=iterate, alpha=alpha, threshold=threshold)

    def fibonacci(self, partition, level = 0, iterate=10, total_iterate=3, alpha=1e-9, total_alpha=1e-9, threshold=0.):
        #written with help from gemini to form recursion, 3x tries makes perfect
        #copying original Andromeda codes intent
        # 1. THE BASE CASE (Bottom of the tree)
        # If we only want 1 partition (or the data is too small to split), stop dividing.
        if partition <= 1 or len(self) <= 1:
            Y = Vector()
            # Decompose this specific chunk of data
            reduced_ranks = self.decompose(partition=1, alpha=alpha, iterate=iterate, threshold=threshold)
            Y += reduced_ranks
            return Y

        # 2. THE RECURSIVE CASE (The "Doubling" step)
        Y = Vector()
                
        # We always split into exactly 2 at this level.
        # This creates the 2 -> 4 -> 8 -> 16 doubling effect as it recurses.
        
        all_ranks = {}
        for i,like_ranks in enumerate(self.set(partition=2)):
            all_ranks[i] = like_ranks
        
        for i in range(2):
            reduced_ranks = all_ranks[i].fibonacci(
                partition=partition // 2 + i * ( partition % 2 ) , 
                level = level+1,
                iterate=iterate, 
                total_iterate=total_iterate, 
                alpha=alpha, 
                total_alpha=total_alpha, 
                threshold=threshold
            )
            # Combine the results from the two branches
            Y += reduced_ranks
        # 3. MERGE & LEARN
        # As the branches merge back together, Y learns the combined T
        return Y.learn(self, iterate=total_iterate, alpha=total_alpha, threshold=threshold)
            
    def dims(self, norm = True):
        """
        an iterator for N 
    
        norm == False will loop over Weights as well...
        """
        if self.components == []:
            return list(range(norm==True,1))
        else:
            return list(range(norm==True, len(self.components)))


    def __getitem__(self, d):
        """
        Now pulls out a complete DIMENSION component on demand.
        my_vector[0] -> The entire Amplitude component object
        my_vector[1] -> The entire Momentum axis component object
        """

        if d < 0 or d >= len(self.dims(False)):
            print(f"Attempted to access dimension index {d}, but valid range is 0 to {len(self.dims(False))-1}")
            raise IndexError("Vector dimension index out of range")
            
        # Simply return the component object sitting at that dimension index
        return self.components[d].values()


    # def __getitem__(self, r):
    #     """
    #     LAZY EVALUATION: Generates the r-th rank state across all dimensions on demand.
    #     Triggered when you do my_vector[r] or iterate: `for rank in my_vector:`
    #     """
    #     if r < 0 or r >= len(self):
    #         raise IndexError("Vector rank index out of range")
            
    #     # Slices the r-th rank from every column component only when requested
    #     return [comp[r] for comp in self.components]

    def __imul__(self, m):
        for r in range(len(self)):
            self.components[0][r] *= m
        return self

    def __add__(self, other):
        assert isinstance(other, Vector)
        assert (self.components == []) or(other.components == []) or (len(self.components) == len(other.components)), "Dimension mismatch"
        
        if self.components == []:
            return other
        if other.components == []:
            return self

        new = Vector()
        # Pair up each dimension (Amp with Amp, MomX with MomX) and add them
        for d in range(len(self.components)):
            # Sopy's Component.add() concatenates or sums the tensors
            combined_comp = self.components[d].copy().add(other.components[d])
            new.components.append(combined_comp)
        return new
    
    def __iadd__(self,other):
        return self + other

    def __isub__(self,other):
        return self-other

    def __sub__(self,other):
        if len(other) == 0:
            return self
        if len(self) == 0:
            return Vector()
        kmeans = KMeans(n_clusters=(len(self)+len(other)), random_state=42, n_init="auto")
        M = (self+other).dot(self+other, sum_ = False)
        kmeans.fit(M)
        new = Vector()
        for ic, canon in enumerate(self):
            if kmeans.labels_[ic] not in kmeans.labels_[len(self):] :
                new += canon
        return new

    def max(self, num = 1):
        """
        max
                
        Returns the maximum absolute value Momentum up to 'num' elements.
        """
        new = Vector()
        # Get the sorting indices directly from TensorFlow
        sorted_indices = tf.argsort((tf.math.abs(self[0])), axis=-1, direction='ASCENDING')
        
        for i, one in enumerate(self):
            if i in sorted_indices[-num:]:  # Check if the index is in the top 'num' indices
                new += one

        return new

    def min(self, num = 1):
        """
        min
                
        Returns the maximum absolute value Momentum up to 'num' elements.
        """
        new = Vector()
        # Get the sorting indices directly from TensorFlow
        sorted_indices = tf.argsort((tf.math.abs(self[0])), axis=-1, direction='ASCENDING')
        
        for i, one in enumerate(self):
            if i in sorted_indices[:num]:  # Check if the index is in the top 'num' indices
                new += one

        return new
    
    def gaussian(self, a:float, positions, sigmas, ls, lattices):
        """
        Generates a multi-dimensional Gaussian wave packet vector.
        Each element in positions/sigmas/ls/lattices represents a single spatial axis.
        """
        # Ensure that every tracking list has exactly 1 entry per spatial dimension
        lens = [len(x) for x in [ls, positions, sigmas, lattices]]
        assert min(lens) == max(lens), "Spatial coordinate array lengths must match!"
        
        # 1. Initialize with your scaled Amplitude component
        v = [Amplitude(a=a)]
        
        # 2. Iterate dynamically over each dimension axis (X, Y, Z...)
        for d, (l, position, sigma, lattice) in enumerate(zip(ls, positions, sigmas, lattices)):
            # Instantiate a clean, dedicated Momentum column for this specific axis
            mom_axis = Momentum(lattice=lattice)
            # Compute the 1D gaussian slice for this axis
            mom_axis.gaussian(position=position, sigma=sigma, l=l)
            # Append this column component to your dimension tracking list
            v.append(mom_axis)
            
        # 3. Assemble your new state vector
        new = Vector()
        new.components = v
        
        # 4. Mathematically unify it with your current state via your component-wise __add__
        return self + new
    def set(self,partition):
        self.partition = partition
        return self
    
    def __iter__(self):
        try:
            self.partition
        except:
            self.partition = len(self)

        # 1. Handle standard initialization variables cleanly
        self.index = 0
        
        if self.partition == len(self):
            # If partitioning isn't requested, labels are just sequential indices
            self.labels = list(range(len(self)))
            return self
        else:
            # 2. Fit K-Means on your overlap dot-product matrix
            # Ensure your .dot() method returns a numpy array compatible with sklearn
            overlap_matrix = self.dot(self, sum_=False)
            kmeans = KMeans(n_clusters=self.partition, random_state=42, n_init="auto")
            kmeans.fit(overlap_matrix)
            
            self.labels = kmeans.labels_
            return self

    def __next__(self):
        # 3. Strict termination guard
        # Stop when the current cluster index reaches the requested number of partitions
        if self.index >= self.partition:
            raise StopIteration
            
        # 4. Find all row/rank indices that belong to the current cluster
        target_indices = [i for i, label in enumerate(self.labels) if label == self.index]
        
        # If a specific cluster turned out empty, advance and try again
        if not target_indices:
            self.index += 1
            if self.index >= self.partition:
                raise StopIteration
            return self.__next__()

        # 5. Build the sub-sliced Vector cleanly using pure TensorFlow gathering
        cluster_vector = Vector()
        
        for comp in self.components:
            # Create a copy of the component shell (Amplitude or Momentum)
            sub_comp = comp.copy()
            # Safely gather ONLY the rows belonging to this cluster in a single C++ step
            sub_comp.contents = tf.gather(comp.values(), target_indices, axis=0)
            cluster_vector.components.append(sub_comp)
            
        # Move the needle forward for the next iteration
        self.index += 1
        return cluster_vector   
    
    def delta(self, a:float , positions  , spacings, lattices  ):
        lens = [ len(x) for x in [positions,spacings, lattices]]
        assert min(lens) == max(lens)
        v =  [ Amplitude(a) ]
        for d,(position, spacing, lattice) in enumerate(zip( positions, spacings, lattices)):
             v +=[ Momentum(lattice = lattice).delta(position = position,spacing = spacing)]
        new = Vector()
        new.components = v
        return self + new
    
    # def resample(self, partition, lattices2 ):
    #     ve = Vector()
    #     for rank in self.set(partition):
    #         rank2 = {0:rank[0]}
    #         dict_lattices2 = {}
    #         for space, lattice2 in zip(self.dims(True), lattices2):
    #             rank2[space] = rank.contents[0][space].resample( lattice2 ).contents
    #             dict_lattices2[space] = lattice2
    #         ve += Vector().transpose( rank2 , dict_lattices2 )
    #     return ve
    
    # def flat(self, lattices ):
    #     v =  [ Amplitude(1) ]

    #     for lattice in lattices:
    #          v +=[ Momentum(lattice = lattice).flat()]
    #     self.contents += [v]
    #     return self

    def sample(self, num_samples ):
        sample_ranks = Amplitude( contents = self[0] ).sample( num_samples ) 
        return tf.convert_to_tensor([ [ self.components[d][r].sample(sample_rank=0,num_samples=1) for d in self.dims() ] for r in sample_ranks ])


    def transpose(self, tl, lattices_dict):
        """
        Inputs a dictionary with integer keys, 0 = amplitude.
        Returns a lazy-evaluated Vector object.
        """
        # 1. Safely extract Amplitude (Key 0)
        amplitude_comp = Amplitude(contents=tl[0])
        # 2. Extract Momentum components in strict sorted order
        momentum_comps = [
            Momentum(contents=tl[key], lattice=lattices_dict[key]) 
            for key in sorted(tl.keys()) if key != 0
        ]
        # 3. Assemble the core dimensions (these are your "columns")
        comps = [amplitude_comp] + momentum_comps   
        
        # 4. Pass the columns directly to the Vector
        other = Vector()
        other.components = comps
        
        return other


    ##### ADDED

    def load( self, components ):
        self.components = components
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
