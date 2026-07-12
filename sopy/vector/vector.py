####################  ##########################
################  SoPy        ##################
####################  ########################## 

################################################
###          by Quantum Galaxies Corp        ###
##           (2023,2024,2025)                 ##
################################################

from os import stat

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

    def n_dist( self, other):
        a = self.mul(1/self.n())
        b = other.mul(1/other.n())
        return tf.math.sqrt(tf.math.abs(a.dot(b) + b.dot(b) - a.dot(b) - b.dot(a)))


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
                if any(tf.reduce_any(tf.math.is_nan(tensor)) for tensor in uv):
                    print("Warning: NaN detected in inner product components!")
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
        if not self.components:
            return new
        if len(self) == 0:
            return new
        components = [self.components[0].copy()]
        for d in self.dims(True):
            components += [self.components[d].copy()]
        new.components = components
        if norm_:
           return new.balance(threshold)
        return new
        
    def balance(self, threshold = 0.):
        ranks = Vector()
        for i,rank in enumerate(self):
            amp = rank.n()
            if (tf.math.is_nan(amp) or (amp == 0.)).numpy() and (i == 0):
                ## sometimes i pass in a single zero term
                content = [Amplitude(a=0)]
                for d in self.dims(True):
                    content += [rank.components[d].flat()]
            else:
                if tf.math.abs(amp) < threshold:
                    continue
                else:
                    content = [Amplitude(a = amp)] 
                for d in self.dims(True):
                    content += [rank.components[d].normalize()]
            new = Vector()
            new.components = content
            ranks += new
        return ranks

    def iterate(self, other, alpha:float, target_dim:int, signage_test:bool = False, threshold:float = 0.):
        u = self  # train
        v = other # origin
        target_dim = (target_dim % len(self.dims(True))) + 1
        
        # 1. EXTRACT RAW TENSORS (Object-heavy prep before XLA)
        # We assume u.dot() handles the SoPy object traversal and returns a raw tf.Tensor.
        overlap_uu = u.dot(u, norm_=True, exclude_dim=target_dim, sum_=False)
        overlap_uv = u.dot(v, norm_=True, exclude_dim=target_dim, sum_=False)
        
        # Construct the 'B' matrix for the solver: B = overlap_uv * (v[0] * v[target_dim])
        v_mult = tf.multiply(v.components[0].values(), v.components[target_dim].values()) # Assuming values() gets the tensor
        b_matrix = tf.linalg.matmul(overlap_uv, v_mult, transpose_b=False)

        # 2. FIRE THE XLA KERNEL
        # The CPU stops here, GPU executes the fused Tikhonov + Solve + Normalize
        component_target, amplitude_target = self._xla_solve_and_normalize(
            overlap_uu, b_matrix, alpha
        )

        # 3. VECTORIZED BEYLKIN-MOHLENKAMP PRUNING (No Python Loops)
        # Squeeze amplitude to 1D for masking: shape [rank]
        amp_1d = tf.reshape(amplitude_target, [-1])    
    
        # Create a boolean mask of survivors (True if >= threshold, False if pruned)
        survivor_mask = tf.math.abs(amp_1d) >= threshold
        
        # Apply mask natively in VRAM (drops rows instantly)
        final_amplitudes = tf.boolean_mask(amplitude_target, survivor_mask)
        final_components = tf.boolean_mask(component_target, survivor_mask)

        # 4. REPACK SOPY OBJECTS
        # Update the non-target dimensions by applying the exact same survivor mask
        for space in self.dims(True):
            if space != target_dim:
                # We use boolean_mask instead of gather/indices_to_remove. It's faster.
                surviving_tensors = tf.boolean_mask(self.components[space].values(), survivor_mask)
                self.components[space] = Momentum(
                    lattice=self.components[space].lattice, 
                    contents=surviving_tensors
                )
            else:
                self.components[target_dim] = Momentum(
                    lattice=self.components[target_dim].lattice, 
                    contents=final_components
                )
                
        self.components[0] = Amplitude(contents=final_amplitudes)
        
        return self

    @staticmethod
    @tf.function(jit_compile=True)
    def _xla_solve_and_normalize(overlap_uu, b_matrix, alpha):
        """
        FUSED XLA KERNEL
        This executes the linear system and normalizations strictly in C++/CUDA.
        No python lists, no inv(), no explicit loops.
        """
        # 1. Tikhonov Regularization (Drain the Swamp)
        # We dynamically get the dimension size to build the identity matrix
        dim_size = tf.shape(overlap_uu)[-1]
        eye = tf.linalg.eye(dim_size, dtype=tf.float64)
        A = overlap_uu + (tf.cast(alpha, tf.float64) * eye)

        # 2. The Linear Solve (Replaces tf.linalg.inv + tf.linalg.matmul)
        # Solves A * vec = b_matrix. This is numerically stable and incredibly fast on GPUs.
        vec = tf.linalg.solve(A, b_matrix)

        # 3. Vectorized Normalization
        # tf.linalg.normalize returns the normalized tensor AND its norms. 
        # Mathematically, inner = vec_l . (vec_l / ||vec_l||) is EXACTLY the L2 norm.
        # We get the components and the amplitudes in one single, parallelized shot.
        component_target, norms = tf.linalg.normalize(vec, axis=1)
        amplitude_target = tf.cast(norms, dtype=tf.float64)

        return component_target, amplitude_target
    
    # def iterate(self, other, alpha:float, target_dim:int, signage_test:bool = False, threshold:float = 0.):
    #     u = self##train
    #     v = other##origin
    #     eye = tf.linalg.eye(len(u),dtype = tf.float64)    
    #     target_dim = (target_dim % len(self.dims(True)))+1
    #     vec = tf.linalg.matmul(  
    #            tf.linalg.inv( u.dot(u,norm_ = True, exclude_dim = target_dim, sum_ = False) + alpha*eye),
    #             tf.linalg.matmul(u.dot(v,norm_ = True, exclude_dim = target_dim, sum_ = False),
    #              tf.multiply(v[0],v[target_dim]), transpose_b = False))
    #     amplitude_target = []
    #     component_target = []
    #     indices_to_remove = []
    #     for l in range(len(self)):
    #         component_target_l = (tf.transpose(tf.linalg.normalize(tf.transpose(vec[l]),axis=0)[0]))
    #         inner = (tf.linalg.matmul([vec[l]], [component_target_l], transpose_a=False, transpose_b=True))[0][0]
    #         amplitude_target_l = tf.cast(inner, dtype = tf.float64)
            
    #         if tf.math.abs(amplitude_target_l) < threshold:
    #             indices_to_remove += [l]
    #         else:
    #             amplitude_target += [[amplitude_target_l]]
    #             component_target += [component_target_l]
    #     for space in self.dims(True):
    #         if space != target_dim:
    #             if indices_to_remove:
    #                 self.components[space] = Momentum( lattice = self.components[space].lattice, contents = tf.gather(self.components[space].values(), [i for i in range(len(self)) if i not in indices_to_remove], axis=0))
    #         else:
    #             self.components[target_dim] = Momentum(lattice = self.components[target_dim].lattice, contents = component_target)
    #     self.components[0] = Amplitude(contents = amplitude_target)
    #     return self
    
    def mul(self, m , norm_ = False):
        other = self.copy()
        other.components[0] *= m
        return other

    def learn(self, other , iterate = 1, alpha = 1e-9, threshold=0., signage_test : bool = False):
        assert isinstance(other, Vector)
        if (len(other) == 0) or ( len(self) == 0 ) :
            return Vector()
        
        train = self
        valid_dims = train.dims(True) # Get actual dimensions (e.g., [1, 2, 3])
        # Loop explicitly through your sweep passes
        for step in range(iterate):
            # Sweep backwards through actual dimension indices cleanly
            for d in reversed(valid_dims):
                train.iterate(
                    other=other, 
                    alpha=alpha, 
                    target_dim=d, 
                    signage_test=signage_test, 
                    threshold=threshold
                )
        return train


    def Decompose(self, other, ambiguity_rate, alpha=1e-5, tune_rate=0.01, iterate=10,max_allowed_distance=2.0):
        """
        Full pipeline: Learns the input, conditionally tunes to target ambiguity, and reduces rank.
        """
        # Step 1: Update
        tolerance = ambiguity_rate * 0.1  # Set tolerance as a fraction of ambiguity rate
        update_dist, was_updated = self.update(other, learn_rate=alpha, tolerance=tolerance, iterate=iterate)
        
        # Step 2: Conditional Tune ("If it calls for it")
        # We define a tiny threshold to prevent micro-tuning if it is already close enough
        epsilon = ambiguity_rate * 0.05 # 5% of the ambiguity rate as a threshold for tuning
        if abs(update_dist - ambiguity_rate) > epsilon:
            tune_dist = self.tune(other, ambiguity_rate=ambiguity_rate, tune_rate=tune_rate)
            was_tuned = True
        else:
            # Skip tuning, pass the current distance forward
            tune_dist = update_dist
            was_tuned = False
            
        # Step 3: Reduce
        final_rank, reduction_error = self.reduce_to_target_distance(
            max_allowed_distance=max_allowed_distance, 
            iterations=iterate,
            ambiguity_rate=ambiguity_rate,
            tune_rate=tune_rate,
            alpha=alpha,
        )
        
        # Step 4: Rich Reporting
        return {
            "update_distance": update_dist,
            "was_updated": was_updated,
            "was_tuned": was_tuned,
            "post_tune_distance": tune_dist,
            "final_rank": final_rank,
            "reduction_error": reduction_error,
            "reduction_stable": reduction_error <= max_allowed_distance 
        }

    def update(self, input_vector, learn_rate, tolerance=0.0, iterate=1):
        """
        Standard update procedure with early stopping to reduce over-learning.
        """
        current_dist = self.dist(input_vector)
        
        if current_dist <= tolerance:
            return current_dist, False 
            
        self = self.learn(input_vector, alpha=learn_rate/100, iterate=iterate)
        return self.dist(input_vector), True

    def tune(self, input_vector, ambiguity_rate, tune_rate=0.01):
        """
        Tunes the vector state so that output.dist(input) ~ ambiguity_rate.
        """
        current_dist = self.dist(input_vector)
        dist_error = current_dist - ambiguity_rate
        effective_alpha = tune_rate * dist_error
        
        self = self.learn(input_vector, alpha=abs(effective_alpha), iterate=1)
        return self.dist(input_vector)

    def reduce_to_target_distance(self, max_allowed_distance, iterations=10, ambiguity_rate=0.1, tune_rate=0.01, alpha=1e-5):
        """
        Binary searches for the smallest Target Partition Size that keeps
        the reduction error (distance to original) <= max_allowed_distance.
        """
        current_rank = len(self)
        if current_rank <= 1:
            return current_rank, 0.0

        original_vector = self
        
        low = 1
        high = current_rank
        best_canon = current_rank
        best_reduced_vector = self
        final_dist = 0.0
        
        while low <= high:
            mid_canon = (low + high) // 2
            
            test_vector = original_vector.Fibonacci(
                canon=mid_canon, 
                iterate=iterations, 
                total_iterate=iterations,
                alpha=alpha,
                total_alpha=alpha,
                ambiguity_rate=ambiguity_rate,
                tune_rate=tune_rate,
                max_allowed_distance=2.0
            )
            if len(test_vector) == 0:
                current_dist = original_vector.n()
            else:
                current_dist = test_vector.n_dist(original_vector)
            
            if current_dist <= max_allowed_distance:
                # Acceptable error, save state, try smaller canon
                best_canon = mid_canon
                best_reduced_vector = test_vector
                final_dist = current_dist
                high = mid_canon - 1
            else:
                # Error too high, need larger canon
                low = mid_canon + 1
                
        self = best_reduced_vector
        return best_canon, final_dist
    
    # def fibonacci(self, canon, level = 0, iterate=25, total_iterate=10, alpha=1e-9, total_alpha=1e-9, threshold=0.):
    #     #written with help from gemini to form recursion, 3x tries makes perfect
    #     #copying original Andromeda codes intent
    #     # 1. THE BASE CASE (Bottom of the tree)
    #     # If we only want 1 canon (or the data is too small to split), stop dividing.
    #     if canon <= 1 or len(self) <= 1:
    #         Y = Vector()
    #         # Decompose this specific chunk of data
    #         stats = Y.Decompose(self, alpha=alpha, iterate=iterate, ambiguity_rate=ambiguity_rate, tune_rate=tune_rate)
    #         return Y

    #     # 2. THE RECURSIVE CASE (The "Doubling" step)
    #     Y = Vector()
                
    #     # We always split into exactly 2 at this level.
    #     # This creates the 2 -> 4 -> 8 -> 16 doubling effect as it recurses.
                
    #     for all_ranks in self.set(partition=2):
    #         reduced_ranks = all_ranks.fibonacci(
    #             canon=canon // 2 + i * ( canon % 2 ) , 
    #             level = level+1,
    #             iterate=iterate, 
    #             total_iterate=total_iterate, 
    #             alpha=alpha, 
    #             total_alpha=total_alpha, 
    #             threshold=threshold
    #         )
    #         # Combine the results from the two branches
    #         Y += reduced_ranks
    #     # 3. MERGE & LEARN
    #     # As the branches merge back together, Y learns the combined T
    #     return Y.learn(self, iterate=total_iterate, alpha=total_alpha, threshold=threshold)
            
    def Fibonacci(self, canon=None, ambiguity_rate = 0.1, level = 0, iterate=10, total_iterate=3, alpha=1e-9, total_alpha=1e-9, tune_rate=0.01, max_allowed_distance=2.0, max_level=10):
        if level > max_level:
            print(f"Recursion safety limit reached at level {level}! Breaking out.")
            return self    
        if canon is None:
            canon = len(self)
        if self.n() < ambiguity_rate:
            if len(self)>0:
                if canon is not None:
                    return self.max(canon)
                else:
                    return self.max(1)
            else:
                return self
        # 1. THE BASE CASE (Bottom of the tree)
        if canon <= 1 or len(self) <= 1:
            Y = self.max(max(1,canon))
            stats = Y.Decompose(self, alpha=alpha, iterate=iterate, ambiguity_rate=ambiguity_rate, tune_rate=tune_rate)
               
            return Y
        
        # 2. THE RECURSIVE CASE (The "Doubling" step)
        Y = Vector()
                
        for i, all_ranks in enumerate(self.set(partition=2)):
            if len(all_ranks) > 0:
                reduced_ranks = all_ranks.Fibonacci(
                    canon=max(1, canon // 2 + i * (canon % 2)), 
                    level=level + 1,
                    iterate=iterate, 
                    total_iterate=total_iterate, 
                    alpha=alpha, 
                    total_alpha=total_alpha, 
                    ambiguity_rate=ambiguity_rate,
                    tune_rate=tune_rate,
                )
                Y += reduced_ranks

        # 3. MERGE & LEARN
        stats = Y.Decompose(self, ambiguity_rate=ambiguity_rate, tune_rate=tune_rate, iterate=total_iterate, alpha=total_alpha, max_allowed_distance=max_allowed_distance)
        
        print("ambiguity_rate", ambiguity_rate, self.dist(Y))
        return Y


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
        try:
            return self.components[d].values()
        except Exception as e:
            print(f"Error accessing dimension {d}: {e}")
            raise

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
        self.components[0] *= m
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
        kmeans = KMeans(n_clusters=min(len(self), len(other)+1), random_state=42, n_init="auto")
        M = (self+other).dot(other, sum_ = False)
        kmeans.fit(M)
        new = Vector()
        for ic, canon in enumerate(self):
            if kmeans.labels_[ic] not in kmeans.labels_[len(self):] :
                new += canon
        return new

    def max(self, canon = 1):
        """
        max
                
        Returns the maximum absolute value Momentum up to 'num' elements.
        """

        # Guard clause: Check if the vector is empty
        if len(self) == 0:
            raise ValueError("Cannot call max() on an empty Vector.")


        new = Vector()
        # Get the sorting indices directly from TensorFlow
        sorted_indices = tf.argsort((tf.math.abs(self[0])), axis=0, direction='ASCENDING')[-canon:]
        for i, one in enumerate(self.iter_all()):
            if i in sorted_indices:  # Check if the index is in the top 'num' indices
                new += one
        if len(new)==0:
            return one
        return new

    def min(self, canon = 1):
        """
        min
                
        Returns the minimum absolute value Momentum up to 'num' elements.
        """
        new = Vector()
        # Get the sorting indices directly from TensorFlow
        sorted_indices = tf.argsort((tf.math.abs(self[0])), axis=0, direction='ASCENDING')[-canon:]
        self.set_partition = False
        for i, one in enumerate(self):
            if i in sorted_indices:  # Check if the index is in the top 'num' indices
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
        self.set_partition = True
        self.partition = partition
        return self
    
    def iter_all(self):
        self.set_partition = False
        return self

    def __iter__(self):

        try:
            self.partition
        except:
            self.partition = len(self)
            self.set_partition = False

        # 1. Handle standard initialization variables cleanly
        self.index = 0
        
        if not self.set_partition:
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
