####################  ##########################
################  SoPy        ##################
####################  ########################## 

################################################
###          by Quantum Galaxies Corp        ###
##           (2023,2024,2025)                 ##
################################################


import tensorflow as tf
from bandlimit.gaussian import compute
pi2 = 6.283185307179586476925286766559005768394338798

class Component :
    """
    d > 0 
    """

    def __init__(self, lattice, contents=None, transform=None):
        if contents is None: contents = []
        if transform is None: transform = []

        self.lattice = lattice
        self.transform = transform
        
        # immutable tensor for contents to prevent C++ heap corruption/fragmentation
        self.contents = tf.convert_to_tensor(contents, dtype=tf.float64)

        # We check to make sure the lattice actually has at least 2 elements first.
        try:
            if len(lattice) > 1:
                self.spacing = lattice[1] - lattice[0]
            else:
                self.spacing = 0.0  # Safe fallback for 1D/single-point lattices
        except TypeError:
            # Fallback if lattice is a scalar instead of a list/array
            self.spacing = 0.0

    def copy(self):
        other = Component(lattice = self.lattice, contents = self.contents, transform = self.transform)
        return other
    
    def __len__(self):
        """
        for SoP like (canonRanks)* N*[R] 
        return CanonRanks, which is the number of product sums
        """
        try:
            return len(self.contents)
        except:
            return 0

    def inner(self, other  ):
        """
        inner product on naturalized collective space

        return matrix across canon ranks
        """
        assert isinstance(other, Component)
        u = self.values()
        v = other.values()
        return tf.linalg.matmul(u,v, transpose_a = False, transpose_b = True)

    def normalize(self):
        self.contents = tf.transpose(tf.linalg.normalize(tf.transpose(self.contents),axis=0)[0])
        return self

    def amplitude(self):
        return tf.linalg.normalize((self.contents),axis=1)[1]

    def batch_add(self, others):
        """
        Safely concatenates a list of Component objects all at once 
        to prevent C++ heap corruption/fragmentation.
        """
        # Collect the current tensor and all incoming tensors in a Python list
        tensor_list = [self.values()]
        for other in others:
            assert isinstance(other, Component)
            tensor_list.append(other.values())
        
        # Hit the C++ memory allocator exactly ONE time
        self.contents = tf.concat(tensor_list, axis=0)
        return self

    def add(self, other  ):
        assert isinstance(other, Component)
        u = self.values()
        v = other.values()
        self.contents = tf.concat([u,v],0)
        return self

    def set_boost(self, transform = []):
        if transform == []:
            u = self.values()
            q,r = tf.linalg.qr(tf.transpose(u) , full_matrices = False)
            self.transform = q
        else:
            self.transform = transform
        return self

    def boost(self):
        u = self.values()
        self.contents =  tf.transpose(tf.linalg.matmul(tf.transpose(self.transform),tf.transpose(u)))
        return self

    def unboost(self):
        u = self.values()
        self.contents =  tf.transpose(tf.linalg.matmul(self.transform,tf.transpose(u)))
        return self
        
    def len(self):
        return len(tf.transpose(self.contents))

    def __getitem__(self, r):
        print(f"Attempting to access component at rank {r} with total ranks {len(self)}")
        if r < len(self):
            return Component(lattice = self.lattice, contents = [self.contents[r]], transform = self.transform )
    
    # --- ADD THIS METHOD ---
    def __setitem__(self, r, value):
        """Allows overwriting an internal component using bracket notation."""
        if r < len(self):
            # Routes the Amplitude object directly into the correct slot in contents
            self.contents[r] = value 
        else:
            raise IndexError("Component assignment index out of range")

    def sample(self, sample_rank, num_samples = 1):
        """
        Manifestly obvious, the reconstruction under frequency sampling of this output would be q
    
        Parameters
        ----------
        q : SoP
        lattices : dict of lattice positions
    
        Returns
        -------
        A sample in q
        """

        u = self.values()
        
        def discrete_inverse_transform_sampling(pdf_values, pdf_domain):
            """
            Generates random samples from a discrete PDF using inverse transform sampling.
        
            Args:
                pdf_values: A NumPy array representing the PDF values.
                pdf_domain: A NumPy array representing the corresponding domain values.
                num_samples: The number of samples to generate.
        
            Returns:
                sample.
    
            advised by Gemini
            """
            pdf_values -= tf.math.reduce_min(pdf_values, axis=0)    
            # 1. Normalize the PDF
            pdf_values = tf.math.abs(pdf_values) / tf.math.abs(tf.math.reduce_sum(pdf_values))
            # 2. Calculate the CDF

            cdf = tf.reshape(tf.math.cumsum(pdf_values),-1)
            # 3. Generate uniform samples
            uniform_samples = tf.random.uniform(shape=(num_samples,), dtype = tf.float64)
            # 4. Inverse lookup

            sampled_value = tf.searchsorted( cdf, uniform_samples)
            return pdf_domain[sampled_value]
    
        return  tf.convert_to_tensor(discrete_inverse_transform_sampling ( u[sample_rank], self.lattice ) )


    def gaussian(self, position :float, sigma :float, l: int = 0 ):
        position   = tf.constant(position, dtype=tf.float64)
        sigma      = tf.constant(sigma, dtype=tf.float64)
        
        self.contents = tf.convert_to_tensor([[ compute(self.spacing, l, 1./sigma**2, position, x) for x in self.lattice ]], dtype = tf.float64)
        return self

    
    def delta(self, position : float , spacing : float ):
        position   = tf.constant(position, dtype=tf.float64)
        spacing    = tf.constant(spacing, dtype=tf.float64)
        
        self.contents =  tf.convert_to_tensor([[ tf.sqrt(self.spacing/spacing)* (tf.math.sin( pi2/2. *  ( x - position )/ spacing )/( pi2/2. *  ( x - position )/ spacing ) if x != position else 1 ) for x in self.lattice ]])            
        return self

    def flat(self):
        norm    = 1.0/tf.sqrt(tf.constant(len(self.lattice), dtype=tf.float64))

        self.contents =  tf.convert_to_tensor([[  norm for _ in self.lattice ]])            
        return self



    def values(self):
        return (self.contents)

