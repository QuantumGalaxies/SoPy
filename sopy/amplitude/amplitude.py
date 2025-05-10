####################  ##########################
################  SoPy        ##################
####################  ########################## 

################################################
###          by Quantum Galaxies Corp        ###
##           (2023,2024,2025)                 ##
################################################


import tensorflow as tf

class amplitude :
    """
    d = 0
    """

    def __init__(self, a = 1 , contents = []):
        if contents == []:
            self.contents = tf.convert_to_tensor([[tf.constant(a, dtype = tf.float64)]], dtype = tf.float64)
        else:    
            self.contents = contents
        
    def boost(self):
        return self

    def set_boost(self, transform = []):
        return self
    
    def unboost(self):
        return self
    
    def copy(self):
        other = amplitude(contents = self.contents)
        return other
    
    def __len__(self):
        """
        for SoP like (canonRanks)* N*[R] 
        return CanonRanks, which is the number of product sums
        """
        try:
            return len(self.values())
        except:
            return 0
            
    def normalize(self):
        """
        take another component and map its canonical norms to amplitudes
        """
        self.contents = tf.linalg.normalize(self.values(),axis=1)[1]
        return self


    def sample(self, num_samples):
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
        
        def discrete_inverse_transform_sampling(pdf_values):
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
            # 1. Normalize             
            pdf_values = tf.math.abs(pdf_values) / tf.math.abs(tf.math.reduce_sum(pdf_values))
            # 2. Calculate the CDF

            cdf = tf.reshape(tf.math.cumsum(pdf_values),-1)
            # 3. Generate uniform samples
            uniform_samples = tf.random.uniform(shape=(num_samples,), dtype = tf.float64)
            # 4. Inverse lookup

            sampled_value = tf.searchsorted( cdf, uniform_samples)
            return sampled_value
    
        return tf.convert_to_tensor( discrete_inverse_transform_sampling ( self.values()) )

    def __getitem__(self,r):
        if r <  len(self): 
            return amplitude( contents = [self.contents[r]] )
            
    def __imul__(self, m):
        self.contents *= tf.constant(m, dtype= tf.float64)
        return self
    
    def __mul__(self, m):
        new = self.copy()
        new *= m
        return new

    def __rmul__(self,m):
        new = self.copy()
        new *= m
        return new


    
    def values(self):
        return (self.contents)