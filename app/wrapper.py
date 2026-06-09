import sopy as sp
import numpy as np

class SoPyWrapper:
    def __init__(self, initial_vector=None):
        self.vector = initial_vector if initial_vector is not None else sp.Vector()

    def update(self, input_vector, learn_rate, tolerance=0.0, iterate=1):
        """
        Standard update procedure with early stopping to reduce over-learning.
        """
        current_dist = self.vector.dist(input_vector)
        
        if current_dist <= tolerance:
            return current_dist, False 
            
        self.vector = self.vector.learn(input_vector, alpha=learn_rate/100, iterate=iterate)
        return self.vector.dist(input_vector), True

    def tune(self, input_vector, ambiguity_rate, tune_rate=0.01):
        """
        Tunes the vector state so that output.dist(input) ~ ambiguity_rate.
        """
        current_dist = self.vector.dist(input_vector)
        dist_error = current_dist - ambiguity_rate
        effective_alpha = tune_rate * dist_error
        
        self.vector = self.vector.learn(input_vector, alpha=abs(effective_alpha), iterate=1)
        return self.vector.dist(input_vector)

    def reduce_to_target_distance(self, max_allowed_distance, iterations=10):
        """
        Binary searches for the smallest Target Partition Size that keeps
        the reduction error (distance to original) <= max_allowed_distance.
        """
        current_rank = len(self.vector)
        if current_rank <= 1:
            return current_rank, 0.0

        original_vector = self.vector 
        
        low = 1
        high = current_rank
        best_partition = current_rank
        best_reduced_vector = self.vector
        final_dist = 0.0
        
        while low <= high:
            mid_partition = (low + high) // 2
            
            test_vector = original_vector.fibonacci(
                partition=mid_partition, 
                iterate=iterations, 
                total_iterate=iterations
            )
            
            current_dist = test_vector.dist(original_vector)
            
            if current_dist <= max_allowed_distance:
                # Acceptable error, save state, try smaller partition
                best_partition = mid_partition
                best_reduced_vector = test_vector
                final_dist = current_dist
                high = mid_partition - 1
            else:
                # Error too high, need larger partition
                low = mid_partition + 1
                
        self.vector = best_reduced_vector
        return best_partition, final_dist