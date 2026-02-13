import numpy as np
import pytest
import sopy as sp

# all data in gaussian vectors comes from this source, checking self consistency!
from bandlimit.gaussian import compute

def mathematica_ground_truth(alpha1, alpha2, k):
    """
    Analytical result for Integrate[g[a1, x] g[a2, x], {x, -inf, inf}]
    """
    numerator = np.exp( - 0.5 *(k/(alpha1+alpha2) )**2 )* np.sqrt(2) * (alpha1 * alpha2)**0.25 
    denominator = np.sqrt(alpha1 + alpha2)
    return numerator / denominator
    

def test_gaussian_overlap_integral():
    """
    test gaussians as vectors
    """

    # Define test parameters (alphas > 0 as per Mathematica assumptions)
    alpha1 = 0.5
    alpha2 = 1.2
    
    # Get the "Golden Standard" from your Mathematica equation
    expected_value = mathematica_ground_truth(alpha1, alpha2,0)
    
    # --- ACT ---
    # alpha = 1/sigma**2
    sigma1 = 1/np.sqrt(alpha1)
    sigma2 = 1/np.sqrt(alpha2)
    lattices = [ np.linspace( - 4*max(sigma1,sigma2), 4 * max(sigma1,sigma2), 1000)]
    
    u1 = sp.Vector().gaussian(a = 1, \
      positions=[0.], \
      sigmas = [ np.sqrt(alpha1)], \
      ls = [0], \
      lattices = lattices)
    
    u2 = sp.Vector().gaussian(a = 1, \
      positions=[0.], \
      sigmas = [ np.sqrt(alpha2)], \
      ls = [0], \
      lattices = lattices)
      
    simulated_sopy_result = u1.dot(u2)
        
    # --- ASSERT ---
    # We use np.isclose for floating point comparisons
    np.testing.assert_allclose(simulated_sopy_result, expected_value, rtol=1e-7)

def test_gaussian_delta_integral():
    """
    test gaussians and deltas
    """
    # Define test parameters (alphas > 0 as per Mathematica assumptions)
    alpha1 = 1
    spacing2 = 0.2
    sigma1 = 1/np.sqrt(alpha1)
    position = 0.1

    # Get the "Silver Standard" from bandwidth
    expected_value = compute(spacing2, 0, alpha1, 0, position)
    
    # --- ACT ---
    # alpha = 1/sigma**2
    lattices = [ np.linspace( - 20, 20, 1000)]
    
    u1 = sp.Vector().gaussian(a = 1, \
      positions=[0.], \
      sigmas = [ np.sqrt(alpha1)], \
      ls = [0], \
      lattices = lattices)
    
    d2 = sp.Vector().delta(a = 1, \
      spacings = [spacing2], \
      positions=[position], \
      lattices = lattices)
      
    simulated_sopy_result = u1.dot(d2)#/np.sqrt(d2.dot(d2)* u1.dot(u1))
        
    # --- ASSERT ---
    # We use np.isclose for floating point comparisons
    np.testing.assert_allclose(simulated_sopy_result, expected_value, rtol=1e-7)

def test_ft_gaussian_overlap_integral():
    """
    test exp_i on gaussians
    """
    # Define test parameters (alphas > 0 as per Mathematica assumptions)
    alpha1 = 0.95
    alpha2 = 1.2
    k = 0.05
    
    # Get the "Golden Standard" from your Mathematica equation
    expected_value = np.real(mathematica_ground_truth(alpha1, alpha2,k))
    
    # --- ACT ---
    # alpha = 1/sigma**2
    sigma1 = 1/np.sqrt(alpha1)
    sigma2 = 1/np.sqrt(alpha2)
    lattices = [ np.linspace( - 20*max(sigma1,sigma2),20 * max(sigma1,sigma2), 2000)]
    
    u1 = sp.Operand( sp.Vector().gaussian(a = 1, \
      positions=[0.], \
      sigmas = [ np.sqrt(alpha1)], \
      ls = [0], \
      lattices = lattices), sp.Vector()
      )
    
    u2 = sp.Vector().gaussian(a = 1, \
      positions=[0.], \
      sigmas = [ np.sqrt(alpha2)], \
      ls = [0], \
      lattices = lattices)
      
    simulated_sopy_result = (u1.exp_i(ks=[k]).re.dot(u2))
        
    # --- ASSERT ---
    # We use np.isclose for floating point comparisons
    np.testing.assert_allclose(simulated_sopy_result, expected_value, rtol=1e-3)



if __name__ == "__main__":
    # Quick manual check
    a1, a2 = 0.5, 1.2
    print(f"Alpha1: {a1}, Alpha2: {a2}")
    print(f"Mathematica Result: {mathematica_ground_truth(a1, a2):.10f}")
    
    
    
    