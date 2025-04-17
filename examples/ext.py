####################  ##########################
################  SoPy -extensions #############
####################  ########################## 

################################################
###          by Quantum Galaxies Corp        ###
##           (2023,2024,2025)                 ##
################################################


import tensorflow as tf

import tensorly as tl
from tensorly.decomposition import parafac
####
##this is an NVIDA package to reduce dense 
##vectors to SoP.
##they have not coded the decompose command
####

def reduce(img, partition:int, init = 'random' , tol = 1e-6):
    """
    reduce Dense Vector to SoP
    Parameters
    ----------
    img : dense vector
    canons : int 
    init , tol internal NVDIA parameters

    Returns
    -------
    SoP
    """
    weights, factors =  parafac(img,partition,init=init,tol=tol)
    q = {}
    q[0] = tf.transpose([weights])
    for space in range(len(factors)):
        q[space+1] = tf.transpose(factors[space])
    return q

def image(q ):
    """
    image SoP back to Dense vector
    Parameters
    ----------
    q: SoP

    Returns
    -------
    image : ?float?
    """
    weight = tf.reshape(q[0],-1)
    factors = []
    for space in range(1,len(q)):
        factors += [ tf.transpose(q[space]) ]
    return tl.cp_to_tensor((weight,factors))
