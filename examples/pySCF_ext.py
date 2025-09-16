import pandas as pd
import sopy as sp
import numpy as np 
from pyscf import gto, scf

TOL = 1e-8

"""
pySCF interface
build the basis and orbitals from foundation up
"""

def tabulate ( mol ):
    Angular_template = {
        0:{
        's' : [{'l':[0,0,0],'amp':1}],
        },
        1:{
        'px': [{'l':[1,0,0],'amp':1}],
        'py': [{'l':[0,1,0],'amp':1}],
        'pz': [{'l':[0,0,1],'amp':1}],
        },
        2:{
        'dxy':[{'l':[1,1,0],'amp':1}],
        'dyz':[{'l':[0,1,1],'amp':1}],
        'dz^2':[{'l':[0,0,2],'amp':1}, {'l':[2,0,0],'amp':-0.5},{'l':[0,2,0],'amp':-0.5}],
        'dxz':[{'l':[1,0,1],'amp':1}],
        'dx2-y2':[{'l':[2,0,0],'amp':np.sqrt(3)/2},{'l':[0,2,0],'amp':-np.sqrt(3)/2}]
        }
    }
    
    N = mol.nbas
    mob = [(mol.bas_ctr_coeff(n)) for n in range(N)]
    exp = [(mol.bas_exp(n)) for n in range(N)]
    ##definition of a1,a2 is including this 2
    ls   = [(mol.bas_angular(n)) for n in range(N)]   
    label  = mol.ao_labels()
    stage = []
    b = 0
    for n,(l,exps,coefs) in enumerate(zip(ls, exp, mob)):
        for ln in range(2*l+1):
            flag = True    
            for ang in Angular_template[l]:
                if flag:
                    id1 = int(label[b][:2])
                    position = mol.atom_coord(id1)
                    if ang in label[b][5:] :
                        flag = False
                        for c in range(coefs.shape[1]):
                            for (exp1, coef1) in (zip( exps, coefs)): 
                                for term in Angular_template[l][ang]:
                                    i = term['l']
                                    amp = term['amp']
                                    stage += [[n,b,c,l,ln+1,i[0],i[1],i[2],amp*coef1[c],exp1,position[0],position[1],position[2]]]
                            b+=1
    return pd.DataFrame((stage), columns =['b','bb','bc','L','ln','lx','ly','lz','coef','exp','x','y','z'])


def get_orbital(mol, orb, b, lattices, tolerance = TOL):
    """
    These dudes should be natively orthogonal
    """
    BA = tabulate(mol)
    orb1 = np.transpose(orb)[b]

    Sum = sp.vector()
    BA1 = BA.set_index(['bb']).copy()
    BA1['coef'] *= pd.Series(orb1)
    BAG = BA1.groupby(['lx','ly','lz','exp','x','y','z'])['coef'].sum()    
    for lx,ly,lz,exp,x,y,z in BAG[BAG.abs()>tolerance].index:
            coef = BAG[lx,ly,lz,exp,x,y,z]
            positions = (x,y,z)
            ls        = (lx,ly,lz)
            sigmas    = np.sqrt(2./np.array([exp,exp,exp]))
            wavenumbers = 3*[0.]
            phis        = 3*[0.]
            v = sp.vector().gaussian(ls=ls,positions=positions, sigmas=sigmas, a = coef, lattices = lattices, wavenumbers = wavenumbers , phis = phis)
            Sum += v
    return Sum


def get_basis(mol, b, lattices, tolerance = TOL):
    """
    Should match overlap matrix of basis
    """
    BA = tabulate(mol)
    B = len(mol.ao_labels())
    orb1 = np.eye(B)[b]

    Sum = sp.vector()
    BA1 = BA.set_index(['bb']).copy()
    BA1['coef'] *= pd.Series(orb1)
    BAG = BA1.groupby(['lx','ly','lz','exp','x','y','z'])['coef'].sum()    
    for lx,ly,lz,exp,x,y,z in BAG[BAG.abs()>tolerance].index:
            coef = BAG[lx,ly,lz,exp,x,y,z]
            positions = (x,y,z)
            ls        = (lx,ly,lz)
            sigmas    = np.sqrt(2./np.array([exp,exp,exp]))
            wavenumbers = 3*[0.]
            phis        = 3*[0.]
            v = sp.vector().gaussian(ls=ls,positions=positions, sigmas=sigmas, a = coef, lattices = lattices, wavenumbers = wavenumbers , phis = phis)
            Sum += v
    return Sum