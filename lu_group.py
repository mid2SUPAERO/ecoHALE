from collections import OrderedDict

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from openmdao.api import Group, ScipyGMRES

import time



class LUSolver(ScipyGMRES): 
    """ only works with LUGroup, assumes that an LU factorization was made during linearize""" 

    def __init__(self): 
        super(LUSolver, self).__init__()

    def setup(self, sub): 

        self.u_vec = sub.unknowns
        self.var_names = self.u_vec.keys() 
        self.var_owners = {}

        self_path = sub.pathname.split(".")
        len_path = len(self_path)
        for var in self.var_names: 
            # have to get the local path name... this is a crappy way to do it, but it works
            owner_path = ".".join(self.u_vec.metadata(var)['pathname'].split('.')[len_path:-1])
            self.var_owners[var] = sub._subsystem(owner_path)

        self.jacobian = np.eye(self.u_vec.vec.size)
            
    def solve(self, rhs_mat, system, mode):

        sol_buf = OrderedDict()
        self.lup = dict()

        for voi, rhs in rhs_mat.items():
            self.voi = None

            self.system = system

            u_vec = self.u_vec

            if system.regen_lu: 
                
                for out_var in self.var_names: 
                    owner = self.var_owners[out_var]
                    jac = owner._jacobian_cache
                    o_start, o_end = u_vec._dat[out_var].slice
                    
                    for in_var in self.var_names: 
                        i_start, i_end = u_vec._dat[in_var].slice
                        try: 
                            self.jacobian[o_start:o_end, i_start:i_end] = jac[out_var, in_var]

                        except KeyError: 
                            pass # that deriv doesn't exist

                # self.lup[voi] = lu_factor(self.jacobian)

            # don't need to explicitly handle transpose, because mode takes care of it in mult            
            # sol_buf[voi] = lu_solve(self.lup[voi], rhs) 
            sol_buf[voi] = np.linalg.solve(self.ja)

        system.regen_lu = False
        return sol_buf


class LUGroup(Group): 

    def __init__(self): 
        super(LUGroup, self).__init__()

        self.ln_solver = LUSolver()

    def linearize(self, params, unknowns, resids): 
        super(LUGroup, self).linearize(params, unknowns, resids)

        self.regen_lu = True



