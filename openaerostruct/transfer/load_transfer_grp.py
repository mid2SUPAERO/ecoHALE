# load transfer group
from openmdao.api import Group, MuxComp
from load_transfer import LoadTransfer

class load_transfer_grp(Group):
    '''
    This group calculates the load transfer components and then muxes them together
    which actually causes units to be mixed (known problem)
    however, changing the mixed units would require other code rewrites, so for now
    we are just muxing the mixed units together and they can be separated at some
    later date.


    Perform aerodynamic load transfer.

    Apply the computed sectional forces on the aerodynamic surfaces to
    obtain the deformed mesh FEM loads.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Flattened array defining the lifting surfaces after deformation.
    sec_forces[nx-1, ny-1, 3] : numpy array
        Flattened array containing the sectional forces acting on each panel.
        Stored in Fortran order (only relevant when more than one chordwise
        panel).

    Returns
    -------
    loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the sectional forces.
        THIS OUTPUT HAS MIXED UNITS!!!!
        LoadsA is output in N
        LoadsB is output in N*m
        
        loads is output in N
    '''
    
    def initialize(self):
        self.options.declare('surface', types=dict)  

    def setup(self):
        surface = self.options['surface']
        
        self.add_subsystem('LoadTransfer', LoadTransfer(surface=surface), \
                           promotes_inputs=['def_mesh','sec_forces'], \
                           promotes_outputs=['loadsA', 'loadsB'] )
        
        # setup the mux
        n = 2 # The number of elements to be muxed
        ny = surface['num_y']
        # m = 'ny,3' # The size of each element to be muxed
        mux_comp = self.add_subsystem('mux', MuxComp(vec_size=n), promotes_outputs=['loads'])
        mux_comp.add_var('loads', shape=(ny,3), axis=1, units='N') # WARNING!! THESE UNITS ARE ACTUALLY MIXED!!
        
        self.connect('loadsA','mux.loads_0')
        self.connect('loadsB','mux.loads_1')
        
