from openmdao.api import Group, IndepVarComp

from openaerostruct.aerodynamics.rotate_to_wind import RotateToWindFrame
from openaerostruct.aerodynamics.scale_to_pg import ScaleToPrandtlGlauert
from openaerostruct.aerodynamics.rotate_from_wind import RotateFromWindFrame
from openaerostruct.aerodynamics.scale_from_pg import ScaleFromPrandtlGlauert


class PGTransform(Group):
    """
    This group is responsible for transforming the VLM geometries from
    physical coordinates to Prandtl-Glauert coordinates. This allows the
    compressible aerodynamic problem to be solved as an equivelent
    incompressible problem.

    The transform can be broken down into two steps:

        1. Rotate the geometry from the body frame to the wind frame so
        x-axis is parallel to freestream velocity.

        2. Scale wind frame coordinates by Prandtl-Glauert factor to retrieve
        equivalent Prandtl-Glauert geometry.
        """

    def initialize(self):
        self.options.declare('surfaces', types=list)
        self.options.declare('rotational', False, types=bool,
                             desc="Set to True to turn on support for computing angular velocities")

    def setup(self):
        surfaces = self.options['surfaces']

        # Create component to rotate mesh geometry to wind frame,
        # s.t. the freestream velocity is along the x-axis
        surface_group.add_subsystem('rotate', RotateToWindFrame(surfaces=surfaces),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        # Scale y and z direction by Prandtl Glauert factor
        surface_group.add_subsystem('scale', ScaleToPrandtlGlauert(surfaces=surfaces),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        # This portion has been commented out because it gives problems when
        # converging the adjoint when used in an aerostructural problem. This
        # is probably because OpenMDAO doesn't like independent variables being
        # introduced inside a coupled cycle. Regardless, these values now default
        # to zero if not specified in the incompressible_states group anyway.
        '''
        # Since we have rotated to the wind frame in the first step, the
        # angle of attack and sideslip angle are by definition zero in PG frame
        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('alpha_pg', val=0., units='deg')
        indep_var_comp.add_output('beta_pg', val=0., units='deg')
        self.add_subsystem('flow_angles', indep_var_comp, promotes_outputs=['*'])
        '''


class InversePGTransform(Group):
    """
    This group is responsible for transforming the solved incompressible
    forces in the Prandtl-Glauert domain to the compressible forces in the
    physical aerodynamic frame. This is the reverse procedure used in the
    PGTransform group.

    The inverse transform can be broken down into two steps:

        1. Scale Prandtl-Glauert force vectors to physical force vectors in wind
        frame.

        2. Rotate physical forces from wind frame to the aerodynamic frame.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)
        self.options.declare('rotational', False, types=bool,
                             desc="Set to True to turn on support for computing angular velocities")

    def setup(self):
        surfaces = self.options['surfaces']

        surfaces_group = Group()

        for surface in surfaces:
            name = surface['name']
            surface_group = Group()

            # Scale force vectors back to compressible values
            surface_group.add_subsystem('scale', ScaleFromPrandtlGlauert(surface=surface),
                promotes_inputs=['Mach_number', 'sec_forces_pg', 'node_forces_pg'],
                promotes_outputs=['sec_forces_w_frame', 'node_forces_w_frame'])

            # Rotate forces back to aerodynamic frame
            surface_group.add_subsystem('rotate', RotateFromWindFrame(surface=surface),
                promotes_inputs=['alpha', 'beta', 'sec_forces_w_frame',
                    'node_forces_w_frame'],
                promotes_outputs=['sec_forces', 'node_forces'])

            surfaces_group.add_subsystem(name,
                 surface_group,
                 promotes_inputs=['alpha', 'beta', 'Mach_number'])

        self.add_subsystem('surfaces', surfaces_group, promotes=['*'])
