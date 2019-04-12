from openmdao.api import Group

from openaerostruct.aerodynamics.rotate_from_wind import RotateFromWindFrame
from openaerostruct.aerodynamics.scale_from_pg import ScaleFromPrandtlGlauert


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
