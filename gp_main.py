# Main python script to test OpenAeroStruct functions

import gp_setup, gp_aero, gp_struct

def_mesh, kwargs = gp_setup.setup()

print "1 --- from main... def_mesh"
print def_mesh

loads = gp_aero.aero(def_mesh,**kwargs)

print "2 --- from main... loads"
print loads

def_mesh = gp_struct.struct(loads,**kwargs)

print "3 --- from main... def_mesh"
print def_mesh
