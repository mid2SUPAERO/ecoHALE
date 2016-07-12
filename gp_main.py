# Main python script to test OpenAeroStruct functions

import gp_setup, gp_aero

def_mesh, kwargs = gp_setup.setup()

print "from main... def_mesh"
print def_mesh

loads = gp_aero.aero(def_mesh,**kwargs)


print "from main... loads"
print loads
