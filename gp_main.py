# Main python script to test OpenAeroStruct functions

import gp_setup, gp_aero

out = gp_setup.setup()

mesh = out['mesh']
num_x = out['num_x']
num_y = out['num_y']
des_vars = out['des_vars']

gp_aero.aero(mesh=mesh, num_x=num_x, num_y=num_y, des_vars=des_vars)
