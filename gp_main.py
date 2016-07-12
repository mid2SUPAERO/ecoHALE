# Main python script to test OpenAeroStruct functions

import gp_setup, gp_aero

kwargs = gp_setup.setup()

gp_aero.aero(kwargs)
