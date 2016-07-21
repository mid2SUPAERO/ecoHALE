% This is a Matlab routine to test the Python aero-struct coupled system

% Setup
[mesh, kwargs] = coupled_setup()

% Aero
loads = coupled_aero(mesh, kwargs)

% Struct
mesh = coupled_struct(loads, kwargs)
