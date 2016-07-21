% This is a Matlab routine to test the Python aero-struct coupled system

% Setup mesh and coupled system parameters
[mesh, params] = coupled_setup(2,3);

% Create anonymous function for easier manipulation
aero = @(mesh) coupled_aero(mesh, params);
struct = @(loads) coupled_struct(loads, params);

% Aero
loads = aero(mesh);

% Struct
mesh = struct(loads);
