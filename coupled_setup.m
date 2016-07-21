function [dm, kwargs] = coupled_setup()

% Setup
kwargs = py.coupled.setup_kwargs();  % parameters for aero and struct modules
def_mesh = py.coupled.setup_mesh();  % initial mesh for wing
dm = np2mat(def_mesh);

end