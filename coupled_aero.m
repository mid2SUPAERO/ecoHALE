function ld = coupled_aero(dm, kwargs)

% Aero
def_mesh = mat2np(dm);
loads = py.coupled.aero(def_mesh,kwargs);
ld = np2mat(loads);

end