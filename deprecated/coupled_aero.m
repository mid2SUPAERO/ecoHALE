function ld = coupled_aero(dm, params)

% Aero
def_mesh = mat2np(dm);  % convert matlab to python
loads = py.coupled.aero(def_mesh,params);
ld = np2mat(loads);     % convert python to matlab

end