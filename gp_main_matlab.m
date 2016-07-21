% This is a Matlab routine to test the Python aero-struct coupled system

% Setup
kwargs = py.coupled.setup_kwargs();  % python dict, parameters for aero and struct modules
def_mesh = py.coupled.setup_mesh();  % python ndarray, initial mesh for wing

dm = np2mat(def_mesh);
def_mesh = mat2np(dm);

% % convert python to matlab
% sh = double(py.array.array('d',def_mesh.shape));
% dm1 = double(py.array.array('d',py.numpy.nditer(def_mesh)));
% dm = reshape(dm1,fliplr(sh))';  % matlab 2d array 
% 
% % convert matlab to python
% sh2 = fliplr(size(dm));
% dm2 = reshape(dm,1,numel(dm));  % [1, n] vector
% dm3 = py.numpy.array(dm2);
% dm4 = dm3.reshape(sh2).transpose()  % python ndarray


% Aero
loads = py.coupled.aero(def_mesh,kwargs)  % python ndarray
sh = double(py.array.array('d',loads.shape));
ld1 = double(py.array.array('d',py.numpy.nditer(loads)));
ld = reshape(ld1,fliplr(sh))'  % matlab 2d array

% Struct
def_mesh = py.coupled.struct(loads, kwargs);  % python ndarray
sh = double(py.array.array('d',def_mesh.shape));
dm1 = double(py.array.array('d',py.numpy.nditer(def_mesh)));
dm = reshape(dm1,fliplr(sh))'  % matlab 2d array 

