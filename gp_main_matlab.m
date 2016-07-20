% This is a Matlab routine to test the Python aero-struct coupled system

% Set it up

 py.gp_main.main()
 
 return

try
    py.gp_main.main()
catch err
    err.message
%     if(isa(err,''))
%         err.ExceptionObject
%     end
end