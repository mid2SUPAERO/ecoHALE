# OpenAeroStruct

To use OpenAeroStruct, you must first install OpenMDAO by following the instructions here: https://github.com/openmdao/openmdao

Next, clone the OpenAeroStruct repository:

    git clone https://github.com/hwangjt/OpenAeroStruct.git

Lastly, from within the OpenAeroStruct folder, make the Fortran files:

    make

Note that the code will run without compiling the Fortran library, but it will run significantly faster with the Fortran files compiled. 
