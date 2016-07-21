# Main python script to test OpenAeroStruct functions


from __future__ import print_function
import coupled
import warnings
import array
import numpy

# warnings.filterwarnings("ignore") # to disable openmdao warnings which will create an error in Matlab

def main(filename=None):

    # out_stream -->  where to write results. Default is None. Set out_stream=sys.stdout to print to screen
    # filename = 'gp_log.txt'
    if filename:
        out_stream = open(filename, 'w')
    else:
        out_stream = None

    # print('testing testing',file=out_stream)
    def_mesh, kwargs = coupled.setup(check=True, out_stream=out_stream)
    print('def_mesh.shape ',def_mesh.shape)
    print(def_mesh)

    dm = array.array('d',numpy.nditer(def_mesh))
    print('type(dm)=',type(dm))
    print(dm)


    # print("1 --- from main... def_mesh")
    # print(def_mesh)

    #
    # loads = coupled.aero(def_mesh,**kwargs)
    #
    # print "2 --- from main... loads"
    # print loads
    #
    # def_mesh = coupled.struct(loads,**kwargs)
    #
    # print "3 --- from main... def_mesh"
    # print def_mesh
    if type(out_stream) is file:
        if out_stream.mode:
            out_stream.close()

if __name__ == '__main__':
    main()
