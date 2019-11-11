.. _Aerostruct_ffd:

Aerostructural FFD
==================

In additional to OpenAeroStruct's internal geometry manipulation group, you can also use `pyGeo` to perform free-form deformation (FFD) manipulation on the mesh.
This allows for more general design shape changes and helps sync up geometry changes between meshes from different levels of fidelity.

.. warning::
  This example requires `pyGeo`, an in-house MDO Lab code. If you have access to the code, you can clone it from `here <https://bitbucket.org/mdolab/pygeo>`_.

.. embed-code::
    openaerostruct.tests.test_aerostruct_ffd.Test.test
