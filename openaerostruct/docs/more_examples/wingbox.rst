.. _Wingbox_Model:

Wingbox Model
=============

In addition to the tubular structural spar available in OpenAeroStruct, you can use a wingbox-based model.
This model is detailed in Chauhan and Martins' paper `here <https://www.researchgate.net/publication/325986597_Low-fidelity_aerostructural_optimization_of_aircraft_wings_with_a_simplified_wingbox_model_using_OpenAeroStruct>`_.
Analytic derivatives are not provided with this model, so any optimization problem will use complex-step to obtain the relevant partial derivatives.

.. embed-code::
    openaerostruct.tests.test_aerostruct_wingbox_analysis.Test.test
    :layout: interleave
