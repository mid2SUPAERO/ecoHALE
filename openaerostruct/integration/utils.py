def connect_aerostruct_old(model, name):
    model.connect(name[:-1] + '.K', 'coupled.' + name[:-1] + '.K')

    # Perform the connections with the modified names within the
    # 'aero_states' group.
    model.connect('coupled.' + name[:-1] + '.def_mesh', 'coupled.aero_states.' + name + 'def_mesh')
    model.connect('coupled.' + name[:-1] + '.b_pts', 'coupled.aero_states.' + name + 'b_pts')
    model.connect('coupled.' + name[:-1] + '.c_pts', 'coupled.aero_states.' + name + 'c_pts')
    model.connect('coupled.' + name[:-1] + '.normals', 'coupled.aero_states.' + name + 'normals')

    # Connect the results from 'aero_states' to the performance groups
    model.connect('coupled.aero_states.' + name + 'sec_forces', name + 'perf' + '.sec_forces')

    # Connect the results from 'coupled' to the performance groups
    model.connect('coupled.' + name[:-1] + '.def_mesh', 'coupled.' + name + 'loads.def_mesh')
    model.connect('coupled.aero_states.' + name + 'sec_forces', 'coupled.' + name + 'loads.sec_forces')

    # Connect the output of the loads component with the FEM
    # displacement parameter. This links the coupling within the coupled
    # group that necessitates the subgroup solver.
    model.connect('coupled.' + name + 'loads.loads', 'coupled.' + name[:-1] + '.loads')

    # Connect aerodyamic mesh to coupled group mesh
    model.connect(name[:-1] + '.mesh', 'coupled.' + name[:-1] + '.mesh')

    # Connect performance calculation variables
    model.connect(name[:-1] + '.radius', name + 'perf.radius')
    model.connect(name[:-1] + '.A', name + 'perf.A')
    model.connect(name[:-1] + '.thickness', name + 'perf.thickness')

    # Connection performance functional variables
    model.connect(name + 'perf.structural_weight', 'total_perf.' + name + 'structural_weight')
    model.connect(name + 'perf.L', 'total_perf.' + name + 'L')
    model.connect(name + 'perf.CL', 'total_perf.' + name + 'CL')
    model.connect(name + 'perf.CD', 'total_perf.' + name + 'CD')
    model.connect('coupled.aero_states.' + name + 'sec_forces', 'total_perf.' + name + 'sec_forces')

    # Connect parameters from the 'coupled' group to the performance
    # groups for the individual surfaces.
    model.connect(name[:-1] + '.nodes', name + 'perf.nodes')
    model.connect('coupled.' + name[:-1] + '.disp', name + 'perf.disp')
    model.connect('coupled.' + name[:-1] + '.S_ref', name + 'perf.S_ref')
    model.connect('coupled.' + name[:-1] + '.widths', name + 'perf.widths')
    model.connect('coupled.' + name[:-1] + '.chords', name + 'perf.chords')
    model.connect('coupled.' + name[:-1] + '.lengths', name + 'perf.lengths')
    model.connect('coupled.' + name[:-1] + '.cos_sweep', name + 'perf.cos_sweep')

    # Connect parameters from the 'coupled' group to the total performance group.
    model.connect('coupled.' + name[:-1] + '.S_ref', 'total_perf.' + name + 'S_ref')
    model.connect('coupled.' + name[:-1] + '.widths', 'total_perf.' + name + 'widths')
    model.connect('coupled.' + name[:-1] + '.chords', 'total_perf.' + name + 'chords')
    model.connect('coupled.' + name[:-1] + '.b_pts', 'total_perf.' + name + 'b_pts')
    model.connect(name + 'perf.cg_location', 'total_perf.' + name + 'cg_location')

def connect_aerostruct(model, point_name, name):

    com_name = point_name + '.' + name

    model.connect(name[:-1] + '.K', point_name + '.coupled.' + name[:-1] + '.K')

    # Connect aerodyamic mesh to coupled group mesh
    model.connect(name[:-1] + '.mesh', point_name + '.coupled.' + name[:-1] + '.mesh')

    # Connect performance calculation variables
    model.connect(name[:-1] + '.radius', com_name + 'perf.radius')
    model.connect(name[:-1] + '.A', com_name + 'perf.A')
    model.connect(name[:-1] + '.thickness', com_name + 'perf.thickness')

    # Connection performance functional variables
    model.connect(com_name + 'perf.structural_weight', point_name + '.total_perf.' + name + 'structural_weight')
    model.connect(com_name + 'perf.L', point_name + '.total_perf.' + name + 'L')
    model.connect(com_name + 'perf.CL', point_name + '.total_perf.' + name + 'CL')
    model.connect(com_name + 'perf.CD', point_name + '.total_perf.' + name + 'CD')
    model.connect(point_name + '.coupled.aero_states.' + name + 'sec_forces', point_name + '.total_perf.' + name + 'sec_forces')

    # Connect parameters from the 'coupled' group to the performance
    # groups for the individual surfaces.
    model.connect(name[:-1] + '.nodes', com_name + 'perf.nodes')
    model.connect(point_name + '.coupled.' + name[:-1] + '.disp', com_name + 'perf.disp')
    model.connect(point_name + '.coupled.' + name[:-1] + '.S_ref', com_name + 'perf.S_ref')
    model.connect(point_name + '.coupled.' + name[:-1] + '.widths', com_name + 'perf.widths')
    model.connect(point_name + '.coupled.' + name[:-1] + '.chords', com_name + 'perf.chords')
    model.connect(point_name + '.coupled.' + name[:-1] + '.lengths', com_name + 'perf.lengths')
    model.connect(point_name + '.coupled.' + name[:-1] + '.cos_sweep', com_name + 'perf.cos_sweep')

    # Connect parameters from the 'coupled' group to the total performance group.
    model.connect(point_name + '.coupled.' + name[:-1] + '.S_ref', point_name + '.total_perf.' + name + 'S_ref')
    model.connect(point_name + '.coupled.' + name[:-1] + '.widths', point_name + '.total_perf.' + name + 'widths')
    model.connect(point_name + '.coupled.' + name[:-1] + '.chords', point_name + '.total_perf.' + name + 'chords')
    model.connect(point_name + '.coupled.' + name[:-1] + '.b_pts', point_name + '.total_perf.' + name + 'b_pts')
    model.connect(com_name + 'perf.cg_location', point_name + '.total_perf.' + name + 'cg_location')
