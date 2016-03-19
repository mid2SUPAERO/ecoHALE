import numpy as np
from numpy import cos, sin

from crm_data import mesh


def rotate(mesh, thetas): 
    """computes rotation matricies given mesh and rotation angles in degress"""

    le = mesh[0]
    te = mesh[1]

    n_points = len(le)
    
    rad_thetas = np.radians(thetas) # the einsum flips the sign, don't know why

    mats = np.zeros((n_points, 3,3))
    mats[:,0,0] = cos(rad_thetas)
    mats[:,0,2] = sin(rad_thetas)
    mats[:,1,1] = 1
    mats[:,2,0] = -sin(rad_thetas)
    mats[:,2,2] = cos(rad_thetas)

    te[:] = np.einsum("ikj, ij -> ik", mats, te-le)
    te += le
   
    # equivalent loop
    # for i,theta in enumerate(rad_thetas): 
    #     mat = np.zeros((3,3))
    #     mat[0,0] = cos(theta)
    #     mat[0,2] = sin(theta)
    #     mat[1,1] = 1
    #     mat[2,0] = -sin(theta)
    #     mat[2,2] = cos(theta)

    #     te[i][:] = mat.dot(te[i]-le[i])
    #     te[i] += le[i]

    return mesh


def sweep(mesh, angle): 
    """shearing sweep angle. Positive sweeps back. """ 

    le = mesh[0]
    te = mesh[1]

    y0 = le[0,1]

    tan_theta = sin(np.radians(angle))
    dx = (le[:,1] - y0) * tan_theta
    
    le[:,0] += dx
    te[:,0] += dx 

    return mesh
        

def stretch(mesh, factor): 
    """proportional change to span. 
    1.1 give a 10% increase in span""" 

    le = mesh[0]
    te = mesh[1]

    n_points = len(le)

    y_max = le[-1,1]
    dy = y_max*(factor-1)/(n_points-1)*np.arange(1,n_points)

    le[1:,1] += dy
    te[1:,1] += dy

    return mesh


def mirror(mesh, right_geom=True):
    """Takes a half geometry and mirrors it across the symmetry plane. 
    If right_geom==True, it mirrors from right to left, 
    assuming that the first point is on the symmetry plane. Else 
    it mirrors from left to right, assuming the last point is on the 
    symmetry plane.""" 
    
    n_points = mesh.shape[1]

    new_mesh = np.empty((2,2*n_points-1,3))

    mirror_y = np.ones(mesh.shape)
    mirror_y[:,:,1] *= -1.0

    if right_geom: 
        new_mesh[:,:n_points,:] = mesh[:,::-1,:]*mirror_y
        new_mesh[:,n_points:,:] = mesh[:,1:,:]
    else: 
        new_mesh[:,:n_points,:] = mesh[:,::-1,:]
        new_mesh[:,n_points:,:] = mesh[:,1:,:]*mirror_y[:,1:,:]


    return new_mesh


if __name__ == "__main__": 
    import plotly.offline as plt
    import plotly.graph_objs as go

    from plot_tools import wire_mesh, build_layout

    thetas = np.zeros(20)
    thetas[10:] += 10
    # new_mesh = rotate(mesh, thetas)

    # new_mesh = sweep(mesh, 20)

    # new_mesh = stretch(mesh, 10)

    new_mesh = mirror(mesh, right_geom=False)

    # wireframe_orig = wire_mesh(mesh)
    wireframe_new = wire_mesh(new_mesh)
    layout = build_layout()

    fig = go.Figure(data=wireframe_new, layout=layout)
    plt.plot(fig, filename="wing_3d.html")

