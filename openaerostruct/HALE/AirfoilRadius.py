# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 17:43:26 2021

@author: Victor M. Guadano
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def RadiusCurvature(upper_x,lower_x,upper_y,lower_y):

    #upper_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype = 'complex128')
    #lower_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype = 'complex128')
    #upper_y = np.array([ 0.0447,  0.046,  0.0472,  0.0484,  0.0495,  0.0505,  0.0514,  0.0523,  0.0531,  0.0538, 0.0545,  0.0551,  0.0557, 0.0563,  0.0568, 0.0573,  0.0577,  0.0581,  0.0585,  0.0588,  0.0591,  0.0593,  0.0595,  0.0597,  0.0599,  0.06,    0.0601,  0.0602,  0.0602,  0.0602,  0.0602,  0.0602,  0.0601,  0.06,    0.0599,  0.0598,  0.0596,  0.0594,  0.0592,  0.0589,  0.0586,  0.0583,  0.058,   0.0576,  0.0572,  0.0568,  0.0563,  0.0558,  0.0553,  0.0547,  0.0541], dtype = 'complex128')
    #lower_y = np.array([-0.0447, -0.046, -0.0473, -0.0485, -0.0496, -0.0506, -0.0515, -0.0524, -0.0532, -0.054, -0.0547, -0.0554, -0.056, -0.0565, -0.057, -0.0575, -0.0579, -0.0583, -0.0586, -0.0589, -0.0592, -0.0594, -0.0595, -0.0596, -0.0597, -0.0598, -0.0598, -0.0598, -0.0598, -0.0597, -0.0596, -0.0594, -0.0592, -0.0589, -0.0586, -0.0582, -0.0578, -0.0573, -0.0567, -0.0561, -0.0554, -0.0546, -0.0538, -0.0529, -0.0519, -0.0509, -0.0497, -0.0485, -0.0472, -0.0458, -0.0444], dtype = 'complex128')
    
    t_c = np.max(upper_y-lower_y)
    
    t_c_new = np.linspace(0.1,0.5,num=100)
    C_new = np.linspace(0.1,1,num=100)
    
    ratio = np.zeros(len(t_c_new))
    xt_new = np.zeros((len(t_c_new),len(upper_x)))
    yt_new = np.zeros((len(t_c_new),len(upper_y)))
    p = np.zeros((len(t_c_new),10))
    p_ = np.zeros((len(t_c_new),len(upper_x)))
    derp = np.zeros((len(t_c_new),9))
    derp_ = np.zeros((len(t_c_new),len(upper_x)))
    der2p = np.zeros((len(t_c_new),8))
    der2p_ = np.zeros((len(t_c_new),len(upper_x)))
    R = np.zeros((len(t_c_new),len(upper_x)))
    Rmean = np.zeros((len(t_c_new),len(C_new)))
    
    for j in range(len(C_new)):
        c_new = C_new[j]
    
        for i in range(len(t_c_new)):
            ratio[i] = t_c_new[i] / t_c
            xt_new[i,0:len(upper_x)] = upper_x * c_new
            yt_new[i,0:len(upper_y)] = upper_y * ratio[i] * c_new
            p[i,0:10] = np.polyfit(xt_new[i,:],yt_new[i,:],9)
            p_[i,0:len(xt_new[i,:])] = np.polyval(p[i,:],xt_new[i,:])
            derp[i,0:9] = np.polyder(p[i,:])
            derp_[i,0:len(xt_new[i,:])] = np.polyval(derp[i,:],xt_new[i,:])
            der2p[i,0:8] = np.polyder(derp[i,:])
            der2p_[i,0:len(xt_new[i,:])] = np.polyval(der2p[i,:],xt_new[i,:])
    
            for k in range(len(xt_new[i,:])):
                R[i,k] = (1+(derp_[i,k])**2)**(3/2)/der2p_[i,k]
    
            Rmean[i,j] = np.mean(R[i,:])
            
    x = C_new.reshape(len(Rmean[0]),1)
    y = t_c_new.reshape(1,len(Rmean))
    Z = np.transpose(Rmean)
    #[x,y] = np.meshgrid(X,Y);
    # fitsurface = np.fit([x(:),y(:)],z(:),'poly55');
    
    # create a 2D-mesh
    X,Y = np.meshgrid(x,y)
    
    # calculate polynomial features based on the input mesh
    features = {}
    features['x^0*y^0'] = np.matmul(x**0,y**0).flatten()
    features['x*y'] = np.matmul(x,y).flatten()
    features['x*y^2'] = np.matmul(x,y**2).flatten()
    features['x^2*y^0'] = np.matmul(x**2, y**0).flatten()
    features['x^2*y'] = np.matmul(x**2, y).flatten()
    features['x^3*y^2'] = np.matmul(x**3, y**2).flatten()
    features['x^3*y'] = np.matmul(x**3, y).flatten()
    features['x^0*y^3'] = np.matmul(x**0, y**3).flatten()
    features['x*y^0'] = np.matmul(x, y**0).flatten()
    features['x^0*y'] = np.matmul(x**0, y).flatten()
    features['x^0*y^2'] = np.matmul(x**0, y**2).flatten()
    features['x^3*y^0'] = np.matmul(x**3, y**0).flatten()
    features['x^4*y^0'] = np.matmul(x**4, y**0).flatten()
    features['x^2*y^2'] = np.matmul(x**2, y**2).flatten()
    features['x^1*y^3'] = np.matmul(x**1, y**3).flatten()
    features['x^0*y^4'] = np.matmul(x**0, y**4).flatten()
    features['x^5*y^0'] = np.matmul(x**5, y**0).flatten()
    features['x^4*y^1'] = np.matmul(x**4, y**1).flatten()
    features['x^2*y^3'] = np.matmul(x**2, y**3).flatten()
    features['x^1*y^4'] = np.matmul(x**1, y**4).flatten()
    features['x^0*y^5'] = np.matmul(x**0, y**5).flatten()
    dataset = pd.DataFrame(features)
    
    # fit a linear regression model
    reg = LinearRegression(fit_intercept=False).fit(dataset.values, Z.flatten())
    # get coefficients and calculate the predictions 
    Rpred = reg.coef_.reshape(-1,1)
    
    return Rpred