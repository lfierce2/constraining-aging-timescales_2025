#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 08:59:23 2018

@author: fiercenator
"""
import numpy as np
def findnth(haystack, needle, n):
    parts= haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def ismember(A, B):
    return np.array([ np.sum(a == B) for a in A ])

def cc(arg):
    '''
    Shorthand to convert 'named' colors to rgba format at 60% opacity.
    '''
    return mcolors.to_rgba(arg, alpha=0.6)


def polygon_under_graph(xlist, ylist):
    '''
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    '''
    return [(xlist[0], 0.)] + list(zip(xlist, ylist)) + [(xlist[-1], 0.)]


def make_num_id(ii):
    import numpy as np
    if np.log10(ii) >= 4:
        print('error: ii>9999')
    elif np.log10(ii) >= 3:
        num_id = str(ii)
    elif np.log10(ii) >= 2:
        num_id = '0' + str(ii)
    elif np.log10(ii) >= 1:
        num_id = '00' + str(ii)
    else:
        num_id = '000' + str(ii)
    return num_id

def scatter3d(x,y,z, cs, colorsMap='jet'):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cmx
    from matplotlib import colors
    
    cm = plt.get_cmap(colorsMap)
    cNorm = colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs),s=10)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.show()
    
def is_empty(value):
    try:
        value = float(value)
    except ValueError:
        pass
    return bool(value)

def gaussian_kernel_scalar(x, x_i, hx):
    import numpy as np
    K = np.exp(-((x-x_i)/hx)**2/2);
    if abs(x-x_i) > hx*10.:
        K = 0.0
    return K

def gaussian_kernel(x, x_i, hx):
    import numpy as np
    K = np.exp(-((x-x_i)/hx)**2/2);
    too_big = abs(x-x_i) > hx*100.
    K[too_big] = 0.0
    return K

def silvermans_rule_of_thumb(x_i):
    import numpy as np
    if len(x_i.shape)==2:
        return 1.06*np.std(x_i)*(x_i.shape[0]*x_i.shape[1])**(-1/5)
    elif len(x_i.shape)==1:
        return 1.06*np.std(x_i)*(x_i.shape[0])**(-1/5)        


