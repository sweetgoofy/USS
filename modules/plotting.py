from matplotlib import pyplot as plt
# 3d map
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def surface(data):
    ''' Surface plot helper'''
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    xv, yv = np.meshgrid(x, y)
    fig = plt.figure()
    ax = plt.gca(projection = '3d')
    ax.plot_surface(xv, yv, data, cmap = cm.coolwarm, rstride = len(y)/10, cstride = len(x)/10, alpha = 0.6)
    plt.show()



def wireframe(data):
    ''' Wireframe plot helper'''
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    xv, yv = np.meshgrid(x, y)
    fig = plt.figure()
    ax = plt.gca(projection = '3d')
    ax.plot_wireframe(xv, yv,
                      data, rstride = len(y)/50,
                      cstride = len(x)/50)
    plt.show()

def plot(data, xy_plot, *args, **kwargs):
    ''' General plot helper'''
    print args, kwargs
    if xy_plot:
        x = data[0]
        for y in data[1:]:
            plt.plot(x, y, *args, **kwargs)
    else:
        for y in data:
            plt.plot(y, *args, **kwargs)
    plt.show()
            
            
