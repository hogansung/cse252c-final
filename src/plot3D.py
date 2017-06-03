import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def plot3D(data=[], labels=[]):
    # font size of the legend
    mpl.rcParams['legend.fontsize'] = 10

    # create image object
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # plot something
    for datum, label in zip(data, labels):
        ax.plot(datum[:,0], datum[:,1], datum[:,2], label=label)
    
    # figure setting
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()

    # plot figure
    plt.show()


# read data
files = []
labels = []
for fname in files:
    with open(fname) as f:
        f.readlines()
#data = [np.array([[1,1,1], [1,2,2], [1,3,3]])]
#labels = ['parametric curve']

# plot data
plot3D(data, labels)
