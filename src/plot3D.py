import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def plot3D(data=[], labels=[]):
    # font size of the legend
    matplotlib.rcParams['legend.fontsize'] = 10

    # create image object
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot something
    for datum, label in zip(data, labels):
        ax.plot(datum[:,0], datum[:,1], datum[:,2], label=label)
    
    # figure setting
    #ax.set_xlim(0, 2)
    #ax.set_ylim(0, 10)
    #ax.set_zlim(0, 10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()

    # plot figure
    plt.show()
    plt.savefig('../res/3D.png')


# read data
with open('../res/g_3D.txt') as f:
    g_datum = []
    for line in f.readlines():
        line = map(float, line.strip().split(','))
        g_datum.append(line)
    g_datum = np.array(g_datum)

with open('../res/v_3D.txt') as f:
    v_datum = []
    for line in f.readlines():
        line = map(float, line.strip().split(','))
        v_datum.append(line)
    v_datum = np.array(v_datum)

data = [g_datum, v_datum]
labels = ['Ground Truth', 'VINet']


# plot data
plot3D(data, labels)
