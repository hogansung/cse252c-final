import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

VERSION = str(27) + 'u'

#TARGET = 'kitti_tn_' + str(VERSION)
TARGET = 'kitti_tt_' + str(VERSION)

def plot3D(data=[], labels=[]):
    # font size of the legend
    matplotlib.rcParams['legend.fontsize'] = 10

    # create image object
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot something
    for datum, label in zip(data, labels):
        ax.plot(datum[:200,0], datum[:200,1], datum[:200,2], label=label)
    
    # figure setting
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()

    # plot figure
    plt.show()
    plt.savefig('../res/' + TARGET + '_3D.png')


# read data
with open('../res/' + TARGET + '_g_3D.txt') as f:
    g_datum = []
    for line in f.readlines():
        line = map(float, line.strip().split(','))
        g_datum.append(line)
    g_datum = np.array(g_datum)

with open('../res/' + TARGET + '_v_3D.txt') as f:
    v_datum = []
    for line in f.readlines():
        line = map(float, line.strip().split(','))
        v_datum.append(line)
    v_datum = np.array(v_datum)

data = [g_datum, v_datum]
labels = ['Ground Truth', 'VINet']


# plot data
plot3D(data, labels)
