# Imports
import argparse
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()

from tensorflow.python.client import device_lib
device_lib.list_local_devices()
# To test obstacle-related functions
from path_prediction.obstacles import image_to_world_xy,raycast,generate_obstacle_polygons,load_image_obstacle_polygons, load_world_obstacle_polygons
import matplotlib.pyplot as plt


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets/',
                        help='glob expression for data files')
    parser.add_argument('--dataset_id', '--id',
                    type=int, default=0,help='dataset id (default: 0)')
    args = parser.parse_args()

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices)>0:
        print('[INF] Using GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("[INF] Using CPU")

    dataset_dir   = args.path
    dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']

    # Load the dataset and perform the split
    idTest = args.dataset_id
    dataset_name = dataset_names[idTest]


    # Determine a list of obstacles for this dataset, from the semantic map and save the results
    generate_obstacle_polygons(dataset_dir,dataset_name)
    # Load the saved obstacles
    obstacles_world = load_world_obstacle_polygons(dataset_dir,dataset_name)

    # Draw obstacles
    for obst in obstacles_world:
        plt.plot(obst[:,0],obst[:,1],"g-")
    # Sample a random position
    xpos = np.random.normal(5.0,3.0)
    ypos = np.random.normal(5.0,3.0)

    for t in range(0,21):
        tpos = t*0.3
        c = np.cos(tpos)
        s = np.sin(tpos)
        plt.plot(xpos,ypos,"ro")
        plt.plot([xpos,xpos+2.0*c],[ypos,ypos+2.0*s],"r-")

        # Test for ray casting: first check if some polygons do intersect the ray.
        # If so, plot in red
        omin,imin,dmin,inters = raycast([xpos,ypos,tpos],obstacles_world)
        if imin>=0:
            plt.plot(obstacles_world[omin][imin:imin+2,0],obstacles_world[omin][imin:imin+2,1],"b-")
            plt.plot(inters[0],inters[1],"bo")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()
