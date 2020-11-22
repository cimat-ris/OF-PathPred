import cv2
import numpy as np
import os

# Image-to-world mapping
def image_to_world_xy(image_xy, H):
    """Convert image (x, y) position to world (x, y) position.
    This function use the homography for do the transform.

    :param image_xy: polygon image (x, y) positions
    :param H: homography matrix
    :return: world (x, y) positions
    """
    image_xy = np.array(image_xy)
    image_xy1 = np.concatenate([image_xy, np.ones((len(image_xy), 1))],axis=1)
    world_xy1 = H.dot(image_xy1.T).T
    return world_xy1[:, :2] / np.expand_dims(world_xy1[:, 2], axis=1)

# This function generates both image and world coordinates of the obstacle polygons.
# It saves both as files
def generate_obstacle_polygons(dataset_paths,dataset_name):
    # Read the map image
    imagenew    = cv2.imread(dataset_paths+dataset_name+'/annotated.png',cv2.IMREAD_GRAYSCALE) # Gray values

    # Binarize
    r,binary1   = cv2.threshold(imagenew, 1, 255, cv2.THRESH_BINARY)
    kernel      = np.ones((11,11),np.uint8)
    # Apply dilation, then erosion
    dilatacion1 = cv2.dilate(binary1,kernel,iterations = 1)
    erosion1    = cv2.erode(dilatacion1,kernel,iterations=1)

    # Detecting shapes in image by selecting region
    # with same colors or intensity.
    erosion_inv= 255 - erosion1
    contours,_ = cv2.findContours(erosion_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img2       = cv2.imread(dataset_paths+dataset_name+'/annotated.png', cv2.IMREAD_COLOR)

    # Searching through every region selected to find the required polygon.
    num       = 0
    obstacles = []
    for cnt in contours :
        area = cv2.contourArea(cnt)
        # Shortlisting the regions based on there area.
        if area > 200:
            approx = cv2.approxPolyDP(cnt, 2.0, True)
            approx = np.append(approx, [approx[0,:,:]], axis=0)
            cv2.drawContours(img2, [approx], 0, (255, 255, 0), 5)
            # Form the filename and save the file
            filename = "/obstacles-img-{:04d}.txt".format(num)
            np.savetxt(dataset_paths+dataset_name+filename, approx[:,0,:])
            obstacles.append(approx[:,0,:])
            num = num + 1

    # Load the homography corresponding to this dataset
    homography_file = os.path.join(dataset_paths+dataset_name, "H.txt")
    H = np.genfromtxt(homography_file)

    # Map the obstacles in world frame
    num             = 0
    for obs in obstacles:
        if((dataset_name=='eth-univ') or (dataset_name=='eth-hotel')):
            obs = np.column_stack((obs[:,1],obs[:,0]))
        obs_world = image_to_world_xy(obs, H)
        # Save them in the same directory
        filename = "/obstacles-world-{:04d}.txt".format(num)
        np.savetxt(dataset_paths+dataset_name+filename, obs_world)
        num = num+1


# Load obstacle polygons (in image coordinates) from files
def load_image_obstacle_polygons(dataset_paths,dataset_name):
    # Load obstacles
    import glob, os
    obstacles = []
    for file in glob.glob(dataset_paths+dataset_name+"/obstacles-img*.txt"):
        obstacles.append(np.loadtxt(file))
    return obstacles

# Load obstacle polygons (in world coordinates) from files
def load_world_obstacle_polygons(dataset_paths,dataset_name):
    # Load obstacles
    import glob, os
    obstacles = []
    for file in glob.glob(dataset_paths+dataset_name+"/obstacles-world*.txt"):
        obstacles.append(np.loadtxt(file))
    return obstacles

# Ray casting from a given position+orientation, through a set of obstacles
def raycast(pos,obstacles):
    xpos = pos[0]
    ypos = pos[1]
    tpos = pos[2]
    c = np.cos(tpos)
    s = np.sin(tpos)
    # Segment id
    imin  = -1
    # Obstacle id
    omin  = -1
    dmin  = 1000.0
    inters= [0,0]
    for o,obst in enumerate(obstacles):
        # Filter out polygons that cannot be intersecting the ray
        mayintersect = False
        # Position of point 0 with respect to the ray
        sgn = (-s*(obst[0,0]-xpos)+c*(obst[0,1]-ypos)>0)
        for i in np.arange(1,obst.shape[0]):
            sgni = (-s*(obst[i,0]-xpos)+c*(obst[i,1]-ypos)>0)
            if sgni != sgn:
                mayintersect = True
                break
        if mayintersect:
            for i in np.arange(0,obst.shape[0]-1):
                if (c*(obst[i,0]-xpos)+s*(obst[i,1]-ypos)>0 or c*(obst[i+1,0]-xpos)+s*(obst[i+1,1]-ypos)>0):
                    dx = obst[i+1,0]-obst[i,0]
                    dy = obst[i+1,1]-obst[i,1]
                    # Compute intersection with line
                    A = np.array([[c,-dx],[s,-dy]])
                    B = np.array([obst[i,0]-xpos,obst[i,1]-ypos])
                    l = np.linalg.solve(A, B)
                    if l[0]>0 and l[1]>0 and l[1]<1:
                        if l[0]<dmin:
                            dmin   = l[0]
                            imin   = i
                            omin   = o
                            inters = [xpos+dmin*c,ypos+dmin*s]
    return omin,imin,dmin,inters

def main():
    dataset_paths = '../datasets/'
    dataset_name  = 'ucy-univ'
    generate_obstacle_polygons(dataset_paths,dataset_name)


if __name__ == '__main__':
    main()
