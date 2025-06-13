"""
utils functions for part 1 (Barcelona)
STUDENTS SHOULDN'T NEED TO CHANGE THIS

"""

import numpy as np
from matplotlib.path import Path


def calculate_interior_pts(image_size, corners):
    """ 
    Calculate_interior_pts takes in the size of an image and a set of corners
    that define a polygon in the image, and returns all (x,y) points within
    the polygon
    
    Input:
        image_size: size of image in [y,x]
        corners: the four corners of a polygon in [x,y] format
    Returns:
        interior_pts: coordinates of points inside polygon in [x,y] format
        
    """

    # YOU SHOULDN'T NEED TO CHANGE THIS
    path = Path(corners)

    xx, yy = np.meshgrid(range(image_size[1]), range(image_size[0]))
    xxyy = np.stack([xx.ravel(), yy.ravel()], 1)

    interior_ind = path.contains_points(xxyy)
    interior_pts = xxyy[interior_ind]
    
    return interior_pts


def inverse_warping(img_initial, img_final, pts_initial, pts_final):
    """ 
    takes two images and a set of correspondences between them, 
    and warps all the pts_inital in img_initial to the pts_final in img_final

    Input:
        img_initial: initial image on top of which we want to overlay img_final
        img_final:   target image to lay on top of img_initial
        pts_initial: Nx2 matrix of (x,y) coordinates of points in video frame
        pts_final:   Nx2 matrix of (x,y) coordinates of points in penn logo
    Returns:
        projected_img: 
        
    """   
    
    # YOU SHOULDN'T NEED TO CHANGE THIS
    pts_final = pts_final.astype(int)
    pts_initial = pts_initial.astype(int)
    
    projected_img = img_initial.copy()
    for i in range(3):
        sub_img_i = img_initial[:,:,i][pts_initial[:,1], pts_initial[:,0]]
        sub_img_f = img_final[:,:,i][pts_final[:,1], pts_final[:,0]]
        
        # sub_img = sub_img_i*1 + sub_img_f*1
        sub_img = sub_img_f*1 
        projected_img[:,:,i][pts_initial[:,1], pts_initial[:,0]] = sub_img
        
    return projected_img