"""
Visualization script for part 1 (Barcelona)

Note: You don't have to change this script for the assignment, but you
can if you'd like to change the images or other parameters

"""

import os
import glob
import numpy as np
import cv2

from warp_pts import warp_pts
import utils


def main():
    # Load Penn logo image, and get its corners
    penn = cv2.imread("../data/barcelona/images/logos/penn_engineering_logo.png")
    penn_y, penn_x, _ = penn.shape
    penn_corners = np.array([[0, 0], [penn_x, 0], [penn_x, penn_y], [0, penn_y]])

    # Load all image paths, and the goal corners in each image
    img_files = sorted(glob.glob("../data/barcelona/images/barca_real/*.png"))
    goal_data = np.load("../data/barcelona/BarcaReal_pts.npy")

    print(goal_data)
    # Process all images
    processed_imgs = []
    for i in range(len(goal_data)):
        goal = cv2.imread(img_files[i])
        goal_corners = goal_data[i]
        # Warping
        int_pts = utils.calculate_interior_pts(goal.shape, goal_corners)
        warped_pts = warp_pts(goal_corners, penn_corners, int_pts)
        projected_img = utils.inverse_warping(goal, penn, int_pts, warped_pts)

        processed_imgs.append(projected_img)

    # Save some examples
    save_ind = [0, 25, 50, 75, 100, 125]
    if not os.path.exists("part_1_results"):
        os.mkdir("part_1_results")

    for ind in save_ind:
        cv2.imwrite("part_1_results/frame_" + str(ind) + ".png", processed_imgs[ind])

    # Visualize the sequence of projected images
    for im in processed_imgs:
        cv2.imshow("display", im)
        cv2.waitKey(200)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
