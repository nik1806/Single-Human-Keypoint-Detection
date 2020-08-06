"""
    convert 2images to skeleton on black background
"""

import glob
import cv2
from WrnchPose import Pose2d
import numpy as np
import argparse

# adding command line interface
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_path', type=str, required=True, help="Input data path")
parser.add_argument('-s', '--save_path', type=str, required=True, help="Path to save skeleton")
arg = parser.parse_args()


data_path = arg.data_path
save_path = arg.save_path

# initialize model
wrnchPose = Pose2d()

# parsing over all images in dataset
for fileN in glob.glob(data_path + '*'):

    filename = fileN.split('/')[-1]

    img = cv2.imread(fileN)

    skeleton = wrnchPose.process_frame_skeleton(img, draw_keypoints=True)

    # comb_frame = np.hstack((img, skeleton))
    # comb_frame = img
    # cv2.imshow("Person pose", comb_frame) # display image

    # k = cv2.waitKey(1) # a small delay to see image
    new_file = save_path + filename
    print(new_file)
    cv2.imwrite(new_file, skeleton)



