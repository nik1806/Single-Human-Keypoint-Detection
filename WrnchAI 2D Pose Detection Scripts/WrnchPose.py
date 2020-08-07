
import cv2
import wrnchAI
import sys
import numpy as np

# CNN model dependencies
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal,constant
from keras.layers import BatchNormalization
from tensorflow.keras import optimizers, regularizers
from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import collections

import math
import time
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from tqdm import tqdm

class Pose2d:
    """ Perform 2D pose estimation """

    def __init__(self, models_dir='/usr/bin/wrModels', video_source=0, width=328, height=184, license_key=None):
        """
            Initialize parameters and load model
        """
        self.models_dir = models_dir
        self.video_source = video_source
        self.width = width
        self.height = height
        self.license_key = license_key
        # load model into system (it will take few minutes)
        self.load_model()
        # drawing paramaters
        self.WRNCH_BLUE = (226.95, 168.3, 38.25)
        self.GREEN = (153, 255, 153)
        self.FILLED = -1
        self.AA_LINE = 16
        # load cnn model for final prediction (rule Engine)
        self.classes = {0:'Correct', 1:'Lean backword', 2:'Lean Forward', 3:'Tilted left', 4:'Tilted right'}
        # self.cnn = tf.keras.models.load_model('../vgg_model_single')
        self.img_size = 224

    def predict_pose(self, img):
        """ Predict the final pose into five categories """
        cur_test = np.zeros((1,self.img_size,self.img_size,3),dtype=np.float32)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation = cv2.INTER_LANCZOS4) #resize
        arr = np.asarray(img)
        cur_test[0,:,:,:] = arr
        cur_test/=255
        prediction = self.cnn.predict(cur_test)
        idx = np.argmax(prediction)
        return self.classes[idx]

    def process_frame_and_predict_pose(self, frame):
        """ Combine pose estimator and rule engine """
        skeleton = self.process_frame_skeleton(frame)
        posture = self.predict_pose(skeleton)
        return posture

    def load_model(self):

        # license key is required for execution
        code = wrnchAI.license_check_string(self.license_key) if self.license_key \
            else wrnchAI.license_check()
        if code != 0:
            raise RuntimeError(wrnchAI.returncode_describe(code))
        
        # settings for pose estimator
        params = wrnchAI.PoseParams()
        params.bone_sensitivity = wrnchAI.Sensitivity.medium
        params.joint_sensitivity = wrnchAI.Sensitivity.medium
        params.enable_tracking = True # 2d pose

        # Default Model resolution
        params.preferred_net_width = self.width
        params.preferred_net_height = self.height

        output_format = wrnchAI.JointDefinitionRegistry.get('j25')

        print('Initializing networks...')
        self.estimator = wrnchAI.PoseEstimator(models_path=self.models_dir,
                                        license_string=self.license_key,
                                        params=params,
                                        gpu_id=0,
                                        output_format=output_format)
        print('Initialization done!')

        self.options = wrnchAI.PoseEstimatorOptions()

        self.joint_definition = self.estimator.human_2d_output_format()
        self.bone_pairs = self.joint_definition.bone_pairs() # indexes for bone pairs
        # adding nose and head connection
        self.bone_pairs.append((self.joint_definition.get_joint_index("NOSE"), self.joint_definition.get_joint_index("NECK")))

    def get_frame(self):
        ''' 
            get recent frame if cap isopen
        '''
        cap = cv2.VideoCapture(self.video_source)

        if not cap.isOpened():
            sys.exit('Cannot open video source')

        try:
            _, frame = cap.read()
            yield frame
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def draw_points(self, frame, points, joint_size=8):
        """ Draw keypoints on the image """

        width = frame.shape[1]
        height = frame.shape[0]
        color = self.GREEN
        for i in range(len(points)//2):
            x = int(points[2 * i] * width)
            y = int(points[2 * i + 1] * height)

            if x >= 0 and y >= 0:
                cv2.circle(frame, (x, y), joint_size,
                           color, self.FILLED, self.AA_LINE)

    def draw_lines(self, frame, points, bone_pairs, bone_width=3):
        """ Draw skeleton lines on image """

        width = frame.shape[1]
        height = frame.shape[0]
        color=self.WRNCH_BLUE

        for joint_idx_0, joint_idx_1 in bone_pairs:
            x1 = int(points[joint_idx_0 * 2] * width)
            y1 = int(points[joint_idx_0 * 2 + 1] * height)
            x2 = int(points[joint_idx_1 * 2] * width)
            y2 = int(points[joint_idx_1 * 2 + 1] * height)

            if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
                cv2.line(frame, (x1, y1), (x2, y2), color,
                         bone_width, self.AA_LINE)

    def process_frame(self, frame):
        '''
            perform keypoint detection on frame and return skeleton drawn on original image
        '''

        # frame = self.get_frame()

        self.estimator.process_frame(frame, self.options)
        # all tracked human
        human2d = self.estimator.humans_2d()

        for human in human2d: # loop over to draw all human # we only use the first one

            joints = human.joints() # store coordinates of joints

            self.draw_points(frame, joints)
            self.draw_lines(frame, joints, self.bone_pairs)

            break # only for first human

        return frame


    def process_frame_skeleton(self, frame, draw_keypoints=False):
        '''
            perform keypoint detection on frame and return skeleton drawn on black background
        '''

        # frame = self.get_frame()

        self.estimator.process_frame(frame, self.options)
        # all tracked human
        human2d = self.estimator.humans_2d()
        # override original image and provide just black background with same size
        frame = np.zeros_like(frame)

        for human in human2d: # loop over to draw all human # we only use the first one

            joints = human.joints() # store coordinates of joints

            if draw_keypoints:
                self.draw_points(frame, joints)
            self.draw_lines(frame, joints, self.bone_pairs)

            break # only for first human

        return frame.copy()


    def process_frame_get_joints(self, frame):
        '''
            perform keypoint detection on frame and return joints coordinates in form (x1, y1, x2, y2)
        '''

        self.estimator.process_frame(frame, self.options)
        # all tracked human
        human2d = self.estimator.humans_2d()

        joints = []

        for human in human2d: # loop over to draw all human # we only use the first one

            points = human.joints() # store coordinates of joints

            width = frame.shape[1]
            height = frame.shape[0]
            
            for joint_idx_0, joint_idx_1 in self.bone_pairs:
                x1 = int(points[joint_idx_0 * 2] * width)
                y1 = int(points[joint_idx_0 * 2 + 1] * height)
                x2 = int(points[joint_idx_1 * 2] * width)
                y2 = int(points[joint_idx_1 * 2 + 1] * height)

                if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
                    joints.append((x1, y1, x2, y2))

            break # only for first human

        return joints


class Pose3d:
    """ Perform 3D pose estimation """

    def __init__(self, models_dir='/usr/bin/wrModels', video_source=0, width=328, height=184, license_key=None):
        """
            Initialize parameters and load model
        """
        self.models_dir = models_dir
        self.video_source = video_source
        self.width = width
        self.height = height
        self.license_key = license_key
        # drawing paramaters
        self.WRNCH_BLUE = (226.95, 168.3, 38.25)
        self.GREEN = (153, 255, 153)
        self.FILLED = -1
        self.AA_LINE = 16
        # load model into system (it will take few minutes)
        self.load_model()

    def load_model(self):
        
        # license key is required for execution 
        code = wrnchAI.license_check_string(self.license_key) if self.license_key \
            else wrnchAI.license_check()
        if code != 0:
            raise RuntimeError(wrnchAI.returncode_describe(code))

        print("Initializing networks...")
        self.estimator = wrnchAI.PoseEstimator(models_path=self.models_dir,
                                        license_string=self.license_key)
        self.estimator.initialize_3d(self.models_dir)
        print("Initialization done!")

        self.options = wrnchAI.PoseEstimatorOptions()
        self.options.estimate_3d = True

    def process_frame(self, frame, draw_skeleton=False):
        """ 
            Perform pose detection and return the skeleton on original or black background  
        """

        self.estimator.process_frame(frame, self.options)
        humans3d = self.estimator.raw_humans_3d()        
        
        # changing original image to black background if flag is set
        if draw_skeleton:
            frame = np.zeros_like(frame)

        for human in humans3d:
            positions = human.positions()
            self.draw_points3d(frame, positions)

            break # only process for first human

        return frame


    def draw_points3d(self, frame, points, joint_size=8):
        width = frame.shape[1]
        height = frame.shape[0]
        color = self.GREEN
        for i in range(len(points)//3):
            x = int(points[3 * i] * width)
            y = int(points[3 * i + 1] * height)
            # z = np.float32(points[3 * i + 2] * height)  Depth is store here

            if x >= 0 and y >= 0:
                cv2.circle(frame, (x, y), joint_size,
                           color, self.FILLED, self.AA_LINE)