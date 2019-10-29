'''
This script defines the class `SkeletonDetector`,
which is used for detecting human skeleton from image.

The code is copied and modified from src/githubs/tf-pose-estimation
'''

# -- Libraries
if True: # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

import sys, os, time, argparse, logging
import cv2

# openpose packages
sys.path.append(ROOT + "src/githubs/tf-pose-estimation")
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common


# -- Settings
MAX_FRACTION_OF_GPU_TO_USE = 0.4
IS_DRAW_FPS = True

# -- Helper functions
def _set_logger():
    logger = logging.getLogger('TfPoseEstimator')
    logger.setLevel(logging.DEBUG)
    logging_stream_handler = logging.StreamHandler()
    logging_stream_handler.setLevel(logging.DEBUG)
    logging_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    logging_stream_handler.setFormatter(logging_formatter)
    logger.addHandler(logging_stream_handler)
    return logger

def _set_config():
    ''' Set the max GPU memory to use '''
    # For tf 1.13.1, The following setting is needed
    import tensorflow as tf
    from tensorflow import keras
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction=MAX_FRACTION_OF_GPU_TO_USE
    return config

def _get_input_img_size_from_string(image_size_str):
    ''' If input image_size_str is "123x456", then output (123, 456) '''
    width, height = map(int, image_size_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)



# -- Main class

class SkeletonDetector(object):
    # This class is mainly copied from https://github.com/ildoonet/tf-pose-estimation

    def __init__(self, model="cmu", image_size="432x368"):
        ''' Arguments:
            model {str}: "cmu" or "mobilenet_thin".        
            image_size {str}: resize input images before they are processed. 
                Recommends : 432x368, 336x288, 304x240, 656x368, 
        '''
        # -- Check input
        assert(model in ["mobilenet_thin", "cmu"])
        self._w, self._h = _get_input_img_size_from_string(image_size)
        
        # -- Set up openpose model
        self._model = model
        self._resize_out_ratio = 4.0 # Resize heatmaps before they are post-processed. If image_size is small, this should be large.
        self._config = _set_config()
        self._tf_pose_estimator = TfPoseEstimator(
            get_graph_path(self._model), 
            target_size=(self._w, self._h),
            tf_config=self._config)
        self._prev_t = time.time()
        self._cnt_image = 0
        
        # -- Set logger
        self._logger = _set_logger()
        

    def detect(self, image):
        ''' Detect human skeleton from image.
        Arguments:
            image: RGB image with arbitrary size. It will be resized to (self._w, self._h).
        Returns:
            humans {list of class Human}: 
                `class Human` is defined in 
                "src/githubs/tf-pose-estimation/tf_pose/estimator.py"
                
                The variable `humans` is returned by the function
                `TfPoseEstimator.inference` which is defined in
                `src/githubs/tf-pose-estimation/tf_pose/estimator.py`.

                I've written a function `self.humans_to_skels_list` to 
                extract the skeleton from this `class Human`. 
        '''

        self._cnt_image += 1
        if self._cnt_image == 1:
            self._image_h = image.shape[0]
            self._image_w = image.shape[1]
            self._scale_h = 1.0 * self._image_h / self._image_w
        t = time.time()

        # Do inference
        humans = self._tf_pose_estimator.inference(
            image, resize_to_default=(self._w > 0 and self._h > 0),
            upsample_size=self._resize_out_ratio)

        # Print result and time cost
        elapsed = time.time() - t
        self._logger.info('inference image in %.4f seconds.' % (elapsed))

        return humans
    
    def draw(self, img_disp, humans):
        ''' Draw human skeleton on img_disp inplace.
        Argument:
            img_disp {RGB image}
            humans {a class returned by self.detect}
        '''
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)
        if IS_DRAW_FPS:
            cv2.putText(img_disp,
                        "fps = {:.1f}".format( (1.0 / (time.time() - self._prev_t) )),
                        (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        self._prev_t = time.time()

    def humans_to_skels_list(self, humans, scale_h = None): 
        ''' Get skeleton data of (x, y * scale_h) from humans.
        Arguments:
            humans {a class returned by self.detect}
            scale_h {float}: scale each skeleton's y coordinate (height) value.
                Default: (image_height / image_widht).
        Returns:
            skeletons {list of list}: a list of skeleton.
                Each skeleton is also a list with a length of 36 (18 joints * 2 coord values).
            scale_h {float}: The resultant height(y coordinate) range.
                The x coordinate is between [0, 1].
                The y coordinate is between [0, scale_h]
        '''
        if scale_h is None:
            scale_h = self._scale_h
        skeletons = []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(18*2)
            for i, body_part in human.body_parts.items(): # iterate dict
                idx = body_part.part_idx
                skeleton[2*idx]=body_part.x
                skeleton[2*idx+1]=body_part.y * scale_h
            skeletons.append(skeleton)
        return skeletons, scale_h
    

def test_openpose_on_webcamera():
    
    # -- Initialize web camera reader
    from utils.lib_images_io import ReadFromWebcam, ImageDisplayer
    webcam_reader = ReadFromWebcam(max_framerate=10)
    img_displayer = ImageDisplayer()
    
    # -- Initialize openpose detector    
    skeleton_detector = SkeletonDetector("mobilenet_thin", "432x368")

    # -- Read image and detect
    import itertools
    for i in itertools.count():
        img = webcam_reader.read_image()
        if img is None:
            break
        print(f"Read {i}th image...")

        # Detect
        humans = skeleton_detector.detect(img)
        
        # Draw
        img_disp = img.copy()
        skeleton_detector.draw(img_disp, humans)
        img_displayer.display(img_disp)
        
    print("Program ends")

if __name__ == "__main__":
    test_openpose_on_webcamera()