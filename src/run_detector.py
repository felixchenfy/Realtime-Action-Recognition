
import numpy as np
import cv2
import sys, os, time, argparse, logging
import simplejson
import argparse
import math

import mylib.io as myio
from mylib.displays import drawActionResult
import mylib.funcs as myfunc
import mylib.feature_proc as myproc 
from mylib.action_classifier import *

# PATHS ==============================================================

CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"


# INPUTS ==============================================================

def parse_input_method():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=False, default='folder',
        help="Choose from (1) webcam, (2) folder, or (3) txtscript")
    return parser.parse_args().source
 
arg_input = parse_input_method()
FROM_WEBCAM = arg_input == "webcam" # from web camera
FROM_TXTSCRIPT = arg_input == "txtscript" # from a txt script (used for training)
FROM_FOLDER = arg_input == "folder" # read images from a folder

# PATHS and SETTINGS =================================

if FROM_WEBCAM:
    save_idx = "4"
    DO_INFER_ACTIONS =  True
    SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = True
    image_size = "304x240" # 14 fps
    # image_size = "240x208" # 14 fps
    OpenPose_MODEL = ["mobilenet_thin", "cmu"][0]

elif FROM_FOLDER:
    save_idx = "5"
    DO_INFER_ACTIONS =  True
    SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = True
    def set_source_images_from_folder():
        # return CURR_PATH + "../data_test/apple/", 1
        # return CURR_PATH + "../data_test/mytest/", 0
        return "/home/qiancheng/DISK/feiyu/TrainYolo/data_yolo/video_bottle/images/", 2
    SRC_IMAGE_FOLDER, SKIP_NUM_IMAGES = set_source_images_from_folder()
    save_idx += SRC_IMAGE_FOLDER.split('/')[-2] # plus file name
    image_size = "304x240" # 14 fps
    OpenPose_MODEL = ["mobilenet_thin", "cmu"][1]

elif FROM_TXTSCRIPT: 
    save_idx = "3"
    DO_INFER_ACTIONS =  False # load groundth data, so no need for inference
    SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = True
    SRC_IMAGE_FOLDER = CURR_PATH + "../data/source_images"+save_idx+"/"
    VALID_IMAGES_TXT = "valid_images.txt"
    image_size = "432x368" # 7 fps
    OpenPose_MODEL = ["mobilenet_thin", "cmu"][1]

else:
    assert False

if DO_INFER_ACTIONS:
    LOAD_MODEL_PATH = CURR_PATH + "../model/trained_classifier.pickle"
    action_labels=  ['jump','kick','punch','run','sit','squat','stand','walk','wave']

if SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE:
    SKELETON_FOLDER = CURR_PATH + "skeleton_data/"
    SAVE_DETECTED_SKELETON_TO =         CURR_PATH + "skeleton_data/skeletons"+save_idx+"/"
    SAVE_DETECTED_SKELETON_IMAGES_TO =  CURR_PATH + "skeleton_data/skeletons"+save_idx+"_images/"
    SAVE_IMAGES_INFO_TO =               CURR_PATH + "skeleton_data/images_info"+save_idx+".txt"

DRAW_FPS = True

# create folders ==============================================================
if not os.path.exists(SKELETON_FOLDER):
    os.makedirs(SKELETON_FOLDER)
if not os.path.exists(SAVE_DETECTED_SKELETON_TO):
    os.makedirs(SAVE_DETECTED_SKELETON_TO)
if not os.path.exists(SAVE_DETECTED_SKELETON_IMAGES_TO):
    os.makedirs(SAVE_DETECTED_SKELETON_IMAGES_TO)

# Openpose ==============================================================

sys.path.append(CURR_PATH + "githubs/tf-pose-estimation")
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# Human pose detection ==============================================================

class SkeletonDetector(object):
    # This func is mostly copied from https://github.com/ildoonet/tf-pose-estimation

    def __init__(self, model=None, image_size=None):
        
        if model is None:
            model = "cmu"

        if image_size is None:
            image_size = "432x368" # 7 fps
            # image_size = "336x288"
            # image_size = "304x240" # 14 fps

        models = set({"mobilenet_thin", "cmu"})
        self.model = model if model in models else "mobilenet_thin"
        # parser = argparse.ArgumentParser(description='tf-pose-estimation run')
        # parser.add_argument('--image', type=str, default='./images/p1.jpg')
        # parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

        # parser.add_argument('--resize', type=str, default='0x0',
        #                     help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
        # parser.add_argument('--resize-out-ratio', type=float, default=4.0,
        #                     help='if provided, resize heatmaps before they are post-processed. default=1.0')
        self.resize_out_ratio = 4.0

        # args = parser.parse_args()

        w, h = model_wh(image_size)
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(self.model),
                                target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))

        # self.args = args
        self.w, self.h = w, h
        self.e = e
        self.fps_time = time.time()
        self.cnt_image = 0

    def detect(self, image):
        self.cnt_image += 1
        if self.cnt_image == 1:
            self.image_h = image.shape[0]
            self.image_w = image.shape[1]
            self.scale_y = 1.0 * self.image_h / self.image_w
        t = time.time()

        # Inference
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0),
                                #   upsample_size=self.args.resize_out_ratio)
                                  upsample_size=self.resize_out_ratio)

        # Print result and time cost
        elapsed = time.time() - t
        logger.info('inference image in %.4f seconds.' % (elapsed))

        return humans
    
    def draw(self, img_disp, humans):
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)

        # logger.debug('show+')
        if DRAW_FPS:
            cv2.putText(img_disp,
                        # "Processing speed: {:.1f} fps".format( (1.0 / (time.time() - self.fps_time) )),
                        "fps = {:.1f}".format( (1.0 / (time.time() - self.fps_time) )),
                        (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        self.fps_time = time.time()

    def humans_to_skelsList(self, humans, scale_y = None): # get (x, y * scale_y)
        if scale_y is None:
            scale_y = self.scale_y
        skeletons = []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(18*2)
            for i, body_part in human.body_parts.items(): # iterate dict
                idx = body_part.part_idx
                skeleton[2*idx]=body_part.x
                skeleton[2*idx+1]=body_part.y * scale_y
            skeletons.append(skeleton)
        return skeletons
    


# ==============================================================

LENGTH_OF_IMAGE_INFO = 5 # see mylib/myio.py: [cnt_action, cnt_clip, cnt_image, img_action_type, filepath]

class DataLoader_usbcam(object):
    def __init__(self, max_framerate = 10):
        self.cam = cv2.VideoCapture(0)
        self.num_images = 9999999
        self.frame_period = 1.0/max_framerate*0.999
        self.prev_image_time = time.time() - self.frame_period

    def wait_for_framerate(self):
        t_curr = time.time()
        t_wait = self.frame_period - (t_curr - self.prev_image_time)
        if t_wait > 0:
            time.sleep(t_wait)

    def load_next_image(self):
        self.wait_for_framerate()
        
        ret_val, img = self.cam.read()
        self.prev_image_time = time.time()

        img =cv2.flip(img, 1)
        img_action_type = "unknown"
        return img, img_action_type, ["none"]*LENGTH_OF_IMAGE_INFO

class DataLoader_folder(object):
    def __init__(self, folder, num_skip = 0):
        self.cnt_image = 0
        self.folder = folder
        self.filenames = myfunc.get_filenames(folder, sort = True)
        self.idx_step = num_skip + 1
        self.num_images = int( len(self.filenames) / self.idx_step)

    def load_next_image(self):
        img =  cv2.imread(self.folder + self.filenames[self.cnt_image])
        self.cnt_image += self.idx_step
        img_action_type = "unknown"
        return img, img_action_type, ["none"]*LENGTH_OF_IMAGE_INFO

class DataLoader_txtscript(object):
    def __init__(self, SRC_IMAGE_FOLDER, VALID_IMAGES_TXT):
        self.images_info = myio.collect_images_info_from_source_images(SRC_IMAGE_FOLDER, VALID_IMAGES_TXT)
        self.imgs_path = SRC_IMAGE_FOLDER
        self.i = 0
        self.num_images = len(self.images_info)
        print("Reading images from txtscript: {}\n".format(SRC_IMAGE_FOLDER))
        print("Reading images information from: {}\n".format(VALID_IMAGES_TXT))
        print("    Num images = {}\n".format(self.num_images))

    def save_images_info(self, path):
        with open(path, 'w') as f:
            simplejson.dump(self.images_info, f)

    def load_next_image(self):
        self.i += 1
        filename = self.get_filename(self.i)
        img = self.imread(self.i)
        img_action_type = self.get_action_type(self.i)
        return img, img_action_type, self.get_image_info(self.i)

    def imread(self, index):
        return cv2.imread(self.imgs_path + self.get_filename(index))
    
    def get_filename(self, index):
        # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.png"]
        # See "myio.collect_images_info_from_source_images" for the data format
        return self.images_info[index-1][4] 

    def get_action_type(self, index):
        # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.png"]
        # See "myio.collect_images_info_from_source_images" for the data format
        return self.images_info[index-1][3]
    
    def get_image_info(self, index):
        return self.images_info[index-1] # with a length of LENGTH_OF_IMAGE_INFO

class OneObjTracker(object):
    def __init__(self):
        self.reset()

    def track(self, skeletons):
        N = len(skeletons)
        if self.prev_skel is None:
            res_idx = 0 # default is zero
        else:
            
            dists = [0]*N
            for i, skel in enumerate(skeletons):
                dists[i] = self.measure_dist(self.prev_skel, skel)
            min_dist = min(dists)
            min_idx = dists.index(min_dist)
            res_idx = min_idx
        self.prev_skel = skeletons[res_idx].copy()
        return res_idx

    def measure_dist(self, prev_skel, curr_skel):
        neck = 1*2
        dx2 = (prev_skel[neck] - curr_skel[neck])**2
        dy2 = (prev_skel[neck+1] - curr_skel[neck+1])**2
        return math.sqrt(dx2+dy2)

    def reset(self):
        self.prev_skel = None

if __name__ == "__main__":
 
    # -- Detect sekelton
    my_detector = SkeletonDetector(OpenPose_MODEL, image_size)

    # -- Load images
    if FROM_WEBCAM:
        images_loader = DataLoader_usbcam()

    elif FROM_FOLDER:
        images_loader = DataLoader_folder(SRC_IMAGE_FOLDER, SKIP_NUM_IMAGES)

    elif FROM_TXTSCRIPT:
        images_loader = DataLoader_txtscript(SRC_IMAGE_FOLDER, VALID_IMAGES_TXT)
        images_loader.save_images_info(path =  SAVE_IMAGES_INFO_TO)

    # -- Classify action
    if DO_INFER_ACTIONS:
        create_classifier = lambda: MyClassifier(
            LOAD_MODEL_PATH,
            action_types = action_labels
            
        )
        humanID_to_classifier = {} 
    tracker = OneObjTracker() # Currently, only supports for 1 person.

    # -- Loop through all images
    ith_img = 1
    while ith_img <= images_loader.num_images:
        img, img_action_type, img_info = images_loader.load_next_image()
        image_disp = img.copy()

        print("\n\n========================================")
        print("\nProcessing {}/{}th image\n".format(ith_img, images_loader.num_images))

        # -- Detect all people's skeletons
        humans = my_detector.detect(img)
        skeletons = my_detector.humans_to_skelsList(humans)

        # -- Track people
        idx_valid_skel, curr_humanIDs = multiperson_tracker.track(skeletons)

        # -- Recognize action for each person
        if DO_INFER_ACTIONS:
            predicted_labels = multipeople_classifier(curr_humanIDs, [skeletons[i] for i in idx_valid_skel])
            print("prediced label is :", predicted_labels[0])
        else:
            print("Ground_truth label is :", img_action_type)


        # -- Draw
        my_detector.draw(image_disp, humans) # Draw all skeletons
        if DO_INFER_ACTIONS: # Draw outer box and label for each person 
            skeletons_no_skew = my_detector.humans_to_skelsList(humans, scale_y = 1.0)
            for i, idx_valid in enumerate(idx_valid_skel):
                drawActionResult(image_disp, skeletons_no_skew[idx_valid], predicted_labels[i])

        if DO_INFER_ACTIONS: # Draw predicting score for only 1 person
            classifier.draw_scores_onto_image(image_disp)


        '''

        list_humanID_and_label = multiperson_action_recognizer.recognize_humans(skeletons, do_recognition = DO_INFER_ACTIONS)



        for ith_skeleton, id in enumerate(curr_humanIDs):
            if id not in humanID_to_classifier:
                humanID_to_classifier[id]=create_classifier()
            humanID_to_classifier[id].classify(skeletons[ith_skeleton])

            else:    
                humanID_to_classifier.pop(id)

        
        # -- Clear: TODO should be put into multipeople_classifier
        def delete_people_not_in_view(curr_humanIDs, humanID_to_classifier):
            curr_IDs = set(curr_humanIDs)
            for prev_ID in humanID_to_classifier.keys:
                if prev_ID not in curr_IDs: 
                    humanID_to_classifier.pop(prev_ID)
        delete_people_not_in_view(curr_humanIDs, humanID_to_classifier)

        '''


        # -- Write skeleton.txt and image.png
        if SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE:

            skel_to_save = [img_info + skeletons[i].tolist() \
                for i, ID in idx_valid_skel if ID is not None]


            myio.save_skeletons(SAVE_DETECTED_SKELETON_TO 
                + myfunc.int2str(ith_img, 5)+".txt", skel_to_save)
            cv2.imwrite(SAVE_DETECTED_SKELETON_IMAGES_TO 
                + myfunc.int2str(ith_img, 5)+".png", image_disp)

            if FROM_TXTSCRIPT or FROM_WEBCAM: # Save source image
                cv2.imwrite(SAVE_DETECTED_SKELETON_IMAGES_TO
                    + myfunc.int2str(ith_img, 5)+"_src.png", img)

        # -- Display
        if 1:
            image_disp = cv2.resize(image_disp,(0,0),fx=1.5,fy=1.5) # resize to make picture bigger
            cv2.imshow("action_recognition", image_disp)
            q = cv2.waitKey(1)
            if q!=-1 and chr(q) == 'q':
                break

        # -- Loop
        print("\n")
        ith_img += 1

