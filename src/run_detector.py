
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
from mylib.action_classifier import MyClassifier
from mylib.action_classifier import * # Import sklearn related libraries

CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"


# INPUTS ==============================================================

def parse_input_method():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=False, default='webcam',
        help="Choose from (1) webcam, (2) folder, or (3) txtscript")
    return parser.parse_args().source
 
arg_input = parse_input_method()
FROM_WEBCAM = arg_input == "webcam" # from web camera
FROM_TXTSCRIPT = arg_input == "txtscript" # from a txt script (for training)
FROM_FOLDER = arg_input == "folder" # read images from a folder

# PATHS and SETTINGS =================================

if FROM_WEBCAM:
    folder_suffix = "3"
    DO_INFER_ACTIONS =  True
    SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = True
    image_size = "304x240" # 14 fps
    # image_size = "240x208" # > 14 fps
    OpenPose_MODEL = ["mobilenet_thin", "cmu"][0]

elif FROM_FOLDER:
    folder_suffix = "4"
    DO_INFER_ACTIONS =  True
    SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = True
    def set_source_images_from_folder():
        return CURR_PATH + "../data_test/apple/", 1
        # return CURR_PATH + "../data_test/mytest/", 0
        # return "/home/qiancheng/DISK/feiyu/TrainYolo/data_yolo/video_bottle/images/", 2
    SRC_IMAGE_FOLDER, SKIP_NUM_IMAGES = set_source_images_from_folder()
    folder_suffix += SRC_IMAGE_FOLDER.split('/')[-2] # plus folder name
    # image_size = "304x240"
    image_size = "240x208"
    OpenPose_MODEL = ["mobilenet_thin", "cmu"][1]

elif FROM_TXTSCRIPT: 
    folder_suffix = "5"
    DO_INFER_ACTIONS =  False # load groundth data, so no need for inference
    SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = True
    SRC_IMAGE_FOLDER = CURR_PATH + "../data/source_images3/"
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
    SAVE_DETECTED_SKELETON_TO =         CURR_PATH + "skeleton_data/skeletons"+folder_suffix+"/"
    SAVE_DETECTED_SKELETON_IMAGES_TO =  CURR_PATH + "skeleton_data/skeletons"+folder_suffix+"_images/"
    SAVE_IMAGES_INFO_TO =               CURR_PATH + "skeleton_data/images_info"+folder_suffix+".txt"

DRAW_FPS = True

# create folders for saving results ==============================================================
if not os.path.exists(SKELETON_FOLDER):
    os.makedirs(SKELETON_FOLDER)
if not os.path.exists(SAVE_DETECTED_SKELETON_TO):
    os.makedirs(SAVE_DETECTED_SKELETON_TO)
if not os.path.exists(SAVE_DETECTED_SKELETON_IMAGES_TO):
    os.makedirs(SAVE_DETECTED_SKELETON_IMAGES_TO)

# Openpose include files==============================================================

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


# Openpose Human pose detection ==============================================================

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
        # type: humans: returned from self.detect()
        # rtype: list[list[]]
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
        return skeletons, scale_y
    


# ==============================================================
class MultiPersonClassifier(object):
    def __init__(self, LOAD_MODEL_PATH, action_labels):
        self.create_classifier = lambda: MyClassifier(
            LOAD_MODEL_PATH, action_types = action_labels)
        self.dict_id2clf = {} # human id -> classifier of this person

    def classify(self, dict_id2skeleton):

        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():
            
            if id not in self.dict_id2clf: # add this new person
                self.dict_id2clf[id] = self.create_classifier()
            
            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton) # predict label
            print("\n\nPredicting label for human{}".format(id))
            print("  skeleton: {}".format(skeleton))
            print("  label: {}".format(id2label[id]))

        return id2label

    def get(self, id):
        # type: id: int or "min"
        if len(self.dict_id2clf) == 0:
            return None 
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]


if __name__ == "__main__":
 
    # -- Detect sekelton
    my_detector = SkeletonDetector(OpenPose_MODEL, image_size)

    # -- Load images
    if FROM_WEBCAM:
        images_loader = myio.DataLoader_usbcam()

    elif FROM_FOLDER:
        images_loader = myio.DataLoader_folder(SRC_IMAGE_FOLDER, SKIP_NUM_IMAGES)

    elif FROM_TXTSCRIPT:
        images_loader = myio.DataLoader_txtscript(SRC_IMAGE_FOLDER, VALID_IMAGES_TXT)
        images_loader.save_images_info(path =  SAVE_IMAGES_INFO_TO)

    # -- Initialize human tracker and action classifier
    if DO_INFER_ACTIONS:
        multipeople_classifier = MultiPersonClassifier(LOAD_MODEL_PATH, action_labels)
    multiperson_tracker = myfunc.Tracker()

    # -- Loop through all images
    ith_img = 1
    while ith_img <= images_loader.num_images:
        img, img_action_type, img_info = images_loader.load_next_image()
        image_disp = img.copy()

        print("\n\n========================================")
        print("\nProcessing {}/{}th image\n".format(ith_img, images_loader.num_images))

        # -- Detect all people's skeletons
        humans = my_detector.detect(img)
        skeletons, scale_y = my_detector.humans_to_skelsList(humans)

        # -- Track people
        dict_id2skeleton = multiperson_tracker.track(skeletons) # int id -> np.array() skeleton

        # -- Recognize action for each person
        if len(dict_id2skeleton):
            if DO_INFER_ACTIONS:
                min_id = min(dict_id2skeleton.keys())
                dict_id2label = multipeople_classifier.classify(dict_id2skeleton)
                print("prediced label is :", dict_id2label[min_id])
            else: # reserve only one skeleton
                min_id = min(dict_id2skeleton.keys()) 
                dict_id2skeleton = {min_id : dict_id2skeleton[min_id]}
                dict_id2label = {min_id : img_action_type}
                print("Ground_truth label is :", dict_id2label[min_id])


        # -- Draw
        my_detector.draw(image_disp, humans) # Draw all skeletons
        if len(dict_id2skeleton): 
            
            # Draw outer box and label for each person 
            for id, label in dict_id2label.items():
                skeleton = dict_id2skeleton[id]
                skeleton[1::2] = skeleton[1::2] / scale_y # scale the y data back to original
                # print("Drawing skeleton: ", dict_id2skeleton[id], "with label:", label, ".")
                drawActionResult(image_disp, id, skeleton, label)

            # Draw predicting score for only 1 person (not using for)
            if DO_INFER_ACTIONS:
                multipeople_classifier.get(id='min').draw_scores_onto_image(image_disp)


        # -- Write skeleton.txt and image.png
        if SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE:

            ids = sorted(dict_id2skeleton.keys())
            skel_to_save = [img_info + dict_id2skeleton[id].tolist() for id in ids]


            myio.save_skeletons(SAVE_DETECTED_SKELETON_TO 
                + myfunc.int2str(ith_img, 5) + ".txt", skel_to_save)
            cv2.imwrite(SAVE_DETECTED_SKELETON_IMAGES_TO 
                + myfunc.int2str(ith_img, 5) + ".png", image_disp)

            if FROM_TXTSCRIPT or FROM_WEBCAM: # Save source image
                cv2.imwrite(SAVE_DETECTED_SKELETON_IMAGES_TO
                    + myfunc.int2str(ith_img, 5) + "_src.png", img)

        # -- Display
        if 1:
            image_disp = cv2.resize(image_disp, (0,0), fx=1.5, fy=1.5) # resize to make picture bigger
            cv2.imshow("action_recognition", image_disp)
            q = cv2.waitKey(1)
            if q != -1 and chr(q) == 'q':
                break

        # -- Loop
        print("\n")
        ith_img += 1

