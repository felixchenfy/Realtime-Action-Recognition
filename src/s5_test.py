'''
Test action recognition on
(1) a video, (2) a folder of images, (3) or web camera.

Input:
    classes: data_proc/classes.csv # TODO: change this to a config file
    model: model/trained_classifier.pickle

Output:
    result video:    output/${video_name}/video.avi
    result skeleton: output/${video_name}/skeleton_res/XXXXX.txt
    visualization by cv2.imshow() in img_displayer
'''

import numpy as np
import cv2
import argparse

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_images_io as lib_images_io
    import utils.lib_io as lib_io
    import utils.lib_commons as lib_commons
    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_plot import draw_action_result
    from utils.lib_tracker import Tracker
    from utils.lib_classifier import ClassifierOnlineTest
    from utils.lib_classifier import *  # Import all sklearn related libraries

# -- Command-line input


def get_command_line_arguments():

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Test action recognition on \n"
            "(1) a video, (2) a folder of images, (3) or web camera.")
        parser.add_argument("--source", required=False, default='webcam',
                            choices=["video", "folder", "webcam"])
        parser.add_argument("--data_path", required=False, default="",
                            help="path to a video file, or images folder, or index of webcam.\n"
                            "The path should be absolute or relative to this project's root.")
        parser.add_argument("--model_path", required=False,
                            default='model/trained_classifier_tmp.pickle')
        parser.add_argument("--classes_path", required=False,
                            default='data_proc/classes.csv')
        args = parser.parse_args()
        return args
    args = parse_args()
    if args.data_path and args.data_path[0] != "/":
        # If the path is not absolute, then its relative to the ROOT.
        args.data_path = ROOT + args.data_path
    return args


def get_dst_folder_name(src_data_type, src_data_path):
    assert(src_data_type in ["video", "folder", "webcam"])
    if src_data_type == "video":  # /root/data/video.avi --> video
        folder_name = os.path.dirname(src_data_type)
    elif src_data_type == "folder":  # /root/data/video/ --> video
        folder_name = src_data_path.rstrip("/").split("/")[-1]
    elif src_data_type == "webcam":
        # month-day-hour-minute-seconds, e.g.: 02-26-15-51-12
        folder_name = lib_commons.get_time_string()
    return folder_name


args = get_command_line_arguments()

SRC_DATA_TYPE = args.source
SRC_DATA_PATH = args.data_path
SRC_MODEL_PATH = args.model_path
SRC_CLASSES_PATH = args.classes_path

DST_FOLDER_NAME = get_dst_folder_name(SRC_DATA_TYPE, SRC_DATA_PATH)

# -- Settings

# Output folder
DST_FOLDER = ROOT + "output/" + DST_FOLDER_NAME + "/"
DST_VIDEO_NAME = "video.avi"
DST_VIDEO_FRAMERATE = 10  # the framerate of the output video.avi
DST_SKELETON_NAME = "{:05d}.txt"

# Video setttings
webcam_max_framerate = 10.0

# Openpose settings
OPENPOSE_MODEL = ["mobilenet_thin", "cmu"][0]
OPENPOSE_IMG_SIZE = "432x368"

# -- Function


def read_classes(classes_path):
    with open(classes_path, 'r') as f:
        classes = [str_class.rstrip() for str_class in f if str_class != '\n']
    classes = np.array(classes)
    return classes


def select_images_loader(src_data_type, src_data_path):
    if src_data_type == "video":
        images_loader = lib_images_io.ReadFromVideo(
            src_data_path, sample_interval=1)
    elif src_data_type == "TODO":
        assert(False)  # TODO
    elif src_data_type == "webcam":
        webcam_idx = 0 if src_data_path == "" else int(src_data_path)
        images_loader = lib_images_io.ReadFromWebcam(
            webcam_max_framerate, webcam_idx)
    return images_loader


def add_white_region_to_left_of_image(img_disp):
    r, c, d = img_disp.shape
    blank = 255 + np.zeros((r, int(c/4), d), np.uint8)
    img_disp = np.hstack((blank, img_disp))
    return img_disp


def remove_skeletons_with_few_joints(skeletons):
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 2:
            # add this skeleton only when all requirements are satisfied
            good_skeletons.append(skeleton)
    return good_skeletons


class MultiPersonClassifier(object):
    def __init__(self, LOAD_MODEL_PATH, action_labels):
        self.create_classifier = lambda human_id: ClassifierOnlineTest(
            LOAD_MODEL_PATH, action_types=action_labels, human_id=human_id)
        self.dict_id2clf = {}  # human id -> classifier of this person

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

            if id not in self.dict_id2clf:  # add this new person
                self.dict_id2clf[id] = self.create_classifier(id)

            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton)  # predict label
            # print("\n\nPredicting label for human{}".format(id))
            # print("  skeleton: {}".format(skeleton))
            # print("  label: {}".format(id2label[id]))

        return id2label

    def get(self, id):
        # type: id: int or "min"
        if len(self.dict_id2clf) == 0:
            return None
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]


def draw_result_img(img_disp, humans, dict_id2skeleton,
                    skeleton_detector, multipeople_classifier):
    # Draw all people's skeleton
    skeleton_detector.draw(img_disp, humans)

    # Draw bounding box and label of each person
    if len(dict_id2skeleton):
        for id, label in dict_id2label.items():
            skeleton = dict_id2skeleton[id]
            # scale the y data back to original
            skeleton[1::2] = skeleton[1::2] / scale_y
            # print("Drawing skeleton: ", dict_id2skeleton[id], "with label:", label, ".")
            draw_action_result(img_disp, id, skeleton, label)

    # Add blank to the left for displaying prediction scores of each class
    img_disp = add_white_region_to_left_of_image(img_disp)

    # Draw predicting score for only 1 person
    if len(dict_id2skeleton):
        classifier_of_a_person = multipeople_classifier.get(id='min')
        classifier_of_a_person.draw_scores_onto_image(img_disp)
    return img_disp


# -- Main
if __name__ == "__main__":

    # -- Detector, tracker, classifier

    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)

    multiperson_tracker = Tracker()

    classes = read_classes(SRC_CLASSES_PATH)
    multipeople_classifier = MultiPersonClassifier(SRC_MODEL_PATH, classes)

    # -- Image reader and displayer
    images_loader = select_images_loader(SRC_DATA_TYPE, SRC_DATA_PATH)
    img_displayer = lib_images_io.ImageDisplayer()

    # -- Init output

    # output folder
    os.makedirs(DST_FOLDER, exist_ok=True)

    # video writer
    video_writer = lib_images_io.VideoWriter(
        DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FRAMERATE)

    # -- Read images and process
    ith_img = -1
    while images_loader.has_image():

        # -- Read image
        img = images_loader.read_image()
        ith_img += 1
        img_disp = img.copy()
        print(f"\nProcessing {ith_img}th image ...")

        # -- Detect skeletons
        humans = skeleton_detector.detect(img)
        skeletons, scale_y = skeleton_detector.humans_to_skels_list(humans)
        skeletons = remove_skeletons_with_few_joints(skeletons)

        # -- Track people
        dict_id2skeleton = multiperson_tracker.track(
            skeletons)  # int id -> np.array() skeleton

        # -- Recognize action of each person
        if len(dict_id2skeleton):
            dict_id2label = multipeople_classifier.classify(
                dict_id2skeleton)

        # -- Draw
        img_disp = draw_result_img(img_disp, humans, dict_id2skeleton,
                                   skeleton_detector, multipeople_classifier)

        # Print label of a person
        if len(dict_id2skeleton):
            min_id = min(dict_id2skeleton.keys())
            print("prediced label is :", dict_id2label[min_id])

        # -- Display image, and write to video.avi
        img_displayer.display(img_disp, wait_key_ms=1)
        video_writer.write(img_disp)

        # -- Get skeleton data and save to file
        skels_to_save = []
        for human_id in dict_id2skeleton.keys():
            label = dict_id2label[human_id]
            skeleton = dict_id2skeleton[human_id]
            skels_to_save.append([[human_id, label] + skeleton.tolist()])
        lib_io.save_skeletons(
            DST_FOLDER + DST_SKELETON_NAME.format(ith_img),
            skels_to_save)

    print("Program ends")
