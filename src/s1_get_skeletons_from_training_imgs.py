'''
Based on a valid_images.txt file, read training images from the folder.
In each image, there should be only 1 person performing one type of action.
Each image is named as 00001.jpg, 00002.jpg, ...

An example of valid_images.txt is shown below:
    jump_03-12-09-18-26-176
    58 680

    jump_03-13-11-27-50-720
    65 393

    kick_03-02-12-36-05-185
    54 62
    75 84
The two indices (such as `56 680` in the first `jump` example)
represents the starting index and ending index of a certain action.
'''

import cv2

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    import utils.lib_images_loader as lib_images_loader
    import utils.lib_io as lib_io


# -- Settings

# Input and output files
SRC_IMAGES_FOLDER = ROOT + "data/source_images3/"
SRC_IMAGES_DESCRIPTION_TXT = ROOT + "data/source_images3/valid_images.txt"
# Store image type, filename, etc.
DST_IMAGES_INFO_FILE = ROOT + "data_proc/raw_skeletons/images_info.txt"
# Each txt stores the skeleton of one image.
DST_DETECTED_SKELETONS_FOLDER = ROOT + "data_proc/raw_skeletons/skeleton_res/"
# Each image is drawn with the detected skeleton.
DST_VIZ_IMGS_FOLDER = ROOT + "data_proc/raw_skeletons/image_viz/"

# Openpose settings
OPENPOSE_MODEL = ["mobilenet_thin", "cmu"][1]
OPENPOSE_IMG_SIZE = "656x368"

# -- Main
if __name__ == "__main__":

    # -- Detector
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    multiperson_tracker = Tracker()

    # -- Image reader and displayer
    images_loader = lib_io.ReadValidImagesAndActionTypesByTxt(
        img_folder=SRC_IMAGES_FOLDER,
        valid_imgs_txt=SRC_IMAGES_DESCRIPTION_TXT,
        img_suffix="png")
    images_loader.save_images_info(filepath=DST_IMAGES_INFO_FILE)
    img_displayer = lib_images_loader.ImageDisplayer()

    # -- Init output path
    os.makedirs(os.path.dirname(DST_IMAGES_INFO_FILE), exist_ok=True)
    os.makedirs(DST_DETECTED_SKELETONS_FOLDER, exist_ok=True)
    os.makedirs(DST_VIZ_IMGS_FOLDER, exist_ok=True)

    # -- Read images and process
    num_total_images = images_loader.num_images
    for ith_img in range(num_total_images):

        # -- Read image
        img, str_action_type, img_info = images_loader.read_image()

        # -- Detect
        humans = skeleton_detector.detect(img)

        # -- Draw
        img_disp = img.copy()
        skeleton_detector.draw(img_disp, humans)
        img_displayer.display(img_disp, wait_key_ms=1)

        # -- Get skeleton data and save to file
        skeletons, _ = skeleton_detector.humans_to_skels_list(humans)
        dict_id2skeleton = multiperson_tracker.track(
            skeletons)  # (int id) -> (np.array() skeleton)
        skels_to_save = [img_info + skeleton.tolist()
                         for skeleton in dict_id2skeleton.values()]

        f = DST_DETECTED_SKELETONS_FOLDER + \
            lib_io.int2str(ith_img, idx_len=5) + ".txt"
        # Save action type, skeleton location, etc.
        lib_io.save_skeletons(f, skels_to_save)

        f = DST_VIZ_IMGS_FOLDER + lib_io.int2str(ith_img, idx_len=5) + ".jpg"
        cv2.imwrite(f, img_disp)  # Save the visualized image for debug.

        print(
            f"{ith_img}/{num_total_images} th image has {len(skeletons)} people in it")

    print("Program ends")
