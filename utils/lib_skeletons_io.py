'''
This script defines the functions for reading/saving images & skeletons data:

* def get_training_imgs_info

    Parse images info from `valid_images.txt`.

* class ReadValidImagesAndActionTypesByTxt

    Read images based on `valid_images.txt`

* def load_skeleton_data
    
    Load data from `skeletons_info.txt`.

'''

import numpy as np
import cv2
import os
import sys
import simplejson
from sklearn.preprocessing import OneHotEncoder


# Image info includes: [cnt_action, cnt_clip, cnt_image, img_action_label, filepath]
LEN_IMG_INFO = 5
LEN_SKELETON_XY = 18*2
NaN = 0  # `Not A Number`, which is the value for invalid data.

# -- Functions


def get_training_imgs_info(
        valid_images_txt,
        img_filename_format="{:05d}.jpg"):
    '''
    Arguments:
        valid_images_txt {str}: path of the txt file that 
            specifies the indices and labels of training images.
    Return:
        images_info {list of list}: shape=PxN, where:
            P: number of training images
            N=5: number of tags of that image, including: 
                [cnt_action, cnt_clip, cnt_image, action_label, filepath]
                An example: [8, 49, 2687, 'wave', 'wave_03-02-12-35-10-194/00439.jpg']                
    '''
    images_info = list()

    with open(valid_images_txt) as f:

        folder_name = None
        action_label = None
        cnt_action = 0
        actions = set()
        action_images_cnt = dict()
        cnt_clip = 0
        cnt_image = 0

        for cnt_line, line in enumerate(f):

            if line.find('_') != -1:  # A new video type
                folder_name = line[:-1]
                action_label = folder_name.split('_')[0]
                if action_label not in actions:
                    cnt_action += 1
                    actions.add(action_label)
                    action_images_cnt[action_label] = 0

            elif len(line) > 1:  # line != "\n"
                # print("Line {}, len ={}, {}".format(cnt_line, len(line), line))
                indices = [int(s) for s in line.split()]
                idx_start = indices[0]
                idx_end = indices[1]
                cnt_clip += 1
                for i in range(idx_start, idx_end+1):
                    filepath = folder_name+"/" + img_filename_format.format(i)
                    cnt_image += 1
                    action_images_cnt[action_label] += 1

                    # Save: 5 values
                    image_info = [cnt_action, cnt_clip,
                                  cnt_image, action_label, filepath]
                    assert(len(image_info) == LEN_IMG_INFO)
                    images_info.append(image_info)
                    # An example: [8, 49, 2687, 'wave', 'wave_03-02-12-35-10-194/00439.jpg']

        print("")
        print("Number of action classes = {}".format(len(actions)))
        print("Number of training images = {}".format(cnt_image))
        print("Number of training images of each action:")
        for action in actions:
            print("  {:>8}| {:>4}|".format(
                action, action_images_cnt[action]))

    return images_info


class ReadValidImagesAndActionTypesByTxt(object):
    ''' This is for reading training images configured by a txt file.
        Each training image should contain a person who is performing certain type of action. 
    '''

    def __init__(self, img_folder, valid_imgs_txt,
                 img_filename_format="{:05d}.jpg"):
        '''
        Arguments:
            img_folder {str}: A folder that contains many sub folders.
                Each subfolder has many images named as xxxxx.jpg.
            valid_imgs_txt {str}: A txt file which specifies the action labels.
                Example:
                    jump_03-12-09-18-26-176
                    58 680

                    jump_03-13-11-27-50-720
                    65 393

                    kick_03-02-12-36-05-185
                    54 62
                    75 84
            img_filename_format {str}: format of the image filename
        '''
        self.images_info = get_training_imgs_info(
            valid_imgs_txt, img_filename_format)
        self.imgs_path = img_folder
        self.i = 0
        self.num_images = len(self.images_info)
        print(f"Reading images from txtscript: {img_folder}")
        print(f"Reading images information from: {valid_imgs_txt}")
        print(f"    Num images = {self.num_images}\n")

    def save_images_info(self, filepath):
        folder_path = os.path.dirname(filepath)
        os.makedirs(folder_path, exist_ok=True)
        with open(filepath, 'w') as f:
            simplejson.dump(self.images_info, f)

    def read_image(self):
        '''
        Returns:
            img {RGB image}: 
                Next RGB image from folder. 
            img_action_label {str}: 
                Action label obtained from folder name.
            img_info {list}: 
                Something like [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]
        Raise:
            RuntimeError, if fail to read next image due to wrong index or wrong filepath.
        '''
        self.i += 1
        if self.i > len(self.images_info):
            raise RuntimeError(f"There are only {len(self.images_info)} images, "
                               f"but you try to read the {self.i}th image")
        filepath = self.get_filename(self.i)
        img = self.imread(self.i)
        if img is None:
            raise RuntimeError("The image file doesn't exist: " + filepath)
        img_action_label = self.get_action_label(self.i)
        img_info = self.get_image_info(self.i)
        return img, img_action_label, img_info

    def imread(self, index):
        return cv2.imread(self.imgs_path + self.get_filename(index))

    def get_filename(self, index):
        # The 4th element of
        # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]
        # See "get_training_imgs_info" for the data format
        return self.images_info[index-1][4]

    def get_action_label(self, index):
        # The 3rd element of
        # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]
        # See "get_training_imgs_info" for the data format
        return self.images_info[index-1][3]

    def get_image_info(self, index):
        # Something like [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]
        return self.images_info[index-1]


def load_skeleton_data(filepath, classes):
    ''' Load training data from skeletons_info.txt.
    Some notations:
        N: number of valid data.
        P: feature dimension. Here P=36.
        C: number of classes.
    Arguments:
        filepath {str}: file path of `skeletons_info.txt`, which stores the skeletons and labels.
    Returns:
        X: {np.array, shape=NxP}:           Skeleton data (feature) of each valid image.
        Y: {list of int, len=N}:            Label of each valid image.
        video_indices {list of int, len=N}:  The video index of which the image belongs to.
        classes {list of string, len=C}:    The classes of all actions.
    '''

    label2index = {c: i for i, c in enumerate(classes)}

    with open(filepath, 'r') as f:

        # Load data
        dataset = simplejson.load(f)

        # Remove bad data. A bad data is filled with zeros.
        def is_good_data(row):
            return row[0] != 0
        dataset = [row for i, row in enumerate(dataset) if is_good_data(row)]

        # Get skeleton data, which is in the pos [5, 41)
        # LEN_IMG_INFO = 5
        # LEN_SKELETON_XY = 36
        X = np.array([row[LEN_IMG_INFO:LEN_IMG_INFO+LEN_SKELETON_XY]
                      for row in dataset])

        # row[1] is the video index of the image
        video_indices = [row[1] for row in dataset]

        # row[3] is the label of the image
        # Y_str = [[row[3]] for row in dataset] # deprecated
        Y_str = [row[3] for row in dataset]

        # Convert string label to indices
        # classes, Y = _get_classes_and_label_indices(Y_str) # deprecated
        Y = [label2index[label] for label in Y_str]

        # Remove data with missing upper body joints
        if 0:
            valid_indices = _get_skeletons_with_complete_upper_body(X, NaN)
            X = X[valid_indices, :]
            Y = [Y[i] for i in valid_indices]
            video_indices = [video_indices[i] for i in valid_indices]
            print("Num samples after removal = ", len(Y))

        # Print data properties
        N = len(Y)
        P = len(X[0])
        C = len(classes)
        print(f"\nNumber of samples = {N} \n"
              f"Raw feature length = {P} \n"
              f"Number of classes = {C}")
        print(f"Classes: {classes}")

        return X, Y, video_indices

    raise RuntimeError("Failed to load skeletons txt: " + filepath)


def _get_skeletons_with_complete_upper_body(X, NaN=0):
    ''' 
    Find good skeletons whose upper body joints don't contain `NaN`.
    Return the indices of these skeletons.
    Arguments:
        X {np.array, shape=NxP}: Feature of each sample. 
            N is number of samples, P is feature dimension.
            P = 36 = 18*2.
        NaN {int}: `Not A Number`, which is the value for invalid data.
    '''

    left_idx, right_idx = 0, 14 * 2  # 1head+1neck+2*(3arms + 3legs)

    def is_valid(x):
        return len(np.where(x[left_idx:right_idx] == NaN)[0]) == 0
    valid_indices = [i for i, x in enumerate(X) if is_valid(x)]
    return valid_indices

# This function is deprecated.
# It was used for getting classes from the data,
# but now I manually set the classes in config/config.yaml.
# def _get_classes_and_label_indices(Y_str):
#     ''' Get classes from labels, and then convert each label to the label index.
#     Arguments:
#         Y_str     {list of str}:     Label of each image.
#     Returns:
#         classes   {list of string}:  e.g. ["run", "sit", "walk", ...].
#         Y_indices {list of int}:     Label index of each image, e.g. [0, 2, 0, ...].
#     '''

#     # -- Get classes
#     enc = OneHotEncoder(handle_unknown='ignore')
#     enc.fit(Y_str)
#     classes = enc.categories_[0]

#     # -- Get One-hot enconding result
#     # Y_one_hot {list of list}:    One-hot encoding of each image. For example:
#     #     The indices [0,2,0,...] are converted to [[1,0,0], [0,0,1], [1,0,0], ...]
#     Y_one_hot = enc.transform(Y_str).toarray()

#     # -- Get label index
#     Y_indices = [np.where(yi == 1)[0][0] for yi in Y_one_hot]
#     return classes, Y_indices
