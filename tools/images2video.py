'''
Description:
    Convert a folder of images to a video file.
    
Example of usage:
    python images2video.py \
        -i /home/feiyu/Desktop/learn_coding/test_data/imgs_of_waving_object \
        -o /home/feiyu/Desktop/learn_coding/test_data/video.avi \
        --framerate 30 \
        --sample_interval 1

'''

import numpy as np
import cv2
import sys
import os
import time
import numpy as np
import simplejson
import sys
import os
import csv
import glob
import argparse
ROOT = os.path.dirname(os.path.abspath(__file__))+"/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a folder of images into a video.")
    parser.add_argument("-i", "--input_folder_path", type=str, required=True)
    parser.add_argument("-o", "--output_video_path", type=str, required=True)
    parser.add_argument("-r", "--framerate", type=int, required=False,
                        default=30)
    parser.add_argument("-s", "--sample_interval", type=int, required=False,
                        default=1,
                        help="Sample every nth image for creating video. Default 1.")

    args = parser.parse_args()
    return args


class ReadFromFolder(object):
    def __init__(self, folder_path):
        fnames = []
        for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
            fnames.extend(glob.glob(folder_path + "/*" + ext))
        self.fnames = sorted(fnames)
        if len(self.fnames) == 0:
            raise IOError("The folder has no images: " + folder_path)
        self.cnt_imgs = 0
        self.cur_filename = ""
        

    def read_image(self):
        if self.cnt_imgs < len(self.fnames):
            self.cur_filename = self.fnames[self.cnt_imgs]
            img = cv2.imread(self.cur_filename, cv2.IMREAD_UNCHANGED)
            self.cnt_imgs += 1
            return img
        else:
            return None

    def __len__(self):
        return len(self.fnames)

    def get_cur_filename(self):
        return self.cur_filename

    def stop(self):
        None


class VideoWriter(object):
    def __init__(self, video_path, framerate):

        # -- Settings
        self.video_path = video_path
        self.framerate = framerate

        # -- Variables
        self.cnt_img = 0
        # initialize later when the 1st image comes
        self.video_writer = None
        self.width = None
        self.height = None

        # -- Create output folder
        folder = os.path.dirname(video_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

    def write(self, img):
        self.cnt_img += 1
        if self.cnt_img == 1:  # initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # define the codec
            self.width = img.shape[1]
            self.height = img.shape[0]
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, self.framerate, (self.width, self.height))
        self.video_writer.write(img)

    def __del__(self):
        if self.cnt_img > 0:
            self.video_writer.release()
            print("Complete writing video: ", self.video_path)


class ImageDisplayer(object):
    def __init__(self):
        self._window_name = "images2video.py"
        cv2.namedWindow(self._window_name)

    def display(self, image, wait_key_ms=1):
        cv2.imshow(self._window_name, image)
        cv2.waitKey(wait_key_ms)

    def __del__(self):
        cv2.destroyWindow(self._window_name)


def main(args):

    images_loader = ReadFromFolder(args.input_folder_path)
    video_writer = VideoWriter(args.output_video_path, args.framerate)
    img_displayer = ImageDisplayer()

    N = len(images_loader)
    i = 0
    while i < N:
        print("Processing {}/{}th image".format(i, N))
        img = images_loader.read_image()
        if i % args.sample_interval == 0:
            video_writer.write(img)
            img_displayer.display(img)
        i += 1


if __name__ == "__main__":
    args = parse_args()
    assert args.sample_interval >= 0 and args.sample_interval
    main(args)
    print("Program stops: " + os.path.basename(__file__))
