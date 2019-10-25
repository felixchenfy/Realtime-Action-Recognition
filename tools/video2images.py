'''
Description:
    Convert a video to a folder of images.
    
Example of usage:
    python video2images.py \
        -i /home/feiyu/Desktop/learn_coding/test_data/video_of_waving_object.avi \
        -o /home/feiyu/Desktop/learn_coding/test_data/video_convert_result \
        --sample_interval 2 \
        --max_frames 30
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
import itertools
ROOT = os.path.dirname(os.path.abspath(__file__))+"/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a folder of images into a video.")
    parser.add_argument("-i", "--input_video_path", type=str, required=True)
    parser.add_argument("-o", "--output_folder_path", type=str, required=True)
    parser.add_argument("-s", "--sample_interval", type=int, required=False,
                        default=1,
                        help="Sample every nth video frame to save to folder. Default 1.")
    parser.add_argument("-m", "--max_frames", type=int, required=False,
                        default=100000,
                        help="Max number of video frames to save to folder. Default 1e5")
    args = parser.parse_args()
    return args


class ReadFromVideo(object):
    def __init__(self, video_path, sample_interval=1):
        ''' A video reader class for reading video frames from video.
        Arguments:
            video_path
            sample_interval {int}: sample every kth image.
        '''
        if not os.path.exists(video_path):
            raise IOError("Video not exist: " + video_path)
        assert isinstance(sample_interval, int) and sample_interval >= 1
        self.cnt_imgs = 0
        self.is_stoped = False
        self.video = cv2.VideoCapture(video_path)
        ret, frame = self.video.read()
        self.next_image = frame
        self.sample_interval = sample_interval
        self.fps = self.get_fps()
        if not self.fps >= 0.0001:
            import warnings
            warnings.warn("Invalid fps of video: {}".format(video_path))

    def has_image(self):
        return self.next_image is not None

    def get_curr_video_time(self):
        return 1.0 / self.fps * self.cnt_imgs

    def read_image(self):
        image = self.next_image
        for i in range(self.sample_interval):
            if self.video.isOpened():
                ret, frame = self.video.read()
                self.next_image = frame
            else:
                self.next_image = None
                break
        self.cnt_imgs += 1
        return image

    def stop(self):
        self.video.release()
        self.is_stoped = True

    def __del__(self):
        if not self.is_stoped:
            self.stop()

    def get_fps(self):

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        # With webcam get(CV_CAP_PROP_FPS) does not work.
        # Let's see for ourselves.

        # Get video properties
        if int(major_ver) < 3:
            fps = self.video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = self.video.get(cv2.CAP_PROP_FPS)
        return fps

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

    video_loader = ReadFromVideo(args.input_video_path)

    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path)

    def set_output_filename(i):
        return args.output_folder_path + "/" + "{:08d}".format(i) + ".jpg"

    img_displayer = ImageDisplayer()
    cnt_img = 0
    for i in itertools.count():
        img = video_loader.read_image()
        if img is None:
            print("Have read all frames from the video file.")
            break
        if i % args.sample_interval == 0:
            cnt_img += 1
            print("Processing {}th image".format(cnt_img))
            cv2.imwrite(set_output_filename(cnt_img), img)
            img_displayer.display(img)
            if cnt_img == args.max_frames:
                print("Read {} frames. ".format(cnt_img) +
                      "Reach the max_frames setting. Stop.")
                break


if __name__ == "__main__":
    args = parse_args()
    assert args.sample_interval >= 0 and args.sample_interval
    main(args)
    print("Program stops: " + os.path.basename(__file__))
