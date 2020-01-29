#!/usr/bin/env python

'''
Classes for reading images from video, folder, or web camera,
    and for writing images to video file.

Main classes and functions:
    * Read:
        class ReadFromFolder
        class ReadFromVideo
        class ReadFromWebcam
    
    * Write:
        class VideoWriter
    
    * Display:
        class ImageDisplayer
    
    * Test:
        def test_ReadFromWebcam

'''

import os
import warnings
import numpy as np
import cv2
import time
import glob
import threading
import queue
import multiprocessing


class ReadFromFolder(object):
    ''' A image reader class for reading images from a folder.
    By default, all files under the folder are considered as image file.
    '''

    def __init__(self, folder_path):
        self.filenames = sorted(glob.glob(folder_path + "/*"))
        self.cnt_imgs = 0
        self.cur_filename = ""

    def read_image(self):
        if self.cnt_imgs >= len(self.filenames):
            return None
        self.cur_filename = self.filenames[self.cnt_imgs]
        img = cv2.imread(self.cur_filename, cv2.IMREAD_UNCHANGED)
        self.cnt_imgs += 1
        return img

    def __len__(self):
        return len(self.filenames)

    def has_image(self):
        return self.cnt_imgs < len(self.filenames)

    def stop(self):
        None


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
        self._is_stoped = False
        self._video = cv2.VideoCapture(video_path)
        ret, image = self._video.read()
        self._next_image = image
        self._sample_interval = sample_interval
        self._fps = self.get_fps()
        if not self._fps >= 0.0001:
            import warnings
            warnings.warn("Invalid fps of video: {}".format(video_path))

    def has_image(self):
        return self._next_image is not None

    def get_curr_video_time(self):
        return 1.0 / self._fps * self.cnt_imgs

    def read_image(self):
        image = self._next_image
        for i in range(self._sample_interval):
            if self._video.isOpened():
                ret, frame = self._video.read()
                self._next_image = frame
            else:
                self._next_image = None
                break
        self.cnt_imgs += 1
        return image

    def stop(self):
        self._video.release()
        self._is_stoped = True

    def __del__(self):
        if not self._is_stoped:
            self.stop()

    def get_fps(self):

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        # With webcam get(CV_CAP_PROP_FPS) does not work.
        # Let's see for ourselves.

        # Get video properties
        if int(major_ver) < 3:
            fps = self._video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = self._video.get(cv2.CAP_PROP_FPS)
        return fps


class ReadFromWebcam(object):
    def __init__(self, max_framerate=30.0, webcam_idx=0):
        ''' Read images from web camera.
        Argument:
            max_framerate {float}: the real framerate will be reduced below this value.
            webcam_idx {int}: index of the web camera on your laptop. It should be 0 by default.
        '''
        # Settings
        self._max_framerate = max_framerate
        queue_size = 3

        # Initialize video reader
        self._video = cv2.VideoCapture(webcam_idx)
        self._is_stoped = False

        # Use a thread to keep on reading images from web camera
        self._imgs_queue = queue.Queue(maxsize=queue_size)
        self._is_thread_alive = multiprocessing.Value('i', 1)
        self._thread = threading.Thread(
            target=self._thread_reading_webcam_images)
        self._thread.start()

        # Manually control the framerate of the webcam by sleeping
        self._min_dt = 1.0 / self._max_framerate
        self._prev_t = time.time() - 1.0 / max_framerate

    def read_image(self):
        dt = time.time() - self._prev_t
        if dt <= self._min_dt:
            time.sleep(self._min_dt - dt)
        self._prev_t = time.time()
        image = self._imgs_queue.get(timeout=10.0)
        return image

    def has_image(self):
        return True  # The web camera always has new image

    def stop(self):
        self._is_thread_alive.value = False
        self._video.release()
        self._is_stoped = True

    def __del__(self):
        if not self._is_stoped:
            self.stop()

    def _thread_reading_webcam_images(self):
        while self._is_thread_alive.value:
            ret, image = self._video.read()
            if self._imgs_queue.full():  # if queue is full, pop one
                img_to_discard = self._imgs_queue.get(timeout=0.001)
            self._imgs_queue.put(image, timeout=0.001)  # push to queue
        print("Web camera thread is dead.")


class VideoWriter(object):
    def __init__(self, video_path, framerate):

        # -- Settings
        self._video_path = video_path
        self._framerate = framerate

        # -- Variables
        self._cnt_img = 0
        # initialize later when the 1st image comes
        self._video_writer = None
        self._width = None
        self._height = None

        # -- Create output folder
        folder = os.path.dirname(video_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
            video_path

    def write(self, img):
        self._cnt_img += 1
        if self._cnt_img == 1:  # initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # define the codec
            self._width = img.shape[1]
            self._height = img.shape[0]
            self._video_writer = cv2.VideoWriter(
                self._video_path, fourcc, self._framerate, (self._width, self._height))
        self._video_writer.write(img)

    def stop(self):
        self.__del__()

    def __del__(self):
        if self._cnt_img > 0:
            self._video_writer.release()
            print("Complete writing {}fps and {}s video to {}".format(
                self._framerate, self._cnt_img/self._framerate, self._video_path))


class ImageDisplayer(object):
    ''' A simple wrapper of using cv2.imshow to display image '''

    def __init__(self):
        self._window_name = "cv2_display_window"
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

    def display(self, image, wait_key_ms=1):
        cv2.imshow(self._window_name, image)
        cv2.waitKey(wait_key_ms)

    def __del__(self):
        cv2.destroyWindow(self._window_name)


def test_ReadFromWebcam():
    ''' Test the class ReadFromWebcam '''
    webcam_reader = ReadFromWebcam(max_framerate=10)
    img_displayer = ImageDisplayer()
    import itertools
    for i in itertools.count():
        img = webcam_reader.read_image()
        if img is None:
            break
        print(f"Read {i}th image...")
        img_displayer.display(img)
    print("Program ends")


if __name__ == "__main__":
    test_ReadFromWebcam()
