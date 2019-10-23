
import numpy as np
import cv2
import math
import datetime
from os import listdir
from os.path import isfile, join
import functools


def int2str(num, idx_len): return ("{:0"+str(idx_len)+"d}").format(num)


def get_filenames(path, use_sort=True, with_folder_path=False):
    ''' Get all filenames under certain path '''
    fnames = [f for f in listdir(path) if isfile(join(path, f))]
    if use_sort:
        fnames.sort()
    if with_folder_path:
        fnames = [path + "/" + f for f in fnames]
    return fnames

def get_time_string():
    ''' Get a formatted string time: `month-day-hour-minute-seconds`,
        such as: `02-26-15-51-12`.
    '''
    s = str(datetime.datetime.now())[5:].replace(
        ' ', '-').replace(":", '-').replace('.', '-')[:-3]
    return s