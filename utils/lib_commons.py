''' This script defines some common functions '''

import numpy as np
import cv2
import math
import time
import os
import glob
import yaml
import datetime
from os import listdir
from os.path import isfile, join
import functools
import simplejson


def int2str(num, idx_len):
    return ("{:0"+str(idx_len)+"d}").format(num)


def save_listlist(filepath, ll):
    ''' Save a list of lists to file '''
    folder_path = os.path.dirname(filepath)
    os.makedirs(folder_path, exist_ok=True)
    with open(filepath, 'w') as f:
        simplejson.dump(ll, f)


def read_listlist(filepath):
    ''' Read a list of lists from file '''
    with open(filepath, 'r') as f:
        ll = simplejson.load(f)
        return ll


def read_yaml(filepath):
    ''' Input a string filepath, 
        output a `dict` containing the contents of the yaml file.
    '''
    with open(filepath, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded


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
