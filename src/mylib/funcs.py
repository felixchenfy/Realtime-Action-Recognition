
import numpy as np
import cv2
import math
from os import listdir
from os.path import isfile, join

int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)

def get_filenames(path, sort = True):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    if sort:
        onlyfiles.sort()
    return onlyfiles
    

