

import numpy as np
if True:
    import sys
    from utils.lib_io import *
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"


data_idx = "3"
SRC_IMAGE_FOLDER = CURR_PATH + "../data/source_images"+data_idx+"/"
VALID_IMAGES_TXT = SRC_IMAGE_FOLDER + "valid_images.txt"

images_info = collect_images_info_from_source_images(
    VALID_IMAGES_TXT, img_suffix="jpg")
