#!/usr/bin/env python
# coding: utf-8

''' 
Load skeleton data from `skeletons_info.txt`, 
process data, 
and then save features and labels to .csv file.
'''

import numpy as np

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_commons as lib_commons
    from utils.lib_skeletons_io import load_skeleton_data
    from utils.lib_feature_proc import extract_multi_frame_features


def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings


cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s3_preprocess_features.py"]

CLASSES = np.array(cfg_all["classes"])

# Action recognition
WINDOW_SIZE = int(cfg_all["features"]["window_size"]) # number of frames used to extract features.

# Input and output
SRC_ALL_SKELETONS_TXT = par(cfg["input"]["all_skeletons_txt"])
DST_PROCESSED_FEATURES = par(cfg["output"]["processed_features"])
DST_PROCESSED_FEATURES_LABELS = par(cfg["output"]["processed_features_labels"])

# -- Functions


def process_features(X0, Y0, video_indices, classes):
    ''' Process features '''
    # Convert features
    # From: raw feature of individual image.
    # To:   time-serials features calculated from multiple raw features
    #       of multiple adjacent images, including speed, normalized pos, etc.
    ADD_NOISE = False
    if ADD_NOISE:
        X1, Y1 = extract_multi_frame_features(
            X0, Y0, video_indices, WINDOW_SIZE, 
            is_adding_noise=True, is_print=True)
        X2, Y2 = extract_multi_frame_features(
            X0, Y0, video_indices, WINDOW_SIZE,
            is_adding_noise=False, is_print=True)
        X = np.vstack((X1, X2))
        Y = np.concatenate((Y1, Y2))
        return X, Y
    else:
        X, Y = extract_multi_frame_features(
            X0, Y0, video_indices, WINDOW_SIZE, 
            is_adding_noise=False, is_print=True)
        return X, Y

# -- Main


def main():
    ''' 
    Load skeleton data from `skeletons_info.txt`, process data, 
    and then save features and labels to .csv file.
    '''

    # Load data
    X0, Y0, video_indices = load_skeleton_data(SRC_ALL_SKELETONS_TXT, CLASSES)

    # Process features
    print("\nExtracting time-serials features ...")
    X, Y = process_features(X0, Y0, video_indices, CLASSES)
    print(f"X.shape = {X.shape}, len(Y) = {len(Y)}")

    # Save data
    print("\nWriting features and labesl to disk ...")

    os.makedirs(os.path.dirname(DST_PROCESSED_FEATURES), exist_ok=True)
    os.makedirs(os.path.dirname(DST_PROCESSED_FEATURES_LABELS), exist_ok=True)

    np.savetxt(DST_PROCESSED_FEATURES, X, fmt="%.5f")
    print("Save features to: " + DST_PROCESSED_FEATURES)

    np.savetxt(DST_PROCESSED_FEATURES_LABELS, Y, fmt="%i")
    print("Save labels to: " + DST_PROCESSED_FEATURES_LABELS)


if __name__ == "__main__":
    main()
