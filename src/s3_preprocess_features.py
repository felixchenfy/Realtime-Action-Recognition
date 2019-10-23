#!/usr/bin/env python
# coding: utf-8

import numpy as np

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    from utils.lib_load_skeleton import load_skeleton_data
    from utils.lib_feature_proc import extract_time_serials_data

# -- Settings

SRC_SKELETONS_DATA_TXT = ROOT + "data_proc/raw_skeletons/skeletons_info.txt"
DST_PROCESSED_FEATURES = ROOT + "data_proc/features_X.csv"
DST_PROCESSED_FEATURES_LABELS = ROOT + "data_proc/features_Y.csv"
DST_CLASSES = ROOT + "data_proc/classes.csv"

# -- Functions


def process_features(X0, Y0, video_indices, classes):
    # Convert features
    # From: raw feature of individual image.
    # To:   time-serials feature. Each feature is calculated from multiple raw features,
    #       by computing the speed, normalization, etc.
    X1, Y1 = extract_time_serials_data(
        X0, Y0, video_indices, is_adding_noise=True)

    # Also, add noise the to features to augment data.
    X2, Y2 = extract_time_serials_data(
        X0, Y0, video_indices, is_adding_noise=False)
    X = np.vstack((X1, X2))
    Y = np.concatenate((Y1, Y2))
    return X, Y


def write_strings(filepath, strings):
    with open(filepath, 'w') as f:
        for s in strings:
            f.write(s + "\n")


def create_folders(filepaths):
    for filepath in filepaths:
        folder = os.path.dirname(filepath)
        os.makedirs(folder, exist_ok=True)

# -- Main


def main():

    # Load data
    X0, Y0, video_indices, classes = load_skeleton_data(SRC_SKELETONS_DATA_TXT)

    # Process features
    print("\nExtracting time-serials features ...")
    X, Y = process_features(X0, Y0, video_indices, classes)
    print(f"X.shape = {X.shape}, len(Y) = {len(Y)}")

    # Save data
    print("\nWriting classes, features, and labesl to disk ...")
    create_folders(filepaths=[DST_CLASSES,
                              DST_PROCESSED_FEATURES,
                              DST_PROCESSED_FEATURES_LABELS])

    write_strings(DST_CLASSES, classes)
    print("Save classes to: " + DST_CLASSES)

    np.savetxt(DST_PROCESSED_FEATURES, X, fmt="%.5f")
    print("Save features to: " + DST_PROCESSED_FEATURES)

    np.savetxt(DST_PROCESSED_FEATURES_LABELS, Y, fmt="%i")
    print("Save labels to: " + DST_PROCESSED_FEATURES_LABELS)


if __name__ == "__main__":
    main()
