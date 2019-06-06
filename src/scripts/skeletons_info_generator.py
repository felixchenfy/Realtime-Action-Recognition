
'''
Input:
    skeletons/00001.txt ~ skeletons/xxxxx.txt
Output:
    skeletons_info.txt
'''

import numpy as np
import simplejson
import sys, os
import csv
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.append(CURR_PATH+"../")
from mylib.io import load_skeletons
from mylib.funcs import get_filenames
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"


if __name__=="__main__":
    # Parameters -------------------------------------------

    data_idx = "5"

    read_from = "../skeleton_data/skeletons"+data_idx+"/"
    output_to = "../skeleton_data/skeletons"+data_idx+"_info.txt"
    NUM_SKELETONS = len(get_filenames(CURR_PATH + read_from))

    # Main -------------------------------------------

    int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)
    all_skeletons = []
    all_skeletons_good_only = []
    for i in range(1, NUM_SKELETONS+1):
        if i==1 or i%100==0:
            print("{}/{}".format(i, NUM_SKELETONS))
        skeletons = load_skeletons(CURR_PATH + read_from + int2str(i, 5) + ".txt")
        idx_person = 0 # Only one person in each image

        # Check if skeleton is valid
        if len(skeletons):
            skeleton = skeletons[idx_person]
            data_size = len(skeleton)
            if i==1:
                print("data size = {}".format(data_size))
        else:
            skeleton = [0] * data_size

        # -- Push to result list
        all_skeletons.append(skeleton) 

    print("There are {} skeleton data.".format(len(all_skeletons)))

    with open(CURR_PATH + output_to, 'w') as f:
        simplejson.dump(all_skeletons, f)

    # def wrote_to_csv(filepath, data):
    #     with open(filepath, 'w') as f:
    #         writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #         first_row = ["nose_x","nose_y","neck_x","neck_y",
    #             "Rshoulder_x","Rshoulder_y","Relbow_x","Relbow_y","Rwrist_x","RWrist_y",
    #             "LShoulder_x","LShoulder_y","LElbow_x","LElbow_y","LWrist_x","LWrist_y",
    #             "RHip_x","RHip_y","RKnee_x","RKnee_y","RAnkle_x","RAnkle_y",
    #             "LHip_x","LHip_y","LKnee_x","LKnee_y","LAnkle_x","LAnkle_y",
    #             "REye_x","REye_y","LEye_x","LEye_y","REar_x","REar_y","LEar_x","LEar_y","class"]
    #         writer.writerow(first_row)
    #         for sk in data:
    #             writer.writerow(sk)
    # wrote_to_csv(CURR_PATH + output_to2, all_skeletons)