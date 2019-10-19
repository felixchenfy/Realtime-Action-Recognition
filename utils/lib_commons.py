
import numpy as np
import cv2
import math
from os import listdir
from os.path import isfile, join
import functools

int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)

def get_filenames(path, sort = True):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    if sort:
        onlyfiles.sort()
    return onlyfiles
    

class Tracker(object):
    def __init__(self):
        self.dict_id2skeleton = {}
        self.cnt_humans = 0
        self.MAX_HUMAN = 5

    def get_neck(self, skeleton):
        x, y = skeleton[2], skeleton[3]
        return x, y

    def sort_skeletons_by_neck(self, skeletons):
        # Skeletons are sorted by which is nearer to the image center 
        calc_dist = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        def cost(skeleton):
            x1, y1 = self.get_neck(skeleton)
            return calc_dist((x1, y1), (0.5, 0.5)) # dist to center
        cmp = lambda a, b: (a>b)-(a<b)
        mycmp = lambda sk1, sk2: cmp(cost(sk1), cost(sk2))
        skeletons_sort = sorted(skeletons, key = functools.cmp_to_key(mycmp))
        return skeletons_sort


    def track(self, curr_skels):
        # type: curr_skels: list[list[]]
        # rtype: a dict, mapping human id to his/her skeleton
        
        curr_skels = self.sort_skeletons_by_neck(curr_skels)
        N = len(curr_skels)
        
        # Match skeletons between curr and prev
        if len(self.dict_id2skeleton) > 0: 
            ids, prev_skels = map(list, zip(*self.dict_id2skeleton.items()))
            good_matches = self.match_features(prev_skels, curr_skels)

            self.dict_id2skeleton = {}
            is_matched = [False]*N
            for i2, i1 in good_matches.items():
                human_id = ids[i1]
                self.dict_id2skeleton[human_id] = np.array(curr_skels[i2])
                is_matched[i2] = True
            unmatched_idx = [i for i, matched in enumerate(is_matched) if not matched]
        else:
            good_matches = []
            unmatched_idx = range(N)

        # Add unmatched skeletons (which are new skeletons) to the list
        num_humans_to_add = min(self.MAX_HUMAN - len(good_matches), len(unmatched_idx))
        for i in range(num_humans_to_add):
            self.cnt_humans += 1
            self.dict_id2skeleton[self.cnt_humans] = np.array(curr_skels[unmatched_idx[i]])

        return self.dict_id2skeleton 

    def match_features(self, features1, features2):
        features1, features2 = np.array(features1), np.array(features2)

        #cost = lambda x1, x2: np.linalg.norm(x1-x2) 
        calc_dist = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        def cost(sk1, sk2): 

            # neck, shoulder, elbow, hip, knee
            joints = np.array([2,3, 4,5,6,7, 10,11,12,13, 16,17,18,19, 22,23,24,25])
            
            sk1, sk2 = sk1[joints], sk2[joints]
            valid_idx = np.logical_and(sk1!=0, sk2!=0)
            sk1, sk2 = sk1[valid_idx], sk2[valid_idx]
            sum_dist, num_points = 0, int(len(sk1)/2)
            if num_points == 0:
                return 99999
            else:
                for i in range(num_points): # compute distance between each pair of joint
                    idx = i * 2
                    sum_dist += calc_dist(sk1[idx:idx+2], sk2[idx:idx+2])
                mean_dist = sum_dist / num_points
                mean_dist /= (1.0 + 0.05*num_points) # more points, the better
                return mean_dist

        DIST_THRESH = 0.4 # TODO: modify this
        good_matches = {} # If f1i is matched to f2j and vice versa, the match is good.
        n1, n2 = len(features1), len(features2)
        if n1 and n2:
            dist_matrix = [[cost(f1,f2) for f2 in features2] for f1 in features1]
            dist_matrix = np.array(dist_matrix)

            matches_f1_to_f2 = [dist_matrix[row,:].argmin() for row in range(n1)]
            matches_f2_to_f1 = [dist_matrix[:,col].argmin() for col in range(n2)]

            for i1, i2 in enumerate(matches_f1_to_f2):
                if matches_f2_to_f1[i2]==i1 and dist_matrix[i1,i2]<DIST_THRESH:
                    good_matches[i2]=i1

            if 0:
                print("distance matrix:", dist_matrix)
                print("matches_f1_to_f2:", matches_f1_to_f2)
                print("matches_f1_to_f2:", matches_f2_to_f1)
                print("good_matches:", good_matches)

        return good_matches