import numpy as np
import math
from collections import deque

# Math-----------------------------------------------
PI = np.pi
Inf = float("inf")

calc_dist = lambda p1, p0: math.sqrt((p1[0]-p0[0])**2+(p1[1]-p0[1])**2)

def pi2pi(x):
    if x>PI:
        x-=2*PI
    if x<=-PI:
        x+=2*PI
    return x

def calc_relative_angle(x1, y1, x0, y0, base_angle):
    # compute rotation from {base_angle} to {(x0,y0)->(x1,y1)}
    if (y1==y0) and (x1==x0):
        return 0
    a1 = np.arctan2(y1-y0, x1-x0)
    return pi2pi(a1 - base_angle)

def calc_relative_angle_v2(p1, p0, base_angle):
    return calc_relative_angle(p1[0], p1[1], p0[0], p0[1], base_angle)

    
# -----------------------------------------------
# ------------------ Process skeleton
# -----------------------------------------------

NECK = 0
L_ARMS = [1,2,3]
R_ARMS = [4,5,6]
L_LEGS = [8,9]
R_LEGS = [11,12]
ARMS_LEGS = L_ARMS + R_ARMS + L_LEGS + R_LEGS
L_THIGH = 7
R_THIGH = 10
NotANum = 0

def get_joint(x, idx):
    px = x[idx]
    py = x[idx+1]
    return px, py

def set_joint(x, idx, px, py):
    x[idx] = px
    x[idx+1] = py
    return


# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# Feature selection/extraction/reduction 
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------


class ProcFtr(object):
        
    @staticmethod
    def retrain_only_body_joints(x): # This is the first step to deal with skeleton.
        # For all other codes, the indexing of joints are formated after this function.
        # Joints: neck, arms, legs.
        x = x.copy()
        return x[2:2+13*2]

    # @staticmethod
    # def drop_arms_and_legs(x):
    #     N = len(ARMS_LEGS)
    #     thre = 0.3 
    #     ra = np.random.random()
    #     if ra<thre:
    #         jonit_idx = int((ra / thre)*N)
    #         set_joint(x, joint_idx, NotANum, NotANum)

    @staticmethod
    def check_valid(x):
        def check_(x0, idx):
            return x0[idx]!=0 and x0[idx+1]!=0 
        return check_(x, NECK) and (check_(x, L_THIGH) or check_(x, R_THIGH))

    # -- Get height of the body
    @staticmethod
    def get_body_height(x):
        if 0:
            px0, py0 = get_joint(x, NECK)
            px_l_thigh, py_l_thigh = get_joint(x, L_THIGH)
            px_r_thigh, py_r_thigh = get_joint(x, R_THIGH)

            if px_l_thigh == NotANum and px_r_thigh == NotANum:
                return 1
                
            if px_l_thigh == NotANum:
                px_l_thigh, py_l_thigh = get_joint(x, R_THIGH)

            if px_r_thigh == NotANum:
                px_r_thigh, py_r_thigh = get_joint(x, L_THIGH)

            assert px_r_thigh != NotANum

            px_mid = (px_l_thigh+px_r_thigh)/2
            py_mid = (py_l_thigh+py_r_thigh)/2

            body_height = math.sqrt((px0-px_mid)**2 + (py0-py_mid)**2)
            return body_height
        else:
            px = x[0::2]
            py = x[1::2]
            return np.max(py) - np.min(py)
    
    @staticmethod
    def remove_body_offset(x):
        x = x.copy()
        if 0:
            # -- Minus the neck
            px0, py0 = get_joint(x, NECK)
            x[0::2] = x[0::2] - px0
            x[1::2] = x[1::2] - py0
        else:
            x[0::2] -= x[0::2].mean()
            x[1::2] -= x[1::2].mean()

        return x

    @staticmethod
    def joint_pos_2_angle_and_length(x):
            
        # ---------------------- Get joint positions ----------------------
        class JointPosExtractor(object):
            def __init__(self, x):
                self.x = x
                self.i = 0
            def get_next_point(self):
                p = [self.x[self.i], self.x[self.i+1]]
                self.i += 2
                return p
        tmp = JointPosExtractor(x)
        
        pneck = tmp.get_next_point()
        
        prshoulder = tmp.get_next_point()
        prelbow = tmp.get_next_point()
        prwrist = tmp.get_next_point()
        
        plshoulder = tmp.get_next_point()
        plelbow = tmp.get_next_point()
        plwrist = tmp.get_next_point()

        prhip = tmp.get_next_point()
        prknee = tmp.get_next_point()
        prankle = tmp.get_next_point()
        
        plhip = tmp.get_next_point()
        plknee = tmp.get_next_point()
        plankle = tmp.get_next_point()
        
        # ---------------------- Get joint angels ----------------------
                
        class Get12Angles(object):
            def __init__(self):
                self.j = 0
                self.f_angles = np.zeros((12,))
                self.x_lengths = np.zeros((12,))
            
            def set_next_angle_len(self, next_joint, base_joint, base_angle):
                angle=calc_relative_angle_v2(next_joint, base_joint, base_angle)
                dist = calc_dist(next_joint, base_joint)
                self.f_angles[self.j]=angle
                self.x_lengths[self.j]=dist
                self.j+=1

        tmp2 = Get12Angles()
        
        tmp2.set_next_angle_len(prshoulder, pneck, PI) # r-shoulder
        tmp2.set_next_angle_len(prelbow, prshoulder, PI/2) # r-elbow
        tmp2.set_next_angle_len(prwrist, prelbow, PI/2) # r-wrist
        
        tmp2.set_next_angle_len(plshoulder, pneck, 0) # l-shoulder
        tmp2.set_next_angle_len(plelbow, plshoulder, PI/2) # l-elbow
        tmp2.set_next_angle_len(plwrist, plelbow, PI/2) # l-wrist
        
        tmp2.set_next_angle_len(prhip, pneck, PI/2+PI/18)
        tmp2.set_next_angle_len(prknee, prhip, PI/2) 
        tmp2.set_next_angle_len(prankle, prknee, PI/2)
        
        tmp2.set_next_angle_len(plhip, pneck, PI/2-PI/18)
        tmp2.set_next_angle_len(plknee, plhip, PI/2) 
        tmp2.set_next_angle_len(plankle, plknee, PI/2)
        
        # Output
        # print([val/PI*180 for val in tmp2.f_angles])
        f_angles = tmp2.f_angles
        f_lens = tmp2.x_lengths
        # x_res = np.concatenate((tmp2.f_angles, tmp2.x_lengths))
        return f_angles, f_lens

# =================================================
# =================================================
# =================================================
# =================================================
# =================================================

class FeatureGenerator(object):
    def __init__(self, config_add_noise=False):
        self.reset()
        self.FEATURE_T_LEN = 5
        self.config_add_noise = config_add_noise
        pass

    def reset(self):
        self.x_deque = deque()
        self.angles_deque = deque()
        self.lens_deque = deque()
        self.prev_x = None

    def maintain_deque_size(self):
        if len(self.x_deque) > self.FEATURE_T_LEN:
            self.x_deque.popleft()
            self.angles_deque.popleft()
            self.lens_deque.popleft()

    def add_curr_skeleton(self, skeleton):
        # return: bool_success, features

        x = ProcFtr.retrain_only_body_joints(skeleton) # return (skeleton.copy())[2:2+13*2]

        if ProcFtr.check_valid(x) == False:
            self.reset()
            return False, None

        else:
            # Fill zeros, compute angles/lens
            self.fill_zeros(x)
            if self.config_add_noise:
                self.add_noises(x)
            angles, lens = ProcFtr.joint_pos_2_angle_and_length(x)

            # Push to deque
            self.x_deque.append(x)
            self.angles_deque.append(angles)
            self.lens_deque.append(lens)

            self.maintain_deque_size()
            self.prev_x = x.copy()

            # Extract features
            if len(self.x_deque)>=self.FEATURE_T_LEN:

                # -- Normalize all 1~t features
                h_list = [ProcFtr.get_body_height(xi) for xi in self.x_deque]
                mean_height = np.mean(h_list)
                xnorm_list = [ ProcFtr.remove_body_offset(xi)/mean_height for xi in self.x_deque]

                # -- Get features of pose/angles/lens
                f_poses = self.deque_to_1darray(xnorm_list)
                f_angles = self.deque_to_1darray(self.angles_deque)
                f_lens = self.deque_to_1darray(self.lens_deque) / mean_height

                # -- Get features of motion
                f_v_center = self.compute_v_center(self.x_deque, step = 1) / mean_height # len = (t=4)*2 = 8
                f_v_center = np.repeat(f_v_center, 10) # Add weights to this feature
                f_v_joints = self.compute_v_all_joints(xnorm_list, step = 1) # len = (t=(5-1)/step)*12*2 = 96

                # -- Output (Choose some features you want)
                # print("f_poses: ",f_poses.shape)
                # print("f_v_joints: ",f_v_joints.shape)
                # print("f_v_center: ",f_v_center.shape)
                features = np.concatenate( (f_poses, f_v_joints, f_v_center) )
                # features = self.deque_to_1darray(self.x_deque)


                return True, features.copy()
            else:
                return False, None
    
    def compute_v_center(self, x_deque, step):
        vel = []
        for i in range(0, len(x_deque) - step, step):
            dxdy = x_deque[i+step][0:2] - x_deque[i][0:2]
            vel += dxdy.tolist()
        return np.array(vel)
    
    def compute_v_all_joints(self, xnorm_list, step):
        vel = []
        for i in range(0, len(xnorm_list) - step, step):
            dxdy = xnorm_list[i+step][:] - xnorm_list[i][:]
            vel += dxdy.tolist()
        return np.array(vel)
        
    def fill_zeros(self, x):
        if self.prev_x is not None:
            def get_x_y_x0_y0(xxx):
                px = xxx[0::2]
                py = xxx[1::2]
                px0, py0 = get_joint(xxx, NECK)
                return px, py, px0, py0
            curr_px, curr_py, curr_px0, curr_py0 = get_x_y_x0_y0(x)
            prev_px, prev_py, prev_px0, prev_py0 = get_x_y_x0_y0(self.prev_x)

            miss_px = np.where(curr_px == NotANum)
            miss_py = np.where(curr_py == NotANum)
            curr_px[miss_px] = curr_px0 + (prev_px[miss_px] - prev_px0)
            curr_py[miss_py] = curr_py0 + (prev_py[miss_px] - prev_py0)
    
    def add_noises(self, x):
        N = len(x)//2 # joints number
        def rand_noise(size, intense):
            return (np.random.random(size)*2 - 1.0) * intense

        if 1: # absolute noise
            NOISE_INTENSE = 0.01 # 200x200 image, 1 pixel = 0.005
            noises = rand_noise((2*N,), NOISE_INTENSE)
        else: # relative noise
            pass
            # NOISE_INTENSE = 0.05
            # width = max(x[::2]) - min(x[::2])
            # width_noises = width * (np.random.random((N,))*2 - 1.0) * NOISE_INTENSE
            # height = max(x[1::2]) - min(x[1::2])
            # height_noises = height * (np.random.random((N,))*2 - 1.0) * NOISE_INTENSE
        for i in range(2*N):
            x[i] += noises[i] if x[i] != 0 else 0

    def deque_to_1darray(self, deque_data):
        features = []
        for i in range(len(deque_data)):
            next_feature = deque_data[i].tolist()
            features += next_feature
        features = np.array(features)
        return features

    def deque_to_2darray(self, deque_data):
        features = []
        for i in range(len(deque_data)):
            next_feature = deque_data[i].tolist()
            features.append( next_feature )
        features = np.array(features)
        return features



