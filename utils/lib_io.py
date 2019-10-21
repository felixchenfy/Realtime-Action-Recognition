import simplejson
import cv2
import time
import glob
import os

# see mylib/py: [cnt_action, cnt_clip, cnt_image, str_img_action_type, filepath]
LENGTH_OF_IMAGE_INFO = 5


class DataLoader_WebCam(object):
    def __init__(self, max_framerate=10):
        self.cam = cv2.VideoCapture(0)
        self.num_images = 9999999
        self.frame_period = 1.0/max_framerate*0.999
        self.prev_image_time = time.time() - self.frame_period

    def read_image(self):
        self._sync_to_framerate()

        ret_val, img = self.cam.read()
        self.prev_image_time = time.time()

        img = cv2.flip(img, 1)
        str_img_action_type = "unknown"
        return img, str_img_action_type, ["none"]*LENGTH_OF_IMAGE_INFO

    def _sync_to_framerate(self):
        t_curr = time.time()
        t_wait = self.frame_period - (t_curr - self.prev_image_time)
        if t_wait > 0:
            time.sleep(t_wait)


class DataLoader_folder(object):
    def __init__(self, folder, num_skip=0):
        self.cnt_image = 0
        self.folder = folder
        self.filenames = sorted(glob.glob(folder+'/*'))  # get_filenames
        self.idx_step = num_skip + 1
        self.num_images = int(len(self.filenames) / self.idx_step)

    def read_image(self):
        img = cv2.imread(self.filenames[self.cnt_image])
        self.cnt_image += self.idx_step
        str_img_action_type = "unknown"
        return img, str_img_action_type, ["none"]*LENGTH_OF_IMAGE_INFO


def save_images_info(filepath, images_info):
    folder_path = os.path.dirname(filepath)
    os.makedirs(folder_path, exist_ok=True)
    with open(filepath, 'w') as f:
        simplejson.dump(images_info, f)


def load_images_info(filepath):
    with open(filepath, 'r') as f:
        images_info = simplejson.load(f)
        return images_info
    return None


def save_skeletons(filepath, skeletons):  # 5 + 2*18
    folder_path = os.path.dirname(filepath)
    os.makedirs(folder_path, exist_ok=True)
    with open(filepath, 'w') as f:
        simplejson.dump(skeletons, f)


def load_skeletons(filepath):
    with open(filepath, 'r') as f:
        skeletons = simplejson.load(f)
        return skeletons
    return None


def print_images_info(images_info):
    for img_info in images_info:
        print(img_info)


def int2str(num, idx_len):
    return ("{:0"+str(idx_len)+"d}").format(num)


def int2name(num, idx_len=5, suffix="png"):
    return int2str(num, idx_len)+"." + suffix


def collect_images_info_from_source_images(valid_images_txt, img_suffix="png"):
    '''
    Arguments:
        valid_images_txt {str}: path of the txt file that specifies the training images and their labels.
    Return:
        images_info {list of list}: shape=PxN, where:
            P: number of training images
            N=5: number of tags of that image, including: 
                [cnt_action, cnt_clip, cnt_image, action_type, filepath]
                An example: [8, 49, 2687, 'wave', 'wave_03-02-12-35-10-194/00439.png']                
    '''
    images_info = list()

    with open(valid_images_txt) as f:

        folder_name = None
        action_type = None
        cnt_action = 0
        actions = set()
        action_images_cnt = dict()
        cnt_clip = 0
        cnt_image = 0

        for cnt_line, line in enumerate(f):

            if line.find('_') != -1:  # A new video type
                folder_name = line[:-1]
                action_type = folder_name.split('_')[0]
                if action_type not in actions:
                    cnt_action += 1
                    actions.add(action_type)
                    action_images_cnt[action_type] = 0

            elif len(line) > 1:  # line != "\n"
                # print("Line {}, len ={}, {}".format(cnt_line, len(line), line))
                indices = [int(s) for s in line.split()]
                idx_start = indices[0]
                idx_end = indices[1]
                cnt_clip += 1
                for i in range(idx_start, idx_end+1):
                    filepath = folder_name+"/" + \
                        int2name(i, idx_len=5, suffix=img_suffix)
                    cnt_image += 1
                    action_images_cnt[action_type] += 1

                    # Save: 5 values
                    image_info = [cnt_action, cnt_clip, cnt_image, action_type, filepath]
                    images_info.append(image_info)
                    # An example: [8, 49, 2687, 'wave', 'wave_03-02-12-35-10-194/00439.png']

        print("Num actions = {}".format(len(actions)))
        print("Num training images = {}".format(cnt_image))
        print("Num training of each action:")
        for action in actions:
            print("  {:>8}| {:>4}|".format(action,
                                           action_images_cnt[action]
                                           ))

    return images_info


class ReadValidImagesAndActionTypesByTxt(object):
    ''' This is for reading training images specified by a txt file.
        Each training image should contain a person who is performing certain type of action. 
    '''

    def __init__(self, img_folder, valid_imgs_txt, img_suffix="png"):
        '''
        Arguments:
            img_folder {str}: A folder that contains many sub folders.
                Each subfolder has many images named as xxxxx.png.
            valid_imgs_txt {str}: A txt file which specifies the action types.
                Example:
                    jump_03-12-09-18-26-176
                    58 680

                    jump_03-13-11-27-50-720
                    65 393

                    kick_03-02-12-36-05-185
                    54 62
                    75 84
        '''
        self.images_info = collect_images_info_from_source_images(
            valid_imgs_txt, img_suffix=img_suffix)
        self.imgs_path = img_folder
        self.i = 0
        self.num_images = len(self.images_info)
        print(f"Reading images from txtscript: {img_folder}")
        print(f"Reading images information from: {valid_imgs_txt}")
        print(f"    Num images = {self.num_images}\n")

    def save_images_info(self, filepath):
        folder_path = os.path.dirname(filepath)
        os.makedirs(folder_path, exist_ok=True)
        with open(filepath, 'w') as f:
            simplejson.dump(self.images_info, f)

    def read_image(self):
        '''
        Returns:
            img {RGB image}: 
                Next RGB image from folder. 
            str_img_action_type {str}: 
                Action type obtained from folder name.
            img_info {list}: 
                Something like [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.png"]
        Raise:
            RuntimeError, if fail to read next image due to wrong index or wrong filepath.
        '''
        self.i += 1
        if self.i > len(self.images_info):
            raise RuntimeError(f"There are only {len(self.images_info)} images, "
                               f"but you try to read the {self.i}th image")
        filepath = self.get_filename(self.i)
        img = self.imread(self.i)
        if img is None:
            raise RuntimeError("The image file doesn't exist: " + filepath)
        str_img_action_type = self.get_action_type(self.i)
        img_info = self.get_image_info(self.i)
        return img, str_img_action_type, img_info

    def imread(self, index):
        return cv2.imread(self.imgs_path + self.get_filename(index))

    def get_filename(self, index):
        # The 4th element of
        # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.png"]
        # See "collect_images_info_from_source_images" for the data format
        return self.images_info[index-1][4]

    def get_action_type(self, index):
        # The 3rd element of
        # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.png"]
        # See "collect_images_info_from_source_images" for the data format
        return self.images_info[index-1][3]

    def get_image_info(self, index):
        # Something like [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.png"]
        return self.images_info[index-1]


if __name__ == "__main__":
    '''
    For test case, see "images_info_save_to_file.py"
    '''
