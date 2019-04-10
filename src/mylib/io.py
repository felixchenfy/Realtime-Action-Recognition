import simplejson
import cv2
import time
import glob
LENGTH_OF_IMAGE_INFO = 5 # see mylib/py: [cnt_action, cnt_clip, cnt_image, img_action_type, filepath]
    
class DataLoader_usbcam(object):
    def __init__(self, max_framerate = 10):
        self.cam = cv2.VideoCapture(0)
        self.num_images = 9999999
        self.frame_period = 1.0/max_framerate*0.999
        self.prev_image_time = time.time() - self.frame_period

    def wait_for_framerate(self):
        t_curr = time.time()
        t_wait = self.frame_period - (t_curr - self.prev_image_time)
        if t_wait > 0:
            time.sleep(t_wait)

    def load_next_image(self):
        self.wait_for_framerate()
        
        ret_val, img = self.cam.read()
        self.prev_image_time = time.time()

        img =cv2.flip(img, 1)
        img_action_type = "unknown"
        return img, img_action_type, ["none"]*LENGTH_OF_IMAGE_INFO

class DataLoader_folder(object):
    def __init__(self, folder, num_skip = 0):
        self.cnt_image = 0
        self.folder = folder
        self.filenames = sorted(glob.glob(folder+'/*')) # get_filenames
        self.idx_step = num_skip + 1
        self.num_images = int( len(self.filenames) / self.idx_step)

    def load_next_image(self):
        img =  cv2.imread(self.filenames[self.cnt_image])
        self.cnt_image += self.idx_step
        img_action_type = "unknown"
        return img, img_action_type, ["none"]*LENGTH_OF_IMAGE_INFO

class DataLoader_txtscript(object):
    def __init__(self, SRC_IMAGE_FOLDER, VALID_IMAGES_TXT):
        self.images_info = collect_images_info_from_source_images(SRC_IMAGE_FOLDER, VALID_IMAGES_TXT)
        self.imgs_path = SRC_IMAGE_FOLDER
        self.i = 0
        self.num_images = len(self.images_info)
        print("Reading images from txtscript: {}\n".format(SRC_IMAGE_FOLDER))
        print("Reading images information from: {}\n".format(VALID_IMAGES_TXT))
        print("    Num images = {}\n".format(self.num_images))

    def save_images_info(self, path):
        with open(path, 'w') as f:
            simplejson.dump(self.images_info, f)

    def load_next_image(self):
        self.i += 1
        filename = self.get_filename(self.i)
        img = self.imread(self.i)
        img_action_type = self.get_action_type(self.i)
        return img, img_action_type, self.get_image_info(self.i)

    def imread(self, index):
        return cv2.imread(self.imgs_path + self.get_filename(index))
    
    def get_filename(self, index):
        # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.png"]
        # See "collect_images_info_from_source_images" for the data format
        return self.images_info[index-1][4] 

    def get_action_type(self, index):
        # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.png"]
        # See "collect_images_info_from_source_images" for the data format
        return self.images_info[index-1][3]
    
    def get_image_info(self, index):
        return self.images_info[index-1] # with a length of LENGTH_OF_IMAGE_INFO


def save_images_info(path, images_info):
    with open(path, 'w') as f:
        simplejson.dump(images_info, f)

def load_images_info(path):
    with open(path, 'r') as f:
        images_info = simplejson.load(f)
        return images_info
    return None

def save_skeletons(filename, skeletons): # 5 + 2*18
    with open(filename, 'w') as f:
        simplejson.dump(skeletons, f)

def load_skeletons(filename):
    with open(filename, 'r') as f:
        skeletons = simplejson.load(f)
        return skeletons
    return None

def print_images_info(images_info):
    for img_info in images_info:
        print(img_info)

def int2str(num, blank):
    return ("{:0"+str(blank)+"d}").format(num)

def int2name(num):
    return int2str(num, 5)+".png"

def collect_images_info_from_source_images(path, valid_images_txt):
    images_info = list()

    with open(path + valid_images_txt) as f:

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
                    action_images_cnt[action_type]=0

            elif len(line) > 1:  # line != "\n"
                # print("Line {}, len ={}, {}".format(cnt_line, len(line), line))
                indices = [int(s) for s in line.split()]
                idx_start = indices[0]
                idx_end = indices[1]
                cnt_clip += 1
                for i in range(idx_start, idx_end+1):
                    filepath = folder_name+"/"+int2name(i)
                    cnt_image += 1
                    action_images_cnt[action_type]+=1
                    
                    # Save: 5 values
                    d = [cnt_action,cnt_clip, cnt_image, action_type, filepath]
                    images_info.append(d)
                    # An example: [8, 49, 2687, 'wave', 'wave_03-02-12-35-10-194/00439.png']
    
        print("Num actions = {}".format(len(actions)))
        print("Num training images = {}".format(cnt_image))
        print("Num training of each action:")
        for action in actions:
            print("  {:>8}| {:>4}|".format(action,
                action_images_cnt[action]
            ))

    return images_info
    '''
    Other notes
    { read line:
        line = fp.readline()
        if fail, then line == None
    }
    '''


        
if __name__=="__main__":
    '''
    For test case, see "images_info_save_to_file.py"
    '''
