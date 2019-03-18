import simplejson
import cv2


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
