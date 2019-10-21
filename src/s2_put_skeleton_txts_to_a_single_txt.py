
'''
Input:
    skeletons/00001.txt ~ skeletons/xxxxx.txt, read from `SRC_DETECTED_SKELETONS_FOLDER`
Output:
    skeletons_info.txt, saved to `DST_SINGLE_SKELETONS_TXT_FILE`
'''

import simplejson

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    from utils.lib_io import load_skeletons
    from utils.lib_commons import get_filenames

# -- Settings
SRC_DETECTED_SKELETONS_FOLDER = ROOT + "data/proc_skeletons/skeleton_res/"
DST_SINGLE_SKELETONS_TXT_FILE = ROOT + "data/proc_skeletons/skeletons_info.txt"
IDX_PERSON = 0  # Only use the skeleton of the 0th person in each image

# -- Helper function


def int2str(num, idx_len): return ("{:0"+str(idx_len)+"d}").format(num)


def load_skeleton_from_ith_txt(i):
    ''' 
    Arguments:
        i {int}: the ith skeleton txt. Zero-based index.
            If there are mutliple people, then there are multiple skeletons' positions in this txt.
    '''
    filename = SRC_DETECTED_SKELETONS_FOLDER + int2str(i, 5) + ".txt"
    skeletons_in_ith_txt = load_skeletons(filename)
    return skeletons_in_ith_txt


def get_content_size_in_one_skeleton_data(filepaths):
    ''' Find a non-empty txt file, and then get the content size of one skeleton data.
    The content size should be 41, where:
    41 = 5 + 36.
        5: [cnt_action, cnt_clip, cnt_image, action_type, filepath]
            See utils.lib_io.collect_images_info_from_source_images for more details
        36: 18 joints * 2 xy positions
    '''
    for i in range(len(filepaths)):
        skeletons = load_skeleton_from_ith_txt(i)
        if len(skeletons):
            skeleton = skeletons[IDX_PERSON]
            data_size = len(skeleton)
            assert(data_size == 41)
            return data_size
    raise RuntimeError(
        f"All txts under {SRC_DETECTED_SKELETONS_FOLDER} have no skeleton data.")


# -- Main
if __name__ == "__main__":

    # -- Get skeleton filenames
    filepaths = get_filenames(SRC_DETECTED_SKELETONS_FOLDER,
                              use_sort=True, with_folder_path=True)
    num_skeletons = len(filepaths)

    # -- Check content size
    content_size = get_content_size_in_one_skeleton_data(filepaths)
    print("content size of one skeleton is {content_size}")

    # -- Read in skeletons and push to all_skeletons
    all_skeletons = []
    for i in range(num_skeletons):

        # Read skeletons from a txt
        skeletons = load_skeleton_from_ith_txt(i)

        # Deal with empty data
        if len(skeletons):
            skeleton = skeletons[IDX_PERSON]
        else:
            skeleton = [0] * content_size

        # Push to result
        all_skeletons.append(skeleton)

        # Print
        if i == 1 or i % 100 == 0:
            print("{}/{}".format(i, num_skeletons))

    # -- Save to txt
    with open(DST_SINGLE_SKELETONS_TXT_FILE, 'w') as f:
        simplejson.dump(all_skeletons, f)
    print(f"There are {len(all_skeletons)} skeleton data.")
    print(f"They are saved to {DST_SINGLE_SKELETONS_TXT_FILE}")