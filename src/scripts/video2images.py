
import cv2
import numpy as np
import sys, os
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

paths = [
    # (CURR_PATH + "../../data_test/", "exercise",".webm"),
    # (CURR_PATH + "../../data_test/", "apple",".mp4"),
    # (CURR_PATH + "../../data_test/", "exercise2",".webm"),
    # (CURR_PATH + "../../data_test/", "walk-stand",".avi"),
    # (CURR_PATH + "../../data_test/", "walk-stand-1",".avi"),
    # (CURR_PATH + "../../data_test/", "walk-stand-2",".avi"),
    # (CURR_PATH + "../../data_test/", "walk-stand-3",".avi"),
    # (CURR_PATH + "../../data_test/", "walk-1",".avi"),
    # (CURR_PATH + "../../data_test/", "sit-1",".avi"),
    (CURR_PATH + "../../data_test/", "sit-2",".avi"),
]

# -- Input
idx = 0
s_folder =  paths[idx][0]
s_video_name_only =  paths[idx][1]
s_video = s_video_name_only+ paths[idx][2] 

# -- Output
s_save_to_folder = s_folder + s_video_name_only + "/"
if not os.path.exists(s_save_to_folder):
    os.makedirs(s_save_to_folder)

# -- Functions
int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)

# -- Read video
cap = cv2.VideoCapture(s_folder + s_video)
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    # if ret == False:
    #     break
    count += 1
    write_to = s_save_to_folder + int2str(count, 5) + ".png"
    if count % 100 ==0:
        print('Read a new frame {} of size {} to {}'.format( count, frame.shape, write_to))
    # if count < 1000:
    #     continue
    # if count > 2000:
    #     break

    # Show and save
    # if count % 10 ==0:
    #     cv2.imshow('frame',frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    cv2.imwrite(write_to, frame)     # save frame as JPEG file      


cap.release()
cv2.destroyAllWindows()