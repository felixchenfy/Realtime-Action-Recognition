
import glob
import cv2
import argparse
import os

SRC_FORMAT = "png"
DST_FORMAT = "jpg"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a folder of images into a video.")
    parser.add_argument("-i", "--src-folder", type=str, required=True)
    parser.add_argument("-o", "--dst-folder", type=str, required=True)
    parser.add_argument("-s", "--is-including-subfolders", 
                        type=str, required=False, default=False)

    args = parser.parse_args()
    return args

def renameImages(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    filenames = glob.glob(src_folder + "/*." + SRC_FORMAT)
    for i, filename in enumerate(filenames):
        img = cv2.imread(filename)
        basename = "".join(os.path.basename(filename).split('.')[:-1])
        new_name = dst_folder + "/" + basename + "." + DST_FORMAT
        cv2.imwrite(new_name, img)
        print("{}/{}: Save image to {}".format(
            i, len(filenames), new_name))

def main():
    args = parse_args()
    
    # Rename images under the folder
    renameImages(args.src_folder, args.dst_folder)

    # Rename images under the subfolders   
    if args.is_including_subfolders in ["1", "true", "True", "Yes", "yes"]:
        subfolders = [f for f in glob.glob(args.src_folder + "/*") if os.path.isdir(f)]
        for src in subfolders:
            dst = args.dst_folder + "/" + src.split("/")[-1]
            renameImages(src, dst)
            
if __name__ == "__main__":
    main()