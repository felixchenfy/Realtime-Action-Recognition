


PROJECT_PATH = "/home/feiyu/Desktop/C1/FinalProject/"
CURRENT_FOLDER = "src/images_to_skeletons"
DATA_FOLDERs = "data/source_images/"
valid_images_txt = "valid_images.txt"

Add path{
    import sys, os
    def get_curr_path():
        p = os.path.join(os.path.dirname(__file__))
        return "" if len(p) == 0 else (p+"/")
    CURR_PATH = get_curr_path()
    sys.path.append(CURR_PATH)
}
or{
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
}
