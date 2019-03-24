

# -------------------------
# Some project related Github repos  



### Action recognition based on Skeleton  
https://github.com/topics/skeleton-based-action-recognition  



### human-pose-estimation-opencv  
https://github.com/quanhua92/human-pose-estimation-opencv  
CPU, OpenCV dnn module,  
Results: 3-4 fps, performance is just so fine  


# ----------------------------
# Installation notes


## Torch  
https://pytorch.org/  

On my CPU: pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl  
pip3 install torchvision  



# ---------------------------------
# Python notes  

PROJECT_PATH = "/home/feiyu/Desktop/C1/FinalProject/"  
CURRENT_FOLDER = "src/images_to_skeletons"  
DATA_FOLDERs = "data/source_images/"  
valid_images_txt = "valid_images.txt" 
 
```
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
```