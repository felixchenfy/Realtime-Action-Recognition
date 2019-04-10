
Multi-person Real-time Action Recognition Based-on Human Skeleton
=========================================================

This is my final project for EECS-433 Pattern Recognition. It's based on this [github]( https://github.com/ChengeYang/Human-Pose-Estimation-Benchmarking-and-Action-Recognition), where Chenge and Zhicheng and me worked out a simpler version.

**Features**: 9 actions; 10 frames/s; multiple (<=5) people.

**Contents:**
- [Multi-person Real-time Action Recognition Based-on Human Skeleton](#multi-person-real-time-action-recognition-based-on-human-skeleton)
- [1. Overview](#1-overview)
  - [1.1 Demo](#11-demo)
  - [1.2 Training Data](#12-training-data)
  - [1.3 Algorithm](#13-algorithm)
- [2. Install OpenPose](#2-install-openpose)
- [3. How to run: Inference](#3-how-to-run-inference)
- [4. How to run: Training](#4-how-to-run-training)

**TODO**: I'll put a multi-person demo later before 2019.4.16.

# 1. Overview
## 1.1 Demo

Old demo:  
![](https://github.com/felixchenfy/Data-Storage/raw/master/EECS-433-Pattern-Recognition/recog_actions.gif)

New demo [TODO: Add this later]: This demo was screen recorded. It ran at 10 fps on GTX 1070. (And it should be running at 10 fps, because I used this framerate for training.)

## 1.2 Training Data
9 Types of actions: wave, stand, punch, kick, squat, sit, walk, run, jump.  
The total video lengths are about 20 mins, containing more than 10000 video frames recorded at 10 frames per second.

## 1.3 Algorithm
*  Get the joints' positions by [OpenPose](https://github.com/ildoonet/tf-pose-estimation).  
*  Track each person. The distance metric is based on joints positions.
*  Fill in a person's missing joints by these joints' relative pos in previous frame.  
*  Use a window size of 0.5s (5 frames) to extract features.    
*  Extract features of (1) body velocity and (2) normalized joint positions and (3) joint velocities. Then apply PCA.  
*  Classify by DNN of 3 layers of 50x50x50 (or switching to other classifiers in one line).
*  Mean filter the prediction scores of 2 frames, and add label above the person if the score is larger than 0.85.   

**Note**:
For more details about the algorithm, please see my report: https://github.com/felixchenfy/Data-Storage/blob/master/EECS-433-Pattern-Recognition/FeiyuChen_Report_EECS433.pdf.   
Difference I've made after writing this report: (1) Support multiple people. (2) Tuned some parameters to improve speed.


# 2. Install OpenPose

I used the OpenPose from this Github: [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation). First download it:

```
export MyRoot=$PWD
cd src/githubs  
git clone https://github.com/ildoonet/tf-pose-estimation  
```

Follow its tutorial [here](https://github.com/ildoonet/tf-pose-estimation#install-1) to download the two models trained by the author, named as "cmu" and "mobilenet_thin" respectively. 

Then install dependencies. I listed my installation steps as bellow:
```
conda create -n tf tensorflow-gpu
conda activate tf
cd $MyRoot
pip install -r requirements1.txt
cd src/githubs/tf-pose-estimation/tf_pose/pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

Make sure you can successfully run its demo examples:
```
cd $MyRoot/src/tf-pose-estimation
python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
```


# 3. How to run: Inference

* 1). Test on webcam
  > $ python src/run_detector.py --source webcam

* 2). Test video frames from a folder
  > $ python src/run_detector.py --source folder  

  But before running, you need to modify the folder path in "**def set_source_images_from_folder()**" in [src/run_detector](src/run_detector).

# 4. How to run: Training

**1).** First, detect skeleton from training images and save result to txt:
> $ python src/run_detector.py --source txtscript

* Input: Images are from "**data/source_images3/**". See [this file](data/download_link.md) for downloading.
* Output:  
    (1) Skeleton of each image in "**src/skeleton_data/skeletons5/**" in txt format.  
    (2) Images with drawn skeleton in "**src/skeleton_data/skeletons5_images/**".

**2).** Second, put skeleton from multiple txt into a single txt file:
> $ python src/scripts/skeletons_info_generator.py

* Input: "**src/skeleton_data/skeletons5/**". 
* Output:  "**src/skeleton_data/skeletons5_info.txt**"

**3).** Third, open jupyter notebook, and run this file: "**src/Train.ipynb**".  
The trained model will be saved to "**model/trained_classifier.pickle**".

