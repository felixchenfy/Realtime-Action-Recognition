
Real-time Action Recognition Based-on Human Skeleton
=========================================================

This is my final project for EECS-433 Pattern Recognition. It's based on this [github]( https://github.com/ChengeYang/Human-Pose-Estimation-Benchmarking-and-Action-Recognition), where Chenge and Zhicheng and me worked out a simpler version.

# 1. Overview
## 1.1 Demo
![](https://github.com/felixchenfy/Data-Storage/raw/master/EECS-433-Pattern-Recognition/recog_actions.gif)

This demo was screen recorded. It ran at 10 fps on GTX 1070. (And it should be running at 10 fps, because I used this framerate for training.)

## 1.2 Training Data
9 Types of actions: wave, stand, punch, kick, squat, sit, walk, run, jump.  
The total video length is 19 mins, containing 11000 video frames recorded at 10 frames per second.

## 1.3 Algorithm
*  Get the joints' positions by [OpenPose](https://github.com/ildoonet/tf-pose-estimation).  
*  Fill in missing joints by their relative pos in previous frame.  
*  Use a window size of 0.5s (5 frames) to extract features.    
*  Extract features of (1) body velocity and (2) normalized joint positions and (3) joint velocities. Then apply PCA.  
*  Classify by DNN of 3 layers of 100x100x100 (or switching to other classifiers in one line).
*  Mean filter the prediction scores of 2 frames, and add label above the person if the score is larger than 0.85.   

For more details about the algorithm, please see my report: https://github.com/felixchenfy/Data-Storage/blob/master/EECS-433-Pattern-Recognition/FeiyuChen_Report_EECS433.pdf.

I'll update this README soon. 

# 2. Installation

## 2.1 Install OpenPose

I used the OpenPose from this Github: [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation). First download it:

```
export MyRoot=$PWD
cd src/githubs  
git clone https://github.com/ildoonet/tf-pose-estimation  
```

Follow its tutorial [here](https://github.com/ildoonet/tf-pose-estimation#install-1) to download 2 models trained by the author, named as "cmu" and "mobilenet_thin" respectively. 

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

## 2.2 Down
## 

## How to run: Test
TODO 

## **How to run: Train**
TODO 

## **Future work**
Track multiple people

