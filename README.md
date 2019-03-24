
Real-time Action Recognition Based-on Human Skeleton
=========================================================

This is my final project for EECS-433 Pattern Recognition. It's based on this [github]( https://github.com/ChengeYang/Human-Pose-Estimation-Benchmarking-and-Action-Recognition), where Chenge and Zhicheng and me worked out a simpler version.

## **Demo:**  
![](https://github.com/felixchenfy/Data-Storage/raw/master/EECS-433-Pattern-Recognition/recog_actions.gif)

## **Training Data:**  
9 Types of actions: wave, stand, punch, kick, squat, sit, walk, run, jump.  
19 mins of video with 10 fps (11000+ images).

## **Method:**
*  Get the joints' positions by [OpenPose](https://github.com/ildoonet/tf-pose-estimation).  
*  Fill in missing joints by their relative pos in previous frame.  
*  Use a window size of 0.5s (5 frames) to extract features.    
*  Extract features of (1) body velocity and (2) normalized joint positions and (3) joint velocities. Then apply PCA.  
*  Classify by DNN of 3 layers of 100x100x100 (or switching to other classifiers in one line).
*  Mean filter the prediction scores of 2 frames, and add label above the person if the score is larger than 0.85.   

For more details about the algorithms, please see my report: https://github.com/felixchenfy/Data-Storage/blob/master/EECS-433-Pattern-Recognition/FeiyuChen_Report_EECS433.pdf 

I'll update this README soon. 

## **Installation**
TODO 

## **How to run: Test**
TODO 

## **How to run: Train**
TODO 

## **Future work**
Track multiple people

