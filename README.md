
Real-time Action Recognition Based-on Human Skeleton
=========================================================

This is my final project for EECS-433 Pattern Recognition. It's based on this [github]( https://github.com/ChengeYang/Human-Pose-Estimation-Benchmarking-and-Action-Recognition), where Chenge and Zhicheng and me worked out a simpler version.

**Demo:**  
![](https://github.com/felixchenfy/Data-Storage/raw/master/EECS-433-Pattern-Recognition/recog_actions.gif)

**Training Data:**  
9 Types of actions: wave, stand, punch, kick, squat, sit, walk, run, jump.  
19 mins of video with 10 fps.

**Method:**
*  Get the joints' positions by [OpenPose](https://github.com/ildoonet/tf-pose-estimation).  
*  Use a window size of 0.5s (5 frames).    
*  Fill in missing joints by their relative pos in previous frame.  
*  Extract features of joint positions and velocities, and apply PCA.  
*  Classify by DNN of 3 layers of 100x100x100 (or switching to other classifiers in one line).  

I will complete the README by 03/20.

