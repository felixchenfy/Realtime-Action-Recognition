
# Multi-person Real-time Action Recognition Based-on Human Skeleton

![](https://github.com/felixchenfy/Data-Storage/raw/master/EECS-433-Pattern-Recognition/recog_actions.gif)

![](https://github.com/felixchenfy/Data-Storage/raw/master/EECS-433-Pattern-Recognition/recog_actions2.gif)


**Highlights**: 
9 actions; multiple people (<=5); Real-time and multi-frame based recognition algorithm.

**Updates**: On 2019-10-26, I refactored the code; added more comments; and put all settings into the [config/config.yaml](config/config.yaml) file, including: action classes, input and output of each file, OpenPose settings, etc. 

**Project**: This is my final project for EECS-433 Pattern Recognition in Northwestern Univeristy on March 2019. A simpler version where two teammates and I worked on is [here](https://github.com/ChengeYang/Human-Pose-Estimation-Benchmarking-and-Action-Recognition).

**Warning:** Since I used the 10 fps video and 0.5s-window for training, you must also limit your video fps to be about 10 fps (7~12 fps) if you want to test my pretrained model on your own video or web camera. 

**Contents:**


# 1. Algorithm


I collected videos of 9 Types of actions: `['stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave']`. The total video lengths are about 20 mins, containing about 10000 video frames recorded at 10 frames per second.

The workflow of the algorithm is:
*  Get the joints' positions by [OpenPose](https://github.com/ildoonet/tf-pose-estimation).  
*  Track each person. Euclidean distance between the joints of two skeletons is used for matching two skeletons. 
See `class Tracker` in [lib_tracker.py](utils/lib_tracker.py)
*  Fill in a person's missing joints by these joints' relative pos in previous frame.  See `class FeatureGenerator` in [lib_feature_proc.py](utils/lib_feature_proc.py). So does the following.
*  Add noise to the (x, y) joint positions to try to augment data.
*  Use a window size of 0.5s (5 frames) to extract features.    
*  Extract features of (1) body velocity and (2) normalized joint positions and (3) joint velocities.
*  Apply PCA to reduce feature dimension to 80.  Classify by DNN of 3 layers of 50x50x50 (or switching to other classifiers in one line). See `class ClassifierOfflineTrain` in [lib_classifier.py](utils/lib_classifier.py)
*  Mean filtering the prediction scores between 2 frames. Add label above the person if the score is larger than 0.8. See `class ClassifierOnlineTest` in [lib_classifier.py](utils/lib_classifier.py)

For more details about how the features are extracted, please see my [report](https://github.com/felixchenfy/Data-Storage/blob/master/EECS-433-Pattern-Recognition/FeiyuChen_Report_EECS433.pdf).





# 2. Install Dependency (OpenPose)

First, Python >= 3.6.

I used the OpenPose from this Github: [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation). First download it:

```
export MyRoot=$PWD
cd src/githubs  
git clone https://github.com/ildoonet/tf-pose-estimation  
```

Follow its tutorial [here](https://github.com/ildoonet/tf-pose-estimation#install-1) to download the "cmu" model. As for the "mobilenet_thin", it's already inside the folder.  

```
$ cd tf-pose-estimation/models/graph/cmu  
$ bash download.sh  
```

Then install dependencies. I listed my installation steps as bellow:
```
conda create -n tf tensorflow-gpu
conda activate tf

cd $MyRoot
pip install -r requirements1.txt
conda install jupyter tqdm
sudo apt install swig

pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

cd $MyRoot/src/githubs/tf-pose-estimation/tf_pose/pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

Make sure you can successfully run its demo examples:
```
cd $MyRoot/src/githubs/tf-pose-estimation
python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
```

# 3. Data


## Download
Follow the instructions in [data/download_link.md](data/download_link.md) to download the data.

## Format

Each data subfolder contains images named as `00001.png`, `00002.png`, etc.

The label and index are configured by this txt [data/source_images3/valid_images.txt](data/source_images3/valid_images.txt). A snapshot is shown below:
```
jump_03-02-12-34-01-795
52 59
72 79
```
The 1st line is the data folder name, which should start with `"${class_name}_"`. In the above example, `jump` is the class.   

The 2nd and following lines specify the `staring index` and `ending index` of the video that corresponds to the `jump` class. In the above example, I'm jumping during the 52~59 & 72~79  frames of the video.

## Classes

The classes are set in [config/config.yaml](config/config.yaml) under the key word `classes`. No matter how many classes you put in the training data (set by the folder name), only the classes set by this **config.yaml** file are used for training and inference.



# 3. How to run: Inference

## Test on webcam
> $ python src/run_detector.py --source webcam

* 2). Test video frames from a folder
  > $ python src/run_detector.py --source folder  

  But before running, you need to modify the folder path in "**def set_source_images_from_folder()**" in [src/run_detector.py](src/run_detector.py).

# 4. How to run: Training

**1).** First, detect skeleton from training images and save result to txt:
> $ python src/run_detector.py --source txtscript

* Input: Images are from "**data/source_images3/**". See [this file](data/download_link.md) for downloading.
* Output:  
    (1) Skeleton positions of each image in "**src/skeleton_data/skeletons5/**" in txt format.  
    (2) Images with drawn skeleton, saved in "**src/skeleton_data/skeletons5_images/**".

**2).** Second, put skeleton from multiple txt into a single txt file:
> $ python src/scripts/skeletons_info_generator.py

* Input: "**src/skeleton_data/skeletons5/**". 
* Output:  "**src/skeleton_data/skeletons5_info.txt**"

**3).** Third, open jupyter notebook, and run this file: "**src/Train.ipynb**".  
The trained model will be saved to "**model/trained_classifier.pickle**".

Now, it's ready to run the real-time demo from your web camera.

# 5. Result and Performance

Unfortunately this project only works well on myself, because I only used the video of myself.

The performance is bad for (1) people who have different body shape, (2) people are far from the camera. **How to improve?** I guess the first thing to do is to collect larger training set from different people. Then, improve the data augmentation and featuer selection.

Besides, my simple tracking algorithm only works for a few number of people (maybe 5). 

So I guess you can only use this project for course demo, but not for any useful applications ... T.T 
