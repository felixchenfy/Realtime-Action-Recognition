The source images for training are stored here.

## Download data

Please download the data from
* Google driver: https://drive.google.com/open?id=1V8rQ5QR5q5zn1NHJhhf-6xIeDdXVtYs9
* or Baidu Cloud: https://pan.baidu.com/s/11M2isbEQnBTQHT3-634AHw

Unzip the data and you will see the folder: `source_images3`. Use it to replace the `data/source_images3`.

Inside the folder, there is a `valid_images.txt`, which describes the label of each image that I used for training. (For your conviniene, I've included it in this repo, and you can view it at [data/source_images3/valid_images.txt](source_images3/valid_images.txt).)

## Data Folder structure

  ```
  data/source_images3
  ├── jump_03-02-12-34-01-795
  ├── jump_03-12-09-18-26-176
  ├── jump_03-13-11-27-50-720
  ├── kick_03-02-12-36-05-185
  ├── kick_03-08-20-32-41-586
  ├── kick_03-12-09-23-41-176
  ├── kick_03-13-16-18-12-361
  ├── punch_03-12-09-21-27-876
  ├── run_03-02-12-31-47-095
  ├── run_03-12-09-15-25-375
  ├── sit_03-02-12-27-51-085
  ├── sit_03-02-12-28-32-893
  ├── sit_03-12-09-25-43-527
  ├── sit_03-13-16-15-56-861
  ├── squat_03-08-20-26-57-195
  ├── squat_03-13-13-21-48-761
  ├── stand_03-08-20-24-55-587
  ├── stand_03-08-20-35-06-287
  ├── stand_03-12-09-17-05-376
  ├── stand_03-13-13-22-37-869
  ├── valid_images.txt
  ├── walk_03-02-12-30-23-393
  ├── walk_03-12-09-13-10-875
  ├── wave_03-08-20-28-39-387
  └── wave_03-13-13-23-25-262
  ```

## Images for training:

  Number of actions = 9  
  Total training images = 11202  
  Number of images of each action:  

  |Label|Number of frames|
  |:---:|:---:|
  punch|  799|  
  walk| 1220|  
  kick| 1162|  
  squat|  964|  
  jump| 1174|  
  run| 1033|  
  wave| 1239|  
  sit| 1908|  
  stand| 1703|  