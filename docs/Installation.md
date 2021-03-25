# Installation & Quick start

### Environment

The code is developed using python 3.6.12, pytorch-1.4.0, and CUDA 10.0 on Ubuntu 18.04. 
For our experiments, we used 2 NVIDIA 2080Ti GPUs.



### Installation

1.Create a conda virtual environment and activate it

```
conda create -n DCPose python=3.6.12
source activate DCPose
```



2.Install dependencies through [DCPose_requirements.txt](../DCPose_requirement.txt)

```
pip install -r DCPose_requirement.txt
```



3.Install DCN

```
cd thirdparty/deform_conv
python setup.py develop
```



4.Download our [pretrained models and supplementary](https://drive.google.com/drive/folders/1VPcwo9jVhnJpf5GVwR2-za1PFE6bCOXE?usp=sharing) .Put it in the directory `${DCPose_SUPP_DIR}`. (Note that part of the pre-trained models are available, and more pre-train models will be released soon later.)
 

### Data preparation

First, create a folder `${DATASET_DIR}`  to store the data of PoseTrack17 and PoseTrack18.

The directory structure should look like this:

```
${DATASET_DIR}
	|--${POSETRACK17_DIR}  
	|--${POSETRACK18_DIR}
	
# For example, our directory structure is as follows.
# If you don't know much about configuration file(.yaml), please refer to our settings.
DataSet
	|--PoseTrack17
	|--PoseTrack18
```

**For PoseTrack17 data**, we use a slightly modified version of the PoseTrack dataset where we rename the frames to follow `%08d` format, with first frame indexed as 1 (i.e. `00000001.jpg`). First, download the data from [PoseTrack download page](https://posetrack.net/users/download.php). Then, rename the frames for each video as described above using [this script](https://github.com/facebookresearch/DetectAndTrack/blob/master/tools/gen_posetrack_json.py).  

Like [PoseWarper](https://github.com/facebookresearch/PoseWarper), We provide all the required JSON files, which have already been converted to COCO format. Evaluation is performed using the official PoseTrack evaluation code, [poseval](https://github.com/leonid-pishchulin/poseval), which uses [py-motmetrics](https://github.com/cheind/py-motmetrics) internally. We also provide required MAT/JSON files that are required for the evaluation.

Your extracted PoseTrack17  directory should look like this:

```
|--${POSETRACK17_DIR}
	|--images
        |-- bonn
        `-- bonn_5sec
        `-- bonn_mpii_test_5sec
        `-- bonn_mpii_test_v2_5sec
        `-- bonn_mpii_train_5sec
        `-- bonn_mpii_train_v2_5sec
        `-- mpii
        `-- mpii_5sec
    |--images_renamed   # first frame indexed as 1  (i.e. 00000001.jpg)
     	|-- bonn
        `-- bonn_5sec
        `-- bonn_mpii_test_5sec
        `-- bonn_mpii_test_v2_5sec
        `-- bonn_mpii_train_5sec
        `-- bonn_mpii_train_v2_5sec
        `-- mpii
        `-- mpii_5sec
```

**For PoseTrack18 data**, please download the data from [PoseTrack download page](https://posetrack.net/users/download.php). Since the video frames are named properly, you only need to extract them into a directory of your choice (no need to rename the video frames). As with PoseTrack17, we provide all required JSON files for PoseTrack18 dataset as well.

Your extracted PoseTrack18 images directory should look like this:
```
${POSETRACK18_DIR}
    |--images
        |-- test
        `-- train
        `-- val
```

### Create Symbolic link

```
ln -s  ${DCPose_SUPP_DIR}  ${DCPose_Project_Dir}  # For DCPose supplementary file
ln -s  ${DATASET_DIR}  ${DCPose_Project_Dir}      #  For Dataset


# For example
${DCPose_Project_Dir} = /your/project/path/Pose_Estimation_DCPose
${DCPose_SUPP_DIR}    = /your/supp/path/DcPose_supp_files
${DATASET_DIR}        = /your/dataset/path/DataSet

ln -s /your/supp/path/DcPose_supp_files  /your/project/path/Pose_Estimation_DCPose  # SUP File Symbolic link 
ln -s /your/dataset/path/DataSet         /your/project/path/Pose_Estimation_DCPose  # DATASET Symbolic link 2
```

### Training from scratch

**For PoseTrack17**

```
cd tools
# train & val 
python run.py --cfg ../configs/posetimation/DcPose/posetrack17/model_RSN.yaml --train --val 
```

The results are saved in `${DCPose_Project_Dir}/output/PE/DcPose/DCPose_Network_Model_RSN/PoseTrack17/{Network_structure _hyperparameters}` by default

**For PoseTrack18**

```
cd tools
# train & val 
python run.py --cfg ../configs/posetimation/DcPose/posetrack18/model_RSN.yaml --train --val 
```

The results are saved in `${DCPose_Project_Dir}/output/PE/DcPose/DCPose_Network_Model_RSN/PoseTrack18/{Network_structure _hyperparameters}` by default



### Validating/Testing from our pretrained models

```
# Evaluate on the PoseTrack17 validation set
python run.py --cfg ../configs/posetimation/DcPose/posetrack17/model_RSN_trained.yaml --val  
# Evaluate on the PoseTrack17 test set
python run.py --cfg ../configs/posetimation/DcPose/posetrack17/model_RSN_trained.yaml --test
```



### Run on video

```
cd demo/                   
mkdir input/
# Put your video in the input directory
python video.py
```

