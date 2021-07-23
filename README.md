# SuperGlue for Visual Place Recognition

## Introduction
We use SuperGlue and SuperPoint for VPR (Visual Place Recognition) task answering the question "Given a query image, where is this place in the map?".

Original SuperGlue works by Magic Leap Team, please see:

* Github repo: [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)

* Full paper PDF: [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763).

* Authors: *Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich*

* Website: [psarlin.com/superglue](https://psarlin.com/superglue) for videos, slides, recent updates, and more visualizations.

* `hloc`: a new toolbox for visual localization and SfM with SuperGlue, available at [cvg/Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization/). Winner of 3 CVPR 2020 competitions on localization and image matching!

## Dependencies
* Python 3
* PyTorch 
* OpenCV (4.1.2.30 recommended for best GUI keyboard interaction, see this [note](#additional-notes))
* Matplotlib
* NumPy 
* Pandas
* FastAPI
* uvicorn

Simply run the following command: `pip3 install numpy pandas opencv-python torch matplotlib fastapi uvicorn`

## Contents
We have contributed these 9 files for VPR task and utilities in addition to original files from original Github repo (examples of usage will be described and provided below sections):

1. `image_from_video.py`: extract frame from videos and save out as image files
2. `superpoints_from_images.py`: extract SuperPoint keypoints from images and save as pickle files
3. `superglue_rank_superpoints_file.py`: do image matching and ranking from SuperPoint keypoints .pickle files using SuperGlue 
4. `superglue_rank_images.py`: do image matching and ranking from images files using SuperGlue 
5. `ranking_viz.py`: visualize the ranking results and save output as a image
6. `rank_from_superpoints_utils.py`: refactored code version of `superglue_rank_superpoints_file.py` in order to use for building an API
7. `place_recognition_api.py`: server side of API built with FastAPI
8. `client_test_api.py`: client side API request example
9. `build_database_from_pickle_dir.ipynb`: example of code for building easy database .csv file for testing SuperGlue for VPR tasks

## Usage
**Note:** you can always run `python (file_name).py -h` or `python (file_name).py --help` for every file receiving arguments to see what arguments you need to pass in (or open the file on our preferred editor and read from the source code)  
**Note 2:** Feel free to edit anything to achieve your desired results!  
  
1. `image_from_video.py` command line arguements:  
* `--input (or -i)` path to input video
* `--output (or -o)` path to directory to save images
* `--skip (or -s)` number of frame to skip, in the other words, save every n frame (default value is 60)
* `--format (or -f)` image file format for saving (only 2 options available 'jpg' or 'png')  

&nbsp;&nbsp;&nbsp;&nbsp;**Example usage:** `python image_from_video.py -i video/country_road.MOV -o saved_frame -s 60 -f png` by running this the code will save every 60 frames from country_road.MOV in video folder into saved_frame folder in the current working directory as .png files

**(MORE INCOMING)**

## BibTeX Citation
If you use any ideas from the paper or code from this repo, please consider citing:

```txt
@inproceedings{sarlin20superglue,
  author    = {Paul-Edouard Sarlin and
               Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  title     = {{SuperGlue}: Learning Feature Matching with Graph Neural Networks},
  booktitle = {CVPR},
  year      = {2020},
  url       = {https://arxiv.org/abs/1911.11763}
}
```

## Legal Disclaimer
Magic Leap is proud to provide its latest samples, toolkits, and research projects on Github to foster development and gather feedback from the spatial computing community. Use of the resources within this repo is subject to (a) the license(s) included herein, or (b) if no license is included, Magic Leap's [Developer Agreement](https://id.magicleap.com/terms/developer), which is available on our [Developer Portal](https://developer.magicleap.com/).
If you need more, just ask on the [forums](https://forum.magicleap.com/hc/en-us/community/topics)!
We're thrilled to be part of a well-meaning, friendly and welcoming community of millions.
