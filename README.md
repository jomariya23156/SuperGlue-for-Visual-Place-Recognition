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
  
1. **`image_from_video.py`** Command line arguements:  
* `--input (or -i)` Path to input video
* `--output (or -o)` Path to directory to save images
* `--skip (or -s)` Number of frame to skip, in the other words, save every n frame (default: 60)
* `--format (or -f)` Image file format for saving (only 2 options available 'jpg' or 'png')  

&nbsp;&nbsp;&nbsp;&nbsp;**Example usage:** `python image_from_video.py -i video/country_road.MOV -o saved_frame -s 60 -f png` by running this, the code will save every 60 frames from country_road.MOV in video folder into saved_frame folder in the current working directory as .png files

2. **`superpoints_from_images.py`** Command line arguements:  
* `--input_dir` Input directory of images to be processed
* `--output_dir` Output directory to save SuperPoints data
* `--max_keypoints` Maximum number of keypoints detected by Superpoint ('-1' keeps all keypoints) (default: 1024)
* `--keypoint_threshold` SuperPoint keypoint detector confidence threshold (default: 0.005)
* `--nms_radius` SuperPoint Non Maximum Suppression (NMS) radius (Must be positive) (default: 4)
* `--resize` Resize the input image before running inference. Two numbers required. (default: 640x480)
* `--cuda` Use cuda GPU to speed up network processing speed (default: False)

&nbsp;&nbsp;&nbsp;&nbsp;**Example usage:** `python superpoints_from_images.py --input_dir test_realdata --output_dir saved_superpoints/pickle/test_realdata --resize 320 240 --cuda` by running this, the code will resize images to 320x240 and extract SuperPoints from images in `test_realdata` folder using CUDA. Then, save to `saved_superpoints/pickle/test_realdata` folder.

3. **`superglue_rank_superpoints_file.py`** Command line arguements:  
* `--query (or -q)` Name of query pickle inside input_dir folder (it must be one of pickle in the input_dir folder) This file does not support inputting image from outside directory. If you wish to do so, use API which we will explain later here.
* `--input_dir (or -i)` Path to database pickle directory
* `--output_dir (or -o)` Path to store .npz files (resuls from matching)
* `--image_size` Image size used to resize image with SuperPoints. If you didn\'t specify resize when running superpoints_from_images.py file. Please enter 640 480. (format: width x height)
* `--max_length` Maximum number of pairs to evaluate. -1 is no maximum. (default: -1)
* `--superglue` SuperGlue weights. There are 2 options available which are `indoor` or `outdoor`. (default: `indoor`)
* `--sinkhorn_iterations` Number of Sinkhorn iterations performed by SuperGlue (default: 20)
* `--match_threshold` SuperGlue match threshold (default: 0.2)
* `--cuda` Use cuda GPU to speed up network processing speed (default: False)

&nbsp;&nbsp;&nbsp;&nbsp;**Example usage:** `python superglue_rank_superpoints_file.py -q 60.pickle -i saved_superpoints/pickle/test_realdata -o rank_output/rank_output_test_realdata --image_size 320 240 --cuda` by running this, the code will rank the SuperPoints file according to the query file and output the ranking result in the command line, save the match output to the output directory, and output `ranking_score.csv` (the result of ranking in csv format) using CUDA.

4. **`superglue_rank_images.py`** Command line arguements:  
* `--query (or -q)` Name of query image inside input_dir folder (it must be one of image in the input_dir folder) This file does not support inputting image from outside directory. If you wish to do so, use API which we will explain later here.
* `--input_dir (or -i)` Path to database image directory
* `--output_dir (or -o)` Path to store .npz files and ranking_score.csv (resuls from matching)
* `--max_length` Maximum number of pairs to evaluate. -1 is no maximum. (default: -1)
* `--resize` Resize the input image before running inference. Two numbers required. (default: 640x480) 
* `--resize_float` Resize the image after casting uint8 to float (default: Fakse)
* `--superglue` SuperGlue weights. There are 2 options available which are `indoor` or `outdoor`. (default: `indoor`)
* `--max_keypoints` Maximum number of keypoints detected by Superpoint ('-1' keeps all keypoints) (default: 1024)
* `--keypoint_threshold` SuperPoint keypoint detector confidence threshold (default: 0.005)
* `--viz` Visualize the matches and dump the plots (default: False)
* `--nms_radius` SuperPoint Non Maximum Suppression (NMS) radius (Must be positive) (default: 4)
* `--sinkhorn_iterations` Number of Sinkhorn iterations performed by SuperGlue (default: 20)
* `--match_threshold` SuperGlue match threshold (default: 0.2)
* `--force_cpu` Force pytorch to run in CPU mode (default: False)
* `--viz_extension` Visualization file extension. Use pdf for highest-quality. There are 2 options which are 'png' and 'pdf'. (default: png)
* `--fast_viz` Use faster image visualization with OpenCV instead of Matplotlib (default: False)
* `--show_keypoints` Plot the keypoints in addition to the matches (default: False)
* `--opencv_display` Visualize via OpenCV before saving output images (default: False)

&nbsp;&nbsp;&nbsp;&nbsp;**Example usage:** `python superglue_rank_images.py -q 60.png -i test_realdata -o rank_output/rank_output_test_realdata_images --resize 320 240 --resize_float --viz --viz_extension png --fast_viz --show_keypoints` by running this, the code will rank the images according to the image query, do the visualization, show the keypoints, and output the ranking result in the command line, save the match output to the output directory, and output `ranking_score.csv` (the result of ranking in csv format) using CUDA.

5. **`ranking_viz.py`** Command line arguements:  
* `--query (or -q)` Name of query image (in the ranking_result that you did ranking from other files)
* `--input_csv (or -i)` Path to ranking result (.csv file) from matching result directory
* `--input_dir (or -id)` Path to original image directory
* `--input_extension` Extension of image in input_dir. There are 2 options which are 'jpg' and 'png'. (default: png)
* `--output_extension` Extension of output visualization image. There are 2 options which are 'jpg' and 'png'. (default: png)
* `--rank (or -r)`Number of rank to show (default: 5)

&nbsp;&nbsp;&nbsp;&nbsp;**Example usage:** `python ranking_viz.py -q 60.png -i test_realdata/ranking_score.csv -id test_realdata --input_extension png --output_extension png -r 10` by running this, the code will visualize the result following the number of rank specify in -r. For example: **(ranking image here)**

6. **`rank_from_superpoints_utils.py`** This code is the refactored code version of `superglue_rank_superpoints_file.py` in order to use for building an API. So, you don't need to do anything with this file

7. **`place_recognition_api.py`** Run `uvicorn place_recognition_api:app` and go to the port showing in the command line. *Note: You will see a lot of online resources adding --reload at the end after :app, but we don't do this here because at the runtime there will be 1 file created named `rank_pairs.txt` for matching to work properly. If you specify --reload, the app api will be reload every time the new file created. This can lead to error and infinite loop* 

8. **`client_test_api.py`** Basically run `python client_test_api.py` to test whether the API works properly. You can change the file name and rank parameter in the source code file to test with other images and sets of parameters.

9. **`build_database_from_pickle_dir.ipynb`** This is the minimum example of .csv database file. You can take a look into the source code and run all cells or you can build it on your own.

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
