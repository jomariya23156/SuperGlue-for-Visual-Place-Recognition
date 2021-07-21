import cv2
import argparse
import numpy as np
from models.superpoint import SuperPoint
import torch
import time
import os
from pathlib import Path
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_dir', type=str, required=True,
    help='Input directory of images to be processed')
parser.add_argument(
    '--output_dir', type=str, required=True,
    help='Output directory to save SuperPoints data')
parser.add_argument(
    '--max_keypoints', type=int, default=1024,
    help='Maximum number of keypoints detected by Superpoint'
    ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.005,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--nms_radius', type=int, default=4,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
    ' (Must be positive)')
parser.add_argument(
    '--resize', type=int, nargs='+', default=[640, 480],
    help='Resize the input image before running inference. Two numbers required.'
    'Default is 640x480')
parser.add_argument('--cuda', action='store_true',
    help='Use cuda GPU to speed up network processing speed (default: False)')

args = parser.parse_args()

torch.set_grad_enabled(False)

total_start_time = time.time()

device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

# setup config and load superpoint model
config = {'nms_radius': args.nms_radius,
      'keypoint_threshold': args.keypoint_threshold,
      'max_keypoints': args.max_keypoints}
    
superpoint = SuperPoint(config).eval().to(device)

input_path = Path(args.input_dir)
all_images_name = os.listdir(input_path)
all_images_name = [image_name for image_name in all_images_name if image_name.endswith('.jpg')
                   or image_name.endswith('.png') or image_name.endswith('jpeg')]
total_images_num = len(all_images_name)
output_path = Path(args.output_dir)
output_path.mkdir(exist_ok=True, parents=True)


for i, image_name in enumerate(all_images_name):
    
    partial_start_time = time.time()
    stem_image_name = Path(image_name).stem
    
    image = cv2.imread(str(input_path / image_name), cv2.IMREAD_GRAYSCALE)
    
    # resize images WxH
    image = cv2.resize(image, (args.resize[0],args.resize[1]))
    
    # model require grayscale image in np.float32
    image = image.astype('float32')
    
    # normalize and convert to tensor
    # [None, None] is to add 2 more dimension to the front
    # the model require 4 dimensional input
    # and .to(device) so it can work with cuda
    tensor_image = torch.from_numpy(image/255.).float()[None, None].to(device)
    
    # debugging
    print('[DEBUGGING] Tensor shape:', tensor_image.shape)
    
    pred = superpoint({'image':tensor_image})
    
    # test value before save
    
    # save with pickle
    with open(f'{output_path}/{stem_image_name}.pickle' ,'wb') as file:
        pickle.dump(pred, file)
    
    # # debugging
    # print(pred0)
    # print('--> Keypoints shape:',pred0['keypoints'][0].shape)
    # print('--> Scores shape:',pred0['scores'][0].shape)
    # print('--> Descriptors shape:',pred0['descriptors'][0].shape)
    # print(type(pred0))
    partial_end_time = time.time()
    print(f"[INFO] Finished  image {i+1} from {total_images_num} with runtime: {partial_end_time - partial_start_time}")
    
total_end_time = time.time()
print(f'[INFO] Total runtime: {total_end_time - total_start_time}')