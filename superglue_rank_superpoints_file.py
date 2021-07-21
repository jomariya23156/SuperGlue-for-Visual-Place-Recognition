import numpy as np
import cv2
import argparse
import pickle

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch

from pathlib import Path
import random
import matplotlib.cm as cm

from models.superglue import SuperGlue
from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

def ranking_score(matches, match_confidence):
    return np.sum(np.multiply(matches,match_confidence)).astype(np.float32)

def load_pickle(path):
    with open(path, 'rb') as file:
        loaded = pickle.load(file)
    return loaded

parser = argparse.ArgumentParser()
parser.add_argument(
    '-q', '--query', type=str, required=True,
    help='Name of query image inside input_dir folder')
parser.add_argument(
    '-i', '--input_dir', type=str, required=True,
    help='Path to database pickle directory')
parser.add_argument(
    '-o', '--output_dir', type=str, default='rank_output/',
    help='Path to store npz')
parser.add_argument(
    '--image_size', type=int, nargs='+', required=True,
    help='Image size used to resize image with SuperPoints'
    'Format: Width x Height'
    'If you didn\'t specify resize when running superpoints_from_images.py file. Please enter 640 480')
parser.add_argument(
    '--max_length', type=int, default=-1,
    help='Maximum number of pairs to evaluate')
parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    help='SuperGlue weights')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')
parser.add_argument(
    '--cuda', action='store_true',
    help='Use cuda GPU to speed up network processing speed (default: False)')
    
args = parser.parse_args()

# score for each image to query image
score_dict = {}

# Create the output directories if they do not exist already.
input_dir = Path(args.input_dir)
print('Looking for data in directory \"{}\"'.format(input_dir))
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
print('Will write matches to directory \"{}\"'.format(output_dir))

all_file_name = os.listdir(input_dir)
total_file_num = len(all_file_name)

with open('rank_pairs.txt', 'w') as file:
    for file_name in all_file_name:
        if file_name.endswith('.pickle'):
            file.write(f'{args.query} {file_name}\n')

with open('rank_pairs.txt', 'r') as f:
    pairs = [l.split() for l in f.readlines()]
    
if args.max_length > -1:
    pairs = pairs[0:np.min([len(pairs), args.max_length])]

# Load the SuperGlue models.
device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
print('Running inference on device \"{}\"'.format(device))

config = {
    'superglue': {
        'weights': args.superglue,
        'sinkhorn_iterations': args.sinkhorn_iterations,
        'match_threshold': args.match_threshold,
    }
}

superglue = SuperGlue(config).eval().to(device)

timer = AverageTimer(newline=True)
for i, pair in enumerate(pairs):
    name0, name1 = pair[:2]
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)

    # Handle --cache logic.
    do_match = True

    if not (do_match):
        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
        continue
    
    superpoints_0 = load_pickle(str(input_dir / name0))
    superpoints_1 = load_pickle(str(input_dir / name1))
    
    # # debugging
    # for k, v in superpoints_0.items():
    #     print('[DEBUGGING]', k, type(v))
    
    superpoints_0 = {k+'0':v for k, v in superpoints_0.items()}
    superpoints_1 = {k+'1':v for k, v in superpoints_1.items()}
    
    if superpoints_0 is None or superpoints_1 is None:
        print('Problem loading pickle pairs: {} {}'.format(
            input_dir/name0, input_dir/name1))
        exit(1)
    timer.update('load_pickle')

    if do_match:
        # Perform the matching.
        # change np to torch tensor
        dummy_data = {'image0': np.zeros((1, 1, args.image_size[1], args.image_size[0])),
                      'image1': np.zeros((1, 1, args.image_size[1], args.image_size[0]))}
        
        data = {**dummy_data, **superpoints_0, **superpoints_1}
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
        
        # print('[DEBUGGING] data', data)
        
        # convert to ndarray to be able to save .npz
        pred = superglue(data)
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        
        # # debugging
        # print(pred.keys())
        # print(type(pred))
        
        # convert to ndarray to be able to save .npz
        kpts0, kpts1 = superpoints_0['keypoints0'][0].cpu().numpy(), superpoints_1['keypoints1'][0].cpu().numpy()
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('matcher')

        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                        'matches': matches, 'match_confidence': conf}
        # # debugging
        # print('[DEBUGGING] matches:', matches)
        # print('[DEBUGGING] matches shape:', matches.shape)
        # print('[DEBUGGING] conf:', conf)
        # print('[DEBUGGING] conf shape:', conf.shape)
        # save score to score dict
        score_dict[stem1] = ranking_score(matches, conf)
        
        # save full score to calculate %
        if name0 == name1:
            full_score = score_dict[stem1]
        
        # save to .npz file
        np.savez(str(matches_path), **out_matches)

        
    timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
        
ranked_images = {k:v for k,v in sorted(score_dict.items(), reverse = True, key= lambda x: x[1])}

ranked_images_percentage = {k:f'{((v/full_score)*100):.3f}%' for k,v in ranked_images.items()}

####write ranked image .csv

import pandas as pd

df = pd.DataFrame.from_dict(ranked_images_percentage,orient='index',columns = ['score'])
df.reset_index(inplace=True)
df.rename(columns = {'index':'image'},inplace=True)
df.to_csv(str(output_dir/'ranking_score.csv'), index=True)

print(ranked_images_percentage)