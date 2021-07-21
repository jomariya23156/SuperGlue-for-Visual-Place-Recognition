import numpy as np
import cv2
import argparse

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch

from pathlib import Path
import random
import matplotlib.cm as cm

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

def ranking_score(matches, match_confidence):
    return np.sum(np.multiply(matches,match_confidence)).astype(np.float32)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-q', '--query', type=str, required=True,
    help='Name of query image inside input_dir folder')
parser.add_argument(
    '-i', '--input_dir', type=str, required=True,
    help='Path to database image directory')
parser.add_argument(
    '-o', '--output_dir', type=str, default='rank_output/',
    help='Path to store npz and visualization files')
parser.add_argument(
    '--max_length', type=int, default=-1,
    help='Maximum number of pairs to evaluate')
parser.add_argument(
    '--resize', type=int, nargs='+', default=[640, 480],
    help='Resize the input image before running inference. If two numbers, '
    'resize to the exact dimensions, if one number, resize the max '
    'dimension, if -1, do not resize')
parser.add_argument(
    '--resize_float', action='store_true',
    help='Resize the image after casting uint8 to float')
parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    help='SuperGlue weights')
parser.add_argument(
    '--max_keypoints', type=int, default=1024,
    help='Maximum number of keypoints detected by Superpoint'
    ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.005,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--viz', action='store_true',
    help='Visualize the matches and dump the plots')
parser.add_argument(
    '--nms_radius', type=int, default=4,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
    ' (Must be positive)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')
parser.add_argument(
    '--force_cpu', action='store_true',
    help='Force pytorch to run in CPU mode.')
parser.add_argument(
    '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
    help='Visualization file extension. Use pdf for highest-quality.')
parser.add_argument(
    '--fast_viz', action='store_true',
    help='Use faster image visualization with OpenCV instead of Matplotlib')
parser.add_argument(
    '--show_keypoints', action='store_true',
    help='Plot the keypoints in addition to the matches')
parser.add_argument(
    '--opencv_display', action='store_true',
    help='Visualize via OpenCV before saving output images')
    
args = parser.parse_args()

assert not (args.opencv_display and not args.viz), 'Must use --viz with --opencv_display'
assert not (args.opencv_display and not args.fast_viz), 'Cannot use --opencv_display without --fast_viz'
assert not (args.fast_viz and not args.viz), 'Must use --viz with --fast_viz'
assert not (args.fast_viz and args.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

# score for each image to query image
score_dict = {}

if len(args.resize) == 2 and args.resize[1] == -1:
    args.resize = args.resize[0:1]
if len(args.resize) == 2:
    print('Will resize to {}x{} (WxH)'.format(
        args.resize[0], args.resize[1]))
elif len(args.resize) == 1 and args.resize[0] > 0:
    print('Will resize max dimension to {}'.format(args.resize[0]))
elif len(args.resize) == 1:
    print('Will not resize images')
else:
    raise ValueError('Cannot specify more than two integers for --resize')

all_image_name = os.listdir(args.input_dir)

with open('rank_pairs.txt', 'w') as file:
    for image_name in all_image_name:
        if (image_name.endswith('.jpg') or image_name.endswith('.png')
                                         or image_name.endswith('.jpeg')):
            file.write(f'{args.query} {image_name}\n')

with open('rank_pairs.txt', 'r') as f:
    pairs = [l.split() for l in f.readlines()]
    
if args.max_length > -1:
    pairs = pairs[0:np.min([len(pairs), args.max_length])]

# Load the SuperPoint and SuperGlue models.
device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
print('Running inference on device \"{}\"'.format(device))
config = {
    'superpoint': {
        'nms_radius': args.nms_radius,
        'keypoint_threshold': args.keypoint_threshold,
        'max_keypoints': args.max_keypoints
    },
    'superglue': {
        'weights': args.superglue,
        'sinkhorn_iterations': args.sinkhorn_iterations,
        'match_threshold': args.match_threshold,
    }
}

matching = Matching(config).eval().to(device)

# Create the output directories if they do not exist already.
input_dir = Path(args.input_dir)
print('Looking for data in directory \"{}\"'.format(input_dir))
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
print('Will write matches to directory \"{}\"'.format(output_dir))

if args.viz:
    print('Will write visualization images to',
          'directory \"{}\"'.format(output_dir))

timer = AverageTimer(newline=True)
for i, pair in enumerate(pairs):
    name0, name1 = pair[:2]
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
    eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
    viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, args.viz_extension)
    viz_eval_path = output_dir / \
        '{}_{}_evaluation.{}'.format(stem0, stem1, args.viz_extension)

    do_match = True
    do_viz = args.viz
    if args.viz and viz_path.exists():
        do_viz = False

    if not (do_match or do_viz):
        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
        continue

    # If a rotation integer is provided (e.g. from EXIF data), use it:
    if len(pair) >= 5:
        rot0, rot1 = int(pair[2]), int(pair[3])
    else:
        rot0, rot1 = 0, 0

    # Load the image pair.
    image0, inp0, scales0 = read_image(
        input_dir / name0, device, args.resize, rot0, args.resize_float)
    image1, inp1, scales1 = read_image(
        input_dir / name1, device, args.resize, rot1, args.resize_float)
    if image0 is None or image1 is None:
        print('Problem reading image pair: {} {}'.format(
            input_dir/name0, input_dir/name1))
        exit(1)
    timer.update('load_image')

    if do_match:
        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('matcher')

        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                       'matches': matches, 'match_confidence': conf}
        
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

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    if do_viz:
        # Visualize the matches.
        color = cm.jet(mconf)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append('Rotation: {}:{}'.format(rot0, rot1))

        # Display extra parameter info.
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {}:{}'.format(stem0, stem1),
        ]

        make_matching_plot(
            image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
            text, viz_path, args.show_keypoints,
            args.fast_viz, args.opencv_display, 'Matches', small_text)

        timer.update('viz_match')
        
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