import numpy as np
import pandas as pd
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
from models.superpoint import SuperPoint
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

# add number of rank to return parameter
def rank_superpoints(query: np.ndarray, images_db: pd.DataFrame, output: str, 
                     image_size: tuple = (640,480), max_length: int = -1, superglue: str = 'indoor', 
                     sinkhorn_iterations: int = 20, match_threshold: int = 0.2, 
                     cuda: bool = True, rank: int = 2, nms_radius: int = 4, 
                     keypoint_threshold: float = 0.005, max_keypoints: int = 1024) -> dict:
    
    device = 'cuda' if torch.cuda.is_available() and cuda else 'cpu'
    
    # setup config and load superpoint+superglue model
    config = {'nms_radius': nms_radius,
          'keypoint_threshold': keypoint_threshold,
          'max_keypoints': max_keypoints}
        
    superpoint = SuperPoint(config).eval().to(device)
    
    config = {
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    
    superglue = SuperGlue(config).eval().to(device)
    
    # resize queryimage
    query = cv2.resize(query, (image_size[0], image_size[1])).astype('float32')
    tensor_query = torch.from_numpy(query/255.).float()[None, None].to(device)
    
    # debugging
    print('[DEBUGGING] Tensor shape:', tensor_query.shape)
    
    query_superpoints = superpoint({'image':tensor_query})
    
    # score for each image to query image
    score_dict = {}
    
    # Create the output directories if they do not exist already.
    # input_dir = Path(images_db)
    # print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    
    all_file_name = list(images_db.index)
    
    with open('rank_pairs.txt', 'w') as file:
        file.write('query query\n')
        for file_name in all_file_name:
            if file_name.endswith('.pickle'):
                file.write(f'query {file_name}\n')
    
    with open('rank_pairs.txt', 'r') as f:
        pairs = [l.split() for l in f.readlines()]
        
    if max_length > -1:
        pairs = pairs[0:np.min([len(pairs), max_length])]
    
    # Load the SuperGlue models.
    
    print('Running inference on device \"{}\"'.format(device))
    
    timer = AverageTimer(newline=True)
    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = name0, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)

        # find maximum possible score to calculate score as percentage
        # benchmark on this maximum possible score
        if name0 == 'query' and name1 == 'query':
            superpoints_0 = query_superpoints.copy()
            superpoints_1 = query_superpoints.copy()
        else:
            superpoints_0 = query_superpoints.copy()
            # because at the first loop we will receive query and query
            # we need to -1 so that the second loop we'll pick the first pickle file in the list 
            superpoints_1 = load_pickle(images_db['abspath'].iloc[i-1])
        
        # # debugging
        # for k, v in superpoints_0.items():
        #     print('[DEBUGGING]', k, type(v))
        
        superpoints_0 = {k+'0':v for k, v in superpoints_0.items()}
        superpoints_1 = {k+'1':v for k, v in superpoints_1.items()}
        
        if superpoints_0 is None or superpoints_1 is None:
            print('Problem loading pickle pairs: {} {}'.format(
                name0, name1))
            exit(1)
        timer.update('load_pickle')
    
        # Perform the matching.
        # change np to torch tensor
        dummy_data = {'image0': np.zeros((1, 1, image_size[1], image_size[0])),
                      'image1': np.zeros((1, 1, image_size[1], image_size[0]))}
        
        data = {**dummy_data, **superpoints_0, **superpoints_1}
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
        
        # convert to ndarray to be able to save .npz
        pred = superglue(data)
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        
        
        # convert to ndarray to be able to save .npz
        kpts0, kpts1 = superpoints_0['keypoints0'][0].cpu().numpy(), superpoints_1['keypoints1'][0].cpu().numpy()
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('matcher')

        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                        'matches': matches, 'match_confidence': conf}
        
        # save full score to calculate %
        if name0 == 'query' and name1 == 'query':
            full_score = ranking_score(matches, conf)
        # save score to score dict
        else:
            score_dict[name1] = ranking_score(matches, conf)
            # save to .npz file
            # we don't need to save query compare to query itself here
            np.savez(str(matches_path), **out_matches)
            
        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
            
    ranked_images = {k:v for k,v in sorted(score_dict.items(), reverse = True, key= lambda x: x[1])}
    
    ranked_images_percentage = {k:f'{((v/full_score)*100):.3f}%' for k,v in ranked_images.items()}
    
    final_result = {}
    
    for i, k in enumerate(ranked_images_percentage, start=1):
        final_result[k] = {'ranking_score':ranked_images_percentage[k]}
        if i == rank:
            break
        
    for k, v in final_result.items():
        v['path'] = images_db.loc[k]['abspath']
        # by default from numpy/pandas it save int as int64/numpy.int64
        # which is not compatible with JSON
        # so we need to cast it to int
        v['location'] = int(images_db.loc[k]['location'])
        v['date_taken'] = images_db.loc[k]['date_taken']
    ####write ranked image .csv
    
    df = pd.DataFrame.from_dict(ranked_images_percentage,orient='index',columns = ['score'])
    df.reset_index(inplace=True)
    df.rename(columns = {'index':'image'},inplace=True)
    df.to_csv(str(output_dir/'ranking_score.csv'), index=True)
    
    # print('[DEBUGGING] Rank result:',ranked_images_percentage)
    # print('[DEBUGGING] Final result value:', final_result)
    
    return final_result