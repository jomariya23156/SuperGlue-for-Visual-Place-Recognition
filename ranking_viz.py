import numpy as np
import pandas as pd
import argparse

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from pathlib import Path
import matplotlib.image as mp
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument(
    '-q', '--query', type=str, required=True,
    help='Name of query image')

parser.add_argument(
    '-i', '--input_csv', type=str, required=True,
    help='Path to ranking result (.csv file) from matching result directory')

parser.add_argument(
    '-id', '--input_dir', type=str, required=True,
    help='Path to original image directory')

parser.add_argument(
    '--input_extension', type=str, default='png', choices={'jpg', 'png'},
    help='Extension of image in input_dir')

parser.add_argument(
    '--output_extension', type=str, default='png', choices={'jpg', 'png'},
    help='Extension of output visualization image')

parser.add_argument(
    '-r', '--rank', type=int, default=5,
    help='Number of rank to show')

args = parser.parse_args()

rank = args.rank
input_csv = Path(args.input_csv)

in_path = Path(args.input_dir)
query = Path(args.query)
print('Looking for data in directory \"{}\"'.format(input_csv))

####start ranking viz process####

#get
df = pd.read_csv(input_csv)
df = df.iloc[:rank+1,1:]

# delete the 100% score (same with query)
df.drop(index=0, axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)

scores = list(df['score'])

#no. of row of table
size = df.shape[0]

#get queried image
imq = mp.imread(os.path.join(in_path, query))

#add extension
df['image'] = df['image'].apply(lambda x: f"{x}.{args.input_extension}")

#create list of matching image
impath = df['image'].apply(lambda x: os.path.join(in_path, str(x)))

image = []
for i in range(size):
    image.append(mp.imread(impath[i]))

#get figsize
h, w, d = image[0].shape
figsize = w*rank/300,h/400

#plotting image
fig, ax = plt.subplots(1,rank+1,figsize = figsize,dpi =150)

#fontsize
fs = 50/rank

#ax[0].spines[['bottom','left','right','top']].set_linewidth(2)
ax[0].imshow(imq)
ax[0].tick_params(bottom = False,left = False ,labelbottom = False, labelleft = False)
#ax[0].axis('off')
ax[0].set_title('Query - {}'.format(query),fontsize = fs)

for i in range(rank):
    ax[i+1].imshow(image[i])
    ax[i+1].axis('off')
    ax[i+1].set_title('Rank {} - {}'.format(i+1,df.image[i]),fontsize = fs)
    ax[i+1].text(0.5, -0.1, scores[i], ha="center", va='center', fontsize=fs, transform=ax[i+1].transAxes)
plt.tight_layout(pad = 2)
plt.show()

#save viz
fig.savefig(f'match_ranking_of_{query}_showing_{rank}_rank.{args.output_extension}',facecolor = 'w')

