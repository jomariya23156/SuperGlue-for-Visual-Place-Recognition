import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True,
                    help='Path to input video')
parser.add_argument('-o', '--output', type=str, required=True,
                    help='Path to save output images')
parser.add_argument('-s', '--skip', type=int, default=30,
                    help='Number of frame to skip')
parser.add_argument('-f', '--format', choices={'jpg', 'png'}, default='png',
                    help='Image file format to save (jpg or png)')
args = vars(parser.parse_args())

count = 0

cap = cv2.VideoCapture(args['input'])
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while count < total_frame:
    success, frame = cap.read()
    if count % args['frame'] == 0:
        print(f'[INFO] Saving frame: {count} from {total_frame}')
        cv2.imwrite(f"{args['output']}/{count}.{args['format']}", frame)
    count += 1
    