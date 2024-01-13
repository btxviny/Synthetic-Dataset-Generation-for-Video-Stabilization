import os
from tqdm import tqdm
import argparse
import subprocess
import cv2
import numpy as np
H,W = 360,640

def parse_args():
    parser = argparse.ArgumentParser(description='Video Stabilization using DMBVS-UNet')
    parser.add_argument('--method', type=str, help='generation method')
    parser.add_argument('--stable_path', type=str, help='Directory of stable videos')
    parser.add_argument('--cropped_stable_path', type=str, help='Directory of cropped stable videos')
    parser.add_argument('--unstable_path', type=str, help='Directory to generate unstable  videos')
    parser.add_argument('--transforms_path', type=str, help='Directory to generate transforms')
    return parser.parse_args()

def fixBorder(frame):
    s = frame.shape
    # Scale the image 1% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def crop_video(in_path,cropped_path):
    cap = cv2.VideoCapture(in_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cropped_path, fourcc, 30.0, (W,H))
    while True:
        ret,img = cap.read()
        if not ret: break
        img = cv2.resize(img,(W,H))
        img = fixBorder(img)
        out.write(img)
    out.release()


def generate(method,s_dir, out_dir, cropped_dir, transform_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)
    if not os.path.exists(transform_dir):
        os.makedirs(transform_dir)

    if method == 'random':
        script = './scripts/random_unstable.py'
    elif method == 'sampling':
        script = './scripts/sampling_unstable.py'
    elif method == 'pca':
        script = './scripts/pca_unstable.py'
    elif method == 'gan':
        script = './scripts/gan_unstable.py'
    else:
        print('Select a method from [random, sampling, pca, gan]')
        exit()
    videos = [x for x in os.listdir(s_dir) if (os.path.splitext(x)[1] == '.avi') or (os.path.splitext(x)[1] == '.mp4')]
    for video in tqdm(videos):
        in_path = os.path.join(s_dir,video)
        out_path = os.path.join(out_dir,video)
        cropped_path = os.path.join(cropped_dir,video)
        transform_path = os.path.join(transform_dir,video[:-4]+'.npy')
        subprocess.run(['python', script, '--in_path', in_path, '--out_path', out_path,'--transforms_path',transform_path])
        crop_video(in_path,cropped_path) # crop original stable video to avoid perspective mismatch

if __name__ == '__main__':
    args = parse_args()
    generate(args.method, args.stable_path, args.unstable_path, args.cropped_stable_path, args.transforms_path)