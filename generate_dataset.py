import os
from tqdm import tqdm
import argparse
import subprocess
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Video Stabilization using DMBVS-UNet')
    parser.add_argument('--method', type=str, help='generation method')
    parser.add_argument('--stable_path', type=str, help='Directory of stable videos')
    parser.add_argument('--unstable_path', type=str, help='Directory to generate unstable  videos')
    return parser.parse_args()

def fixBorder(frame):
    s = frame.shape
    # Scale the image 1% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def crop_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret,img = cap.read()
        if not ret: break
        frames.append(img)
    cap.release()
    frames = np.array(frames, dtype = np.uint8)
    num_frames,height,width,_ = frames.shape
    cropped_frames = np.array([fixBorder(x) for x in frames],dtype = np.uint8)
    output_path = path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust codec if needed
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    for frame in cropped_frames: out.write(frame)
    out.release()


def generate(method,s_dir,out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
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
        subprocess.run(['python', script, '--in_path', in_path, '--out_path', out_path])
        crop_video(in_path) # crop original stable video to avoid perspective mismatch

if __name__ == '__main__':
    args = parse_args()
    generate(args.method, args.stable_path, args.unstable_path)