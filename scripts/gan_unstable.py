import os 
import random
import numpy as np
import cv2
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
sequence_length = 90
H,W = 360,640
device = 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description='Video Stabilization using DMBVS-UNet')
    parser.add_argument('--in_path', type=str, help='Input video file path')
    parser.add_argument('--out_path', type=str, help='Output stabilized video file path')
    return parser.parse_args()

def save_video(frames, path):
    frame_count,h,w,_ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (w,h))
    for idx in range(frame_count):
        out.write(frames[idx,...])
    out.release()

def fixBorder(frame):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame

def warp_frames(frames,transforms):
    n,h,w,c = frames.shape
    cx, cy = (w-1) // 2, (h-1) // 2
    # Compute the translation matrix to shift the center to the origin
    translation_matrix1 = np.array([[1, 0, -cx],
                                    [0, 1, -cy],
                                    [0, 0, 1]],dtype=np.float32)
    # Compute the translation matrix to shift the origin back to the center
    translation_matrix2 = np.array([[1, 0, cx],
                                    [0, 1, cy],
                                    [0, 0, 1]],dtype=np.float32)
    unstable_frames = np.zeros_like(frames)
    for i in range(num_frames):
        # Read next frame
        frame = frames[i, ...]
        # Extract transformations from the new transformation array
        dx = transforms[i, 0]
        dy = transforms[i, 1]
        da = transforms[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        M = np.zeros((3, 3), np.float32)
        deg = da/(180*np.pi)
        M[0, 0] = np.cos(deg)
        M[0, 1] = -np.sin(deg)
        M[1, 0] = np.sin(deg)
        M[1, 1] = np.cos(deg)
        M[0, 2] = dx
        M[1, 2] = dy
        M[2,2] = 1.0
        transformation = translation_matrix2 @ M @ translation_matrix1
        frame_unstabilized = cv2.warpPerspective(frame, transformation , (w, h))

        # Fix border artifacts
        frame_unstabilized = fixBorder(frame_unstabilized)
        unstable_frames[i, ...] = frame_unstabilized
    return unstable_frames

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 90 * 3)  # Output shape: [batch, 90 * 3]
    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        return z.view(-1, 90, 3)



if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.in_path):
        print(f"The input file '{args.in_path}' does not exist.")
        exit()
    _,ext = os.path.splitext(args.in_path)
    if ext not in ['.mp4','.avi']:
        print(f"The input file '{args.in_path}' is not a supported video file (only .mp4 and .avi are supported).")
        exit()
    #load stable video
    cap = cv2.VideoCapture(args.in_path)
    frames = []
    while True:
        ret,frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame,(W,H))
        frames.append(frame)
    cap.release()
    frames = np.array(frames,dtype = np.uint8)
    num_frames = frames.shape[0]
    #load model
    generator = Generator(latent_dim = 1000, hidden_size = 2048).train().to(device)
    state_dict = torch.load('./data/generator_0.pth')
    generator.load_state_dict(state_dict)
    #create transforms
    transforms_noisy = np.zeros((num_frames,3),dtype=np.float32)
    sequence_length = min(90,num_frames)
    for idx in range(0,num_frames,sequence_length):
        torch.manual_seed(idx)
        noise = torch.randn(1, 1000).to(device)
        with torch.no_grad():
            transforms = generator(noise)
        transforms = transforms.squeeze(0).cpu().numpy()
        transforms[:,0] *= W
        transforms[:,1] *= H
        if idx > num_frames - sequence_length:
            transforms_noisy[idx : idx + (num_frames-sequence_length),:] = transforms[:(num_frames - idx),:]
        else:
            transforms_noisy[idx : idx + sequence_length,:] = transforms[:sequence_length,:]
    #make them more natural with low-pass
    smooth_noise_transforms = np.zeros_like(transforms_noisy)
    alpha = 0.5  # Smoothing factor
    smooth_noise_transforms[0] = transforms_noisy[0]  # Initialize the first value
    for i in range(1, len(transforms_noisy)):
        smooth_noise_transforms[i] = alpha * transforms_noisy[i] + (1 - alpha) * smooth_noise_transforms[i - 1]
    unstable_frames = warp_frames(frames,smooth_noise_transforms)
    save_video(unstable_frames,args.out_path)
    
