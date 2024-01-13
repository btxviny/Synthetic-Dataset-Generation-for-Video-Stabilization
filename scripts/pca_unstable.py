import os 
import random
import numpy as np
import cv2
import argparse
H,W = 256,256

def parse_args():
    parser = argparse.ArgumentParser(description='Video Stabilization using DMBVS-UNet')
    parser.add_argument('--in_path', type=str, help='Input video file path')
    parser.add_argument('--out_path', type=str, help='Output stabilized video file path')
    parser.add_argument('--transforms_path', type=str, help='transforms file path')
    return parser.parse_args()

def save_video(frames, path):
    frame_count,h,w,_ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (w,h))
    for idx in range(frame_count):
        out.write(frames[idx,...])
    del frames
    out.release()

def fixBorder(frame):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame

def warp_frame(frame,transform):
    h,w,c = frame.shape
    cx, cy = (w-1) // 2, (h-1) // 2
    # Compute the translation matrix to shift the center to the origin
    translation_matrix1 = np.array([[1, 0, -cx],
                                    [0, 1, -cy],
                                    [0, 0, 1]],dtype=np.float32)
    # Compute the translation matrix to shift the origin back to the center
    translation_matrix2 = np.array([[1, 0, cx],
                                    [0, 1, cy],
                                    [0, 0, 1]],dtype=np.float32)
    
    
    # Extract transformations from the new transformation array
    dx = transform[0]
    dy = transform[1]
    da = transform[2]

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
    return frame_unstabilized




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
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #load principal components
    principal_components = np.load('data/principal_components.npy')
    x_principal_components = principal_components[:,0:120]
    y_principal_components = principal_components[:,120:240]
    theta_principal_components = principal_components[:,240:360]
    #create random transformations
    transforms_noisy = np.zeros((num_frames,3),dtype=np.float32)
    sequence_length = min(120,num_frames)
    for idx in range(0,num_frames,sequence_length):
        np.random.seed(idx)
        random_weights = np.random.dirichlet(np.ones(20), size=1).flatten()
        x_new = np.dot(random_weights, x_principal_components[:20,:])
        y_new = np.dot(random_weights, y_principal_components[:20,:])
        theta_new = np.dot(random_weights, theta_principal_components[:20,:])
        transforms = np.stack([x_new,y_new,theta_new],axis = 1)
        transforms[:,0] *= W
        transforms[:,1] *= H
        if idx > num_frames - sequence_length:
            transforms_noisy[idx : idx + (num_frames-sequence_length),:] = transforms[:(num_frames - idx),:]
        else:
            transforms_noisy[idx : idx + sequence_length,:] = transforms[:sequence_length,:]
        print(f'\r{idx}',end = '')
    #make them more natural with low-pass
    smooth_noise_transforms = np.zeros_like(transforms_noisy)
    alpha = 0.5  # Smoothing factor
    smooth_noise_transforms[0] = transforms_noisy[0]  # Initialize the first value
    for i in range(1, len(transforms_noisy)):
        smooth_noise_transforms[i] = alpha * transforms_noisy[i] + (1 - alpha) * smooth_noise_transforms[i - 1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_path, fourcc, 30.0, (W,H))
    for idx in range(num_frames):
        ret,img = cap.read()
        img = cv2.resize(img,(W,H))
        img = warp_frame(img,smooth_noise_transforms[idx,...])
        out.write(img)
    out.release()
    np.save(args.transforms_path,smooth_noise_transforms)
    
