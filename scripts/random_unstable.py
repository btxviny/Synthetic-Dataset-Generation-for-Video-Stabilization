import os 
import random
import numpy as np
import cv2
import argparse
H,W = 360,640

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

def get_transformation(shape):
    h,w, = shape
    cx, cy = (w -1) // 2, (h-1) // 2
    # Compute the translation matrix to shift the center to the origin
    T1 = np.array([[1, 0, -cx],
                                    [0, 1, -cy],
                                    [0, 0, 1]],dtype=np.float32)
    # Compute the translation matrix to shift the origin back to the center
    T2 = np.array([[1, 0, cx],
                                    [0, 1, cy],
                                    [0, 0, 1]],dtype=np.float32)
    #translation
    tx,ty = np.random.uniform(-0.05, 0.05, 2) * (w,h)
    translation_matrix = np.eye(3)
    translation_matrix[0,2] = tx
    translation_matrix[1,2] = ty

    #scaling
    sx,sy = np.random.uniform(0.9, 1.1, 2)
    scale_matrix = np.eye(3)
    scale_matrix[0,0] = sx
    scale_matrix[1,1] = sy

    #rotation
    theta = np.random.uniform(-5,5) * np.pi / 180
    rotation_matrix  = np.eye(3)
    rotation_matrix[0,0] = rotation_matrix[1,1] = np.cos(theta)
    rotation_matrix[0,1] = -np.sin(theta)
    rotation_matrix[1,0] = np.sin(theta)

    #shear
    theta = np.random.uniform(-5,5) * np.pi / 180
    shear_matrix  = np.eye(3)
    shear_matrix[0,1] = np.tan(theta)

    transformation = translation_matrix @ rotation_matrix @ shear_matrix @ scale_matrix
    mat = T2 @ transformation @ T1
    return mat

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
        ret,img = cap.read()
        if not ret: break
        img  = cv2.resize(img,(W,H))
        frames.append(img)
    frames = np.array(frames,dtype = np.uint8)
    frame_count  = frames.shape[0]
    #create transformations
    noisy_transforms = np.zeros((frame_count,3,3),dtype = np.float32)
    for idx in range(frame_count):
        noisy_transforms[idx,...] = get_transformation((H,W))

    smooth_transforms = np.zeros_like(noisy_transforms)
    alpha = 0.1  # Smoothing factor
    smooth_transforms[0] = noisy_transforms[0]  # Initialize the first value

    for i in range(1, len(noisy_transforms)):
        smooth_transforms[i] = alpha * noisy_transforms[i] + (1 - alpha) * smooth_transforms[i - 1]
    #create unstable video
    unstable_frames = np.zeros_like(frames)
    for idx in range(1,frame_count-1):
        img = frames[idx,...]
        mat = smooth_transforms[idx,...]
        unstable_frames[idx,...]  = cv2.warpPerspective(img,mat,(W,H))
    save_video(unstable_frames,args.out_path)
    
