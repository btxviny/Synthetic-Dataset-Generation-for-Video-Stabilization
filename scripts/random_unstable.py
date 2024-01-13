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
    parser.add_argument('--transforms_path', type=str, help='transforms file path')
    return parser.parse_args()

def fixBorder(frame):
    s = frame.shape
    # Scale the image 1% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

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
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_path, fourcc, 30.0, (W,H))
    for idx in range(frame_count):
        ret,img = cap.read()
        mat = smooth_transforms[idx,...]
        img = cv2.resize(img,(W,H))
        img = cv2.warpPerspective(img,mat,(W,H))
        img = fixBorder(img)
        out.write(img)
    out.release()
    np.save(args.transforms_path,smooth_transforms)
    
E:\Datasets\Learning_DeepStab\stable\