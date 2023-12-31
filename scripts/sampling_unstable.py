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
    #load transformations bank and chose transformations
    transformations_bank = np.load('./data/transformations_bank.npy')
    matrices = np.array(random.choices(transformations_bank,k = num_frames))
    noise_transforms = np.zeros((num_frames,3))
    noise_transforms[:,0] = matrices[:,0,2] * W
    noise_transforms[:,1] = matrices[:,1,2] * H
    noise_transforms[:,2] = np.arctan2(matrices[:,1, 0], matrices[:,0, 0])
    #make them more natural with low-pass
    #smooth it with EWMA
    smooth_noise_transforms = np.zeros_like(noise_transforms)
    alpha = 0.5  # Smoothing factor
    smooth_noise_transforms[0] = noise_transforms[0]  # Initialize the first value

    for i in range(1, len(noise_transforms)):
        smooth_noise_transforms[i] = alpha * noise_transforms[i] + (1 - alpha) * smooth_noise_transforms[i - 1]

    unstable_frames = warp_frames(frames,smooth_noise_transforms)
    np.save(args.transforms_path,smooth_noise_transforms)
    save_video(unstable_frames,args.out_path)
    
