# Synthetic Dataset Generation for Video Stabilization

I provide four different methods ['random', 'sampling', 'pca', 'gan'] for unstable video synthesis as described in my thesis.

![Example](https://github.com/btxviny/Synthetic-Dataset-Generation-for-Video-Stabilization/blob/main/result.gif)

## Instructions
1. Gather the original stable videos from any online repository and place them in 'stable'.
2.  - Run the following command:
       ```bash
       python generate_dataset.py --method pca --stable_path ./stable/ --unstable_path ./unstable/
       ```
   - Chose any method from ['random', 'sampling', 'pca', 'gan'].
   - Replace `./stable/` with the path to your input stable videos.
   - Replace `./unstable/` with the directory you want the generated videos to be placed in.
