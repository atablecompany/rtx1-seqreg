# This script is used to test the fundus.py functions
from skimage.color import rgb2gray
from cv2 import imread
import matplotlib.pyplot as plt
import fundus
from fundus import SHARPNESS_METRIC
# SHARPNESS_METRIC is set within fundus.py


#%% Import video file
video_path = ""

reference_path = video_path.replace(".mpg", ".png")
reference = rgb2gray(imread(reference_path))
frames = fundus.import_video(video_path)
print(frames.shape)

#%% Determine the sharpness of frames
sharpness = fundus.calculate_sharpness(frames, blur=True)
sharpness_threshold = 0.6
selected_frames = fundus.select_frames(frames, sharpness)

#%% Show individual frames
for i in range(len(frames)):
    fundus.show_frame(frames[i], sharpness=sharpness[i], custom_note=i)

#%% Plot sharpness over frames
plt.figure(figsize=(10, 6))
plt.plot(sharpness, marker='o', linestyle='-', color='b')
plt.axhline(y=min(sharpness) + sharpness_threshold * (max(sharpness) - min(sharpness)), color='r', linestyle='--', label='Sharpness Threshold')
plt.title(f'Sharpness over frames ({SHARPNESS_METRIC})')
plt.xlabel('Frame index')
plt.ylabel('Sharpness')
plt.grid(True)
plt.show()

#%% Perform registration and averaging
reg = fundus.register(selected_frames, sharpness, reference='best', crop=True)
cum = fundus.cumulate(reg)

#%% Show result

fundus.show_frame(cum)
fundus.show_frame(imread(reference_path), custom_note="Reference image")

brisque_image, brisque_reference = fundus.assess_quality(cum, path=reference_path, generate_report=False)[0]
print("BRISQUE Image: ", brisque_image)
print("BRISQUE Reference: ", brisque_reference)
