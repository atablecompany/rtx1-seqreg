# This script is used to test the fundus.py functions
from skimage.color import rgb2gray
from cv2 import imread
import matplotlib.pyplot as plt
import fundus
from fundus import SHARPNESS_METRIC
import time

# SHARPNESS_METRIC is set within fundus.py


#%% Import video file
# video_path = "G:\PapyrusSorted\AHMED_Madeleine_19790728_FEMALE\OS_20231017114436\OS_20231017114436_X0.0N_Y2.0_Z0.0_AHMED_Madeleine_121.mpg"  # S timto register2 nefunguje dobre
video_path = "G:\PapyrusSorted\ABDEL_Eman_19860604_FEMALE\OD_20231108153957\OD_20231108153957_X0.0T_Y2.0_Z10.0_ABDEL_Eman_201.mpg"

reference_path = video_path.replace(".mpg", ".png")
reference = rgb2gray(imread(reference_path))
start_time = time.time()
frames = fundus.import_video(video_path)
print(frames.shape)

#%% Determine the sharpness of frames
sharpness = fundus.calculate_sharpness(frames, blur=True)
sharpness_threshold = 0.6
selected_frames = fundus.select_frames(frames, sharpness)

#%% Show individual frames
# for i in range(len(frames)):
#     fundus.show_frame(frames[i], sharpness=sharpness[i], custom_note=i)

#%% Plot sharpness over frames
plt.figure(figsize=(10, 6))
plt.plot(sharpness, marker='o', linestyle='-', color='b')
plt.axhline(y=min(sharpness) + sharpness_threshold * (max(sharpness) - min(sharpness)), color='r', linestyle='--', label='Sharpness Threshold')
plt.title(f'Sharpness over frames ({SHARPNESS_METRIC})')
plt.xlabel('Frame index')
plt.ylabel('Sharpness')
plt.grid(True)
# plt.savefig('sharpness_plot.svg', format='svg')
plt.show()

#%% Perform registration and averaging
reg = fundus.register(selected_frames, sharpness, reference='previous', crop=True)
cum = fundus.cumulate(reg)

#%% Show result
fundus.show_frame(cum)
fundus.show_frame(imread(reference_path), custom_note="Reference image\n")

brisque = fundus.assess_quality(cum, path=reference_path)
elapsed_time = time.time() - start_time
print("Processing took: {:.2f} seconds".format(elapsed_time))
print("BRISQUE Image: ", brisque[0])
print("BRISQUE Reference: ", brisque[1])
