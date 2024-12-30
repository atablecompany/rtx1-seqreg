# Description: This script is used to test the fundus.py functions
from skimage.color import rgb2gray
from cv2 import imread, resize
import matplotlib.pyplot as plt
import fundus
import skimage.restoration
from fundus import SHARPNESS_METRIC
# SHARPNESS_METRIC is set within fundus.py
#%% Import video file
# Path to the .mpg file
# video_path = "G:/PapyrusSorted/test_auto.mpg"
# video_path = "G:\PapyrusSorted\AHMED_Madeleine_19790728_FEMALE\OS_20231017114311\OS_20231017114311_X2.0N_Y0.0_Z0.0_AHMED_Madeleine_121.mpg"
# video_path = "G:\PapyrusSorted\ABADZHIEVA_Polina_19940415_FEMALE\OD_20240506114845\OD_20240506114845_X2.0N_Y0.0_Z0.0_ABADZHIEVA_Polina_496.mpg"
# video_path = "G:\PapyrusSorted\ALTSTEDT_Lia_20050421_FEMALE\OD_20231219141316\OD_20231219141316_X0.0T_Y-2.0_Z0.0_ALTSTEDT_Lia_NS017.mpg"
# video_path = "G:\PapyrusSorted\ABDEL_Eman_19860604_FEMALE\OD_20231108153844\OD_20231108153844_X2.0T_Y0.0_Z0.0_ABDEL_Eman_201.mpg"
# video_path = "G:\PapyrusSorted\ABDULMANOVA_Fariza_19900611_FEMALE\OD_20240222130817\OD_20240222130817_X2.0T_Y0.0_Z30.0_ABDULMANOVA_Fariza_423.mpg"
# video_path = "G:\PapyrusSorted\ABDEL_Eman_19860604_FEMALE\OD_20231108153928\OD_20231108153928_X0.0T_Y-2.0_Z10.0_ABDEL_Eman_201.mpg"
video_path = "G:\PapyrusSorted\ABDEL_Eman_19860604_FEMALE\OD_20231108154058\OD_20231108154058_X11.7N_Y7.4_Z180.0_ABDEL_Eman_201.mpg"

reference_path = video_path.replace(".mpg", ".png")
reference = rgb2gray(imread(reference_path))
frames = fundus.import_video(video_path)
print(frames.shape)

#%% Determine the sharpness of frames
sharpness = fundus.calculate_sharpness(frames, blur=True)
sharpness_threshold = 0.6
#%% Show frames
# for i in range(len(frames)):
#     fundus.show_frame(frames[i], sharpness=sharpness[i], note=i)

#%% Plot sharpness over frames
plt.figure(figsize=(10, 6))
plt.plot(sharpness, marker='o', linestyle='-', color='b')
plt.axhline(y=min(sharpness) + sharpness_threshold * (max(sharpness) - min(sharpness)), color='r', linestyle='--', label='Sharpness Threshold')
plt.title(f'Sharpness over frames ({SHARPNESS_METRIC})')
plt.xlabel('Frame index')
plt.ylabel('Sharpness')
plt.grid(True)
plt.show()

#%%
reg = fundus.register(frames, sharpness, threshold=sharpness_threshold, reference='best', crop=True)
cum = fundus.cumulate(reg)
# registered_stack = fundus.register_cumulate(frames, sharpness, threshold=sharpness_threshold, cumulate=False, reference='best')
# denoised = fundus.denoise(cum)
# fundus.show_frame(denoised)
#%% Average registered frames

# fundus.show_frame(cum, note=cum_note, save=True, filename="cum.png")
fundus.show_frame(cum)
fundus.show_frame(imread(reference_path), custom_note="Reference image")

# resize reference to the same size as the registered image
reference = resize(reference, (cum.shape[1], cum.shape[0]))
brisque_image, brisque_reference = fundus.assess_quality(cum, path=reference_path, generate_report=False)[0]
snr_image, snr_reference = fundus.assess_quality(cum, path=reference_path, generate_report=False)[1]
print("BRISQUE Image: ", brisque_image)
print("BRISQUE Reference: ", brisque_reference)
print("SNR Image: ", snr_image)
print("SNR Reference: ", snr_reference)
