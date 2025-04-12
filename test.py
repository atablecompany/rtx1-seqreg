# This script is used to test the fundus.py functions
from skimage.color import rgb2gray
from cv2 import imread
import matplotlib.pyplot as plt
import fundus
from fundus import load_reference_image, select_frames
import time


#%% Import video file
video_path = "G:\PapyrusSorted\AHMED_Madeleine_19790728_FEMALE\OS_20231017114409\OS_20231017114409_X0.0N_Y-2.0_Z0.0_AHMED_Madeleine_121.mpg"  # S timto register2 nefunguje dobre
# video_path = "G:\PapyrusSorted\ADAMS KIPPING_Elena_19920504_FEMALE\OD_20240418114137\OD_20240418114137_X11.3N_Y8.4_Z130.0_ADAMS KIPPING_Elena_473.mpg"
# video_path = "G:\PapyrusSorted\ABADZHIEVA_Polina_19940415_FEMALE\OD_20240506114909\OD_20240506114909_X2.0T_Y0.0_Z0.0_ABADZHIEVA_Polina_496.mpg"
# video_path = "G:\PapyrusSorted\ABDEL_Eman_19860604_FEMALE\OD_20231108153809\OD_20231108153809_X2.0N_Y0.0_Z0.0_ABDEL_Eman_201.mpg"

reference_path = video_path.replace(".mpg", ".png")
report_path = video_path.replace(".mpg", "_report.txt")
reference = load_reference_image(reference_path)
start_time = time.time()
frames = fundus.load_video(video_path)
print(frames.shape)

#%% Determine the sharpness of frames
# metric = 'loc_var_of_gray'
sharpness = fundus.calculate_sharpness(frames)
sharpness_threshold = 0.6
selected_frames = fundus.select_frames(frames, sharpness, threshold=sharpness_threshold)
# TODO: Zda se mi ze to ted bere moc framu jako ostre, asi to chce postelovat threshold

#%% Show individual frames
# for i in range(len(frames)):
#     fundus.show_frame(frames[i], sharpness=sharpness[i], custom_note=i)

#%% Plot sharpness over frames
plt.figure(figsize=(10, 6))
plt.plot(sharpness, marker='o', linestyle='-', color='b')
plt.axhline(y=sharpness_threshold, color='r', linestyle='--', label='Sharpness Threshold')
plt.title(f'Sharpness over frames (avg)')
# plt.title(f'Sharpness over frames ({metric})')
plt.xlabel('Frame index')
plt.ylabel('Sharpness')
plt.grid(True)
# plt.savefig('sharpness_plot.svg', format='svg')
plt.show()

#%% Perform registration and averaging
reg = fundus.register(selected_frames, sharpness, reference='best', pad='same')
cum = fundus.cumulate(reg)

#%% Show result
fundus.show_frame(cum)
fundus.show_frame(reference, custom_note="Reference image\n")

elapsed_time = time.time() - start_time
brisque, piqe = fundus.assess_quality(cum, report_path)
print("Processing took: {:.2f} seconds".format(elapsed_time))
print("BRISQUE Image: ", brisque[0])
print("BRISQUE Reference: ", brisque[1])
print("PIQE Image: ", piqe[0])
print("PIQE Reference: ", piqe[1])
