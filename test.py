# This script is used to test the fundus.py functions
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fundus
from fundus import load_reference_image
import time
import skimage

#%% Import video file
# video_path = "G:\PapyrusSorted\AVINA ZAVALA_Marta Ester_19860214_FEMALE\OD_20240405143339\OD_20240405143339_X2.0N_Y0.0_Z0.0_AVINA ZAVALA_Marta Ester_451.mpg"
# video_path = "G:\PapyrusSorted\ABADZHIEVA_Polina_19940415_FEMALE\OD_20240506114845\OD_20240506114845_X2.0N_Y0.0_Z0.0_ABADZHIEVA_Polina_496.mpg"

# Nektere pripady
# video_path = "G:\PapyrusSorted\ALLERT_Mandy_19840402_FEMALE\OD_20231120102058\OD_20231120102058_X10.7N_Y8.1_Z70.0_ALLERT_Mandy_234.mpg"
# video_path = "G:\PapyrusSorted\AUERBACH_Nancy_19920617_FEMALE\OD_20231219143706\OD_20231219143706_X2.0N_Y0.0_Z0.0_AUERBACH_Nancy_NS018.mpg"
# video_path = "G:\PapyrusSorted\AL KHAKANI_Denise_19890818_FEMALE\OD_20240611115725\OD_20240611115725_X0.0T_Y-2.0_Z0.0_AL KHAKANI_Denise_550.mpg"
# video_path = "G:\PapyrusSorted\AWADALLAH_Sara_19980829_FEMALE\OD_20240503142217\OD_20240503142217_X10.2N_Y6.5_Z120.0_AWADALLAH_Sara_155.mpg"
video_path = "G:\PapyrusSorted\ABDEL_Eman_19860604_FEMALE\OD_20240320145026\OD_20240320145026_X11.7N_Y7.4_Z180.0_ABDEL_Eman_201.mpg"

reference_path = video_path.replace(".mpg", ".png")
report_path = video_path.replace(".mpg", "_report_test.txt")
reference = load_reference_image(reference_path)
start_time = time.time()
frames = fundus.load_video(video_path)
print(frames.shape)

#%% Determine the sharpness of frames
# metric = 'loc_var_of_gray'
sharpness = fundus.calculate_sharpness(frames)
sharpness_threshold = 0.8
# selected_frames = fundus.select_frames(frames, sharpness, threshold=sharpness_threshold)
selected_frames = fundus.select_frames2(frames, sharpness)

#%% Show individual frames
# for i in range(len(frames)):
#     fundus.show_frame(frames[i], sharpness=sharpness[i], custom_note=i)

#%% Plot sharpness over frames
plt.figure(figsize=(10, 6))
plt.plot(sharpness, marker='o', linestyle='-', color='b')
plt.axhline(y=sharpness_threshold, color='r', linestyle='--', label='Sharpness Threshold')
plt.title(f'Sharpness over frames (avg)')
plt.xlabel('Frame index')
plt.ylabel('Sharpness')
plt.grid(True)
# plt.savefig('sharpness_plot.svg', format='svg')
plt.show()

#%% Perform registration
reg = fundus.register(selected_frames, sharpness, reference='best', pad='same')
# TODO: Register pomoci elastixu v podstate vubec nefunguje
# for i in range(len(reg)):
#     fundus.save_frame(reg[i], f"C:/Users/tengl/PycharmProjects/dp/reg/frame_{i}.png")

#%% Export registered frames as lossless avi
output_path = "C:/Users/tengl/PycharmProjects/dp/registered_frames.avi"
height, width = frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Lossless codec
out = cv2.VideoWriter(output_path, fourcc, 15, (width, height), False)

for frame in reg:
    # Ensure frame is in the correct format (uint8)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    out.write(frame)

out.release()
print(f"Video saved to {output_path}")

#%% Cumulate registered frames
cum = fundus.cumulate(reg, method='mean')
# TODO: Zkusit rozdily mezi mean a median

# fundus.save_frame(cum, "C:/Users/tengl/PycharmProjects/dp/pokus/cum_wt3.png")
# fundus.assess_quality(cum, "C:/Users/tengl/PycharmProjects/dp/pokus/cum_wt3.txt")

#%% Show result
fundus.show_frame(cum)
fundus.show_frame(reference, custom_note="Reference image\n")

#%% Apply additional denoising
weight = 2
denoised = fundus.denoise(cum, method='hamgf', weight=weight)
fundus.show_frame(denoised, custom_note=f"Denoised weight={weight}\n")

#%% Assess quality
elapsed_time = time.time() - start_time
sharpness, sharpness_log, brisque, piqe = fundus.assess_quality(cum, report_path)
print("Processing took: {:.2f} seconds".format(elapsed_time))
print("Sharpness Image: ", sharpness[0])
print("Sharpness Reference: ", sharpness[1])
print("Sharpness LoG Image: ", sharpness_log[0])
print("Sharpness LoG Reference: ", sharpness_log[1])
print("BRISQUE Image: ", brisque[0])
print("BRISQUE Reference: ", brisque[1])
print("PIQE Image: ", piqe[0])
print("PIQE Reference: ", piqe[1])
