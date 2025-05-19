# This script is used to test the fundus.py functions
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fundus
import time

#%% Import video file
video_path = ""

reference_path = video_path.replace(".mpg", ".png")
report_path = video_path.replace(".mpg", "_report_test.txt")
start_time = time.time()
frames = fundus.load_video(video_path)
reference = fundus.load_reference_image(reference_path)
print(frames.shape)

#%% Determine the sharpness of frames
# Use adaptive sharp frame selection based on the distribution of sharpness values
sharpness = fundus.calculate_sharpness(frames)
selected_frames = fundus.select_frames(frames, sharpness)

# Use a hard threshold for sharp frame selection
# sharpness_threshold = 0.8
# selected_frames = fundus.select_frames2(frames, sharpness, threshold=sharpness_threshold)

#%% Show individual frames
# for i in range(len(frames)):
#     fundus.show_frame(frames[i], sharpness=sharpness[i], custom_note=i)

#%% Plot sharpness over frames
# plt.figure(figsize=(10, 6))
# plt.plot(sharpness, marker='o', linestyle='-', color='b')
# plt.title(f'Sharpness over frames (avg)')
# plt.xlabel('Frame index')
# plt.ylabel('Sharpness')
# plt.grid(True)
# plt.show()

#%% Perform registration
# Register with pyStackReg
# reg = fundus.register(selected_frames, sharpness, reference='previous', pad='same')

# Register with elastix
reg = fundus.register2(selected_frames, sharpness, reference='previous', pad='same')

#%% Save registered frames
# for i in range(len(reg)):
#     fundus.save_frame(reg[i], path="")

#%% Export registered frames as video
# output_path = ""
# height, width = frames[0].shape[:2]
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Lossless codec
# out = cv2.VideoWriter(output_path, fourcc, 15, (width, height), False)
#
# for frame in reg:
#     # Ensure frame is in the correct format (uint8)
#     if frame.dtype != np.uint8:
#         frame = frame.astype(np.uint8)
#     out.write(frame)
#
# out.release()
# print(f"Video saved to {output_path}")

#%% Cumulate registered frames
cum = fundus.cumulate(reg, method='mean')

#%% Show result
fundus.show_frame(cum)
fundus.show_frame(reference, custom_note="Reference image\n")

#%% Apply additional denoising
weight = 2
cum = fundus.denoise(cum, method='bm3d')
fundus.show_frame(cum, custom_note=f"Denoised\n")

#%% Resize image to match reference
fundus.resize(cum, reference)

#%% Assess quality
elapsed_time = time.time() - start_time
sharpness_log, brisque, piqe, niqe = fundus.assess_quality(cum, report_path)
print("Processing took: {:.2f} seconds".format(elapsed_time))
print("Sharpness LoG Image: ", sharpness_log[0])
print("Sharpness LoG Reference: ", sharpness_log[1])
print("BRISQUE Image: ", brisque[0])
print("BRISQUE Reference: ", brisque[1])
print("PIQE Image: ", piqe[0])
print("PIQE Reference: ", piqe[1])
print("NIQE Image: ", niqe[0])
print("NIQE Reference: ", niqe[1])
