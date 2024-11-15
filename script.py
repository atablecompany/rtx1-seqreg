from cv2 import imread
import fundus
import matplotlib.pyplot as plt

# SHARPNESS_METRIC is set within fundus.py
#%% Import video file
# Path to the .mpg file
# video_path = "G:/PapyrusSorted/test.mpg"
video_path = "C:\PapyrusSorted\AHMED_Madeleine_19790728_FEMALE\OS_20231017114311\OS_20231017114311_X2.0N_Y0.0_Z0.0_AHMED_Madeleine_121.mpg"
reference_path = "C:\PapyrusSorted\AHMED_Madeleine_19790728_FEMALE\OS_20231017114311\OS_20231017114311_X2.0N_Y0.0_Z0.0_AHMED_Madeleine_121.png"

frames = fundus.import_video(video_path)
print(frames.shape)

#%% Determine the variance of laplacian of frames
# var_of_lap = []
# for frame in frames:
#     var_of_lap.append(cv2.Laplacian(frame, cv2.CV_64F).var())

#%% Determine the sharpness of frames
sharpness = fundus.calculate_sharpness(frames)

#%% Show frames and plot sharpness over frame indices
for i in range(len(frames)):
     fundus.show_frame(frames[i], sharpness[i], i)

plt.figure(figsize=(10, 6))
plt.plot(sharpness, marker='o', linestyle='-', color='b')
plt.title('Sharpness over frames')
plt.xlabel('Frame index')
plt.ylabel('Sharpness')
plt.grid(True)
plt.show()

# quality = [fundus.assess_quality(frame) for frame in frames]
# plt.figure(figsize=(10, 6))
# plt.plot(quality, marker='o', linestyle='-', color='r')
# plt.title('Quality over frames')
# plt.xlabel('Frame index')
# plt.ylabel('Sharpness')
# plt.grid(True)
# plt.show()

# Proc je to od i=7 vsechno rozmazany?? pro C:\PapyrusSorted\AHMED_Madeleine_19790728_FEMALE\OS_20231017114311\OS_20231017114311_X2.0N_Y0.0_Z0.0_AHMED_Madeleine_121.mpg

#%% Select the sharpest frames
# threshold = 0.92 * max(sharpness)
# selected_frames_indices = [i for i, var in enumerate(sharpness) if var > threshold]
# best_frame_index = np.argmax(sharpness)

 # oriznout nebo pospojovat a rozsirit??
#%%
cum, cum_note = fundus.register_cumulate(frames, sharpness, threshold=0.92 * max(sharpness), reference='previous', cumulate=True)

#%% Average registered frames
# cum = np.mean(out_rigid_stack, axis=0)
# cum_note = f"Mean of {len(selected_frames_indices)} registered frames"
# fundus.show_frame(cum, note=cum_note, save=True, filename="cum.png")
fundus.show_frame(cum, note=cum_note)
print(f"Registered image quality: {fundus.assess_quality(cum)}")
print(f"Reference image quality: {fundus.assess_quality(imread(reference_path))}")
