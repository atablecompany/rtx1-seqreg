#%% Import video file
# Path to the .mpg file
video_path = "G:/PapyrusSorted/test.mpg"
# video_path = "G:\PapyrusSorted\ALNAMROD_Islam_20060101_FEMALE\OD_20240602140937\OD_20240602140937_X0.0T_Y-2.0_Z0.0_ALNAMROD_Islam_492.mpg"

frames = import_video(video_path)
print(frames.shape)

#%% Determine the variance of laplacian of frames
# var_of_lap = []
# for frame in frames:
#     var_of_lap.append(cv2.Laplacian(frame, cv2.CV_64F).var())

#%% Determine the sharpness of frames
sharpness = []
for frame in frames:
    sharpness.append(calculate_sharpness(frame, metric=SHARPNESS_METRIC))  # Using local gray variance

#%% Show frames and plot sharpness over frame indices
for i in range(len(frames)):
    show_frame(frames[i], sharpness[i], i)

plt.figure(figsize=(10, 6))
plt.plot(sharpness, marker='o', linestyle='-', color='b')
plt.title(f'Sharpness over frames, metric={SHARPNESS_METRIC}')
plt.xlabel('Frame index')
plt.ylabel('Sharpness')
plt.grid(True)
plt.show()

#%% Select the sharpest frames
threshold = 0.92 * max(sharpness)
selected_frames_indices = [i for i, var in enumerate(sharpness) if var > threshold]
best_frame_index = np.argmax(sharpness)

#%% Perform registration to reference
# ref_image = frames[best_frame_index]
# offset_image = frames[selected_frames_indices[4]]
# sr = StackReg(StackReg.RIGID_BODY)
# out_rigid = sr.register_transform(ref_image, offset_image)
# # Show registered frames
# show_frame(out_rigid, note="Registered image")
# show_frame(ref_image, frame_number=best_frame_index, note="Reference image")
# show_frame(offset_image, frame_number=selected_frames_indices[4], note="Unregistered image")

#%% Perform stack registration
selected_frames = frames[selected_frames_indices]
sr = StackReg(StackReg.RIGID_BODY)
out_rigid_stack = sr.register_transform_stack(selected_frames, reference='previous')
# Show registered frames
for i, frame in enumerate(out_rigid_stack):
    show_frame(frame, frame_number=selected_frames_indices[i], note="Stack registered image")

# oriznout nebo pospojovat a rozsirit??
#%% Average registered frames
cum = np.mean(out_rigid_stack, axis=0)
cum_note = f"Mean of {len(selected_frames_indices)} registered frames"
show_frame(cum, note=cum_note, save=True, filename="cum.png")
show_frame(cum, note=cum_note)
