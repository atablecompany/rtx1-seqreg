import glob
import os
import fundus


# Function to process each video
def process_video(video_path, reference_path):
    frames = fundus.import_video(video_path)
    sharpness = fundus.calculate_sharpness(frames)
    cum, cum_note = fundus.register_cumulate(frames, sharpness, reference='best', crop=True)
    filename = video_path.replace(".mpg", "_processed.png")
    fundus.show_frame(cum, overlay=False, note=cum_note, save=True, filename=filename)
    fundus.assess_quality(cum, reference_path, cum_note)


# Specify the directory containing video files
video_directory = "G:\PapyrusSorted\ABDEL_Eman_19860604_FEMALE"

video_files = glob.glob(os.path.join(video_directory, "**", "*.mpg"), recursive=True)
reference_files = [f.replace(".mpg", ".png") for f in video_files]

# Loop through all files and process each
for i, _ in enumerate(video_files):
    print(f"Processing ({i+1} out of {len(video_files)}) {video_files[i]}")
    process_video(video_files[i], reference_files[i])
print("Done!")
