import glob
import os
import fundus


# Function to process each video
def process_video(video_path):
    frames = fundus.import_video(video_path)
    sharpness = fundus.calculate_sharpness(frames)
    cum, cum_note = fundus.register_cumulate(frames, sharpness, threshold=0.92 * max(sharpness), reference='previous')
    filename = os.path.basename(video_path)
    filename = filename[:-4] + "_cumulated.png"
    fundus.show_frame(cum, note=cum_note, save=True, filename=filename)


# Specify the directory containing video files
video_directory = "G:\PapyrusSorted\ABADZHIEVA_Polina_19940415_FEMALE"
video_files = glob.glob(os.path.join(video_directory, "**", "*.mpg"), recursive=True)

# Loop through all files and process each
for video_file in video_files:
    print(f"Processing {video_file}")
    process_video(video_file)
