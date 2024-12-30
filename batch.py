import glob
import os
import fundus
# Control function to process all videos in a directory (including subdirectories)


def process_video(video_path):
    """
    Process a video file and save the processed image.
    :param video_path: Path to the video file to be processed
    :return:
    """
    reference_path = video_path.replace(".mpg", ".png")
    frames = fundus.import_video(video_path)
    sharpness = fundus.calculate_sharpness(frames)
    reg = fundus.register(frames, sharpness, reference='best', crop=True)
    cum = fundus.cumulate(reg)
    output_path = video_path.replace(".mpg", "_processed.png")
    fundus.save_frame(cum, path=output_path)
    fundus.assess_quality(cum, reference_path)


# Specify the directory containing video files
video_directory = "G:\PapyrusSorted"
video_files = glob.glob(os.path.join(video_directory, "**", "*.mpg"), recursive=True)

# Loop through all files and process each
for i, video_file in enumerate(video_files):
    print(f"Processing ({i+1} out of {len(video_files)}) {video_file}")
    process_video(video_file)

print("Done!")
