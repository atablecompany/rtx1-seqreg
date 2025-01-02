# Control script to process all videos in a directory (including subdirectories)
import glob
import os
import fundus


def process_video(video_path):
    """
    Process a video file and save the processed image.
    :param video_path: Path to the video file to be processed
    """
    reference_path = video_path.replace(".mpg", ".png")  # Path to the reference .png image
    frames = fundus.import_video(video_path)  # Import the video file as a numpy array
    sharpness = fundus.calculate_sharpness(frames)  # Calculate the sharpness of each frame
    selected_frames = fundus.select_frames(frames, sharpness)  # Select sharp frames
    reg = fundus.register(selected_frames, sharpness)  # Perform registration of sharp frames
    cum = fundus.cumulate(reg)  # Cumulate the registered frames
    output_path = video_path.replace(".mpg", "_processed.png")  # Path to save the processed image
    fundus.save_frame(cum, path=output_path)  # Save the processed image
    fundus.assess_quality(cum, reference_path)  # Generate quality assessment report


# Specify the directory containing video files
video_directory = "G:\PapyrusSorted"
video_files = glob.glob(os.path.join(video_directory, "**", "*.mpg"), recursive=True)

# Loop through all files and process each
for i, video_file in enumerate(video_files):
    print(f"Processing ({i+1} out of {len(video_files)}) {video_file}")
    process_video(video_file)

print("Done!")
