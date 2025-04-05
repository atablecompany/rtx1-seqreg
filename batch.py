# Control script to process all videos in a directory (including subdirectories)
import glob
import os
import fundus
from tqdm import tqdm


def process_video(video_path):
    """
    Process a video file and save the processed image.
    :param video_path: Path to the video file to be processed
    """
    frames = fundus.load_video(video_path)  # Load the video file as a numpy array
    reference_path = video_path.replace(".mpg", ".png")  # Path to the reference .png image
    fundus.load_reference_image(reference_path)  # Load the reference image
    sharpness = fundus.calculate_sharpness(frames)  # Calculate the sharpness of each frame
    selected_frames = fundus.select_frames(frames, sharpness)  # Select sharp frames
    reg = fundus.register2(selected_frames, sharpness, reference='previous')  # Perform registration of sharp frames
    cum = fundus.cumulate(reg)  # Cumulate the registered frames
    output_path = video_path.replace(".mpg", "_processed.png")  # Path to save the processed image
    fundus.save_frame(cum, output_path)  # Save the processed image
    report_path = video_path.replace(".mpg", "_report.txt")  # Path to save the quality report
    fundus.assess_quality(cum, report_path)  # Generate quality assessment report


# Specify the directory containing video files
video_directory = "G:\PapyrusSorted\AHMED_Madeleine_19790728_FEMALE"
video_files = glob.glob(os.path.join(video_directory, "**", "*.mpg"), recursive=True)

# Loop through all files and process each
for video_file in tqdm(video_files, desc="Processing video files"):
    process_video(video_file)
