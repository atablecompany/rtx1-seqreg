import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import fundus
from tqdm import tqdm
import random


def process_video(video_path):
    frames = fundus.load_video(video_path)  # Load video frames
    reference_path = video_path.replace(".mpg", ".png")
    fundus.load_reference_image(reference_path)  # Load reference image
    sharpness = fundus.calculate_sharpness(frames)  # Calculate sharpness of frames
    selected_frames = fundus.select_frames(frames, sharpness)  # Select sharp frames
    reg = fundus.register(selected_frames, sharpness, reference='best')  # Register selected frames
    # TODO: Možná použít pro necentrální oblasti flexibilní registraci
    cum = fundus.cumulate(reg)  # Cumulate registered frames
    if fundus.is_central_region:
        cum = fundus.denoise(cum)  # Always denoise if the region is central
    elif len(selected_frames) < 12:
        cum = fundus.denoise(cum)  # If region is not central and less than 12 frames, denoise
    # TODO: Možná forcnout hamgf denoising pro všechny když jsou míň jak 3 snímky?
    # TODO: Možná měnit váhu denoisingu podle počtu snímků?
    output_path = video_path.replace(".mpg", "_processed_adaptive5.png")
    fundus.save_frame(cum, output_path)  # Save processed image
    report_path = video_path.replace(".mpg", "_report_adaptive5.txt")
    fundus.assess_quality(cum, report_path)  # Save quality assessment report, compare with reference if it was loaded
    return video_path  # Return path for progress tracking


if __name__ == "__main__":
    video_directory = "G:\PapyrusSorted"
    video_files = glob.glob(os.path.join(video_directory, "**", "*.mpg"), recursive=True)

    # Select 300 random files based on a seed
    random.seed(37)  # Set seed for reproducibility
    video_files = random.sample(video_files, 20)

    with ProcessPoolExecutor(max_workers=8) as executor:
        # Submit all video processing tasks
        futures = [executor.submit(process_video, file) for file in video_files]

        # Initialize progress bar
        with tqdm(total=len(video_files), desc="Processing videos") as pbar:
            # Update progress as tasks complete
            for future in as_completed(futures):
                future.result()  # Get result (or raise exception)
                pbar.update(1)
