import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import fundus
from tqdm import tqdm
import random


def process_video(video_path):
    frames = fundus.load_video(video_path)
    reference_path = video_path.replace(".mpg", ".png")
    fundus.load_reference_image(reference_path)
    sharpness = fundus.calculate_sharpness(frames)
    # selected_frames = fundus.select_frames(frames, sharpness, threshold=0.8)
    selected_frames = fundus.select_frames(frames, sharpness)
    reg = fundus.register(selected_frames, sharpness, reference='best')
    cum = fundus.cumulate(reg)
    _, _, brisque, piqe = fundus.assess_quality(cum)
    if brisque[0] > brisque[1] or piqe[0] > piqe[1] or len(selected_frames) < 10:
        cum = fundus.denoise(cum)  # Apply denoising if quality is worse than reference or if noise level is too high
    output_path = video_path.replace(".mpg", "_processed_adaptive.png")
    fundus.save_frame(cum, output_path)
    report_path = video_path.replace(".mpg", "_report_adaptive.txt")
    fundus.assess_quality(cum, report_path)
    return video_path  # Return path for progress tracking


if __name__ == "__main__":
    video_directory = "G:\PapyrusSorted"
    video_files = glob.glob(os.path.join(video_directory, "**", "*.mpg"), recursive=True)

    # Select 300 random files based on a seed
    random.seed(23)  # Set seed for reproducibility
    video_files = random.sample(video_files, 300)

    with ProcessPoolExecutor(max_workers=8) as executor:
        # Submit all video processing tasks
        futures = [executor.submit(process_video, file) for file in video_files]

        # Initialize progress bar
        with tqdm(total=len(video_files), desc="Processing videos") as pbar:
            # Update progress as tasks complete
            for future in as_completed(futures):
                future.result()  # Get result (or raise exception)
                pbar.update(1)
