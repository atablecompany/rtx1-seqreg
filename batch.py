import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import fundus
from tqdm import tqdm


def process_video(video_path):
    frames = fundus.load_video(video_path)  # Load video frames
    reference_path = video_path.replace(".mpg", ".png")
    reference = fundus.load_reference_image(reference_path)  # Load reference image
    sharpness = fundus.calculate_sharpness(frames)  # Calculate sharpness of frames
    selected_frames = fundus.select_frames(frames, sharpness)  # Select sharp frames
    reg = fundus.register(selected_frames, sharpness, reference='mean')  # Register selected frames
    cum = fundus.cumulate(reg)  # Cumulate registered frames
    if fundus.is_central_region:  # Apply additional denoising
        if len(selected_frames) < 4:
            cum = fundus.denoise(cum, method='hmgf')  # Denoise using HMGF if very few frames are selected
        else:
            cum = fundus.denoise(cum)  # Always denoise using BM3D if the region is central
    else:
        cum = fundus.denoise(cum)  # If region is not central, always denoise
    cum = fundus.resize(cum, reference)  # Resize processed image to match reference

    output_path = video_path.replace(".mpg", "_processed.png")
    fundus.save_frame(cum, output_path)  # Save processed image
    report_path = video_path.replace(".mpg", "_report.txt")
    fundus.assess_quality(cum, report_path)  # Save quality assessment report, compare with reference if it was loaded

    return video_path  # Return path for progress tracking


if __name__ == "__main__":
    video_directory = "path/to/video/directory"  # Input directory containing video files
    video_files = glob.glob(os.path.join(video_directory, "**", "*.mpg"), recursive=True)

    # Parallel processing of video files
    with ProcessPoolExecutor(max_workers=4) as executor:  # Select the number of CPU threads to be used
        # Submit all video processing tasks
        futures = [executor.submit(process_video, file) for file in video_files]

        # Initialize progress bar
        with tqdm(total=len(video_files), desc="Processing videos") as pbar:
            # Update progress as tasks complete
            for future in as_completed(futures):
                future.result()  # Get result (or raise exception)
                pbar.update(1)
