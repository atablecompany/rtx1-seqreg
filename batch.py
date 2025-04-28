import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import fundus
from tqdm import tqdm


def process_video(video_path):
    # Existing processing logic remains unchanged
    frames = fundus.load_video(video_path)
    reference_path = video_path.replace(".mpg", ".png")
    fundus.load_reference_image(reference_path)
    sharpness = fundus.calculate_sharpness(frames)
    # selected_frames = fundus.select_frames(frames, sharpness, threshold=0.8)
    selected_frames = fundus.select_frames2(frames, sharpness)
    reg = fundus.register2(selected_frames, sharpness, reference='best')
    cum = fundus.cumulate(reg)
    # TODO: Mozna denoising udelat na zaklade hodnoty BRISQUE a PIQE namisto poctu framu
    if len(selected_frames) < 4:
        cum = fundus.denoise(cum, method='tv')
    elif len(selected_frames) < 9:
        cum = fundus.denoise(cum, method='bm3d')
    output_path = video_path.replace(".mpg", "_processed_new.png")
    fundus.save_frame(cum, output_path)
    report_path = video_path.replace(".mpg", "_report_new.txt")
    fundus.assess_quality(cum, report_path)
    return video_path  # Return path for progress tracking


if __name__ == "__main__":
    video_directory = "G:\PapyrusSorted"
    video_files = glob.glob(os.path.join(video_directory, "**", "*.mpg"), recursive=True)

    with ProcessPoolExecutor(max_workers=8) as executor:
        # Submit all video processing tasks
        futures = [executor.submit(process_video, file) for file in video_files]

        # Initialize progress bar
        with tqdm(total=len(video_files), desc="Processing videos") as pbar:
            # Update progress as tasks complete
            for future in as_completed(futures):
                future.result()  # Get result (or raise exception)
                pbar.update(1)
