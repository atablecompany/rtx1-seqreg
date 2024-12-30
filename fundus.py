import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.restoration
import torch
from pystackreg import StackReg
import piq
from skimage.color import rgb2gray

# pip install -r requirements.txt
SHARPNESS_METRIC = 'var_of_laplacian'  # Choose between 'loc_var_of_gray', 'var_of_laplacian', 'tenengrad', 'var_of_tenengrad'
note = ""  # Note to be displayed in the title of the image or printed in the report


def import_video(video_path):
    """
    Opens a video file and returns a np.ndarray of non-repeated frames. Only every third frame of the video file is saved.
    :param video_path: Video file path.
    :return: Reduced frame stack as np.ndarray.
    """
    global note
    note = ""  # Reset the note for each video

    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error opening video file")
        exit()

    frames_reduced = []
    frames = []

    # Read the first frame
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        exit()
    else:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # Read frames and save them to a list
    while True:
        ok, frame = video.read()
        if not ok:
            break
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    video.release()

    # Only save every third frame
    frames_reduced = frames[2::3]

    return np.array(frames_reduced).astype("uint8")  # Output array of frames


def calculate_sharpness(frames, metric=SHARPNESS_METRIC, blur=True, threshold=None):
    """
    Calculates the sharpness of a frame or frame stack using a specified metric.
    :param blur: If True, the input frames will be blurred using a Gaussian filter to reduce the noise level.
    :param frames: Input frame as np.ndarray.
    :param metric: Can be either 'loc_var_of_gray', 'var_of_laplacian', 'tenengrad', or 'var_of_tenengrad'.
    :param threshold: Threshold between 0 and 1 for selecting the sharpest frames (0: all frames are selected, 1: only the sharpest frame is selected). If not provided, the threshold is calculated automatically.
    :return: Estimated sharpness value if input is a single frame or list of estimated sharpness values if input is a frame stack.
    """
    global note
    window_size = 36  # Window size for local variance of gray

    if len(frames.shape) == 2:
        # If a single frame is given
        if frames.ndim == 3:
            frames = rgb2gray(frames)
        if blur:
            frames = cv2.GaussianBlur(frames, (7, 7), 0)
        if metric == 'loc_var_of_gray':
            # Determine the local gray level variance in a window https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=903548
            height, width = frames.shape
            local_variances = []
            for y in range(0, height, window_size):
                for x in range(0, width, window_size):
                    window = frames[y:y + window_size, x:x + window_size]
                    if window.size == 0:
                        continue
                    local_variances.append(np.var(window))
            var_of_gray = np.mean(local_variances)
            return var_of_gray
        elif metric == 'var_of_laplacian':
            var_of_lap = cv2.Laplacian(frames, cv2.CV_64F).var()
            return var_of_lap
        elif metric == 'tenengrad':
            sobel_x = cv2.Sobel(frames, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(frames, cv2.CV_64F, 0, 1, ksize=3)
            gradient = sobel_x ** 2 + sobel_y ** 2
            tenengrad = np.mean(gradient)
            return tenengrad
        elif metric == 'var_of_tenengrad':
            sobel_x = cv2.Sobel(frames, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(frames, cv2.CV_64F, 0, 1, ksize=3)
            gradient = sobel_x**2 + sobel_y**2
            var_of_tenengrad = np.var(gradient)
            return var_of_tenengrad
        else:
            raise ValueError("Invalid metric parameter")

    elif len(frames.shape) == 3:
        # If a stack of frames is given
        note += f"Sharpness metric used: {metric}\n"
        # Calculate sharpness for each frame in the stack by calling the function recursively
        sharpness = [calculate_sharpness(f, metric) for f in frames]
        return sharpness

    else:
        raise ValueError("Invalid input shape")


def select_frames(frames, sharpness, threshold=None, metric=SHARPNESS_METRIC):
    """
    Selects sharp frames based on a threshold.
    :param frames: Input frame stack as np.ndarray.
    :param sharpness: List of sharpness values for frames.
    :param threshold: Threshold between 0 and 1 for selecting the sharpest frames (0: all frames are selected, 1: only the sharpest frame is selected). If not provided, the threshold is calculated automatically.
    :param metric: Sharpness metric used by the calculate_sharpness function. Used for automatic threshold calculation.
    :return: Selected frames as np.ndarray.
    """
    global note
    sharpness_metrics = ['var_of_laplacian', 'loc_var_of_gray', 'tenengrad', 'var_of_tenengrad']

    # If no threshold is given, calculate it automatically
    if threshold is None:
        if metric == 'loc_var_of_gray':
            threshold = min(sharpness) + 0.8 * (max(sharpness) - min(sharpness))
        elif metric == 'var_of_laplacian':
            threshold = min(sharpness) + 0.6 * (max(sharpness) - min(sharpness))
        elif metric == 'tenengrad':
            threshold = min(sharpness) + 0.65 * (max(sharpness) - min(sharpness))
        elif metric == 'var_of_tenengrad':
            threshold = min(sharpness) + 0.6 * (max(sharpness) - min(sharpness))
    elif threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be between 0 and 1")
    else:
        threshold = min(sharpness) + threshold * (max(sharpness) - min(sharpness))

    # Detect ineffective sharpness metric
    # If only one frame is above the threshold, return the sharpest frame
    if sum([1 for var in sharpness if var >= threshold]) <= 1 and threshold == 1:
        note += "Only one frame above threshold found, no registration performed\n"
        return frames[np.argmax(sharpness)]

    # If few sharp frames are detected, switch to a different sharpness metric and try again
    if sum([1 for var in sharpness if var >= threshold]) <= 2 and threshold != 1:
        # Check if this iteration is the last available sharpness metric
        if metric == sharpness_metrics[-1]:
            # If the last sharpness metric is reached, return the sharpest frame
            note += "Few sharp frames detected. May not register\n"
            print("Few sharp frames detected. May not register")
            return frames[np.where(sharpness >= threshold)]
        else:
            # If this iteration is not the last available sharpness metric, switch to the next one
            note += "Few sharp frames detected, switching to another sharpness metric...\n"
            print("Few sharp frames detected, switching to another sharpness metric")
            new_metric = sharpness_metrics[(sharpness_metrics.index(metric) + 1) % len(sharpness_metrics)]
            # Get a new list of sharpness values
            new_sharpness = calculate_sharpness(frames, metric=new_metric)
            # Recursively call the function with the new sharpness values
            return select_frames(frames, new_sharpness, metric=new_metric)

    # Make a list of frames above the threshold and return it
    selected_frames = frames[np.where(sharpness >= threshold)]
    return selected_frames


def show_frame(image, sharpness=None, frame_number=None, custom_note=None):
    """
    Displays a frame with a title overlay. Optionally saves the frame as a .png file.
    :param image: Input frame as np.ndarray.
    :param sharpness: Sharpness value to be printed in the title. If None, it will be calculated.
    :param frame_number: Frame index to be printed in the title.
    :param custom_note: Custom note to be displayed in the title instead of the default note.
    """
    global note
    # Convert RGB input image to grayscale
    if image.ndim == 3:
        image = rgb2gray(image)

    height, width = image.shape

    if sharpness is None:
        try:
            sharpness = calculate_sharpness(image, blur=False)
        except:
            sharpness = 0

    # Create a figure matching the original image size (1:1)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)

    # Adjust subplot to remove borders
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Show the image without any interpolation (1:1 scale)
    plt.imshow(image, cmap='gray', interpolation='none')

    # Overlay the title on top of the image
    if frame_number is None:
        plt.text(width / 2, 10, f"{note if custom_note is None else custom_note}\nsharpness={sharpness:.2f}", color='white', fontsize=12, ha='center', va='top', backgroundcolor='black')
    else:
        plt.text(width / 2, 10, f"{note if custom_note is None else custom_note}\nsharpness={sharpness:.2f}, i={frame_number}", color='white', fontsize=12, ha='center', va='top', backgroundcolor='black')
    # Hide axis
    plt.axis('off')
    plt.show()


def save_frame(image, path):
    """
    Saves a frame as a .png file.
    :param image: Input frame as np.ndarray.
    :param path: Name of the file to be saved (full path).
    """
    cv2.imwrite(path, image)
    print(f"Saved image as {path}")


def crop_to_intersection(frames):
    """
    Crops a stack of frames to their intersection.
    :param frames: np.ndarray of registered images.
    :return: Cropped images as np.ndarray.
    """
    mask = np.ones_like(frames[0], dtype=bool)

    for frame in frames:
        mask &= (frame > 0)

    # Find the bounding box of the remaining mask
    y_indices, x_indices = np.where(mask)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    cropped_images = [frame[y_min:y_max + 1, x_min:x_max + 1] for frame in frames]
    return cropped_images


def register(selected_frames, sharpness, reference='best', crop=True):
    """
    Registers the sharpest frames and returns a stack of registered and optionally cropped frames.
    :param reference: Either 'previous' or 'best'. If 'previous', each frame is registered to the previous frame in the stack. If 'best', each frame is registered to the sharpest frame in the stack.
    :param selected_frames: Selected sharp frames as np.ndarray.
    :param sharpness: List of sharpness values for frames.
    :param crop: Controls whether to crop the registered frames to their intersection.
    :return: If cumulate is False, returns the registered frame stack as np.ndarray. If cumulate is True, returns the cumulated image (np.ndarray) alongside a note (string).
    """
    global note
    if selected_frames.ndim == 2:
        # If only one frame is given, return it
        note += "Only one frame given, nothing to register\n"
        print("Only one frame given, nothing to register")
        return selected_frames
    else:
        if reference == 'previous':
            # Register to previous frame in the stack
            # Find the sharpest frames and add them to a list
            # Perform registration
            sr = StackReg(StackReg.RIGID_BODY)
            out_rigid_stack = sr.register_transform_stack(selected_frames, reference='previous')
            out_rigid_stack = np.clip(out_rigid_stack, 0, 255).astype(np.uint8)  # Ensure pixel values are within [0, 255]
        elif reference == 'best':
            # Register to the sharpest frame
            # Find the sharpest frame and move it to the first position in the stack
            best_frame_index = np.argmax(sharpness)  # Find the sharpest frame
            # Move the sharpest frame to the first position in the stack
            selected_frames = np.roll(selected_frames, -best_frame_index, axis=0)

            # Perform registration
            sr = StackReg(StackReg.RIGID_BODY)
            out_rigid_stack = sr.register_transform_stack(selected_frames, reference='first')
            out_rigid_stack = np.clip(out_rigid_stack, 0, 255).astype(np.uint8)  # Ensure pixel values are within [0, 255]
        else:
            raise ValueError("Invalid reference")

        note += "Output image given by: "
        if crop:
            # Crop images to their intersection
            out_rigid_stack = crop_to_intersection(out_rigid_stack)
            note += "cropped "

        return out_rigid_stack


def cumulate(frames):
    """
    Averages a stack of frames to reduce noise.
    :param frames: Input frame stack as np.ndarray.
    :return: Averaged frame as np.ndarray.
    """
    global note
    # If only one frame is given, return it
    if np.array(frames).ndim == 2:
        return frames
    note += f"mean of {len(frames)} frames\n"
    return np.mean(frames, axis=0)


def denoise(image, sigma=0.7):
    """
    Denoises an image using bilateral filtering.
    :param sigma: Standard deviation for range distance. A larger value results in averaging of pixels with larger spatial differences.
    :param image: Input image as np.ndarray.
    :return: Denoised image as np.ndarray.
    """
    global note
    note += f"Denoised with sigma={sigma}\n"
    return skimage.restoration.denoise_bilateral(image, sigma_spatial=sigma)


def normalize(image):
    """
    Normalizes an image to the range [0, 255].
    :param image: Input image as np.ndarray.
    :return: Normalized image as np.ndarray.
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def assess_quality(image, path, generate_report=True):
    """
    Calculates the BRISQUE index for an image and optionally compares it to a reference image.
    :param generate_report: Parameter to control whether a report text file is generated.
    :param image: Processed image as np.ndarray.
    :param path: Path to the reference image.
    :return:
    """
    global note
    reference = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Convert to grayscale if necessary
    if image.ndim == 3:
        image = rgb2gray(image)

    # Ensure all pixel values are in 0-255 range
    if np.min(image) < 0 or np.max(image) > 255:
        image = normalize(image)

    # Calculate SNR
    snr_image = np.mean(image) / np.std(image)
    # Convert image to 4D tensor
    image = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(image), 0), 0)
    # Calculate BRISQUE index
    brisque_image = piq.brisque(image, data_range=255).item()

    if reference is not None:
        # Convert to grayscale if necessary
        if reference.ndim == 3:
            reference = rgb2gray(reference)

        # Ensure all pixel values are in 0-255 range
        reference = normalize(reference)

        # Calculate SNR
        snr_reference = np.mean(reference) / np.std(reference)
        # Convert reference to 4D tensor
        reference = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(reference), 0), 0)
        # Calculate BRISQUE index
        brisque_reference = piq.brisque(reference, data_range=255).item()

    else:
        # If no reference is provided, return None
        snr_reference = None
        brisque_reference = None

    # Create a text file containing the quality values
    if generate_report:
        with open(path[:-4] + "_report.txt", "w") as f:
            f.write(f"Processed file: \"{path[:-4] + ".mpg"}\" \n\n"
                    f"{note}\n"
                    f"=== Image statistics ===\n"
                    f"Processed image: \n"
                    f"BRISQUE index = {brisque_image:.2f}\n"
                    f"\n")
            f.write(f"No reference image provided\n" if reference is None else f"Reference image: \n"
                    f"BRISQUE index = {brisque_reference:.2f}\n")
    note = ""
    return [brisque_image, brisque_reference], [snr_image, snr_reference]
