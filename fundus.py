import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.color import rgb2gray
from pystackreg import StackReg
from brisque import BRISQUE
from pypiqe import piqe
import SimpleITK as sitk
import warnings
import bm3d
import re
import os
from typing import Literal
import sys
import types
from torchvision.transforms.functional import rgb_to_grayscale
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
from basicsr.metrics.niqe import calculate_niqe


note = ""  # Initialize a note to be displayed in the title of the image or printed in the report
video_path = None
reference_image = None
is_central_region = True  # Flag to indicate if the video is of the central region (macula)


def load_video(video_file_path: str) -> np.ndarray:
    """
    Opens a video file and returns a np.ndarray of non-repeated frames. Only every third frame of the video file is saved.
    Checks if the video is of the central region (macula) by looking for the _X<number><letter>_Y<number>_Z<number> pattern in the filename.
    :param video_file_path: Full video filepath.
    :return: Reduced frame stack as np.ndarray.
    """
    global note, video_path, is_central_region, reference_image
    reference_image = None  # Reset the reference image for each video
    video_path = video_file_path  # Reset the video path for each video
    note = ""  # Reset the note for each video

    # Check if the video is of the central region
    filename = os.path.basename(video_file_path)
    # Regex to match _X<number><letter>_Y<number>_Z
    match = re.search(r'_X([-\d.]+)\D_Y([-\d.]+)_Z', filename)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
    else:
        raise warnings.warn("Filename does not match expected format. Unable to determine the region. Assuming video is of the central region.")

    if x > 4 or y > 4:
        is_central_region = False
        note += "Video is not of the central region\n"
    else:
        is_central_region = True
        note += "Video is of the central region\n"

    # Open the video file
    video = cv2.VideoCapture(video_file_path)

    if not video.isOpened():
        print("Error opening video file")
        exit()

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


def load_reference_image(reference_path: str) -> np.ndarray:
    """
    Loads a reference image.
    Always load a video first, then load the reference image. Otherwise, the reference image will be discarded.
    :param reference_path: Path to the reference image.
    :return: Reference image as np.ndarray.
    """
    global reference_image

    reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    reference_image = normalize(reference_image)

    return reference_image


def calculate_sharpness2(
        frames: np.ndarray,
        metric: Literal['loc_var_of_gray', 'var_of_laplacian', 'tenengrad', 'var_of_tenengrad'] = 'var_of_laplacian',
        blur=True,
        update_note=True
) -> float | list[float]:
    """
    Calculates the sharpness of a frame or frame stack using a specified metric.
    Helper function used by the calculate_sharpness function but can be also used independently to override the default combination of metrics.
    :param update_note: Controls whether to update the note variable. Mainly for debug use.
    :param blur: If True, the input frames will first be blurred using a Gaussian filter to reduce the noise level.
    :param frames: Input frame or frame stack as np.ndarray.
    :param metric: Sharpness metric to be used. Can be either 'loc_var_of_gray', 'var_of_laplacian', 'tenengrad', or 'var_of_tenengrad'. If no metric is selected, the default metric is 'var_of_laplacian'.
    :return: Estimated sharpness value if input is a single frame or list of estimated sharpness values if input is a frame stack.
    """
    assert metric in ['var_of_laplacian', 'loc_var_of_gray', 'tenengrad', 'var_of_tenengrad'], "Invalid metric parameter. Supported values are 'loc_var_of_gray', 'var_of_laplacian', 'tenengrad', 'var_of_tenengrad'"
    global note

    if len(frames.shape) == 2:
        # If a single frame is given
        if frames.ndim == 3:
            frames = rgb2gray(frames)
        if blur:
            frames = cv2.GaussianBlur(frames, (7, 7), 0)

        if metric == 'loc_var_of_gray':
            # Determine the local gray level variance in a window https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=903548
            window_size = 36  # Balance between speed and accuracy
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
            raise ValueError("Invalid metric parameter. Supported values are 'loc_var_of_gray', 'var_of_laplacian', 'tenengrad', 'var_of_tenengrad'")

    elif len(frames.shape) == 3:
        # If a stack of frames is given
        if update_note:
            note += f"Sharpness metric: '{metric}'\n"
        # Calculate sharpness for each frame in the stack by calling the function recursively
        sharpness = [calculate_sharpness2(f, metric) for f in frames]
        return sharpness

    else:
        raise ValueError("Invalid input shape")


def calculate_sharpness(frames: np.ndarray) -> list[float]:
    """
    Calculates the sharpness of a frame stack using multiple metrics and combines them into a single value.
    :param frames: Input frame stack as np.ndarray.
    :return: List of estimated sharpness values.
    """
    # Get the sharpness values for each metric and normalize them to 0-1 range
    gray = calculate_sharpness2(frames, metric='loc_var_of_gray', blur=True, update_note=False)
    gray = gray / max(gray)
    laplacian = calculate_sharpness2(frames, metric='var_of_laplacian', blur=True, update_note=False)
    laplacian = laplacian / max(laplacian)
    tenengrad = calculate_sharpness2(frames, metric='var_of_tenengrad', blur=True, update_note=False)
    tenengrad = tenengrad / max(tenengrad)

    # Combine the sharpness values into a single metric
    avg = (gray + laplacian + tenengrad) / 3

    return avg


def select_frames2(frames: np.ndarray, sharpness: list, threshold=0.7) -> np.ndarray:
    """
    Selects sharp frames based on a hard threshold.
    :param frames: Input frame stack as np.ndarray.
    :param sharpness: List of sharpness values for frames.
    :param threshold: Threshold between 0 and 1 for selecting the sharpest frames (0: all frames are selected, 1: only the sharpest frame is selected).
        Higher values indicate that fewer frames will be selected for registration and averaging, resulting in a noisier but sharper image.
        Lower values indicate that more frames will be selected for registration and averaging, resulting in a smoother but less sharp image.
        Threshold value also affects the speed of the registration process.
    :return: List of selected frames as np.ndarray.
    """
    global note
    selected_frames = frames[np.where(sharpness >= threshold)]
    note += "Sharpness threshold: " + str(threshold) + "\n"

    return selected_frames


def select_frames(frames: np.ndarray, sharpness: list) -> np.ndarray:
    """
    Selects sharp frames adaptively based on sharpness distribution.
    :param frames: Input frame stack as np.ndarray.
    :param sharpness: List or np.ndarray of sharpness values for frames.
    :return: List of selected frames as np.ndarray.
    """
    global note

    sharpness = np.array(sharpness)
    num_frames = len(sharpness)
    min_select = max(1, int(num_frames * 0.05))  # At least 5 % of frames (2 frames min out of 40)
    max_select = max(1, int(num_frames * 0.7))   # Up to 70 % of frames (28 frames max out of 40)

    # Measure sharpness variability
    std_sharp = np.std(sharpness)
    mean_sharp = np.mean(sharpness)
    cv = std_sharp / (mean_sharp + 1e-8)  # Coefficient of variation; + 1e-8 to avoid division by zero
    cv_max = 0.5  # Controls the sensitivity to sharpness variability
    cv = min(cv, cv_max)
    frac = 1 - (cv / cv_max)  # 1 when cv=0, 0 when cv=cv_max
    frac = frac ** 2  # Make the drop-off sharper by squaring frac

    n_select = int(min_select + frac * (max_select - min_select))
    n_select = max(min_select, min(n_select, max_select))

    # Select the n_select sharpest frames
    sorted_indices = np.argsort(sharpness)[::-1]
    selected_indices = sorted_indices[:n_select]
    selected_frames = frames[selected_indices]

    note += f"Selected {n_select} sharpest frames adaptively\n"

    return selected_frames


def show_frame(image: np.ndarray, sharpness=None, frame_number=None, custom_note=None):
    """
    Displays a frame with a title overlay.
    :param image: Input frame as np.ndarray.
    :param sharpness: Sharpness value to be printed in the title. If None, it will be calculated.
    :param frame_number: Frame index to be printed in the title.
    :param custom_note: Custom note to be displayed in the title instead of the default note.
    """
    global note

    # Convert RGB input image to grayscale
    if image.ndim == 3:
        image = rgb2gray(image)

    # Normalize the image to the range [0, 255] and convert to uint8
    image = normalize(image)

    height, width = image.shape

    if sharpness is None:
        try:
            sharpness = calculate_sharpness2(image, blur=False)
        except:
            sharpness = 0

    # Create a figure matching the original image size (1:1)
    plt.figure(figsize=(width / 100, height / 100), dpi=100)

    # Adjust subplot to remove borders
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Show the image without any interpolation (1:1 scale)
    plt.imshow(image, cmap='gray', interpolation='none')

    # Overlay the title on top of the image
    if frame_number is None:
        plt.text(width / 2, 10, f"{note if custom_note is None else custom_note}sharpness={sharpness:.2f}", color='white', fontsize=12, ha='center', va='top', backgroundcolor='black')
    else:
        plt.text(width / 2, 10, f"{note if custom_note is None else custom_note}sharpness={sharpness:.2f}, i={frame_number}", color='white', fontsize=12, ha='center', va='top', backgroundcolor='black')
    # Hide axis
    plt.axis('off')
    plt.show()


def save_frame(image: np.ndarray, path: str):
    """
    Saves a frame as a .png file.
    :param image: Input frame as np.ndarray.
    :param path: Name of the file to be saved (full path).
    """
    cv2.imwrite(path, image)
    print(f"Saved image as {path}")


def crop_to_intersection(frames: np.ndarray, update_note=True) -> np.ndarray:
    """
    Crops a stack of frames to their intersection.
    :param update_note: Controls whether to update the note variable. Mainly for debug use.
    :param frames: np.ndarray of registered images.
    :return: Cropped images as np.ndarray.
    """
    global note

    mask = np.ones_like(frames[0], dtype=bool)

    for frame in frames:
        mask &= (frame > 0)

    # Find the bounding box of the remaining mask
    y_indices, x_indices = np.where(mask)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    cropped_images = [frame[y_min:y_max + 1, x_min:x_max + 1] for frame in frames]

    # Update the global note if update_note is True
    if update_note:
        note += "cropped "

    return np.array(cropped_images)


def extend_to_union(frames: np.ndarray, update_note=True) -> np.ndarray:
    """
    Expands all frames to cover the union of non-zero regions across the stack.
    Replaces zero values in union areas with average non-zero values from other frames.
    :param update_note: Controls whether to update the note variable. Mainly for debug use.
    :param frames: Input stack of registered frames (np.ndarray)
    :return: Frame stack with consistent non-zero regions (np.ndarray)
    """
    global note

    # Create union mask where any frame has non-zero values
    union_mask = np.any(frames != 0, axis=0)

    extended_frames = []
    for i, frame in enumerate(frames):
        # Create working copy and identify fill targets
        modified_frame = frame.copy()
        fill_mask = np.logical_and(union_mask, modified_frame == 0)

        # Calculate average non-zero values from other frames
        other_frames = np.delete(frames, i, axis=0)
        non_zero_sum = np.sum(np.where(other_frames != 0, other_frames, 0), axis=0)
        non_zero_count = np.maximum(np.sum(other_frames != 0, axis=0), 1)  # Prevent division by zero

        # Apply averaged values to zero regions
        modified_frame[fill_mask] = (non_zero_sum / non_zero_count)[fill_mask]
        extended_frames.append(modified_frame)

    # Update the global note if update_note is True
    if update_note:
        note += "extended to union "

    return np.array(extended_frames)


def register(
        selected_frames: np.ndarray,
        sharpness: list,
        reference: Literal['best', 'previous', 'mean'] = None,
        pad: Literal['same', 'crop', 'zeros'] = 'same',
        update_note=True
) -> np.ndarray:
    """
    Registers the sharpest frames using the pyStackReg library and returns a stack of registered frames.
    :param update_note: Controls whether to update the note variable. Mainly for debug use.
    :param reference: Either 'previous', 'best' or 'mean'.
        If 'previous', each frame is registered to its previous (already registered) frame in the stack.
        If 'best', each frame is registered to the sharpest frame in the stack.
        If 'mean', each frame is registered to a mean frame of reference.
    :param selected_frames: Stack of frames to be registered as np.ndarray.
    :param sharpness: List of sharpness values for frames.
    :param pad: Either 'same', 'intersection', or 'zeros'.
        If 'same', all frames are expanded to cover the union of non-zero regions across the stack and the output shape is the same as that of the input. Regions near the borders may be of lower quality as a result.
        If 'intersection', all frames are cropped to their intersection, which results in an output with a smaller shape.
        If 'zeros', all frames are padded with zeros.
    :return: Stack of registered frames as np.ndarray.
    """
    assert pad in ['same', 'crop', 'zeros'], "Invalid pad parameter. Supported values are 'same', 'intersection', or 'zeros'."
    assert reference in ['best', 'previous', 'mean'], "Invalid reference parameter. Supported values are 'best', 'previous' and 'mean'."
    global note, reference_image
    local_note = f"Registration method: pyStackReg\nReference: '{reference}'\n"

    if selected_frames.ndim == 2 or selected_frames.shape[0] == 1:
        # If only one frame is given, return it
        local_note += "Only one frame given, nothing to register\n"
        warnings.warn("Only a single frame was given to the registration function. No registration was performed and the frame was returned.")
        if update_note:
            note += local_note
        return selected_frames

    else:
        if reference is None:
            reference = 'best' if is_central_region else 'mean'  # Default to 'best' for central region, 'mean' for non-central region

        if reference == 'previous':
            # Register to previous frame in the stack
            # Perform registration
            sr = StackReg(StackReg.RIGID_BODY)
            out_rigid_stack = sr.register_transform_stack(selected_frames, reference='previous')
            out_rigid_stack = np.clip(out_rigid_stack, 0, 255).astype(np.uint8)  # Ensure pixel values are within [0, 255]

        elif reference == 'mean':
            # Register to the mean of the stack
            # Perform registration
            sr = StackReg(StackReg.RIGID_BODY)
            out_rigid_stack = sr.register_transform_stack(selected_frames, reference='mean')
            out_rigid_stack = np.clip(out_rigid_stack, 0, 255).astype(np.uint8)

        elif reference == 'best':
            # Register to the sharpest frame
            # Find the sharpest frame and move it to the first position in the stack
            best_frame_index = np.argmax(sharpness)  # Find the sharpest frame
            # Move the sharpest frame to the first position in the stack
            selected_frames = np.roll(selected_frames, -best_frame_index, axis=0)

            # Perform registration to the previous frame in the stack
            sr = StackReg(StackReg.RIGID_BODY)
            out_rigid_stack = sr.register_transform_stack(selected_frames, reference='first')
            out_rigid_stack = np.clip(out_rigid_stack, 0, 255).astype(np.uint8)  # Ensure pixel values are within [0, 255]

        else:
            raise ValueError("Invalid reference parameter. Supported values are 'best', 'previous' and 'mean'.")

        local_note += "Output image given by: "
        if pad == 'crop':
            # Crop images to their intersection
            out_rigid_stack = crop_to_intersection(out_rigid_stack, update_note=False)
            local_note += "cropped "
        elif pad == 'same':
            # Expand all frames to cover the union of non-zero regions across the stack
            out_rigid_stack = extend_to_union(out_rigid_stack, update_note=False)
            local_note += "extended to union "
        elif pad == 'zeros':
            pass
        else:
            raise ValueError("Invalid pad parameter. Supported values are 'same', 'intersection', or 'zeros'.")

        # Update the global note if update_note is True
        if update_note:
            note += local_note

        return out_rigid_stack


def register2(
        selected_frames: np.ndarray,
        sharpness: list,
        reference: Literal['best', 'previous'] = 'best',
        pad: Literal['same', 'crop', 'zeros'] = 'same'
) -> np.ndarray:
    """
    Registers the sharpest frames using the SimpleElastix library and returns a stack of registered frames.
    :param selected_frames: Stack of frames to be registered as np.ndarray.
    :param sharpness: List of sharpness values for frames.
    :param reference: Either 'previous' or 'best'. If 'previous', each frame is registered to its previous (already registered) frame in the stack. If 'best', each frame is registered to the sharpest frame in the stack.
    :param pad: Either 'same', 'intersection', or 'zeros'.
        If 'same', all frames are expanded to cover the union of non-zero regions across the stack and the output shape is the same as that of the input. Regions near the borders may be of lower quality as a result.
        If 'intersection', all frames are cropped to their intersection, which results in an output with a smaller shape.
        If 'zeros', all frames are padded with zeros.
    :return: Stack of registered frames as np.ndarray.
    """
    assert reference in ['best', 'previous'], "Invalid reference parameter. Supported values are 'best' and 'previous'."
    assert pad in ['same', 'crop', 'zeros'], "Invalid pad parameter. Supported values are 'same', 'intersection', or 'zeros'."
    global note, reference_image
    local_note = f"Registration method: SimpleElastix\nReference: '{reference}'\n"

    if selected_frames.ndim == 2:
        # If only one frame is given, return it
        note += "Only one frame given, nothing to register\n"
        warnings.warn("Only a single frame was given to the registration function. No registration was performed and the frame was returned.")
        return selected_frames
    else:
        # Set parameters for registration
        param_map = sitk.GetDefaultParameterMap("rigid")
        param_map["NumberOfResolutions"] = ["5"]  # Set number of resolutions for pyramidal registration
        # Uncomment to turn off pyramidal registration
        # param_map["NumberOfResolutions"] = ["1"]
        param_map["ShrinkFactorsPerLevel"] = ["2"]
        param_map["SmoothingSigmasPerLevel"] = ["5"]
        param_map["Metric"] = ["AdvancedNormalizedCorrelation"]
        param_map["MaximumNumberOfIterations"] = ["200"]
        # sitk.PrintParameterMap(param_map)  # Uncomment to print the parameter map

        if reference == 'previous':
            out_rigid_stack = [selected_frames[0]]
            # Register to previous frame in the stack
            for i, moving_frame in enumerate(selected_frames[1:]):
                # print(i + 1)
                # Convert reference image to SimpleITK format
                reference_frame = sitk.GetImageFromArray(out_rigid_stack[i].astype(np.float32))
                # Convert moving image to SimpleITK format
                moving_image = sitk.GetImageFromArray(moving_frame.astype(np.float32))

                # Perform registration using SimpleElastix
                elastix_image_filter = sitk.ElastixImageFilter()
                elastix_image_filter.SetFixedImage(reference_frame)
                elastix_image_filter.SetMovingImage(moving_image)
                elastix_image_filter.SetParameterMap(param_map)
                elastix_image_filter.LogToConsoleOff()
                elastix_image_filter.Execute()

                # Get the registered image and convert it back to NumPy format
                registered_image = sitk.GetArrayFromImage(elastix_image_filter.GetResultImage())
                out_rigid_stack.append(registered_image)

            # Update note
            local_note += "Output image given by: "

        elif reference == 'best':
            # Register to the sharpest frame
            best_frame_index = np.argmax(sharpness)  # Find the sharpest frame

            # Move the sharpest frame to the first position in the stack
            selected_frames = np.roll(selected_frames, -best_frame_index, axis=0)

            reference_frame = selected_frames[0]

            # Initialize output stack with the reference image
            out_rigid_stack = [reference_frame]

            # Convert reference image to SimpleITK format
            reference_frame = sitk.GetImageFromArray(reference_frame.astype(np.float32))

            i = 0
            for moving_frame in selected_frames[1:]:
                i += 1
                # print(i)
                # Convert moving image to SimpleITK format
                moving_image = sitk.GetImageFromArray(moving_frame.astype(np.float32))

                # Perform registration using SimpleElastix
                elastix_image_filter = sitk.ElastixImageFilter()
                elastix_image_filter.SetFixedImage(reference_frame)
                elastix_image_filter.SetMovingImage(moving_image)
                elastix_image_filter.SetParameterMap(param_map)
                elastix_image_filter.LogToConsoleOff()
                elastix_image_filter.Execute()

                # Get the registered image and convert it back to NumPy format
                registered_image = sitk.GetArrayFromImage(elastix_image_filter.GetResultImage())
                out_rigid_stack.append(registered_image)

            # Update note
            local_note += "Output image given by: "

        else:
            raise ValueError("Invalid reference parameter. Supported values are 'best' and 'previous'.")

        if pad == 'crop':
            # Crop images to their intersection
            out_rigid_stack = crop_to_intersection(out_rigid_stack, update_note=False)
            local_note += "cropped "
        elif pad == 'same':
            # Expand all frames to cover the union of non-zero regions across the stack
            out_rigid_stack = extend_to_union(out_rigid_stack, update_note=False)
            local_note += "extended to union "
        elif pad == 'zeros':
            pass
        else:
            raise ValueError("Invalid pad parameter. Supported values are 'same', 'intersection', or 'zeros'.")

        # This is the output of the registration
        out_rigid_stack = np.clip(out_rigid_stack, 0, 255).astype(np.uint8)  # Ensure pixel values are within [0, 255]

        # Update note
        note += local_note

        return out_rigid_stack


def cumulate(frames: np.ndarray, method: Literal['mean', 'median'] = 'mean', update_note=True) -> np.ndarray:
    """
    Fuses a stack of frames to reduce noise.
    :param method: Method for fusing frames. Can be either 'mean' or 'median'.
    :param update_note: Controls whether to update the note variable. Mainly for debug use.
    :param frames: Input list of frames as np.ndarrays.
    :return: Resulting image as np.ndarray.
    """
    global note
    assert method in ['mean', 'median'], "Invalid method parameter. Supported values are 'mean' or 'median'."

    # If only one frame is given, return it
    if np.array(frames).ndim == 2:
        return frames

    if method == 'mean':
        cum = np.mean(frames, axis=0).astype(np.uint8)
        local_note = f"mean of {len(frames)} frames\n"

    elif method == 'median':
        # Takes the median value of each pixel across all frames
        cum = np.median(frames, axis=0).astype(np.uint8)
        local_note = f"median of {len(frames)} frames\n"

    else:
        raise ValueError("Invalid method parameter.")

    # Update the global note if update_note is True
    if update_note:
        note += local_note

    # Normalize the output image to the range [0, 255]
    cum = normalize(cum)

    return cum


def denoise(image: np.ndarray, method: Literal['bm3d', 'tv', 'hmgf'] = None, weight: int | float = None) -> np.ndarray[np.uint8]:
    """
    Denoises an image using BM3D, TV, bilateral filtering, or hybrid median Gaussian filtering.
    :param method: Method for denoising. Can be one of the following:
        'bm3d': Block-Matching and 3D filtering, recommended for images with lots of small details (like cells).
        'tv': Total Variation denoising.
        'hmgf': Hybrid median Gaussian filter. Uses median filtration plus additional Gaussian smoothing in homogeneous regions.
        If None, the method is chosen based on the region of the image:
            If the image captures the central region of the eye, 'bm3d' is used. Otherwise, 'hmgf' is used.
    :param weight: Universal weight parameter affecting the amount of denoising done. Higher values result in more aggressive denoising.
        Suggested values are 6 for BM3D, 0.04 for TV, and 2 for HMGF.
    :param image: Input image as np.ndarray.
    :return: Denoised image as np.ndarray.
    """
    assert method in ['bm3d', 'tv', 'hmgf', None], "Invalid method parameter. Supported values are 'bm3d', 'tv', 'hmgf' and None."
    global note, is_central_region

    if method is None:
        method = 'bm3d' if is_central_region else 'hmgf'

    if method == 'bm3d':
        weight = 6 if weight is None else weight
        # Ensure float32 input in [0,1] range
        image_norm = image.astype(np.float32) / 255
        denoised = bm3d.bm3d(image_norm, sigma_psd=weight / 255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        note += f"Denoised with BM3D (sigma={weight})\n"

    elif method == 'tv':
        weight = 0.04 if weight is None else weight
        denoised = skimage.restoration.denoise_tv_chambolle(image, weight=weight)
        note += f"Denoised with TV (weight={weight})\n"

    elif method == 'hmgf':
        weight = 2 if weight is None else weight
        # Normalize image to [0,1] range
        img_float = image.astype(np.float32) / 255.0

        # Apply adaptive median filter for impulse noise removal
        # Scale the median filter size based on the weight parameter
        median_size = min(7, max(3, 3 + 2 * int(weight / 3)))

        # First apply basic median filter
        median_filtered = skimage.filters.median(
            img_float,
            footprint=skimage.morphology.disk(median_size // 2)
        )

        # Apply edge-preserving Gaussian filtering
        # Calculate edge response to adapt filter parameters
        edges = skimage.filters.sobel(median_filtered)
        edges = edges / edges.max() if edges.max() > 0 else edges

        # Apply Gaussian filter with different sigma based on weight
        sigma = min(2.0, max(0.5, 0.5 + weight * 0.15))
        gaussian_filtered = skimage.filters.gaussian(median_filtered, sigma=sigma)

        # Blend results: use median filter result near vessel edges and Gaussian filter result in smooth regions
        blend_factor = 5  # Control the amount of median vs Gaussian filtering (higher = more median)
        blend_ratio = np.clip(edges * blend_factor, 0, 1)  # Scale up edge response for blending
        denoised = blend_ratio * median_filtered + (1 - blend_ratio) * gaussian_filtered
        note += f"Denoised with HMGF (weight={weight})\n"

    else:
        raise ValueError("Invalid method parameter. Supported values are 'bm3d', 'tv' and 'hmgf'.")

    # Clamp values to [0,1] range before conversion
    denoised = np.clip(denoised, 0, 1)

    return (denoised * 255).astype(np.uint8)


def normalize(image: np.ndarray) -> np.ndarray[np.uint8]:
    """
    Normalizes an image to the range [0, 255].
    :param image: Input image as np.ndarray.
    :return: Normalized image as uint8 np.ndarray.
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def resize(image: np.ndarray, target_dimensions: tuple[int, int] | np.ndarray) -> np.ndarray:
    """
    Resizes an image using bicubic interpolation. Either a tuple or an image of target shape can be provided as the target.
    If target_dimensions is a tuple, it is treated as the target size.
    If target_dimensions is an image, its shape is used as the target size.
    :param image: Image to be resized as np.ndarray.
    :param target_dimensions: Dimensions of the target image as tuple or image to be used as a target as np.ndarray.
    :return: Resized image1 as np.ndarray.
    """
    global note

    if isinstance(target_dimensions, tuple):
        image_resized = cv2.resize(image, target_dimensions, interpolation=cv2.INTER_CUBIC)
        note += f"Resized to {target_dimensions}\n"

    elif isinstance(target_dimensions, np.ndarray):
        image_resized = cv2.resize(image, (target_dimensions.shape[1], target_dimensions.shape[0]), interpolation=cv2.INTER_CUBIC)
        note += f"Resized to ({target_dimensions.shape[1]}, {target_dimensions.shape[0]})\n"

    else:
        raise ValueError("Parameter target_dimensions must be a tuple or an image (np.ndarray).")

    return image_resized


def assess_quality(image_processed: np.ndarray, report_path: str = None):
    """
    Assesses the quality of a processed image and compares it with a reference image if provided.
    Calculates BRISQUE index, PIQE score and sharpness (according to variance of Laplacian) for an image and compares its scores with those of the reference image if it was loaded previously using load_reference_image().
    :param image_processed: Processed image as np.ndarray.
    :param report_path: Path to save the report text file. If no path is provided, no report will be generated.
    :return: Sharpness, BRISQUE index and PIQE score for the processed image. If a reference image was provided, it will return both sets of scores as tuples (processed_image, reference_image).
    """
    global note, reference_image, video_path
    reference = reference_image

    # Convert to grayscale if necessary
    if image_processed.ndim == 3:
        image_processed = rgb2gray(image_processed)

    # Ensure all pixel values are in 0-255 range
    if np.min(image_processed) < 0 or np.max(image_processed) > 255:
        image_processed = normalize(image_processed)

    # Calculate sharpness for processed image
    sharpness_image = calculate_sharpness2(image_processed, blur=False)
    sharpness_log_image = calculate_sharpness2(image_processed, blur=True)

    # Calculate BRISQUE index for processed image
    obj = BRISQUE(url=False)
    brisque_image = obj.score(np.stack((image_processed,)*3, axis=-1))  # Convert image to RGB first

    # Calculate PIQE score for processed image
    piqe_image, _, _, _ = piqe(image_processed)

    # Calculate NIQE score for processed image
    niqe_image = calculate_niqe(
        image_processed.astype(np.float32),
        crop_border=0,
        input_order='HW',
        convert_to='y'
    )

    if reference is not None:
        # Calculate sharpness of the reference image
        sharpness_reference = calculate_sharpness2(reference, blur=False)
        sharpness_log_reference = calculate_sharpness2(reference, blur=True)

        # Calculate BRISQUE index of the reference image
        brisque_reference = obj.score(np.stack((reference,)*3, axis=-1))

        # Calculate PIQE score of the reference image
        piqe_reference, _, _, _ = piqe(reference)

        # Calculate NIQE score of the reference image
        # noinspection PyUnresolvedReferences
        niqe_reference = calculate_niqe(
            reference.astype(np.float32),
            crop_border=0,
            input_order='HW',
            convert_to='y'
        )

    else:
        # If no reference is provided, return None
        brisque_reference = None
        piqe_reference = None
        sharpness_reference = None
        sharpness_log_reference = None

    # Create a text file containing the quality values
    if report_path is not None:
        with open(report_path, "w") as f:
            f.write(f"Processed file: \"{video_path}\"\n\n"
                    f"{note}\n"
                    f"=== Image statistics ===\n"
                    f"Processed image:\n"
                    f"Sharpness (var_of_laplacian) = {sharpness_image:.2f}\n"
                    f"Sharpness (var_of_LoG) = {sharpness_log_image:.2f}\n"
                    f"BRISQUE index = {brisque_image:.2f}\n"
                    f"PIQE index = {piqe_image:.2f}\n"
                    f"NIQE index = {niqe_image:.2f}\n"
                    f"\n")
            f.write(f"No reference image provided\n" if reference is None else
                    f"Reference image:\n"
                    f"Sharpness (var_of_laplacian) = {sharpness_reference:.2f}\n"
                    f"Sharpness (var_of_LoG) = {sharpness_log_reference:.2f}\n"
                    f"BRISQUE index = {brisque_reference:.2f}\n"
                    f"PIQE index = {piqe_reference:.2f}\n"
                    f"NIQE index = {niqe_reference:.2f}\n"
                    f"\n")

    if reference is not None:
        return (sharpness_log_image, sharpness_log_reference), (brisque_image, brisque_reference), (piqe_image, piqe_reference), (niqe_image, niqe_reference)
    else:
        return sharpness_log_image, brisque_image, piqe_image, niqe_image
