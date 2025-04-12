import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.restoration
import skimage.registration
import skimage.transform
from skimage.color import rgb2gray
import torch
from pystackreg import StackReg
from piq import brisque
from brisque import BRISQUE
from pypiqe import piqe
import SimpleITK as sitk
import warnings

note = ""  # Initialize a note to be displayed in the title of the image or printed in the report
video_path = None
reference_image = None


def load_video(video_file_path):
    """
    Opens a video file and returns a np.ndarray of non-repeated frames. Only every third frame of the video file is saved.
    :param video_file_path: Video file path.
    :return: Reduced frame stack as np.ndarray.
    """
    global note, video_path
    video_path = video_file_path
    note = ""  # Reset the note for each video

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


def load_reference_image(reference_path):
    """
    Loads a reference image.
    :param reference_path: Path to the reference image.
    :return: Reference image as np.ndarray.
    """
    global reference_image
    reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    reference_image = match_dimensions(reference_image)  # Resize reference image to match the dimensions of the input frames
    reference_image = normalize(reference_image)
    return reference_image


def calculate_sharpness2(frames, metric='var_of_laplacian', blur=True, update_note=True):
    """
    Calculates the sharpness of a frame or frame stack using a specified metric.
    Used by the calculate_sharpness function but can be also used independently to override the default combination of metrics.
    :param update_note: Controls whether to update the note variable. Mainly for debug use.
    :param blur: If True, the input frames will be blurred using a Gaussian filter to reduce the noise level.
    :param frames: Input frame as np.ndarray.
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
            window_size = 36
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


def calculate_sharpness(frames):
    """
    Calculates the sharpness of a frame stack using multiple metrics and combines them into a single metric.
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


def select_frames(frames, sharpness, threshold=0.6):
    """
    Selects sharp frames based on a threshold.
    :param frames: Input frame stack as np.ndarray.
    :param sharpness: List of sharpness values for frames.
    :param threshold: Threshold between 0 and 1 for selecting the sharpest frames (0: all frames are selected, 1: only the sharpest frame is selected).
    Higher values mean that fewer frames will be selected for registration and averaging, resulting in a noisier but sharper image.
    Lower values mean that more frames will be selected for registration and averaging, resulting in a smoother but less sharp image.
    This parameter is used to control the trade-off between sharpness and smoothness.
    :return: List of selected frames as np.ndarray.
    """
    selected_frames = frames[np.where(sharpness >= threshold)]
    return selected_frames


def show_frame(image, sharpness=None, frame_number=None, custom_note=None):
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


def save_frame(image, path):
    """
    Saves a frame as a .png file.
    :param image: Input frame as np.ndarray.
    :param path: Name of the file to be saved (full path).
    """
    cv2.imwrite(path, image)
    print(f"Saved image as {path}")


def crop_to_intersection(frames, update_note=True):
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

    return cropped_images


def extend_to_union(frames, update_note=True):
    """
    Expands all frames to cover the union of non-zero regions across the stack.
    Replaces zero values in union areas with average non-zero values from other frames.
    :param update_note: Controls whether to update the note variable. Mainly for debug use.
    :param frames: Input stack of registered frames (np.ndarray)
    :return: Expanded frame stack with consistent non-zero regions (np.ndarray)
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


def register2(selected_frames, sharpness, reference='best', pad='same', update_note=True):
    """
    Registers the sharpest frames using the pyStackReg library and returns a stack of registered and optionally cropped frames.
    Used by the register function but can be also used independently to register using PyStackReg specifically.
    :param update_note: Controls whether to update the note variable. Mainly for debug use.
    :param reference: Either 'previous' or 'best'. If 'previous', each frame is registered to its previous (already registered) frame in the stack. If 'best', each frame is registered to the sharpest frame in the stack.
    :param selected_frames: Stack of frames to be registered as np.ndarray.
    :param sharpness: List of sharpness values for frames.
    :param pad: Either 'same', 'intersection', or 'zeros'. If 'same', all frames are expanded to cover the union of non-zero regions across the stack. If 'intersection', all frames are cropped to their intersection. If 'zeros', all frames are padded with zeros.
    :return: Stack of registered frames as np.ndarray.
    """
    assert pad in ['same', 'crop', 'zeros'], "Invalid pad parameter. Supported values are 'same', 'intersection', or 'zeros'."
    global note
    local_note = ""

    assert reference in ['best', 'previous'], "Invalid reference parameter. Supported values are 'best' and 'previous'."
    local_note += f"Registration method: pyStackReg\nReference: '{reference}'\n"

    if selected_frames.ndim == 2:
        # If only one frame is given, return it
        local_note += "Only one frame given, nothing to register\n"
        warnings.warn("Only a single frame was given to the registration function. No registration was performed and the frame was returned.")
        if update_note:
            note += local_note
        return selected_frames
    else:
        if reference == 'previous':
            # Register to previous frame in the stack
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
            raise ValueError("Invalid reference parameter. Supported values are 'best' and 'previous'.")

        local_note += "Output image given by: "
        # Update the global note if update_note is True
        if update_note:
            note += local_note

        if pad == 'crop':
            # Crop images to their intersection
            out_rigid_stack = crop_to_intersection(out_rigid_stack, update_note=update_note)
        elif pad == 'same':
            # Expand all frames to cover the union of non-zero regions across the stack
            out_rigid_stack = extend_to_union(out_rigid_stack, update_note=update_note)
        elif pad == 'zeros':
            pass
        else:
            raise ValueError("Invalid pad parameter. Supported values are 'same', 'intersection', or 'zeros'.")

        return out_rigid_stack


def register(selected_frames, sharpness, reference='best', pad='same'):
    """
    Registers the sharpest frames using the SimpleElastix library and returns a stack of registered and optionally cropped frames.
    :param selected_frames: Stack of frames to be registered as np.ndarray.
    :param sharpness: List of sharpness values for frames.
    :param reference: Either 'previous' or 'best'. If 'previous', each frame is registered to its previous (already registered) frame in the stack. If 'best', each frame is registered to the sharpest frame in the stack.
    :param pad: Either 'same', 'intersection', or 'zeros'. If 'same', all frames are expanded to cover the union of non-zero regions across the stack. If 'intersection', all frames are cropped to their intersection. If 'zeros', all frames are padded with zeros.
    :return: Stack of registered frames as np.ndarray.
    """
    assert reference in ['best', 'previous'], "Invalid reference parameter. Supported values are 'best' and 'previous'."
    assert pad in ['same', 'crop', 'zeros'], "Invalid pad parameter. Supported values are 'same', 'intersection', or 'zeros'."
    global note, reference_image
    note_addition = f"Registration method: SimpleElastix\nReference: '{reference}'\n"

    if selected_frames.ndim == 2:
        # If only one frame is given, return it
        note += "Only one frame given, nothing to register\n"
        warnings.warn("Only a single frame was given to the registration function. No registration was performed and the frame was returned.")
        return selected_frames
    else:
        # Set parameters for registration
        # TODO: Rotace možná špatně propaguje pro reference='previous'?
        param_map = sitk.GetDefaultParameterMap("rigid")
        param_map["NumberOfResolutions"] = ["4"]  # Set number of resolutions for pyramidal registration
        # Uncomment to turn off pyramidal registration
        # param_map["NumberOfResolutions"] = ["1"]
        # param_map["ShrinkFactorsPerLevel"] = ["1"]
        # param_map["SmoothingSigmasPerLevel"] = ["0"]
        param_map["Metric"] = ["AdvancedNormalizedCorrelation"]
        param_map["MaximumNumberOfIterations"] = ["100"]
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
            note_addition += "Output image given by: "

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
            note_addition += "Output image given by: "

        else:
            raise ValueError("Invalid reference parameter. Supported values are 'best' and 'previous'.")

        if pad == 'crop':
            # Crop images to their intersection
            out_rigid_stack = crop_to_intersection(out_rigid_stack, update_note=False)
            note_addition += "cropped "
        elif pad == 'same':
            # Expand all frames to cover the union of non-zero regions across the stack
            out_rigid_stack = extend_to_union(out_rigid_stack, update_note=False)
            note_addition += "extended to union "
        elif pad == 'zeros':
            pass
        else:
            raise ValueError("Invalid pad parameter. Supported values are 'same', 'intersection', or 'zeros'.")

        # This is the output of the registration
        out_rigid_stack = np.clip(out_rigid_stack, 0, 255).astype(np.uint8)  # Ensure pixel values are within [0, 255]
        out_rigid_stack1 = out_rigid_stack.copy()

        # This section is used to check if the processed image is sharper than the reference image and to re-register if necessary
        if reference_image is not None:
            # Cumulate the registered frame
            cum = cumulate(out_rigid_stack1, update_note=False)
            # Check if the processed image is sharper than reference
            # noinspection PyTypeChecker
            sharpness_reference = calculate_sharpness2(reference_image, blur=False)
            sharpness_processed1 = calculate_sharpness2(cum, blur=False)

            if sharpness_reference > sharpness_processed1:
                # print("zkusme to znova")
                # Create lists to store the registered frame stacks and sharpness values
                outs = [out_rigid_stack1]
                sharps = [sharpness_processed1]

                # Register again, this time using pyStackReg
                outs.append(register2(selected_frames, sharpness, reference='previous', pad=pad, update_note=False))

                # Cumulate the registered frame
                cum = cumulate(outs[-1], update_note=False)
                sharps.append(calculate_sharpness2(cum, blur=False))

                if sharpness_reference > sharps[-1]:
                    # If the processed image is still not sharper than the reference, register one last time using overkill parameters
                    # print("zkusme to jeste jednou")
                    param_map["NumberOfResolutions"] = ["6"]  # Use pyramidal registration
                    param_map["MaximumNumberOfIterations"] = ["500"]

                    out_rigid_stack3 = [selected_frames[0]]

                    # Register to previous frame in the stack
                    for i, moving_frame in enumerate(selected_frames[1:]):
                        # Convert reference image to SimpleITK format
                        reference_frame = sitk.GetImageFromArray(out_rigid_stack3[i].astype(np.float32))
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
                        out_rigid_stack3.append(registered_image)

                    if pad == 'same':
                        # Expand all frames to cover the union of non-zero regions across the stack
                        out_rigid_stack3 = extend_to_union(out_rigid_stack3, update_note=False)
                    elif pad == 'crop':
                        # Crop images to their intersection
                        out_rigid_stack3 = crop_to_intersection(out_rigid_stack3, update_note=False)

                    outs.append(np.clip(out_rigid_stack3, 0, 255).astype(np.uint8))  # Ensure pixel values are within [0, 255])
                    cum = cumulate(out_rigid_stack3, update_note=False)
                    sharps.append(calculate_sharpness2(cum, blur=False))

                # Return the sharpest registration result
                sharpest_index = np.argmax(sharps)
                if sharpest_index == 1:
                    note_addition = f"Registration method: pyStackReg\nReference: 'previous'\n"
                elif sharpest_index == 2:
                    note_addition = f"Registration method: SimpleElastix\nReference: 'previous'\n"

                if sharpest_index != 0:
                    if pad == 'crop':
                        note_addition += "Output image given by: cropped "
                    elif pad == 'same':
                        note_addition += "Output image given by: extended to union "

                out_rigid_stack = outs[sharpest_index]

                # print(sharpest_index)
                # show_frame(cumulate(outs[0], update_note=False), custom_note="0")
                # show_frame(cumulate(outs[1], update_note=False), custom_note="1")
                # show_frame(cumulate(outs[2], update_note=False), custom_note="2")

        # Update note
        note += note_addition

        return out_rigid_stack


def cumulate(frames, update_note=True):
    """
    Averages a stack of frames to reduce noise.
    :param update_note: Controls whether to update the note variable. Mainly for debug use.
    :param frames: Input list of frames as np.ndarrays.
    :return: Averaged frame as np.ndarray.
    """
    global note

    # If only one frame is given, return it
    if np.array(frames).ndim == 2:
        return frames
    cum = np.mean(frames, axis=0).astype(np.uint8)

    # Update the global note if update_note is True
    if update_note:
        note += f"mean of {len(frames)} frames\n"

    # Normalize the output image to the range [0, 255]
    cum = normalize(cum)

    return cum


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
    :return: Normalized image as uint8 np.ndarray.
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def match_dimensions(image1, image2=None):
    """
    Resizes image1 to 1066 x 1066 pixels. If image2 is provided, it will be resized to match its dimensions instead.
    :param image1: Image to be resized as np.ndarray.
    :param image2: Image to match dimensions with as np.ndarray.
    :return: Resized image1 as np.ndarray.
    """
    if image2 is None:
        image2 = np.zeros((1066, 1066), dtype=np.uint8)

    return cv2.resize(image1, (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_CUBIC)


def assess_quality(image_processed, report_path=None, generate_report=True):
    """
    Calculates the BRISQUE index for an image and optionally compares it to a reference image.
    :param generate_report: Parameter to control whether a report text file is generated.
    :param image_processed: Processed image as np.ndarray.
    :param report_path: Path to save the report text file.
    :return: BRISQUE index for the processed image and the reference image (if provided).
    """
    global note, reference_image, video_path
    reference = reference_image

    # Convert to grayscale if necessary
    if image_processed.ndim == 3:
        image_processed = rgb2gray(image_processed)

    # Ensure all pixel values are in 0-255 range
    if np.min(image_processed) < 0 or np.max(image_processed) > 255:
        image_processed = normalize(image_processed)

    # piq brisque implementation
    # Convert image to 4D tensor
    # image_processed_tensor = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(image_processed), 0), 0)
    # Calculate BRISQUE index
    # brisque_image = brisque(image_processed_tensor, data_range=255).item()

    # Calculate BRISQUE index
    obj = BRISQUE(url=False)
    brisque_image = obj.score(np.stack((image_processed,)*3, axis=-1))

    # Calculate PIQE score
    piqe_image, _, _, _ = piqe(image_processed)

    if reference is not None:
        # piq brisque implementation
        # Convert reference to 4D tensor
        # reference_tensor = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(reference), 0), 0)
        # Calculate BRISQUE index
        # brisque_reference = brisque(reference_tensor, data_range=255).item()

        # Calculate BRISQUE index
        brisque_reference = obj.score(np.stack((reference,)*3, axis=-1))

        # Calculate PIQE score
        piqe_reference, _, _, _ = piqe(reference)

    else:
        # If no reference is provided, return None
        brisque_reference = None
        piqe_reference = None

    # Create a text file containing the quality values
    if generate_report is True and report_path is not None:
        with open(report_path, "w") as f:
            f.write(f"Processed file: \"{video_path}\"\n\n"
                    f"{note}\n"
                    f"=== Image statistics ===\n"
                    f"Processed image:\n"
                    f"Sharpness = {calculate_sharpness2(image_processed, blur=False):.2f}\n"
                    f"BRISQUE index = {brisque_image:.2f}\n"
                    f"PIQE index = {piqe_image:.2f}\n"
                    f"\n")
            # noinspection PyTypeChecker
            f.write(f"No reference image provided\n" if reference is None else
                    f"Reference image:\n"
                    f"Sharpness = {calculate_sharpness2(reference):.2f}\n"
                    f"BRISQUE index = {brisque_reference:.2f}\n"
                    f"PIQE index = {piqe_reference:.2f}\n")

    note = ""  # Reset note
    return [brisque_image, brisque_reference], [piqe_image, piqe_reference]
