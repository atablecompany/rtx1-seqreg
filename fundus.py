import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pystackreg import StackReg
from dom import DOM
import piq
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy

# pip install -r requirements.txt
SHARPNESS_METRIC = 'variance_of_gray'  # Choose between 'variance_of_gray', 'dom' or 'variance_of_laplacian'
if SHARPNESS_METRIC == 'dom':
    iqa = DOM()  # Initialize DOM


def import_video(video_path, similarity_threshold=0.92):
    """
    Opens a video file and returns a np.ndarray of averaged frames based on similarity.
    :param video_path: Video file path.
    :param similarity_threshold: Threshold for frame similarity to group frames.
    :return: Averaged frames as np.ndarray.
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error opening video file")
        exit()
    frames_list = []

    # Read the first frame
    ok, prev_frame = video.read()
    if not ok:
        print("Cannot read video file")
        exit()
    prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    group_frames = [prev_gray_frame]

    while True:
        ok, frame = video.read()
        if not ok:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM between the current frame and the previous frame
        similarity = ssim(prev_gray_frame, gray_frame)

        if similarity >= similarity_threshold:
            # If frames are similar, add to the current group
            group_frames.append(gray_frame)
        else:
            # If frames are not similar, average the current group and start a new group
            group_avg = np.mean(group_frames, axis=0).astype("uint8")
            frames_list.append(group_avg)
            group_frames = [gray_frame]

        prev_gray_frame = gray_frame

    # Average the last group of frames
    if group_frames:
        group_avg = np.mean(group_frames, axis=0).astype("uint8")
        frames_list.append(group_avg)

    video.release()

    return np.array(frames_list).astype("uint8")  # Output array of frames


# def import_video_manual(video_path):
#     """
#     Opens a video file and returns a np.ndarray of valid frames (4 averaged initial frames, then averaged triplets of frames).
#     :param video_path: Video file path.
#     :return: Valid frames as np.ndarray.
#     """
#     # Open the video file
#     video = cv2.VideoCapture(video_path)
#
#     if not video.isOpened():
#         print("Error opening video file")
#         exit()
#     frames_list = []
#
#     # Read and average the first 4 frames
#     initial_frames = []
#     for _ in range(4):
#         ok, frame = video.read()
#         if not ok:
#             print("Cannot read video file")
#             exit()
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         initial_frames.append(gray_frame)
#     initial_avg = np.mean(initial_frames, axis=0).astype("uint8")
#     frames_list.append(initial_avg)
#
#     # Read and average every triplet of frames
#     while True:
#         triplet_frames = []
#         for _ in range(3):
#             ok, frame = video.read()
#             if not ok:
#                 break
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             triplet_frames.append(gray_frame)
#         if len(triplet_frames) < 3:
#             break
#         triplet_avg = np.mean(triplet_frames, axis=0)
#         frames_list.append(triplet_avg)
#
#     video.release()
#
#     return np.array(frames_list).astype("uint8")  # Output array of frames


def calculate_sharpness(frames, metric=SHARPNESS_METRIC, window_size=36):
    """
    Calculates the sharpness of a frame or frame stack using a specified metric.
    :param frames: Input frame as np.ndarray.
    :param metric: Can be either 'variance_of_gray', 'dom', 'variance_of_laplacian'.
    :param window_size: Size of window for local variance of gray.
    :return: Estimated sharpness value if input is a single frame or list of estimated sharpness values if input is a frame stack.
    """
    if len(frames.shape) == 2:
        # If a single frame is given
        if metric == 'variance_of_gray':
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
        elif metric == 'dom':
            # Using DOM https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2012/media/files/0043.pdf
            dom = iqa.get_sharpness(frames) ** 4
            return -dom  # -dom -> lower values = better
        elif metric == 'variance_of_laplacian':
            var_of_lap = cv2.Laplacian(frames, cv2.CV_64F).var()
            return var_of_lap
    elif len(frames.shape) == 3:
        # If a frame stack is given
        return [calculate_sharpness(f, metric, window_size) for f in frames]
    else:
        raise ValueError("Invalid input shape")


def show_frame(image, sharpness=None, frame_number=None, note="", save=False, filename="out.png"):
    """
    Displays a frame with a title overlay. Optionally saves the frame as a .png file.
    :param image: Input frame as np.ndarray.
    :param sharpness: Sharpness value to be printed in the title. If None, it will be calculated according to the SHARPNESS_METRIC global variable.
    :param frame_number: Frame index to be printed in the title.
    :param note: Optional note to be printed in the title.
    :param save: If True, the frame will be saved as a .png file.
    :param filename: Name of the file to be saved.
    :return:
    """
    dpi = 100  # Use your preferred DPI for display
    height, width = image.shape

    if sharpness is None:
        try:
            sharpness = calculate_sharpness(image)
        except:
            sharpness = 0

    quality = assess_quality(image)
    entropy = shannon_entropy(image)
    # Create a figure matching the original image size (1:1)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Adjust subplot to remove borders
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Show the image without any interpolation (1:1 scale)
    plt.imshow(image, cmap='gray', interpolation='none')

    # Overlay the title on top of the image
    if frame_number is None:
        plt.text(width / 2, 10, f"{note}\nsharpness={sharpness:.2f}, quality={quality:.0f}", color='white', fontsize=12, ha='center', va='top', backgroundcolor='black')
    else:
        plt.text(width / 2, 10, f"{note}\nsharpness={sharpness:.2f}, entropy={entropy}, i={frame_number}", color='white', fontsize=12, ha='center', va='top', backgroundcolor='black')
    # Hide axis
    plt.axis('off')

    if save:
        # Save the figure
        path = f"C:/Users/tengl/PycharmProjects/dp/{filename}"
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved figure as {path}")
        # show_frame(image, sharpness, frame_number, note, save=False)
    
    plt.show()


def crop_to_intersection(frames, threshold=0):
    """
    Crops a stack of frames to their intersection.
    :param frames: np.ndarray of registered images.
    :param threshold: Pixel value below which a pixel is considered black.
    :return:
    """
    # TODO: implement rotation
    # # Start with the maximum possible bounding box
    # h, w = frames[0].shape[:2]
    # x_min, x_max = 0, w - 1
    # y_min, y_max = 0, h - 1
    #
    # for frame in frames:
    #     # Find non-black pixels in the current image
    #     non_black_mask = (frame > threshold)
    #     y_indices, x_indices = np.where(non_black_mask)
    #
    #     # Update the bounding box
    #     if len(x_indices) > 0 and len(y_indices) > 0:  # Ensure there are non-black pixels
    #         x_min = max(x_min, x_indices.min())
    #         x_max = min(x_max, x_indices.max())
    #         y_min = max(y_min, y_indices.min())
    #         y_max = min(y_max, y_indices.max())
    #     else:
    #         raise ValueError("One of the images is completely black within the threshold!")
    #
    # cropped_images = [frame[y_min:y_max + 1, x_min:x_max + 1] for frame in frames]
    # return cropped_images
    mask = np.ones_like(frames[0], dtype=bool)

    for frame in frames:
        mask &= (frame > threshold)

    # Find the bounding box of the remaining mask
    y_indices, x_indices = np.where(mask)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    cropped_images = [frame[y_min:y_max + 1, x_min:x_max + 1] for frame in frames]
    return cropped_images


def register_cumulate(frames, sharpness, threshold, reference='previous', cumulate=True, crop=False):
    """
    Registers the sharpest frames and optionally averages them to reduce noise.
    :param reference: Either 'previous' or 'best'. If 'previous', each frame is registered to the previous frame in the stack. If 'best', each frame is registered to the sharpest frame in the stack.
    :param frames: Frames as np.ndarray.
    :param sharpness: List of sharpness values for frames.
    :param threshold: Threshold for selecting the sharpest frames.
    :param cumulate: Controls whether to average the registered frames.
    :param crop: Controls whether to crop the registered frames to their intersection.
    :return: If cumulate is False, returns the registered frame stack as np.ndarray. If cumulate is True, returns the cumulated image (np.ndarray) alongside a note (string).
    """
    if reference == 'previous':
        # Register to previous frame in the stack
        # Find the sharpest frames and add them to a list
        selected_frames_indices = [i for i, var in enumerate(sharpness) if var < threshold]  # Select frames above threshold
        selected_frames = frames[selected_frames_indices]  # Add the selected frames to the list. The frames are in chronological order

        # Perform registration
        sr = StackReg(StackReg.RIGID_BODY)
        out_rigid_stack = sr.register_transform_stack(selected_frames, reference='previous')
        out_rigid_stack = np.clip(out_rigid_stack, 0, 255).astype(np.uint8)  # Ensure pixel values are within [0, 255]
    elif reference == 'best':
        # Register to the sharpest frame
        # Find the sharpest frame and move it to the first position in the stack
        best_frame_index = np.argmin(sharpness)  # Find sharpest frame
        selected_frames = frames[best_frame_index]  # Add the sharpest frame to the array in position 0
        selected_frames_indices = [i for i, var in enumerate(sharpness) if var < threshold]  # Select frames above threshold
        selected_frames_indices.remove(best_frame_index)  # Remove the sharpest frame from the indices list (it's already in selected_frames)
        selected_frames = np.vstack((selected_frames[np.newaxis, ...], frames[selected_frames_indices]))  # Add the selected frames to the list. The sharpest frame is in position 0, followed by the selected frames in chronological order

        # Perform registration
        sr = StackReg(StackReg.RIGID_BODY)
        out_rigid_stack = sr.register_transform_stack(selected_frames, reference='first')
        out_rigid_stack = np.clip(out_rigid_stack, 0, 255).astype(np.uint8)  # Ensure pixel values are within [0, 255]
    else:
        raise ValueError("Invalid reference")

    cum_note = ""
    if crop:
        # Crop images to their intersection
        out_rigid_stack = crop_to_intersection(out_rigid_stack)
        cum_note += "Cropped "
    if cumulate:
        # Average registered frames
        cum = np.mean(out_rigid_stack, axis=0)
        cum_note += f"Mean of {selected_frames.shape[0]} registered frames ({reference})"
        return cum, cum_note
    else:
        return out_rigid_stack


def assess_quality(image):
    """
    Evaluates the quality of an image using the BRISQUE metric.
    :param image: Input image as np.ndarray.
    :return: BRISQUE index from 0 to 100. Lower scores indicate better quality.
    """
    # Convert RGB input image to grayscale
    if image.ndim == 3:
        image = rgb2gray(image)

    # Ensure all pixel values are non-negative
    if np.min(image) < 0:
        image = image + abs(np.min(image))

    intensity_range = np.max(image)
    # Convert image to 4D tensor
    image = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(image), 0), 0)
    score = piq.brisque(image, data_range=intensity_range)
    return score.item()
