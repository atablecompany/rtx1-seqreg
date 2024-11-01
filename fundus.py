import cv2
import numpy as np
import matplotlib.pyplot as plt
from pystackreg import StackReg
from dom import DOM

SHARPNESS_METRIC = 'variance_of_gray'  # Choose between 'variance_of_gray', 'dom' or 'variance_of_laplacian'
iqa = DOM()  # Initialize DOM


def import_video(video_path):
    """
    Opens a video file and returns a np.array of valid frames (4 averaged initial frames, then averaged triplets of frames).
    :param video_path: Video file path.
    :return: Valid frames as np.array.
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error opening video file")
        exit()
    frames_list = []

    # Read and average the first 4 frames
    initial_frames = []
    for _ in range(4):
        ok, frame = video.read()
        if not ok:
            print("Cannot read video file")
            exit()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        initial_frames.append(gray_frame)
    initial_avg = np.mean(initial_frames, axis=0).astype("uint8")
    frames_list.append(initial_avg)

    # Read and average every triplet of frames
    while True:
        triplet_frames = []
        for _ in range(3):
            ok, frame = video.read()
            if not ok:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            triplet_frames.append(gray_frame)
        if len(triplet_frames) < 3:
            break
        triplet_avg = np.mean(triplet_frames, axis=0)
        frames_list.append(triplet_avg)

    video.release()

    return np.array(frames_list).astype("uint8")  # Output array of frames


def calculate_sharpness(frame, metric=SHARPNESS_METRIC, window_size=36):
    """
    Calculates the sharpness of a frame using a specified metric.
    :param frame: Input frame as np.array.
    :param metric: Can be either 'variance_of_gray', 'dom' or 'variance_of_laplacian'.
    :param window_size: Size of window for local variance of gray.
    :return: Estimated sharpness value.
    """
    if metric == 'variance_of_gray':
        # Determine the local gray level variance in a window https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=903548
        height, width = frame.shape
        local_variances = []
        for y in range(0, height, window_size):
            for x in range(0, width, window_size):
                window = frame[y:y + window_size, x:x + window_size]
                if window.size == 0:
                    continue
                local_variances.append(np.var(window))
        var_of_gray = np.mean(local_variances)
        return var_of_gray
    elif metric == 'dom':
        # Using DOM https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2012/media/files/0043.pdf
        dom = iqa.get_sharpness(frame) ** 4
        return dom
    elif metric == 'variance_of_laplacian':
        var_of_lap = cv2.Laplacian(frame, cv2.CV_64F).var()
        return var_of_lap


def show_frame(image, sharpness=None, frame_number=None, note="", save=False, filename="out.png"):
    """
    Displays a frame with a title overlay. Optionally saves the frame as a .png file.
    :param image: Input frame as np.array.
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
        sharpness = calculate_sharpness(image)

    # Create a figure matching the original image size (1:1)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Adjust subplot to remove borders
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Show the image without any interpolation (1:1 scale)
    plt.imshow(image, cmap='gray', interpolation='none')

    # Overlay the title on top of the image
    if frame_number is None:
        plt.text(width / 2, 10, f"{note}\nsharpness={sharpness}", color='white', fontsize=12,
                 ha='center', va='top', backgroundcolor='black')
    else:
        plt.text(width / 2, 10, f"{note}\nsharpness={sharpness}, i={frame_number}", color='white', fontsize=12,
                 ha='center',
                 va='top', backgroundcolor='black')
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


def register_cumulate(frames, sharpness, threshold, cumulate=True):
    """
    Registers the sharpest frames and optionally averages them to reduce noise.
    :param frames: Frames as np.array.
    :param sharpness: List of sharpness values for each frame.
    :param threshold: Threshold for selecting the sharpest frames.
    :param cumulate: Controls whether to average the registered frames.
    :return: If cumulate is False, returns the registered frames. If cumulate is True, returns the cumulated image alongside a note.
    """
    selected_frames_indices = [i for i, var in enumerate(sharpness) if var > threshold]

    # Perform stack registration
    selected_frames = frames[selected_frames_indices]
    sr = StackReg(StackReg.RIGID_BODY)
    out_rigid_stack = sr.register_transform_stack(selected_frames, reference='previous')
    if not cumulate:
        return out_rigid_stack
    else:
        # Average registered frames
        cum = np.mean(out_rigid_stack, axis=0)
        cum_note = f"Mean of {len(selected_frames_indices)} registered frames"
        return cum, cum_note

