import cv2
import numpy as np
import matplotlib.pyplot as plt
from pystackreg import StackReg

def variance_of_gray(frame, window_size=36):
    # Determine the local gray level variance in a window https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=903548
    height, width = frame.shape
    local_variances = []
    for y in range(0, height, window_size):
        for x in range(0, width, window_size):
            window = frame[y:y+window_size, x:x+window_size]
            if window.size == 0:
                continue
            local_variances.append(np.var(window))
    var_of_gray = np.mean(local_variances)
    return var_of_gray


def show_frame(image, blurriness=None, frame_number=None, note=""):
    dpi = 100  # Use your preferred DPI for display
    height, width = image.shape
    
    if blurriness == None:
        blurriness = variance_of_gray(image)
        
    # Create a figure matching the original image size (1:1)
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Adjust subplot to remove borders
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Show the image without any interpolation (1:1 scale)
    plt.imshow(image, cmap='gray', interpolation='none')
    
    # Overlay the title on top of the image
    plt.text(width / 2, 10, f"{note}\nvar={blurriness}, i={frame_number}", color='white', fontsize=12, ha='center', va='top', backgroundcolor='black')
    # Hide axis
    plt.axis('off')
    plt.show()


def save_frame(image, filename, blurriness=None, frame_number=None, note=""):
    dpi = 100  # Use your preferred DPI for display
    height, width = image.shape

    if blurriness is None:
        blurriness = variance_of_gray(image)

    # Create a figure matching the original image size (1:1)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Adjust subplot to remove borders
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Show the image without any interpolation (1:1 scale)
    plt.imshow(image, cmap='gray', interpolation='none')

    # Overlay the title on top of the image
    plt.text(width / 2, 10, f"{note}\nvar={blurriness}, i={frame_number}", color='white', fontsize=12, ha='center', va='top', backgroundcolor='black')

    # Hide axis
    plt.axis('off')

    # Save the figure
    plt.savefig(f"C:/Users/tengl/PycharmProjects/dp/{filename}", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

#%% Import video file
# Path to the .mpg file
video_path = "G:\PapyrusSorted\AIZUATOVA_Imira_19970420_FEMALE\OD_20231124121205\OD_20231124121205_X0.0T_Y-2.0_Z0.0_AIZUATOVA_Imira_246.mpg"

# Open the video file
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Error opening video file")
    exit()

frames_list = []

# Read the first 4 frames
initial_frames = []
for _ in range(4):
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        exit()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    initial_frames.append(gray_frame)

# Average the 4 initial frames
initial_avg = np.mean(initial_frames, axis=0)
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

frames = np.array(frames_list)  # Output array of frames
print(frames.shape)

#%% Determine the variance of laplacian of frames
# var_of_lap = []
# for frame in frames:
#     var_of_lap.append(cv2.Laplacian(frame, cv2.CV_64F).var())

#%% Determine the blurriness of frames
blurriness = []
for frame in frames:
    blurriness.append(variance_of_gray(frame))

#%% Show frames and plot variance over frame indices
for i in range(len(frames)):
    show_frame(frames[i], blurriness[i], i)

plt.figure(figsize=(10, 6))
plt.plot(blurriness, marker='o', linestyle='-', color='b')
plt.title('Local Gray Level Variance Over Frames')
plt.xlabel('Frame Index')
plt.ylabel('Local Gray Level Variance')
plt.grid(True)
plt.show()

#%% Select the sharpest frames
threshold = 1400
selected_frames_indices = [i for i, var in enumerate(blurriness) if var > threshold]
best_frame_index = np.argmax(blurriness)

#%% Perform registration to reference
ref_image = frames[best_frame_index]
offset_image = frames[selected_frames_indices[4]]
sr = StackReg(StackReg.RIGID_BODY)
out_rigid = sr.register_transform(ref_image, offset_image)
# Show registered frames
show_frame(out_rigid, note="Registered image")
show_frame(ref_image, frame_number=best_frame_index, note="Reference image")
show_frame(offset_image, frame_number=selected_frames_indices[4], note="Unregistered image")

#%% Perform stack registration
selected_frames = frames[selected_frames_indices]
sr = StackReg(StackReg.RIGID_BODY)
out_rigid_stack = sr.register_transform_stack(selected_frames, reference='previous')
# Show registered frames
for i, frame in enumerate(out_rigid_stack):
    show_frame(frame, frame_number=i, note="Stack registered image")

#%% Average registered frames
cum = np.mean(out_rigid_stack, axis=0)
show_frame(cum, note="Mean of registered frames")
save_frame(cum, "out")
