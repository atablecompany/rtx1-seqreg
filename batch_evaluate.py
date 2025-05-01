# This script is used to evaluate the results of batch processing of a dataset
import glob
import os
import matplotlib.pyplot as plt
import numpy as np


#%% Parse data
def parse_metrics_file(filename):
    current_section = None

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Skip section headers and empty lines
            if line.startswith('===') or not line:
                continue

            # Detect sections
            if line.startswith('Processed image:'):
                current_section = 'processed'
                continue
            elif line.startswith('Reference image:'):
                current_section = 'reference'
                continue

            # Process key-value pairs
            if '=' in line:
                try:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = float(value.strip())

                    if current_section == 'processed':
                        if 'var_of_laplacian' in key:
                            sharpness_images.append(value)
                        elif 'var_of_LoG' in key:
                            sharpness_log_images.append(value)
                        elif 'BRISQUE' in key:
                            brisque_images.append(value)
                        elif 'PIQE' in key:
                            piqe_images.append(value)

                    elif current_section == 'reference':
                        if 'var_of_laplacian' in key:
                            sharpness_references.append(value)
                        elif 'var_of_LoG' in key:
                            sharpness_log_references.append(value)
                        elif 'BRISQUE' in key:
                            brisque_references.append(value)
                        elif 'PIQE' in key:
                            piqe_references.append(value)
                except ValueError as e:
                    print(f"Skipping malformed line: {line} - {str(e)}")


# Specify the directory containing text files
project_directory = "G:\PapyrusSorted"
report_files = glob.glob(os.path.join(project_directory, "**", "*report_adaptive.txt"), recursive=True)

sharpness_images = []
sharpness_log_images = []
brisque_images = []
piqe_images = []

sharpness_references = []
sharpness_log_references = []
brisque_references = []
piqe_references = []

# Loop through all files and process each
for report_file in report_files:
    parse_metrics_file(report_file)


#%% Create boxplots
def create_standard_boxplots(image_data, reference_data, metrics):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (ax, metric) in enumerate(zip(axs.flat, metrics)):
        combined_data = [image_data[idx], reference_data[idx]]

        # Create basic boxplot with default styling
        bp = ax.boxplot(combined_data,
                        patch_artist=True,
                        labels=['Processed Images', 'Reference Images'],
                        boxprops=dict(facecolor='#1f77b4', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))

        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Image Quality Metrics Comparison: Processed vs Reference. Sharpness threshold=adaptive (0.3 + **2)',
                 y=0.98,  # Position slightly below top
                 fontsize=14,
                 fontweight='bold')

    # Adjust layout with increased top padding
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # rect=[left, bottom, right, top]
    plt.show()


# Prepare data (using your lists)
metrics = [
    'Sharpness (var_of_laplacian)',
    'Sharpness (var_of_LoG)',
    'BRISQUE Index',
    'PIQE Index'
]

data_arrays = [
    (sharpness_images, sharpness_references),
    (sharpness_log_images, sharpness_log_references),
    (brisque_images, brisque_references),
    (piqe_images, piqe_references)
]

# Convert to numpy arrays
image_data = [np.array(pair[0]) for pair in data_arrays]
reference_data = [np.array(pair[1]) for pair in data_arrays]

create_standard_boxplots(image_data, reference_data, metrics)


#%% Identify outliers
def detect_outliers_iqr(data):
    data = np.array(data)
    q1 = np.percentile(data, 15)
    q3 = np.percentile(data, 85)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Indices where data is an outlier
    outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
    return outliers.tolist()


metrics = {
    # 'sharpness_images': sharpness_images,
    'sharpness_log_images': sharpness_log_images,
    'brisque_images': brisque_images,
    'piqe_images': piqe_images
}

outlier_files = set()
for metric, values in metrics.items():
    outlier_indices = detect_outliers_iqr(values)
    for idx in outlier_indices:
        outlier_files.add(report_files[idx])

print("Outlier files:")
for filename in outlier_files:
    print(filename)
