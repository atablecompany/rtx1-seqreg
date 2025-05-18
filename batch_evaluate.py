# This script is used to evaluate the results of batch processing of a dataset
import glob
import os
import matplotlib.pyplot as plt
import numpy as np


# %% Parse data
def parse_metrics_file(filename, central_metrics, noncentral_metrics):
    current_section = None
    is_central = None

    with open(filename, 'r') as file:
        lines = file.readlines()

        # First, scan the file for region information
        for line in lines:
            if "Video is of the central region" in line:
                is_central = True
                break
            elif "Video is not of the central region" in line:
                is_central = False
                break

        # If region information was not found, return None
        if is_central is None:
            return None

        # Now process the file again to collect metrics
        for line in lines:
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

                    metrics = central_metrics if is_central else noncentral_metrics

                    # Store metrics based on section
                    if current_section == 'processed':
                        if 'NIQE' in key:
                            metrics['niqe_images'].append(value)
                        elif 'var_of_LoG' in key:
                            metrics['sharpness_log_images'].append(value)
                        elif 'BRISQUE' in key:
                            metrics['brisque_images'].append(value)
                        elif 'PIQE' in key:
                            metrics['piqe_images'].append(value)
                    elif current_section == 'reference':
                        if 'NIQE' in key:
                            metrics['niqe_references'].append(value)
                        elif 'var_of_LoG' in key:
                            metrics['sharpness_log_references'].append(value)
                        elif 'BRISQUE' in key:
                            metrics['brisque_references'].append(value)
                        elif 'PIQE' in key:
                            metrics['piqe_references'].append(value)
                except ValueError as e:
                    print(f"Skipping malformed line: {line} - {str(e)}")

    return is_central


# Specify the directory containing text files
project_directory = "G:\PapyrusSorted"
report_files = glob.glob(os.path.join(project_directory, "**", "*report_adaptive8.txt"), recursive=True)

# Initialize metrics dictionaries
central_metrics = {
    'niqe_images': [],
    'sharpness_log_images': [],
    'brisque_images': [],
    'piqe_images': [],
    'niqe_references': [],
    'sharpness_log_references': [],
    'brisque_references': [],
    'piqe_references': [],
    'report_files': []
}

noncentral_metrics = {
    'niqe_images': [],
    'sharpness_log_images': [],
    'brisque_images': [],
    'piqe_images': [],
    'niqe_references': [],
    'sharpness_log_references': [],
    'brisque_references': [],
    'piqe_references': [],
    'report_files': []
}

# Loop through all files and process each
for report_file in report_files:
    is_central = parse_metrics_file(report_file, central_metrics, noncentral_metrics)
    if is_central is not None:  # Only add to list if region was detected
        if is_central:
            central_metrics['report_files'].append(report_file)
        else:
            noncentral_metrics['report_files'].append(report_file)


# %% Create boxplots
def create_region_boxplots(region_metrics, metrics_names, region_type):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    image_data = [
        np.array(region_metrics['niqe_images']),
        np.array(region_metrics['sharpness_log_images']),
        np.array(region_metrics['brisque_images']),
        np.array(region_metrics['piqe_images'])
    ]

    reference_data = [
        np.array(region_metrics['niqe_references']),
        np.array(region_metrics['sharpness_log_references']),
        np.array(region_metrics['brisque_references']),
        np.array(region_metrics['piqe_references'])
    ]

    for idx, (ax, metric) in enumerate(zip(axs.flat, metrics_names)):
        combined_data = [image_data[idx], reference_data[idx]]

        # Create basic boxplot with default styling
        bp = ax.boxplot(combined_data,
                        patch_artist=True,
                        labels=['Processed Images', 'Reference Images'],
                        boxprops=dict(facecolor='#1f77b4', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))

        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'{region_type} region only. Sharpness threshold=adaptive (min 2 frames, max 28 frames, 0.5 + **2 (default)).\nReg reference=\'mean\'. Mean cumulation. Resized to 1500x1500.',
        y=0.98,  # Position slightly below top
        fontsize=14,
        fontweight='bold')

    # Adjust layout with increased top padding
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # rect=[left, bottom, right, top]
    plt.show()


# Define metrics names for plots
metrics_names = [
    'NIQE Index',
    'Sharpness (var_of_LoG)',
    'BRISQUE Index',
    'PIQE Index'
]

# Create boxplots for central region
if central_metrics['niqe_images']:  # Check if we have data for central region
    create_region_boxplots(central_metrics, metrics_names, "Central")
else:
    print("No data available for central region.")

# Create boxplots for non-central region
if noncentral_metrics['niqe_images']:  # Check if we have data for non-central region
    create_region_boxplots(noncentral_metrics, metrics_names, "Non-Central")
else:
    print("No data available for non-central region.")


# %% Identify outliers
def detect_outliers_iqr(data):
    data = np.array(data)
    q1 = np.percentile(data, 15)  # 15
    q3 = np.percentile(data, 85)  # 85
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Indices where data is an outlier
    outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
    return outliers.tolist()


def detect_outliers_for_region(region_metrics, region_type):
    outlier_folders = set()

    metrics_to_check = {
        'sharpness_log_images': region_metrics['sharpness_log_images'],
        'brisque_images': region_metrics['brisque_images'],
        'piqe_images': region_metrics['piqe_images'],
        'niqe_images': region_metrics['niqe_images']
    }

    for metric, values in metrics_to_check.items():
        outlier_indices = detect_outliers_iqr(values)
        for idx in outlier_indices:
            folder = os.path.dirname(region_metrics['report_files'][idx])
            outlier_folders.add(folder)

    print(f"Outlier folders ({region_type} Region):")
    for folder in outlier_folders:
        print(folder)

    return outlier_folders


# Detect outliers for central region
if central_metrics['niqe_images']:
    central_outliers = detect_outliers_for_region(central_metrics, "Central")

# Detect outliers for non-central region
if noncentral_metrics['niqe_images']:
    noncentral_outliers = detect_outliers_for_region(noncentral_metrics, "Non-Central")


#%% Export data into csv
import csv


def export_metrics_to_csv(region_metrics, region_type, output_file):
    """
    Export region metrics to a CSV file.
    :param region_metrics: Dictionary containing metrics data.
    :param region_type: Type of region (e.g., "Central" or "Non-Central").
    :param output_file: Path to the output CSV file.
    """
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow([
            "NIQE (Images)", "Sharpness (var_of_LoG)",
            "BRISQUE (Images)", "PIQE (Images)",
            "NIQE (References)", "Sharpness (var_of_LoG References)",
            "BRISQUE (References)", "PIQE (References)"
        ])
        # Write data rows
        for i, report_file in enumerate(region_metrics['report_files']):
            writer.writerow([
                region_metrics['niqe_images'][i],
                region_metrics['sharpness_log_images'][i],
                region_metrics['brisque_images'][i],
                region_metrics['piqe_images'][i],
                region_metrics['niqe_references'][i] if i < len(region_metrics['niqe_references']) else None,
                region_metrics['sharpness_log_references'][i] if i < len(region_metrics['sharpness_log_references']) else None,
                region_metrics['brisque_references'][i] if i < len(region_metrics['brisque_references']) else None,
                region_metrics['piqe_references'][i] if i < len(region_metrics['piqe_references']) else None
            ])

# Export central region metrics
if central_metrics['report_files']:
    export_metrics_to_csv(central_metrics, "Central", "G:/PapyrusSorted/Results/Region-separate/central_metrics.csv")

# Export non-central region metrics
if noncentral_metrics['report_files']:
    export_metrics_to_csv(noncentral_metrics, "Non-Central", "G:/PapyrusSorted/Results/Region-separate/noncentral_metrics.csv")
