# This script is used to evaluate the results of batch processing of a dataset
import glob
import os


def parse_metrics_file(filename):
    current_section = None

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Detect sections
            if line.startswith('Processed image:'):
                current_section = 'processed'
            elif line.startswith('Reference image:'):
                current_section = 'reference'

            # Split key-value pairs
            if '=' in line:
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


# Specify the directory containing text files
project_directory = "G:\PapyrusSorted"
report_files = glob.glob(os.path.join(project_directory, "**", "*report_new.txt"), recursive=True)

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
