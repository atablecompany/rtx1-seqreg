# rtx1-seqreg
A Python toolkit for processing fundus image sequences from the rtx1 camera. This project processes MPEG-1 video files, registers sharp video frames and reduces noise through frame averaging and denoising.

![Original image sequence](https://i.imgur.com/POrsoiR.gif)
 → 
<img src="https://i.imgur.com/rkV157D.png" alt="Processed image" width="400" height="400">

## Features

-   **Frame selection**: Adaptive algorithm that selects the sharpest frames from video sequences
-   **Motion correction**: Rigid body registration to align frames and remove motion artifacts
-   **Region-specific processing**: Different processing parameters for central and peripheral regions
-   **Advanced denoising**: Multiple denoising algorithms (BM3D, TV, HMGF) optimized for rtx1 fundus imagery
-   **Quality assessment**: Comprehensive image quality metrics (NIQE, BRISQUE, PIQE, sharpness)
-   **Batch processing**: Parallel processing of multiple video files
-   **Processing evaluation**: Tools to evaluate and compare processing results

## Installation
        # Clone repository
	    git clone https://github.com/atablecompany/rtx1-seqreg
	    cd rtx1-seqreg
    
	    # Install dependencies
	    pip install -r requirements.txt

## Processing Pipeline

The standard processing pipeline includes:
1.  **Video loading**: Import MPEG-1 data from rtx1 fundus camera
2.  **Frame analysis**: Calculate sharpness metrics for all frames
3.  **Frame selection**: Select the sharpest frames adaptively
4.  **Registration**: Align frames to remove motion artifacts
5.  **Frame averaging**: Combine aligned frames to reduce noise
6.  **Denoising**: Apply advanced denoising algorithms
7.  **Quality assessment**: Calculate quality metrics for processed images

## Example usage
### Single video processing

    import fundus
    
    # Load video frames 
    frames = fundus.load_video("path/to/video.mpg")
    
    # Calculate sharpness and select the best frames
    sharpness = fundus.calculate_sharpness(frames)
    selected_frames = fundus.select_frames(frames, sharpness)
	
	# Register selected frames
	registered_frames = fundus.register(selected_frames, sharpness, reference='mean')
	
	# Cumulate registered frames
	averaged_image = fundus.cumulate(registered_frames)
	
	# Apply additional denoising
	if fundus.is_central_region:
	  if len(selected_frames) < 4:  
        denoised_image = fundus.denoise(cum, method='hmgf')  # Denoise using HMGF if few frames are selected  
	  else:  
        denoised_image = fundus.denoise(cum)  # Always denoise using BM3D if the region is central  
	else:  
    denoised_image = fundus.denoise(cum)  # If region is not central, always denoise
	
	# Resize the image
	final_image = fundus.resize(denoised_image, (1500, 1500))
	
	# Save result 
	fundus.save_frame(final_image, "output_image.png")

### Batch processing
Modify and run `batch.py` for processing multiple files.

### Evaluation
Use the function `fundus.assess_quality(final_image, "path/to/report.txt)` in your processing script to generate a report file containing details about the performed image processing alongside various quality metrics.

Modify and run `batch_evaluate.py` to batch analyze existing report files and compare with reference.

## Key parameters
### Frame selection
 - **Adaptive selection**: Automatically determines optimal number of frames based on sharpness distribution across time series
 - **Manual threshold**: Use the function `fundus.select_frames2(frames, sharpness, threshold=0.7)` to use a hard threshold instead
### Registration
 - **Reference type**: 'best', 'previous', or 'mean' frame as reference. Mean usually works the best.
 - **Registration method**: pyStackReg (robust) or SimpleElastix (faster but less accurate, doesn't support 'mean' reference). Use the function `fundus.register2(frames, sharpness)` to register using SimpleElastix instead.
 - **Border handling**: Different methods for handling borders ('same', 'crop', 'zeros')

### Denoising

-   **Method**:
    -   'bm3d': Block-matching and 3D filtering (best for central regions)
    -   'tv': Total Variation denoising
    -   'hmgf': Hybrid median-Gaussian filtering (good for peripheral regions). Only uses median filtering by default, but Gaussian can be enabled by modifying the `blend_factor` constant within fundus.denoise().
-   **Weight**: Controls denoising strength

### Quality Metrics
The toolkit calculates multiple quality metrics to evaluate processing results:

-   **Sharpness**: Variance of Laplacian of Gaussian (higher is better)
-   **NIQE**: Natural Image Quality Evaluator (lower is better)
-   **BRISQUE**: Blind/Referenceless Image Spatial Quality Evaluator (lower is better)
-   **PIQE**: Perception-based Image Quality Evaluator (lower is better)


## Credit
This work leverages [pyStackReg](https://github.com/glichtner/pystackreg) for registration.

Credit goes to [Gregor Lichtner](https://github.com/glichtner) for the Python implementation and Philippe Thevenaz for the original [StackReg](https://bigwww.epfl.ch/thevenaz/stackreg/) Java code.
