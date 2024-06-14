import os
import subprocess

# Directories
ms4_dir = 'results/tiling/ms4'
pan_dir = 'results/tiling/pan'
sharp_dir = 'results/tiling/sharp'

# Create the sharp directory if it doesn't exist
os.makedirs(sharp_dir, exist_ok=True)

# Iterate over .tif files in the ms4 directory
for file_name in os.listdir(ms4_dir):
    if file_name.endswith('.tif'):
        # Extract the file name without the directory
        base_name = os.path.splitext(file_name)[0]
        
        # Construct the corresponding pan file path
        pan_file = os.path.join(pan_dir, f"{base_name}.tif")
        
        # Check if the pan file exists
        if os.path.exists(pan_file):
            # Construct the output file path
            output_file = os.path.join(sharp_dir, f"{base_name}.tif")
            
            # Run the pansharp.py script
            subprocess.run(['python', './pansharp.py', 
                            os.path.join(ms4_dir, file_name), 
                            pan_file, 
                            output_file, 
                            '-m', 'esri', 
                            '-i', 'bgr'])
        else:
            print(f'Pan file "{pan_file}" does not exist.')
