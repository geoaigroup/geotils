import os
import argparse
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling

def mosaic_rasters(input_folder, output_path):
    """
    Mosaic multiple raster files in the input folder and save the result to the output path.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing raster files to mosaic.
    output_path : str
        Path to save the mosaic result.
    """
    raster_files = glob.glob(os.path.join(input_folder, '*.tif'))

    src_files_to_mosaic = []
    for fp in raster_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src.meta.copy()

    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": src.crs,
        #"nodata": 0
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Mosaic raster saved to {output_path}")

def main():
    """
    Main function to parse arguments and run the mosaic_rasters function.
    """
    parser = argparse.ArgumentParser(description="Mosaic all tiles in the input directory.")
    parser.add_argument('tiles_folder_path', metavar="tiles_folder_path", type=str, help="Path to the folder containing the tiles.")
    parser.add_argument('output_dir', metavar='output_dir', type=str, help='Path to the output directory.')
    args = parser.parse_args()

    input_folder, output_dir = args.tiles_folder_path, args.output_dir

    # Check if input folder exists
    if not os.path.isdir(input_folder):
        print("Input folder doesn't exist. Enter a valid path.")
        return

    # Create or clear output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create the output file path
    output_path = os.path.join(output_dir, 'Mosaic.tif')

    # Mosaic the raster tiles
    mosaic_rasters(input_folder, output_path)

if __name__ == "__main__":
    main()
