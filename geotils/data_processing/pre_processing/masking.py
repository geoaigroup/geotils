import os
import argparse
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping, box

def masking(raster_path, shapefile_path, output_directory):
    """
    Tile the raster according to the shapefile and save the tiles in the output directory.

    Parameters
    ----------
    raster_path : str
        Path to the raster file.
    shapefile_path : str
        Path to the shapefile.
    output_directory : str
        Directory to save the tiled raster tiles.
    """
    # Read the raster file
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds

        shapefile = gpd.read_file(shapefile_path)

        if shapefile.crs != raster_crs:
            shapefile = shapefile.to_crs(raster_crs)

        for idx, row in shapefile.iterrows():
            geometry = row.geometry

            if geometry.intersects(box(*raster_bounds)):
                shapes = [mapping(geometry)]

                out_image, out_transform = mask(src, shapes, crop=True)

                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "crs": raster_crs,
                    "nodata": 0
                })

                tile_filename = f"{output_directory}/tile_{idx}.tif"

                with rasterio.open(tile_filename, 'w', **out_meta) as dest:
                    dest.write(out_image)
                
                print(f"Saved clipped tile to {tile_filename}")
            else:
                print(f"Shape {idx} does not intersect with the raster. Skipping.")

    print("Tiling process completed.")

def main():
    """
    Main function to parse arguments and run the masking function.
    """
    parser = argparse.ArgumentParser(description="Tile a raster according to a shapefile fishnet.")
    parser.add_argument('raster_path', metavar="raster_path", type=str, help="Path to the raster file.")
    parser.add_argument('shapefile_path', metavar='shapefile_path', type=str, help='Path to the shapefile.')
    parser.add_argument('output_dir', metavar='output_dir', type=str, help='Path to the output directory.')
    args = parser.parse_args()

    raster_path, shapefile_path, output_directory = args.raster_path, args.shapefile_path, args.output_dir

    # Check if paths exist and formats are correct
    if not all(os.path.isfile(path) for path in [raster_path, shapefile_path]):
        print("Raster or shapefile doesn't exist. Enter valid paths.")
        return
    if not raster_path.endswith(('.tif', '.tiff')) or not shapefile_path.endswith('.shp'):
        print("Incorrect file formats. Raster should be .tif or .tiff, and shapefile should be .shp.")
        return

     #Create or clear output directory
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    else:
        for filename in os.listdir(output_directory):
            file_path = os.path.join(output_directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        print("Output directory cleared.")

    # Tile the raster
    masking(raster_path=raster_path, shapefile_path=shapefile_path, output_directory=output_directory)

if __name__ == "__main__":
    main()
