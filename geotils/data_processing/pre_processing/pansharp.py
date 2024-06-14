import os,sys,argparse
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pywt
                                   
def WriteImage(dist_ds, output_path, red, green, blue, nir=None):
    """
    Create a new TIFF file with the specified bands and resolution.

    Parameters
    ----------
    dist_ds : gdal.Dataset
        The source dataset from which to copy projection and geotransform.
    output_path : str
        The file path where the output TIFF will be saved.
    red : array_like
        The red band data.
    green : array_like
        The green band data.
    blue : array_like
        The blue band data.
    nir : array_like, optional
        The near-infrared band data, if available.

    Returns
    -------
    None
    """
    bands_count = 4 if nir is not None else 3
    output_ds = gdal.GetDriverByName("GTiff").Create(output_path, dist_ds.RasterXSize, dist_ds.RasterYSize,
                                                     bands_count, gdal.GDT_Float32)

    output_ds.SetProjection(dist_ds.GetProjection())
    output_ds.SetGeoTransform(dist_ds.GetGeoTransform())

    output_ds.GetRasterBand(3).WriteArray(blue)
    output_ds.GetRasterBand(3).SetNoDataValue(0)

    output_ds.GetRasterBand(2).WriteArray(green)
    output_ds.GetRasterBand(2).SetNoDataValue(0)

    output_ds.GetRasterBand(1).WriteArray(red)
    output_ds.GetRasterBand(1).SetNoDataValue(0)

    if nir is not None:
        output_ds.GetRasterBand(4).WriteArray(nir)
        #output_ds.GetRasterBand(4).SetNoDataValue(0)

    output_ds = None

def resample(source_ds, dest_ds, output_path):
    """
    Resample the source image dataset to match the spatial resolution of the 
    destination image using cubic interpolation.

    Parameters
    ----------
    source_ds : gdal.Dataset
        GDAL dataset of the source image.
    dest_ds : gdal.Dataset
        GDAL dataset of the destination image to match resolution.
    output_path : str
        Path to save the resampled raster.

    Returns
    -------
    None
    """
    output_ds = gdal.GetDriverByName("GTiff").Create(
        output_path, 
        dest_ds.RasterXSize, 
        dest_ds.RasterYSize, 
        source_ds.RasterCount, 
        gdal.GDT_Float32
    )
    
    output_ds.SetProjection(dest_ds.GetProjection())
    output_ds.SetGeoTransform(dest_ds.GetGeoTransform())

    gdal.ReprojectImage(
        source_ds, 
        output_ds, 
        source_ds.GetProjection(), 
        dest_ds.GetProjection(), 
        gdal.GRA_Cubic
    )

    for band_index in range(1, output_ds.RasterCount + 1):
        band = output_ds.GetRasterBand(band_index)
        band.SetNoDataValue(0)
    
    output_ds = None

def brovey(pan, red, green, blue, nir=None, weights=(0.3, 0.3, 0.3, 0.2)):
    """
    Brovey method implementation.

    Parameters
    ----------
    pan : numpy.ndarray
        Panchromatic image.
    red : numpy.ndarray
        Red band of the multispectral image.
    green : numpy.ndarray
        Green band of the multispectral image.
    blue : numpy.ndarray
        Blue band of the multispectral image.
    nir : numpy.ndarray, optional
        Near-infrared band of the multispectral image.
    weights : tuple, optional
        Weights for red, green, blue, and near-infrared bands.

    Returns
    -------
    tuple
        New red, green, blue, and optionally near-infrared bands.
    """
    if nir is None:
        brovey_ratio = np.true_divide(pan, red + green + blue + 3)
    else:
        brovey_ratio = np.true_divide(
            pan - nir * weights[3], 
            red * weights[0] + green * weights[1] + blue * weights[2] + 3
        )

    red = brovey_ratio * red
    green = brovey_ratio * green
    blue = brovey_ratio * blue

    if nir is None:
        return red, green, blue
    else:
        nir = brovey_ratio * nir
        return red, green, blue, nir

def rgb2ihs(r, g, b):
    """
    Convert an RGB image to IHS color space.

    Parameters
    ----------
    r : numpy.ndarray
        Red channel of the RGB image.
    g : numpy.ndarray
        Green channel of the RGB image.
    b : numpy.ndarray
        Blue channel of the RGB image.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing the intensity (I), hue (V1), and saturation (V2) components of the IHS image.
    """
    I = (r + g + b) / 3.0
    V1 = (-np.sqrt(2) * r - np.sqrt(2) * g + 2 * np.sqrt(2) * b) / 6
    V2 = (r - g) / np.sqrt(2)

    return I, V1, V2

def ihs2rgb(I, V1, V2):
    """
    Convert an IHS image to RGB color space.

    Parameters
    ----------
    I : numpy.ndarray
        Intensity component of the IHS image.
    V1 : numpy.ndarray
        Hue component of the IHS image.
    V2 : numpy.ndarray
        Saturation component of the IHS image.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing the red (R), green (G), and blue (B) channels of the RGB image.
    """
    R = I - V1 / np.sqrt(2) + V2 / np.sqrt(2)
    G = I - V1 / np.sqrt(2) - V2 / np.sqrt(2)
    B = I + np.sqrt(2) * V1

    return R, G, B

def IHS(pan, red, green, blue, nir=None, weights=(0.299, 0.587, 0.144, 0.15)):
    """
    IHS method implementation for pansharpening.

    Parameters
    ----------
    pan : numpy.ndarray
        Panchromatic image.
    red : numpy.ndarray
        Red band of the multispectral image.
    green : numpy.ndarray
        Green band of the multispectral image.
    blue : numpy.ndarray
        Blue band of the multispectral image.
    nir : numpy.ndarray, optional
        Near-infrared band of the multispectral image.
    weights : tuple, optional
        Weights for intensity calculation (default: (0.299, 0.587, 0.144, 0.15)).

    Returns
    -------
    tuple
        Adjusted red, green, blue, and optionally NIR bands.
    """
    I, v1, v2 = rgb2ihs(red, green, blue)
    if nir is not None:
        red, green, blue = ihs2rgb(pan - I*weights[3], v1, v2)
    else:
        red, green, blue = ihs2rgb(pan, v1, v2)

    return red, green, blue, nir
 
def esri(pan, red, green, blue, nir=None, weights=(1, 1, 1, 0.1)):
    """
    ESRI method implementation with optional infrared band integration.

    Parameters
    ----------
    pan : numpy.ndarray
        Panchromatic image.
    red : numpy.ndarray
        Red band of the multispectral image.
    green : numpy.ndarray
        Green band of the multispectral image.
    blue : numpy.ndarray
        Blue band of the multispectral image.
    nir : numpy.ndarray, optional
        Near-infrared band of the multispectral image.
    weights : tuple, optional
        Weights for red, green, blue, and near-infrared bands.

    Returns
    -------
    tuple
        Adjusted red, green, blue, and optionally near-infrared bands.
    """
    if nir is None:
        WA = (weights[0] * red + weights[1] * green + weights[2] * blue) / (weights[0] + weights[1] + weights[2])
    else:
        WA = (weights[0] * red + weights[1] * green + weights[2] * blue + weights[3] * nir) / (weights[0] + weights[1] + weights[2] + weights[3])
    
    ADJ = pan - WA
    
    red = red + ADJ
    green = green + ADJ
    blue = blue + ADJ
    
    if nir is None:
        return red, green, blue
    else:
        #nir = nir + ADJ
        return red, green, blue, nir

class PanSharpening:
    """
    A class to perform pan-sharpening using various methods.

    Methods
    -------
    check_method(method_name)
        Check if the given pan-sharpening method is available.
    apply(pan, red, green, blue, nir=None, method_name='esri')
        Apply the selected pan-sharpening method.
    """

    def __init__(self):
        """
        Initialize the PanSharpening class with available methods.
        """
        self.method_dict = {
            'brovey': brovey,
            'ihs': IHS,
            'esri': esri,
            
        }
        self.available_methods = list(self.method_dict.keys())

    def check_method(self, method_name):
        """
        Check if the given pan-sharpening method is available.

        Parameters
        ----------
        method_name : str
            The name of the pan-sharpening method.

        Raises
        ------
        AssertionError
            If the method name is not in the list of available methods.
        """
        assert method_name.lower() in self.available_methods, \
            f'Pan Sharpening Method : {method_name} is not available, select one of these options {self.available_methods}.'

    def apply(self, pan, red, green, blue, nir=None, method_name='esri'):
        """
        Apply the selected pan-sharpening method.

        Parameters
        ----------
        pan : numpy.ndarray
            Panchromatic image.
        red : numpy.ndarray
            Red band of the multispectral image.
        green : numpy.ndarray
            Green band of the multispectral image.
        blue : numpy.ndarray
            Blue band of the multispectral image.
        nir : numpy.ndarray, optional
            Near-infrared band of the multispectral image.
        method_name : str, optional
            The name of the pan-sharpening method (default is 'esri').

        Returns
        -------
        tuple
            Adjusted red, green, blue, and optionally NIR bands.
        """
        # Check if the pan-sharpening method is implemented
        self.check_method(method_name)
        
        # Choose the pan-sharpening method function to call
        func = self.method_dict[method_name.lower()]
        
        # Apply pan-sharpening
        if nir is None:
            new_red, new_green, new_blue = func(pan, red, green, blue)
            return new_red, new_green, new_blue
        
        new_red, new_green, new_blue, new_nir = func(pan, red, green, blue, nir)
        return new_red, new_green, new_blue, new_nir
 
def main():
    
    parser = argparse.ArgumentParser(description="Pansharpening.")
    parser.add_argument('spectral_path', metavar="spectral file path", type=str, help="Path to the .tif file of the spectral image.")
    parser.add_argument('pan_path', metavar="panchromatic file path", type=str, help="Path to the .tif file of the panchromatic image.")
    parser.add_argument('output_path', metavar='output path', type=str, help='Path to the output directory.')
    parser.add_argument('-m','--method', metavar='method', type=str,
                         help='Pansharpening method [ESRI,IHS,Brovey].',
                         choices=["esri","ihs","brovey"],default='esri')
    parser.add_argument('-i','--input', metavar='input_format', type=str,
                         help='Input format [RGB,BGR].',
                         choices=["rgb","bgr"],default='bgr')
    args = parser.parse_args()


    #check if the file names are correct
    multispectral_image_path = args.spectral_path
    panchromatic_image_path = args.pan_path
    destination_path = args.output_path
    method = args.method
    format = args.input


    #check if files are found
    if not os.path.isfile(multispectral_image_path):
       print("The path for the multispectral image is not found")
       sys.exit()
    
    if not os.path.isfile(panchromatic_image_path):
       print("The path for the panchromatic image is not found")
       sys.exit()
    
    if os.path.isfile(destination_path):
       os.remove(destination_path)
       print("deleted old copies")

    #check if files are int the correct format
    if '.tif' not in multispectral_image_path and '.tiff' not in multispectral_image_path:
       print("The Multispectral image file is not in the correct format, please enter a GEOTIFF file")
       sys.exit()
    
    if '.tif' not in panchromatic_image_path and '.tiff' not in panchromatic_image_path:
       print("The Panachromatic image file is not in the correct format, please enter a GEOTIFF file")
       sys.exit()
    
    if '.tif' not in destination_path and '.tiff' not in destination_path:
       print("The destination file format ins wrong, enter a path to a GEOTIFF file")
    

    #check if the first file is multisepectral
    multispectral_image = gdal.Open(multispectral_image_path)
    if(multispectral_image.RasterCount < 3): 
        print("The provided multispectral file is not multispectral, enter a valid file ")
        sys.exit()
   
        
    panchromatic_image = gdal.Open(panchromatic_image_path)
    if(panchromatic_image.RasterCount != 1): 
       print("The provided panchromatic file is not panachromatic, enter a valid file ")
       sys.exit()
    
    #resample the Mulstispectral image into the resolution of the panachromatic image.
    resample_path = multispectral_image_path.replace(".tif","_resamapled.tif")
    resample(multispectral_image,panchromatic_image,resample_path)
    multispectral_image = gdal.Open(resample_path)

    #copy the Bands into numpy arrays
    
    pan = panchromatic_image.GetRasterBand(1).ReadAsArray()
    if format =='bgr':
        blue = multispectral_image.GetRasterBand(1).ReadAsArray()
        green = multispectral_image.GetRasterBand(2).ReadAsArray()
        red = multispectral_image.GetRasterBand(3).ReadAsArray()
    elif format == 'rgb':      
        blue = multispectral_image.GetRasterBand(3).ReadAsArray()
        green = multispectral_image.GetRasterBand(2).ReadAsArray()
        red = multispectral_image.GetRasterBand(1).ReadAsArray()
    if multispectral_image.RasterCount >= 4:
       nir = multispectral_image.GetRasterBand(4).ReadAsArray()

    #get the sharp colors
    pansharp = PanSharpening()
    if multispectral_image.RasterCount >=4:
        red,green,blue,nir = pansharp.apply(pan=pan,red=red,green=green,blue=blue,nir=nir,method_name=method)
        WriteImage(blue=blue,red=red,green=green,nir=nir,dist_ds=multispectral_image,output_path=destination_path)
    else:
        red,green,blue = pansharp.apply(pan=pan,red=red,green=green,blue=blue,method_name=method)
        WriteImage(blue=blue,red=red,green=green,dist_ds=multispectral_image,output_path=destination_path)
    print(f"pansharpening done in {destination_path}")
  

    #close file streams to apply changes
    multispectral_image = panchromatic_image = None
    os.remove(resample_path)
    
if __name__ == "__main__":
    main()