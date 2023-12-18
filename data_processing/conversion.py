import rasterio as rio 
import geopandas as gpd
from rasterio.features import shapes

def instance_mask_to_gdf(
        instance_mask,
        transform = None,
        crs=None
        ):
    """
    Input:
        - instance_mask : np.array of shape (H,W), where each instance is labeled by a unique id/number
        - transform : geospatial transform of the raster - default is None
        - crs : crs of the raster - default is None
    Output:
        - GeoDataFrame of the shapes projected to the specified crs using the transform
    """

    #transform should be Identity if None is provided
    transform = rio.transform.IDENTITY if transform is None else transform
    
    all_shapes = shapes(instance_mask,mask=None,transform=transform)
    data = [
            {'properties' : {'id' : v} , 'geometry' : s} for i,(s,v) in enumerate(all_shapes) if v!=0
    ]
    
    if len(data) == 0:
        ##return empty dataframe
        return gpd.GeoDataFrame(columns=['id','geometry'], geometry='geometry',crs=crs)
    
    gdf = gpd.GeoDataFrame.from_features(data,crs=crs)
    gdf = gdf.dissolve(by='id')
    
    return gdf