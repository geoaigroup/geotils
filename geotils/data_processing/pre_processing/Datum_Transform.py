# This tool is based on ArcGis code logic
import geopandas as gpd
import math
from tqdm import tqdm
from shapely.geometry import MultiPolygon , Polygon, Point
from fiona.crs import CRS 
from pyproj import Transformer

def GetFeatureName(s):
    """
    Extracts the feature name from a path string.

    Parameters
    ----------
    s : str
        The path string from which to extract the feature name.

    Returns
    -------
    str
        The last segment of the path, typically representing the feature name.
    """
    l = s.split('\\')
    n = len(l)
    return l[n-1]


def GetFeaturePath(s):
    """
    Extracts the path without the feature name from a full path string.

    Parameters
    ----------
    s : str
        The full path string from which to extract the path without the feature name.

    Returns
    -------
    str
        The path string excluding the feature name.
    """
    l = s.split('\\')
    n = len(l)
    path = ''
    for i in range(0, n-1):
        path += l[i] + "\\"
    return path

def eccentricity(flattening: float):
    """
    Calculate the eccentricity of an ellipse or ellipsoid from its flattening.

    Parameters
    ----------
    flattening : float
        The flattening factor of the ellipse or ellipsoid 
    Returns
    -------
    float
        The eccentricity, calculated using the formula sqrt(2f - f^2) where f is the flattening.
    """

    inv_flatten = 1 / flattening
    eccentricity = math.sqrt(inv_flatten*(2 - inv_flatten))
    return eccentricity


def SterioToGeo(point, p0, a, f, m, x0, y0):
    """
    Converts a point from stereographic projection coordinates to geographic coordinates.

    Parameters
    ----------
    point : Point
        The point in stereographic projection coordinates.
    p0 : Point
        The reference point for the projection in geographic coordinates.
    a : float
        Semi-major axis of the ellipsoid.
    f : float
        Flattening of the ellipsoid.
    m : float
        Scale factor.
    x0 : float
        False Easting.
    y0 : float
        False Northing.

    Returns
    -------
    Point
        The geographic coordinates of the point.
    """
    x=point.x - x0
    y=point.y - y0
    long0=p0.x*math.pi/180 # num5
    lat0=p0.y*math.pi/180 # num6
    e = eccentricity(f)
    e2=math.pow(e,2) # num7 
    e1=e2/(1-e2) #num9
    q = math.sqrt(1 + e1 * math.pow(math.cos(lat0), 4)) #num10
    s=a * m * math.sqrt(1 - e2) / (1 - e2 * math.pow(math.sin(lat0), 2)) #num11
    w=math.asin(math.sin(lat0)/q) #num13
    w2=math.log(math.tan((math.pi/4)+w/2)) #num14
    t=math.log(math.tan((math.pi/4)+lat0/2) * math.pow((1 - e * math.sin(lat0)) / (1 + e * math.sin(lat0)), e / 2)) #num15
    z=w2-q*t #num16
    r=math.sqrt(math.pow(x,2)+math.pow(y,2)) #num17
    if r >= math.pow(10,-13):
        l=2 * math.atan(x / (r - y)) #num1
    else:
        l=0 #num1
    i=(math.pi/2)-2*math.atan(r/(2*s)) #num18
    w3=w-(math.pi/2) #num19
    u1=math.cos(i) * math.cos(l) * math.cos(w3) - math.sin(i) * math.sin(w3) #num20
    u2=math.cos(i) * math.sin(l) #num21
    u3=math.sin(i) * math.cos(w3) + math.cos(i) * math.cos(l) * math.sin(w3) #num22
    if math.sqrt(math.pow(u1,2)+math.pow(u3,2)) >= math.pow(10,-13) :
        h1=math.atan(u3 / math.sqrt(math.pow(u1, 2) + math.pow(u2, 2))) #num3
        h2=2*math.atan(u2 / (math.sqrt(math.pow(u1, 2) + math.pow(u2, 2))+u1)) #num
    else:
        h1=u3/abs(u3) * math.pi /2 #num3
        h2=0 #num
    w2=math.log(math.tan((math.pi/4)+h1/2)) #num14
    lat0=-(math.pi/2 ) + 2 * math.atan(math.exp(w2 - z) / q) #num6
    while True:
        g=-(math.pi/2) + 2 * math.atan(math.pow((1+e*math.sin(lat0))/(1-e*math.sin(lat0)),e/2)*math.exp((w2 - z) / q))  #num2
        g2=lat0 #num4
        lat0=g #num6
        if abs(g2-g) <= math.pow(10,-12):
            break
    lat=g * 180 / math.pi
    lon=(long0+h2/q) * 180 / math.pi
    point2=Point(lon,lat)
    return point2


def GeographicToGeocentric(point, a, f):
    """
    Converts geographic coordinates to geocentric coordinates.

    Parameters
    ----------
    point : Point
        The point in geographic coordinates (longitude, latitude).
    a : float
        Semi-major axis of the ellipsoid.
    f : float
        Flattening of the ellipsoid.

    Returns
    -------
    Point
        The point in geocentric coordinates.
    """
    e = eccentricity(f)
    lon=point.x
    lat=point.y
    alt=0.0
    n=a/math.sqrt(1-(e*e*math.pow(math.sin(lat*math.pi/180),2)))
    x=(n+alt)*math.cos(lat*math.pi/180)*math.cos(lon*math.pi/180)
    y=(n+alt)*math.cos(lat*math.pi/180)*math.sin(lon*math.pi/180)
    z=(n*(1-math.pow(e,2))+alt)*math.sin(lat*math.pi/180)
    p=Point(x,y,z)
    return p


def GeocentricToGeographic(point, a, f):
    """
    Converts geocentric coordinates to geographic coordinates.

    Parameters
    ----------
    point : Point
        The point in geocentric coordinates.
    a : float
        Semi-major axis of the ellipsoid.
    f : float
        Flattening of the ellipsoid.

    Returns
    -------
    Point
        The geographic coordinates of the point.
    """
    e = eccentricity(f)
    e2=math.pow(e,2) # num1
    r=math.sqrt(math.pow(point.x,2)+math.pow(point.y,2)) # num2
    phi0=math.atan(point.z/r) # num3
    phi=phi0 # num4
    while True:
        phi0=phi
        c=math.sqrt(1-e2*math.pow(math.sin(phi0),2)) # num5
        o=a/c # num6
        k=r*math.cos(phi0)+point.z*math.sin(phi0)-a*c # num
        phi=math.atan(point.z/r*math.pow(1-o*e2/(o+k),-1))
        if abs(phi-phi0) <= math.pow(10,-13):
            break
    phi=phi*180/math.pi
    lamda=math.atan(point.y/point.x)*180/math.pi
    p=Point(lamda,phi,0)
    return p


def SterioToWgs84(point):
    """
    Converts a point from stereographic projection to WGS 84 geographic coordinates through
    a series of transformations.

    Parameters
    ----------
    point : Point
        The point in stereographic projection coordinates.

    Returns
    -------
    Point
        The geographic coordinates of the point in the WGS 84 system.
    """
    p0=Point(39.15,34.2)
    print("Original Point:", point)
    p1 = SterioToGeo(point , p0,6378249.2,293.4660212936265,0.999534104,0,0) 
    print("After SterioToGeo:", p1)
    p2 = GeographicToGeocentric(p1 , 6378249.2,293.4660212936265)
    print("After GeographicToGeocentric:", p2)
    p3 = DatumToGRS80(p2, 176.44534,124.582493,-244.842726,17.257592,12.12001,10.508606,-6.123272) 
    print("After DatumToGRS80:", p3)
    p4 = GeocentricToGeographic(p3 , 6378137.0,298.257223563)
    print("Final Output (GeocentricToGeographic):", p4)
    return p4


def DatumToGRS80(point, tx, ty, tz, rx, ry, rz, f):
    """
    Applies a datum transformation from GRS 80 using provided shift and rotation parameters.

    Parameters
    ----------
    point : Point
        The point in geocentric coordinates.
    tx, ty, tz : float
        Translation parameters for the x, y, and z coordinates.
    rx, ry, rz : float
        Rotation parameters (in arc seconds) for the x, y, and z axes.
    f : float
        Scale factor in parts per million.

    Returns
    -------
    Point
        The transformed point in geocentric coordinates.
    """
    
    vx = point.x - tx
    vy = point.y - ty
    vz = point.z - tz
    rx = rx * math.pi / 648000
    ry = ry * math.pi / 648000
    rz = rz * math.pi / 648000
    e = 1 + (f / 1000000)
    det = e * (math.pow(e, 2) + math.pow(rx, 2) + math.pow(ry, 2) + math.pow(rz, 2))
    x2 = ((math.pow(e, 2) + math.pow(rx, 2)) * vx + (e * rz + rx * ry) * vy + (rx * rz - e * ry) * vz) / det
    y2 = ((-e * rz + rx * ry) * vx + (e * e + math.pow(ry, 2)) * vy + (rx * e + rz * ry) * vz) / det
    z2 = ((e * ry + rx * rz) * vx + (-e * rx + rz * ry) * vy + (e * e + math.pow(rz, 2)) * vz) / det
    p = Point(x2, y2, z2)
    return p


def clean_geodataframe(gdf): 
    """
    Cleans a GeoDataFrame by:
    - Removing rows with missing geometries.
    - Fixing invalid geometries using buffering.
    - Filtering out invalid geometries after the fix.
    - Resetting the index.
    
    Args:
        gdf (GeoDataFrame): Input GeoDataFrame.

    Returns:
        GeoDataFrame: Cleaned GeoDataFrame.
    """
    # Remove rows where the geometry column is NaN
    gdf = gdf[gdf['geometry'].notna()] 
    
    # Fix invalid geometries by applying a zero-width buffer
    gdf['geometry'] = gdf.buffer(0) 
    
    # Keep only rows with valid geometries
    gdf = gdf[gdf.is_valid] 
    
    # Reset the index to be consecutive and drop the old index
    gdf.reset_index(inplace=True, drop=True) 
    
    return gdf



def ProjectLayer_sterioToWgs84(Layer, output):
    """
    Projects a geographic layer from stereographic to WGS 84 coordinates, supporting
    transformations into both UTM and geographic coordinate systems.

    Parameters
    ----------
    Layer : str
        The file path to the layer containing geographic data.
    output : str
        The output path where the transformed data will be saved.

    Notes
    -----
    The function handles both Polygon and MultiPolygon geometries, converting them
    through a series of transformations and saving the results in both UTM and
    WGS 84 formats.
    """
    # WGS 84 to UTM Zone 36N
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32636', always_xy=True, accuracy=1)
    name = GetFeatureName(output)
    path = GetFeaturePath(output)
    sr_utm = CRS.from_epsg(32636)  # UTM Zone 36N
    sr_wgs = CRS.from_epsg(4326)   # WGS 84

    # Load and clean the input GeoDataFrame
    gdf = gpd.read_file(Layer)
    gdf = clean_geodataframe(gdf)  # Clean input GeoDataFrame
    type = gdf.geometry.geom_type[0]
    new_gdf_utm = gpd.GeoDataFrame(columns=['geometry'], crs=sr_utm)
    new_gdf_wgs = gpd.GeoDataFrame(columns=['geometry'], crs=sr_wgs)

    if type == 'Polygon' or type == 'MultiPolygon':
        for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
            s = row.geometry
            newfeature_utm = []
            newfeature_wgs = []
            if s.geom_type == 'Polygon':
                polygons = [s]  
            elif s.geom_type == 'MultiPolygon':
                polygons = list(s.geoms)  
            for polygon in polygons:
                newpart_utm = []
                newpart_wgs = []
                for point in polygon.exterior.coords:
                    if point is not None:
                        point2 = SterioToWgs84(Point(point)) 
                        # Use transformer to project point to UTM
                        point_utm = transformer.transform(point2.x, point2.y)
                        point_wgs = (point2.x, point2.y)
                        newpart_utm.append(point_utm)
                        newpart_wgs.append(point_wgs)
                newfeature_utm.append(Polygon(newpart_utm))
                newfeature_wgs.append(Polygon(newpart_wgs))
            pol_utm = MultiPolygon(newfeature_utm) if len(newfeature_utm) > 1 else newfeature_utm[0]
            pol_wgs = MultiPolygon(newfeature_wgs) if len(newfeature_wgs) > 1 else newfeature_wgs[0]
            new_gdf_utm.loc[len(new_gdf_utm)] = [pol_utm]
            new_gdf_wgs.loc[len(new_gdf_wgs)] = [pol_wgs]

    # Clean the output GeoDataFrames
    new_gdf_utm = clean_geodataframe(new_gdf_utm)
    new_gdf_wgs = clean_geodataframe(new_gdf_wgs)

    # Save the cleaned GeoDataFrames
    new_gdf_utm.to_file(path + name + '_utm.shp', driver='ESRI Shapefile')
    new_gdf_wgs.to_file(path + name + '_wgs.shp', driver='ESRI Shapefile')



    
    
input_shapefile = ""
output_path = ""

ProjectLayer_sterioToWgs84(input_shapefile, output_path)


