from shapely.geometry import Polygon,MultiPolygon
import types
​
def pix_to_utm(crd,A,B,C,D,E,F):
    #convert a single pixel's (x,y) coordinates into the georeferenced coordinates
    #A,B,C,D,E,F are parameters that can be read from the TFW file ...
    # associated with the TIFF File that The Tile was taken from
    x,y=crd
    return x*A+y*B+C,x*D+y*E+F
​
def utm_to_pix(crd,A,B,C,D,E,F):
    #convert a single pixel's georeferenced coords (lon,lat) coordinates into the pixel's coords
    x1,y1 = crd
	return (E * x1 + B * y1 + B * F - E * C)/(A * E - D * B),(-D * x1 + A * y1 + D * C - A * F )/(A *E  - D * B)
​
def tf(func,poly,A,B,C,D,E,F):
    #apply a transformation to a list of coordinates
    #transformation options ( pix_to_utm & utm_to_pix)
    k=[]
    for tup in poly:
        k.append(func(tup,A,B,C,D,E,F))
    return k
​
def tf_polygon(func,poly,A,B,C,D,E,F):
    #apply transformation to shapely.geometry polygon
    k = tf(func,list(poly.exterior.coords),A,B,C,D,E,F)
    return Polygon(k)
​
def tf_multipoly(func,mpoly,A,B,C,D,E,F):
    #apply transformation to shapely.geometry multi-polygon
    k=[]
    for poly in list(mpoly):
        k.append(tf_polygon(func,poly,A,B,C,D,E,F))
    return MultiPolygon(k)
​
def _tf(func,coords,A,B,C,D,E,F):
    try:
        typ = coords.geom_type
    except AttributeError:
        if(type(coords) is list):
            typ = 'List'
        else:
            typ='ínvalid'
    except:
        typ='ínvalid'
    if(typ == 'ínvalid'):
        raise AttributeError('INVALID TYPE FOR COORDS')
    elif(typ == 'MultiPolygon'):
        return tf_multipoly(func,coords,A,B,C,D,E,F)
    elif(typ == 'Polygon'):
        return tf_polygon(func,coords,A,B,C,D,E,F)
    else:
        return tf(func,coords,A,B,C,D,E,F)
    
def parse_tfw(path): 
    #given the path of the .TFW  File, this function returs A,B,C,D,E,F params ...
    #needed for the conversion between pixel and georeferenced coordinates
     with open(path,'r') as lol:
         a=lol.read()
         a=a.split('\n')
     permutation = [0,2,4,1,3,5]
     #return [float(a[i].strip('      ')) for i in permutation]
     #or
     return list(map(lambda i : float(a[i].strip('      ')),permutation))
​
def tf_upper(polys,x,y):
    #given a list of polygons   
    #and a list of x & y offsets from top left corner
    #this function adds the offsets to the polygon's pixel coords
    ipolys=[]
    for i,poly in enumerate(polys):
        k=[]
        for tup in poly:
            k.append((x[i]+tup[0],y[i]+tup[1]))
        ipolys.append(k)
    npolys =[Polygon(poly) for poly in ipolys]
    return npolys
    
def tf_utm(polys,func,A,B,C,D,E,F):
    #transform all the polys in a list from pixel to georeferenced coordinates
    #the input polygon should have pixel coordinates w.r.t top-left corner 
    #( i.e : the offsets should be added using previous function (tf_upper))
    ipolys=[]
    for poly in polys:
        ipolys.append(_tf(func,poly,A,B,C,D,E,F))
    return ipolys