import os
import glob
import rioxarray as rio
import rasterio
import numpy

def get_tile_info(tile_name,sentinel_path):
	prod,date = search_product(sentinel_path,tile_name)
	return(get_prod_info(prod[0]))

def get_prod_info(SAFE):
    prod_info={}
    SAFE=os.path.basename(SAFE)
   
    prod_info['s2_file_name']=SAFE
    SAFE=SAFE.split('_')
    #prod_info['mission_id']=SAFE[0]
    #prod_info['product_level']=SAFE[1]
    prod_info['sensing_start']=SAFE[2]
    #prod_info['PDGS']=SAFE[3]
    prod_info['Relative_orbit']=SAFE[4]
    prod_info['Tile']=SAFE[5]
    prod_info['prod_discriminator']=SAFE[6].split('.')[0]
    return(prod_info) 
    
def get_jp2_path(prod_path,bands):
    src = os.path.join(prod_path,'GRANULE/*/IMG_DATA/*_{}.jp2'.format(bands))
    return(glob.glob(src)) 
    
    
#takes in tile name and returns products for tile
#
def search_product(path,tile_name='*',source='*',date = '*',ext='tif'):
    
    if not source =='*':
        source = '*_'+source
        
    src = os.path.join(path,'S2*_MSIL*_{}_*_*_T{}_{}.{}'.format(date,tile_name,source,ext))

    found_prod = [i for i in glob.glob(src)]
    found_dates = [get_prod_info(product)['sensing_start'] for product in found_prod]
    return((found_prod,found_dates))
    
    
def clip_tile(tile,clip_shape):
    bounds  = clip_shape.to_crs(tile.rio.crs).total_bounds
    patch = tile.rio.clip_box(minx=bounds[0],
             miny=bounds[1],
             maxx=bounds[2],
             maxy=bounds[3])
    return(patch)

def open_tile(tile_name,sentinel_path,source='*',bands='TCI',ext='tif'): 
    
    prod,date = search_product(sentinel_path,tile_name=tile_name,source=source,ext=ext)
    if os.path.splitext(prod[0])[1] == '.SAFE':
        src = get_jp2_path(prod[0],bands)[0]
    else:
        src = prod[0]
    with rio.open_rasterio(src) as rds:
        rds.name='raw'
        return(rds)

#def open_clip_tile(tile_name,curr_patch,sentinel_path,tidsperiod):
#    prod,date = search_product(sentinel_path,tile_name)
#    src = get_jp2_path(prod[0],'TCI')[0]
#    #rds = rioxarray.open_rasterio(src,dtype=numpy.uint8)
#    with rio.open_rasterio(src,dtype=numpy.uint16) as rds:
#       #rds = clip_tile(rds,curr_patch)
        #rds.attrs['type']='TCI'
        #rds = rds.to_dataset('band') #convert to dataset 
        #rds=rds.rename({1:'R',2:'G',3:'B'}) #rename variables
        
#        
#    return(rds)

  
#s2_folder = os.path.join(sentinel_path,'sentinel-2')   
### not used remove later   
#def open_bands(SAFE_fn,bands,clip_shape,sentinel_path):
#    s2_patches = []
#    for i,band in enumerate(bands):
#        src = os.path.join(os.path.join(sentinel_path,'{}/GRANULE/*/IMG_DATA/*_{}.jp2'.format(SAFE_fn,band))
#        s2_prod_path = glob.glob(src)[0]
#        try: #open s2 tile and clip
#            s2_patch = open_s2_tile_clipped_vir(s2_prod_path, clip_shape)
#            s2_patch.name = band
#            print(band)
#            s2_patches.append(s2_patch)
#        except ValueError:
#            print("Exception: Input patch id: {} do not overlap tile {}".format(patch.id.values[0],SAFE_fn))
#        except NameError:
#            print(NameError)
#        if i>1:
#            break
#    return(s2_patches)
    
    
### not used remove later   
#def open_TCI(prod_path,clip_shape):
#        src= get_jp2_path(prod_path,'TCI')[0]
#        try: #open s2 tile and clip
#            s2_patch = open_s2_tile_clipp2(src,clip_shape)
#        except ValueError:
#            print("Exception: Input patch id: {} do not overlap tile {}".format(clip_shape.id.values[0],prod_path))
#        except NameError:
#            print(NameError)
#        return(s2_patch)


### not used remove later
#def open_s2_tile_clipp(tile_path,clip_shape):
#    env = rasterio.Env(
#        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
#        CPL_VSIL_CURL_USE_HEAD=False,
#        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="TIF",
#    )
#    with env:
#        with rasterio.open(tile_path) as src:
#            with rasterio.vrt.WarpedVRT(src, crs="EPSG:3035",) as vrt:
#                rds = rio.open_rasterio(vrt,dtype=numpy.uint16).rio.clip_box(patch.geometry, from_disk=False,all_touched=True)
#    return(rds)


### not used remove later 
#def open_s2_tile_clipp2(tile_path,patch):
#    env = rasterio.Env(
#        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
#        CPL_VSIL_CURL_USE_HEAD=False,
#        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="TIF",
#    )
#    with env:
#        with rasterio.open(tile_path) as src:
#            with rasterio.vrt.WarpedVRT(src, crs="EPSG:3035",) as vrt:
#                rds = rio.open_rasterio(vrt,dtype=numpy.uint16,chunks={'x':512,'y':512}).rio.clip_box(
#                    minx=patch.total_bounds[0],
#                    miny=patch.total_bounds[1],
#                    maxx=patch.total_bounds[2],
#                    maxy=patch.total_bounds[3]
#                )
#    return(rds)


### not used remove later
#Reproject-Large-Rasters-with-Virtual-Warping
#https://corteva.github.io/rioxarray/stable/examples/reproject.html
# reproject file and open clip
#def open_s2_tile_clipped_vir(tile_path,clip_shape):
#    env = rasterio.Env(
#        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
#        CPL_VSIL_CURL_USE_HEAD=False,
#        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="TIF",
#    )
#    with env:
#        with rasterio.open(tile_path) as src:
#            with rasterio.vrt.WarpedVRT(src, crs="EPSG:3035",) as vrt:
#                rds = rio.open_rasterio(vrt,chunks={'x':1000,'y':1000},lock=True).rio.clip(patch.geometry, from_disk=False,all_touched=True)
#    return(rds)
    
