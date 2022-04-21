import os
import glob
import geopandas as gp


    
## returns layer names for urban atlas gpkg files
def get_layers(filepath):
    basename ="_".join(os.path.basename(filepath).split('_')[0:3])
    urbanBoundary =  basename + "_UrbanCore"
    baseboundary = basename + "_Boundary"
    return (basename,baseboundary,urbanBoundary)


#open UA data and return FUA class labels and boundary layer (geopandas)
def open_fua(path):
    # get layer names (from filename)
    labels,boundary,_ = get_layers(path)
    # open layers
    fua_labls = gp.read_file(path,layer = labels)
    fua_boundary = gp.read_file(path,layer=boundary)
    return(fua_labls,fua_boundary)


