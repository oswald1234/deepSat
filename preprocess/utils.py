import geopandas
import os

from geocube.api.core import make_geocube
from functools import partial
from geocube.rasterize import rasterize_image
from preprocess.classDict import class_dict
import shutil
import numpy
import glob


def rasterize(patch_labls):
    
    patch_labls['train_id'] = patch_labls['code_2018'].apply(lambda x: numpy.uint16(class_dict[str(x)]['train_id']))
    patch_labls['class_code'] = patch_labls['code_2018'].apply(lambda  x:  numpy.uint16(int(x)))
    roads = patch_labls[patch_labls.code_2018.isin(['12210','12220','12230'])]
    
    cube  = make_geocube(
        vector_data=patch_labls,
        measurements=['class_code','train_id'],
        resolution=(10,-10),
        rasterize_function=partial(rasterize_image,dtype=numpy.dtype('uint16')),
        fill = 0
    )
    
    roads = make_geocube(
        vector_data=roads,
        measurements=['class_code','train_id'],
        resolution=(10,-10),
        rasterize_function=partial(rasterize_image,all_touched=True,dtype=numpy.dtype('uint16')),
            fill=0
    )
    
    cube = roads.where(roads.train_id!=0).combine_first(cube)  #merge roads  
    cube = cube.assign(train_id = lambda ds: ds.train_id.astype('uint16'))
    cube = cube.assign(class_code = lambda ds: ds.class_code.astype('uint16'))
    
    return(cube)
    
    # spatialy joins. returns obj (geodataframe) within  roi (geodataframe)
def get_obj_within(obj,roi):
    return obj[(obj.id.isin(obj.sjoin(roi,predicate='within').id))]
    

def save_cube(cube,savedir):
                # patch filename and 
    filename = "PATCH_{i_d}_{tile}_{start}_{fua}_{size}.nc".format(i_d=cube.attrs['patch_id'],
                                                                   tile=cube.attrs['Tile'],
                                                                   start=cube.attrs['sensing_start'],
                                                                   fua=cube.attrs['FUA'],
                                                                   size=cube.dims['x'])
              
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    cube.to_netcdf(path=os.path.join(savedir,filename),format='NETCDF4')



def org_files(df,mode='train'):
    for path in df:
        source = path
        dirn,fn = os.path.split(path)
        timeperiod = dirn.split('/')[-3] 
        prod = dirn.split('/')[-4] 
        root = dirn.split(prod)[0]
        prod = prod + '_split'
        dest_dir = os.path.join(root,prod,timeperiod,mode)
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            print(dest_dir)
        destcop = shutil.copy(source, os.path.join(dest_dir,fn))

        
def rename_ext(df, ext='.h5'):
    for path in df:
        fn = os.path.basename(path)
        dirn = os.path.dirname(path)
        new_fn = fn.split('.')[0] + ext
        new_path = os.path.join(dirn,new_fn)
        os.rename(path,new_path)
        
        
def str_to_oao(string: str):
    #test_list = ['å','ä','ö']
    replace_list= {'å':"a",
                   'ä':"a",
                   'ö':"o"}
    res = [ele for ele in ['å','ä','ö'] if(ele in string)]
    for ele in res:
        string = string.replace(ele,replace_list[ele])
    return(string.upper())
