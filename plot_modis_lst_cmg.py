import os ; import numpy as np ; import datetime as dt ; from dateutil import rrule ; import glob; import pandas as pd ; 

from pyhdf.SD import SD, SDC ; from pyhdf import HDF,VS #, V  ; 

import matplotlib.pyplot as plt ; import matplotlib.colors as mcolors

import cartopy ; import cartopy.crs as crs ; from cartopy.feature import NaturalEarthFeature ;

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import pprint
import calendar

file_name='/home/vkvalappil/Data/modis_data/modis_lst/MOD11C1.A2017335.006.2017336085723.hdf'
        
           sdf_f = SD(file_name, SDC.READ)          
           datasets_dic = sdf_f.datasets() 
#           for idx,sds in enumerate(datasets_dic.keys()):
#               print idx,sds

           dset = sdf_f.select('LST_Day_CMG')  
           lst_d = dset[:,:] 

           # Read attributes.
           attrs_c = dset.attributes(full=1) ; 
           lna_c=attrs_c["long_name"] ; long_name_c = lna_c[0]
           sfa_c=attrs_c["scale_factor"] ;    scale_factor_c = sfa_c[0]        
           vra_c=attrs_c["valid_range"] ; valid_min_c = vra_c[0][0]        
           valid_max_c = vra_c[0][1]      ; 
           offset_c = attrs_c['add_offset'][0]

           invalid_c = np.logical_or(lst_d < valid_min_c, lst_d > valid_max_c)
           lstd_f = lst_d.astype(float)
           lstd_f[invalid_c] = np.nan
           # Apply scale factor according to [1].
           lst_d_f = (lstd_f - offset_c) * scale_factor_c
           lst_d_f =np.flipud(lst_d_f)
           lat=np.arange(-90,90,0.05) ; lon=np.arange(-180,180,0.05)
           
             

           
#for key, value in sds_obj.attributes().iteritems():
#    print key, value
#    if key == 'add_offset':
#        add_offset = value  
#    if key == 'scale_factor':
#        scale_factor = value           
        

    lons,lats =np.meshgrid(lon,lat)

    fig = plt.figure(figsize=(12,9)) ; ax = plt.axes(projection=crs.PlateCarree())
    ax.set_extent([50, 60,20,30])

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',name='admin_1_states_provinces_shp')
    ax.add_feature(states, linewidth=0.5) ; ax.coastlines('50m', linewidth=0.8)

    shp_file='/home/vkvalappil/Data/shapeFiles/uae/UAE_map/Export_Output.shp'
    shape_feature = cartopy.feature.ShapelyFeature(cartopy.io.shapereader.Reader(shp_file).geometries(), crs.PlateCarree(), facecolor='none')
    ax.add_feature(shape_feature,linewidth=1.5,color='k')

    levels = [280,288,291,295,298,301,305,308,311,314,317,320]
    tmp_contours = plt.contourf(lons, lats, lst_d_f,levels=levels,cmap=get_cmap("Oranges"),origin='lower',extend='both',transform=crs.PlateCarree())

    tmp_contours.cmap.set_under('forestgreen') ;    tmp_contours.cmap.set_over('yellow')

    plt.colorbar(tmp_contours, ax=ax, orientation="horizontal", pad=.05,shrink=0.60)

    ax.gridlines()

    plt.title(" MODIS Land Surface Temperature (K) ")

    o_file=out_file_path+'/modis_lst.png'
    plt.savefig(o_file)
    plt.close()



 
 

c
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 