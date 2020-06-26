#/usr/bin/python

import os ; import numpy as np ; import datetime as dt ; from dateutil import rrule ; import glob; import pandas as pd ; 

from pyhdf.SD import SD, SDC ; from pyhdf import HDF,VS #, V  ; 

#import pprint ; 

from mpl_toolkits.axes_grid1 import make_axes_locatable ; from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt ; import matplotlib as mpl ; import matplotlib.cm as cm
##############################################################################################################

main='/home/masdar-fog/vvalappil/cloudsat/'

date='20160101'

date_1=dt.datetime.strptime(date,'%Y%m%d')+dt.timedelta(days=0)
date_2=date_1+dt.timedelta(days=364) ;           
date_list=[x.strftime('%Y%m%d') for x in rrule.rrule(rrule.DAILY,dtstart=date_1,until=date_2)]
################################################################################################################# 
 
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
################################################################################################################# 
cloudy_date=[] ;cloudy_date1=[] ;

for d in date_list: 
        
    dte=dt.datetime.strptime(d,'%Y%m%d') ;  year= dte.timetuple().tm_year
    day_y= dte.timetuple().tm_yday 
    
    print(dte)
    file_path=main+str(year)+'/'+str(day_y).zfill(3)

    if (os.path.exists(file_path)):
        print("file_exist")
        file_list=glob.glob(file_path+'/*.hdf')
        for f in file_list:
            
            file_name =f   
            
            hdf_f = HDF.HDF(file_name, SDC.READ) 
            vs = hdf_f.vstart()
            #data_info_list = vs.vdatainfo() ; pprint.pprint( data_info_list )
            
            lat = np.array(vs.attach('Latitude')[:]) ; lon = np.array(vs.attach('Longitude')[:])  ; 
            tim=np.array(vs.attach('Profile_time')[:])
            hdf_f.close()
 
 
            h = HDF.HDF(file_name)
            vs = h.vstart()

#            xid = vs.find('Latitude')
#            latid = vs.attach(xid)
#            latid.setfields('Latitude')
#            nrecs, _, _, _, _ = latid.inquire()
#            latitude = latid.read(nRec=nrecs)
#            latid.detach()
#
#            lonid = vs.attach(vs.find('Longitude'))
#            lonid.setfields('Longitude')
#            nrecs, _, _, _, _ = lonid.inquire()
#            longitude = lonid.read(nRec=nrecs)
#            lonid.detach()
#
            timeid = vs.attach(vs.find('Profile_time'))
            timeid.setfields('Profile_time')
            nrecs, _, _, _, _ = timeid.inquire()
            time = timeid.read(nRec=nrecs)
            units_t =  timeid.attr('units').get()
            longname_t = timeid.attr('long_name').get()
            timeid.detach() 
 
 
#################################################################################################################################          
            sdf_f = SD(file_name, SDC.READ)          
            datasets_dic = sdf_f.datasets() 

            dset = sdf_f.select('CPR_Cloud_mask') # select sds
            #cloud_mask = sds_obj.get()
            cloud_mask = dset[:,:]
            
            # Read attributes.
            attrs_c = dset.attributes(full=1) ; 
            lna_c=attrs_c["long_name"] ; long_name_c = lna_c[0]
            sfa_c=attrs_c["factor"] ;    scale_factor_c = sfa_c[0]        
            vra_c=attrs_c["valid_range"] ; valid_min_c = vra_c[0][0]        
            valid_max_c = vra_c[0][1]      ; 
#            ua=attrs_c["units"] ; units = ua[0]            
            # Process valid range.
            
            invalid_c = np.logical_or(cloud_mask < valid_min_c, cloud_mask > valid_max_c)
            cloud_maskf = cloud_mask.astype(float)
            cloud_maskf[invalid_c] = np.nan
            # Apply scale factor according to [1].
            cloud_maskf = cloud_maskf / scale_factor_c

#######################################################################################################################################           
            
            dset_r=sdf_f.select('Radar_Reflectivity')
            rad_refl=dset_r[:,:]    

            attrs_r = dset_r.attributes(full=1)
            lna_r=attrs_r["long_name"] ;  long_name_r = lna_r[0]
            sfa_r=attrs_r["factor"]    ;  scale_factor_r = sfa_r[0]        
            vra_r=attrs_r["valid_range"] ; valid_min_r = vra_r[0][0] ; valid_max_r = vra_r[0][1]        
            ua_r=attrs_r["units"] ; units_r = ua_r[0]            

            invalid_r = np.logical_or(rad_refl < valid_min_r, rad_refl > valid_max_r)
            rad_reflf = rad_refl.astype(float)
            rad_reflf[invalid_r] = np.nan
            # Apply scale factor according to [1].
            rad_reflf = rad_reflf / scale_factor_r
            
########################################################################################################################################            

            dset_h=sdf_f.select('Height')
            #hgt=sds_obj1.get()   
            hgt = dset_h[:,:]
                        
            attrs_h = dset_h.attributes(full=1)
            uah=attrs_h["units"]
            units_h = uah[0]            
            


############################################################################################################################################

            # Make a split window plot.
#            fig = plt.figure(figsize = (10, 10)) ; ax1 = plt.subplot(3, 1, 1)  
#            cmap=plt.get_cmap('RdYlGn_r')
#            t, h = np.meshgrid(tim, hgt[0,:])
#            im = ax1.contourf(t, h, rad_reflf.T,cmap=cmap)
#            ax1.set_xlabel(longname_t+' ('+units_t+')')
#            ax1.set_ylabel('Height ('+units_h+')')
#            basename = os.path.basename(file_name)
#            ax1.set_title('{0}\n{1}'.format(basename,  long_name_r))
#            cb = plt.colorbar(im) ; cb.set_label(units_r)
##############################################################################################################################################
#            ax1 = plt.subplot(3, 1, 2)  
#            #cmap=plt.get_cmap('RdYlGn_r')
#            cmap = [(1.0,1.0,1.0)] + [(0.1,0.1,0.1)]
#            cmap = mpl.colors.ListedColormap(cmap)   
#            bounds = [0,20,40]
#            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)                             
#            t, h = np.meshgrid(tim, hgt[0,:])
#            im1 = ax1.contourf(t, h, cloud_maskf.T,cmap=cmap,norm=norm)
#            ax1.set_xlabel(longname_t+' ('+units_t+')')
#            ax1.set_ylabel('Height ('+units_h+')')
#            basename = os.path.basename(file_name)
#            ax1.set_title('{0}\n{1}'.format(basename,  long_name_c))
#            cbar_ticks = [10,20,30]   ; cbar_ticks_labels = ['','20','Cloudy']
#            cb1 = plt.colorbar(im1,fraction=0.01, cmap=cmap, norm=norm, boundaries=bounds, ticks=cbar_ticks) ; 
#            cb1.ax.set_yticklabels(cbar_ticks_labels, fontsize=8)
############################################################################################################################################
#            # The 3rd plot is the trajectory.
#            ax1 = plt.subplot(3, 1, 3)
#            m = Basemap(projection='cyl', resolution='l',llcrnrlat=-90, urcrnrlat = 90,llcrnrlon=-180, urcrnrlon = 180)
#            m.drawcoastlines(linewidth=0.5)
#            m.drawparallels(np.arange(-90, 91, 45),labels=[True,False,False,True])
#            m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
#            # x, y = m(longitude, latitude)
#            m.plot(lon, lat, linestyle='None', marker='.', color='blue', latlon=True)
#
#            # Annotate the starting point.
#            m.plot(lon[0], lat[0], marker='o', color='red')
#
#            plt.title('Trajectory of Flight Path (starting point in red)')
#
#            fig = plt.gcf()
#            pngfile = file_path+'/'+"{0}.png".format(basename)
#            fig.savefig(pngfile)
#            plt.close()
######################################################################################################################################
            
            lat_1=10 ; lat_2=35 ; lon_1=35 ; lon_2=60

            idx=((lat >=lat_1) & (lat<=lat_2)) & ((lon>=lon_1) & (lon<=lon_2))
            idx1=np.tile(idx,125)
            
            lat_r=lat[idx] ; lon_r=lon[idx] ; cloud_mask_r=cloud_maskf[idx1]; tim_r=tim[idx]
            cloud_mask_r=cloud_mask_r.reshape(lat_r.shape[0],125)                                     
            hgt_r=hgt[idx1] ; hgt_r=hgt_r.reshape(lat_r.shape[0],125)
            
            if hgt_r.shape[0] >0:
                print("Domain have cloud profiles"); print(f) ; 
                invalid_hgt = np.logical_not(hgt_r > 0)
                hgt_r[invalid_hgt]=0 ; cloud_mask_r[invalid_hgt]=0 ; 
             
                valid_cld=np.logical_not(cloud_mask_r <= 20)  #cld_indx=(cloud_mask_r >20)           
                invalid_cld=np.logical_not(cloud_mask_r > 20)  #cld_indx=(cloud_mask_r <=20)           
                
                cloud_mask_r_v=np.copy(cloud_mask_r) ; cloud_mask_r_v[invalid_cld]=np.nan
                if np.count_nonzero(~np.isnan(cloud_mask_r_v)) >0 :
                    cloudy_file=f
                    f_split=f.split('/')
                    cloudy_date.append(f_split[-1][0:19])
                    cloudy_date1.append(d)
                    
                    cloud_mask_r_v_1=np.copy(cloud_mask_r_v) ; cloud_mask_r_v_1[invalid_cld]=0
#########################################################################################################################################

                    fig = plt.figure(figsize = (10, 10)) ; ax1 = plt.subplot(2, 1, 1)  
                    #cmap=plt.get_cmap('RdYlGn_r')
                    cmap = [(1.0,1.0,1.0)] + [(0.0,0.0,0.0)]
                    cmap = mpl.colors.ListedColormap(cmap)   
                    bounds = [0,20,40]
                    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)                             
                    t, h = np.meshgrid(tim_r, hgt_r[0,:])
                    im1 = ax1.contourf(t, h, cloud_mask_r_v_1.T,cmap=cmap,norm=norm)
                    ax1.set_xlabel(longname_t+' ('+units_t+')')
                    ax1.set_ylabel('Height ('+units_h+')')
                    basename = os.path.basename(file_name)
                    ax1.set_title('{0}\n{1}'.format(basename,  long_name_c))
                    cbar_ticks = [10,20,30]   ; cbar_ticks_labels = ['','20','Cloudy']
                    cb1 = plt.colorbar(im1,fraction=0.01, cmap=cmap, norm=norm, boundaries=bounds, ticks=cbar_ticks) ; 
                    cb1.ax.set_yticklabels(cbar_ticks_labels, fontsize=8)
###########################################################################################################################################
                    # The 3rd plot is the trajectory.
                    ax2 = plt.subplot(3, 1, 3)
                    m = Basemap(projection='cyl', resolution='l',llcrnrlat=-90, urcrnrlat = 90,llcrnrlon=-180, urcrnrlon = 180)
                    m.drawcoastlines(linewidth=0.5)
                    m.drawparallels(np.arange(-90, 91, 45),labels=[True,False,False,True])
                    m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
                    # x, y = m(longitude, latitude)
                    m.plot(lon_r, lat_r, linestyle='None', marker='.', color='blue', latlon=True)

                    # Annotate the starting point.
                    #m.plot(lon[0], lat[0], marker='o', color='red')

                    m.plot(35, 10, marker='o', color='red')
                    m.plot(60, 35, marker='o', color='red')
                    
                    m.plot(35, 35, marker='o', color='red')
                    m.plot(60, 10, marker='o', color='red')
                    plt.title('Trajectory of Flight Path (starting point in red)')
                    pngfile = file_path+'/'+"{0}.png".format(basename)
                    plt.savefig(pngfile)
                    plt.close()
############################################################################################################################################            
cloudy_date=pd.DataFrame(cloudy_date)  
cloudy_date1=pd.DataFrame(cloudy_date1)
cloudy_date2=pd.concat([cloudy_date,cloudy_date1],axis=1)          
cloudy_date2.to_csv(file_path+'cloudy_date_'+date[0:4]+'.csv')

#quit()            
#            fig = plt.figure() ; ax = fig.add_subplot(111)
#
#            cmap = [(0.0,0.0,0.0)] + [(0.75,0.75,0.75)]
#            cmap = mpl.colors.ListedColormap(cmap)
#    
#            bounds = [0,20,40]
#
#            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#
#            img = plt.imshow(cloud_mask_r_v.T, extent=[0,1000,0,25], aspect='auto',cmap=cmap, norm=norm, interpolation='none')
#
#            cbar_bounds = [0,20,40] ;  cbar_ticks = [10,30]   
#            cbar_ticks_labels = ['', 'Cloudy']
#               
#            cbar = plt.colorbar(img, fraction=0.01, cmap=cmap, norm=norm, boundaries=cbar_bounds, ticks=cbar_ticks)
#            cbar.ax.set_yticklabels(cbar_ticks_labels, fontsize=8)
#
#            plt.title('CloudSat CPR cloud mask', fontsize=8)
#
#            forceAspect(ax,aspect=3)
#
#            plt.xticks(fontsize=8)
#            plt.yticks(fontsize=8)
#            plt.show()           
#
#            plt.savefig("cloudsat_cpr_cloud_mask.png", bbox_inches='tight', dpi=100)
            
            
            