 #!/usr/bin/python

from __future__ import print_function
import ncepbufr
from netCDF4 import Dataset
import numpy as np
import sys, datetime
from utils import quantize

#hdstr1 ='SAID SIID FOVN YEAR MNTH DAYS HOUR MINU SECO CLAT CLON CLATH CLONH HOLS'
hdstr2 ='SAZA SOZA BEARAZ SOLAZI'
hdstr1 ='SAID SIID FOVN YEAR MNTH DAYS HOUR MINU SECO CLATH CLONH' # SAZA BEARAZ SOZA SOLAZI'


# input and output file names from command line args.
bufr_filename ='/home/vkvalappil/Data/modelWRF/input/2017090118/gdas.t18z.mtiasi.tm00.bufr_d' 
nc_filename ='/home/vkvalappil/Data/modelWRF/input/iasi_18.nc' 

if bufr_filename == nc_filename:
    raise IOError('cannot overwrite input bufr file')
########################################################################################################
bufr = ncepbufr.open(bufr_filename)
# get number of channels from 1st message in bufr file.
nchanl = None
while bufr.advance() == 0:
    while bufr.load_subset() == 0:
        obs = bufr.read_subset('SCRA',rep=True).squeeze()
        nchanl = len(obs) ; print(nchanl)
        break
    if nchanl is not None: break
bufr.rewind()
######################################################################################################

f = Dataset(nc_filename,'w',format='NETCDF4_CLASSIC')
# create netCDF variables.
obsdim = f.createDimension('nobs',None)
complevel = 4 # compression level
nobs_chunk = 20000 # write data in nobs_chunk chunks
# chunking and compression parameters for all variables.
kwargs = {'zlib':True,'shuffle':True,'complevel':complevel,'chunksizes':(nobs_chunk,)}
#sat_idv = f.createVariable("sat_id","i4","nobs",**kwargs)
#sensor_idv = f.createVariable("sensor_id","i4","nobs",**kwargs)
latv = f.createVariable("lat","f4","nobs",**kwargs)
lonv = f.createVariable("lon","f4","nobs",**kwargs)
yyyymmddhhmmssv = f.createVariable("yyyymmddhhmmss","f8","nobs",**kwargs)
#land_surface_heightv = f.createVariable("land_surface_height","f4","nobs",**kwargs)
#field_of_view_numberv = f.createVariable("field_of_view_number","i4","nobs",**kwargs)
#sat_zenith_anglev = f.createVariable("sat_zenith_angle","f4","nobs",**kwargs)
#solar_zenith_anglev = f.createVariable("solar_zenith_angle","f4","nobs",**kwargs)
#solar_azimuth_anglev = f.createVariable("solar_azimuth_angle","f4","nobs",**kwargs)
#local_azimuth_anglev = f.createVariable("local_azimuth_angle","f4","nobs",**kwargs)
channel_numberv = f.createVariable("channel_number","i2","nobs",**kwargs)
tbv = f.createVariable("tb","f4","nobs",**kwargs)
vnames = f.variables.keys()

# these will be filled in later (merged from GSI diagnostic file)
#tb_modelv = f.createVariable("tb_model","f4","nobs",**kwargs)
#tb_biascorrv = f.createVariable("tb_biascorr","f4","nobs",**kwargs)
#water_fracv = f.createVariable("water_frac","f4","nobs",**kwargs)
#ice_fracv = f.createVariable("ice_frac","f4","nobs",**kwargs)
#snow_fracv = f.createVariable("snow_frac","f4","nobs",**kwargs)
#oberr_origv = f.createVariable("oberr_orig","f4","nobs",**kwargs)
#oberr_usedv = f.createVariable("oberr_used","f4","nobs",**kwargs)
#use_flagv = f.createVariable("use_flag","i1","nobs",**kwargs)
#qc_flagv = f.createVariable("qc_flag","i1","nobs",**kwargs)

# dict holding netcdf variable instances and corresponding numpy arrays
var_dict = {}
for vname in vnames:
    var_dict[vname] = f[vname],np.empty(nobs_chunk, f[vname].dtype)

nob = 0; nobtot = 0; nob1 = 0
obid_set = set()

while bufr.advance() == 0:
    #print(bufr.msg_counter, bufr.msg_type, bufr.msg_date)
    while bufr.load_subset() == 0:
        #print(bufr.msg_counter, bufr.msg_type, bufr.msg_date)

        hdr1 = bufr.read_subset(hdstr1).squeeze()           
        hdr2 = bufr.read_subset(hdstr2).squeeze()
        hour = int(hdr1[6])
        min = int(hdr1[7])
        sec = int(hdr1[8])
        if sec == 60:
           sec = 59
           d = datetime.datetime(int(hdr1[3]),int(hdr1[4]),int(hdr1[5]),hour,min,sec)
           d = d + datetime.timedelta(seconds=1)
        else:
           d = datetime.datetime(int(hdr1[3]),int(hdr1[4]),int(hdr1[5]),hour,min,sec)
        yyyymmddhhmmss = d.strftime('%Y%m%d%H%M%S') ; print(yyyymmddhhmmss)
        
        obs = bufr.read_subset('SCRA',rep=True).squeeze()
        channum = bufr.read_subset('CHNM',rep=True).squeeze()
        #coldspacecorr = bufr.read_subset('CSTC',rep=True).squeeze()
        nchanls = len(obs)
        if nchanl != nchanls:
            raise ValueError('unexpected number of channels')
        lat1 = hdr1[9]; lon1 = hdr1[10]
        #lat2 = hdr1[11]; lon2 = hdr1[12]
        #if np.abs(lat2) <= 90 and np.abs(lon2) <= 360:
        #    lat = lat2; lon = lon2
        #else:
        lat = lat1; lon = lon1
        if lon > 180: lon -= 360
        latstr = '%8.4f' % quantize(lat,2)
        lonstr = '%9.4f' % quantize(lon,2)
        
        sat_id = int(hdr1[0])
        sensor_id = int(hdr1[1])
        #for ob,channel_number in zip(obs.filled(),channum):
        ob=obs[0] ; channel_number=channum[0]
        obid = "%3s %3s %8s %9s %4s %14s" %(sat_id,sensor_id,latstr,lonstr,int(channel_number),yyyymmddhhmmss)
            # skip ob if there is already a matching obid string
            # (don't want any duplicates in netcdf file)
        if obid not in obid_set:
                obid_set.add(obid)
                #var_dict['sat_id'][1][nob] = sat_id
                #var_dict['sensor_id'][1][nob] = sensor_id
                var_dict['lat'][1][nob] = lat
                var_dict['lon'][1][nob] = lon
                var_dict['yyyymmddhhmmss'][1][nob] = float(yyyymmddhhmmss)
                #var_dict['land_surface_height'][1][nob] = hdr1[13]
                #var_dict['field_of_view_number'][1][nob] = int(hdr1[2])
                #var_dict['sat_zenith_angle'][1][nob] = hdr2[0]
                #var_dict['solar_zenith_angle'][1][nob] = hdr2[1]
                #var_dict['solar_azimuth_angle'][1][nob] = hdr2[3]
                #var_dict['local_azimuth_angle'][1][nob] = hdr2[2]
                var_dict['tb'][1][nob] = ob
                var_dict['channel_number'][1][nob] = int(channel_number)
                nob += 1; nobtot += 1

                if nob == nobs_chunk:
                    # write nobs_chunk obs.
                    for vname in vnames:
                        vv,va = var_dict[vname]
                        vv[nob1:nob1+nobs_chunk] = va
                    nob = 0; nob1 = nobtot # reset counters
                    f.sync() # flush to disk.
        else:
                print('skipping duplicate ob with id %s' % obid)

# write remaining obs.
if nob != 0:
    for vname in vnames:
        vv,va = var_dict[vname]
        vv[nob1:nob1+nob] = va[0:nob]
    f.sync()

# check that total number of obs processed is the same as
# total number of obs written to file.
if nobtot != len(obsdim):
    raise ValueError('not all obs written to file! (wrote %s, processed %s)' % (len(obsdim),nobtot))
print('total number of observations written to netcdf file= %s' % nobtot)

# all done, close files.
f.close()
bufr.close()        
quit()        
        
