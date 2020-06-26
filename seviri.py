from __future__ import print_function
import ncepbufr
from netCDF4 import Dataset
import numpy as np
import sys, datetime
from utils import quantize


# input and output file names from command line args.
bufr_filename ='/home/vkvalappil/Data/modelWRF/input/2017122006/gdas.t00z.sevcsr.tm00.bufr_d' 
nc_filename ='/home/vkvalappil/Data/modelWRF/input/severi_06.nc' 

if bufr_filename == nc_filename:
    raise IOError('cannot overwrite input bufr file')
########################################################################################################
bufr = ncepbufr.open(bufr_filename)


# get number of channels from 1st message in bufr file.
nchanl = None
hdr_data=[] ; brt_data=[] ; ncld_data=[]

while bufr.advance() == 0:
    while bufr.load_subset() == 0:
        msg_type= bufr.msg_type ;  msg_date= bufr.msg_date
        #print(msg_type) ;  print(msg_date)
        print(bufr.msg_counter, bufr.msg_type, bufr.msg_date)
        
        if msg_type=='NC021043' :
           sky_flag='clearsky'          
           hdrsevi='SAID YEAR MNTH DAYS HOUR MINU SECO CLATH CLONH SAZA SOZA'
           nhdr=11 ; nchn=12 ; ncld=nchn ; nbrst=nchn                     
        elif msg_type=='NC021042' :
           sky_flag='allsky' 
           hdrsevi='SAID YEAR MNTH DAYS HOUR MINU SECO CLATH CLONH'
           nhdr=9 ; nchn=11 ; ncld=2 ;  nbrst=nchn*6 
        hdr1=bufr.read_subset(hdrsevi).squeeze()
        ncld=bufr.read_subset('NCLDMNT').squeeze()
        brt=bufr.read_subset('TMBRST').squeeze()
        hdr_data.append(hdr1)
        brt_data.append(brt)
        ncld_data.append(ncld)
        
        #break

bufr.rewind()

