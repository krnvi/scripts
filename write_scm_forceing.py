#/usr/bin/python

import os ; import sys; import numpy as np ; import netCDF4 as nf ; import datetime as dt ; from dateutil import rrule, tz ; import pandas as pd ; 
from scipy.interpolate import interp1d
##############################################################################################################################

main='/home/vkvalappil/Data/modelWRF/WRF_SCM/sounding_files/'
date='2017110100'

date_1=dt.datetime.strptime(date,'%Y%m%d%H') ; date_2=dt.datetime.strptime(date,'%Y%m%d%H')+dt.timedelta(days=2)
date_list=[x.strftime('%Y%m%d%H') for x in rrule.rrule(rrule.HOURLY,dtstart=date_1,until=date_2)][0::12]  ; 

req_hgts=np.array([0.0,100,200,300,400,500,1000,16000])
Data=[] ; Times_1=[] ;  Times_3=[]
for dte in  date_list[:]:
    
    file_name=main+'force_sounding_'+dte+'.csv'
    data=pd.read_csv(file_name)
    
    f_rso = interp1d(data.HGHT,data.u_wind,bounds_error=False) ; u_wind_new=f_rso(req_hgts)
    f_rso = interp1d(data.HGHT,data.v_wind,bounds_error=False) ; v_wind_new=f_rso(req_hgts)
    f_rso = interp1d(data.HGHT,data.W,bounds_error=False) ; w_wind_new=f_rso(req_hgts)
    
    data_new=pd.DataFrame({'HGHT':req_hgts,'u_wind':u_wind_new,'v_wind':v_wind_new,'W':w_wind_new})  
    Data.append(data_new)
    Times_2=np.chararray((1, 19))
    Times_2[0,0]=dte[0] ;  Times_2[0,1]=dte[1] ;  Times_2[0,2]=dte[2] ;  Times_2[0,3]=dte[3] ;
    Times_2[0,4]='-' ;  Times_2[0,5]=dte[4] ;  Times_2[0,6]=dte[5] ;  Times_2[0,7]='-' ;
    Times_2[0,8]=dte[6] ;  Times_2[0,9]=dte[7] ;  Times_2[0,10]='_' ;  Times_2[0,11]=dte[8] ;
    Times_2[0,12]=dte[9] ;  Times_2[0,13]=':' ;  Times_2[0,14]='0' ;  Times_2[0,15]='0' ;
    Times_2[0,16]=':' ;  Times_2[0,17]='0' ;  Times_2[0,18]='0'  ;   
      
    Times_3.append(dte[0:4]+'-'+dte[4:6]+'-'+dte[6:8]+'_'+dte[8:10]+':00:00')    
    Times_1.append(Times_2)    
    
Data_1=pd.concat(Data)    
z = Data_1.HGHT ; u_g = Data_1.u_wind ; v_g = Data_1.v_wind ; w = Data_1.W
Times_1=np.array(Times_1,dtype=object) ; z_mt=np.zeros(z.shape,dtype=float)     
Times_3=np.array(Times_3)

######################################################################################################################################

fnme=main+'forcing_file.nc'
if os.path.isfile(fnme):
        os.remove(fnme)
nc_ff=nf.Dataset(fnme,'w')#,format='NETCDF_')
nc_ff.setncattr('TITLE', 'AUXILIARY FORCING FOR SCM')
nc_ff.setncattr('START_DATE',Times_3[0])
nc_ff.setncattr('SIMULATION_START_DATE',Times_3[0])
nc_ff.setncattr('DX',3000.0)
nc_ff.setncattr('DY',3000.0)
nc_ff.setncattr('GRID_ID',1)
nc_ff.setncattr('PARENT_ID',0)
nc_ff.setncattr('I_PARENT_START',1)
nc_ff.setncattr('I_PARENT_START',1)
nc_ff.setncattr('DT',20.0)
nc_ff.setncattr('CEN_LAT',24.0)
nc_ff.setncattr('CEN_LON',54.0)
nc_ff.setncattr('TRUELAT1',24.0)
nc_ff.setncattr('TRUELAT2',24.0)
nc_ff.setncattr('MOAD_CEN_LAT',24.0)
nc_ff.setncattr('STAND_LON',54.0)
nc_ff.setncattr('GMT',0.0)
nc_ff.setncattr('JULYR',1.0)
nc_ff.setncattr('JULDAY',1.0)
nc_ff.setncattr('MAP_PROJ',0.0)
nc_ff.setncattr('MMINLU','USGS')
nc_ff.setncattr('ISWATER',16.0)
nc_ff.setncattr('ISICE',0.0)
nc_ff.setncattr('ISURBAN',0.0)
nc_ff.setncattr('ISOILWATER',0.0)


nc_ff.createDimension('Time',len(date_list)) ;  
nc_ff.createDimension('DateStrLen',19) ;  
nc_ff.createDimension('force_layers',req_hgts.shape[0]) ;  

Times=nc_ff.createVariable('Times',str,('Time',))
#Times=nc_ff.createVariable('Times',str,('Time','DateStrLen'))
Z_FORCE=nc_ff.createVariable('Z_FORCE',('f4'),('Time','force_layers'))
U_G=nc_ff.createVariable('U_G',('f4'),('Time','force_layers'))
V_G=nc_ff.createVariable('V_G',('f4'),('Time','force_layers'))
W_SUBS=nc_ff.createVariable('W_SUBS',('f4'),('Time','force_layers'))
TH_UPSTREAM_X=nc_ff.createVariable('TH_UPSTREAM_X',('f4'),('Time','force_layers'))
TH_UPSTREAM_Y=nc_ff.createVariable('TH_UPSTREAM_Y',('f4'),('Time','force_layers'))
QV_UPSTREAM_X=nc_ff.createVariable('QV_UPSTREAM_X',('f4'),('Time','force_layers'))
QV_UPSTREAM_Y=nc_ff.createVariable('QV_UPSTREAM_Y',('f4'),('Time','force_layers'))
U_UPSTREAM_X=nc_ff.createVariable('U_UPSTREAM_X',('f4'),('Time','force_layers'))
U_UPSTREAM_Y=nc_ff.createVariable('U_UPSTREAM_Y',('f4'),('Time','force_layers'))
V_UPSTREAM_X=nc_ff.createVariable('V_UPSTREAM_X',('f4'),('Time','force_layers'))
V_UPSTREAM_Y=nc_ff.createVariable('V_UPSTREAM_Y',('f4'),('Time','force_layers'))
Z_FORCE_TEND=nc_ff.createVariable('Z_FORCE_TEND',('f4'),('Time','force_layers'))
U_G_TEND=nc_ff.createVariable('U_G_TEND',('f4'),('Time','force_layers'))
V_G_TEND=nc_ff.createVariable('V_G_TEND',('f4'),('Time','force_layers'))
W_SUBS_TEND=nc_ff.createVariable('W_SUBS_TEND',('f4'),('Time','force_layers'))
TH_UPSTREAM_X_TEND=nc_ff.createVariable('TH_UPSTREAM_X_TEND',('f4'),('Time','force_layers'))
TH_UPSTREAM_Y_TEND=nc_ff.createVariable('TH_UPSTREAM_Y_TEND',('f4'),('Time','force_layers'))
QV_UPSTREAM_X_TEND=nc_ff.createVariable('QV_UPSTREAM_X_TEND',('f4'),('Time','force_layers'))
QV_UPSTREAM_Y_TEND=nc_ff.createVariable('QV_UPSTREAM_Y_TEND',('f4'),('Time','force_layers'))
U_UPSTREAM_X_TEND=nc_ff.createVariable('U_UPSTREAM_X_TEND',('f4'),('Time','force_layers'))
U_UPSTREAM_Y_TEND=nc_ff.createVariable('U_UPSTREAM_Y_TEND',('f4'),('Time','force_layers'))
V_UPSTREAM_X_TEND=nc_ff.createVariable('V_UPSTREAM_X_TEND',('f4'),('Time','force_layers'))
V_UPSTREAM_Y_TEND=nc_ff.createVariable('V_UPSTREAM_Y_TEND',('f4'),('Time','force_layers'))
TAU_X=nc_ff.createVariable('TAU_X',('f4'),('Time','force_layers'))
TAU_X_TEND=nc_ff.createVariable('TAU_X_TEND',('f4'),('Time','force_layers'))
TAU_Y=nc_ff.createVariable('TAU_Y',('f4'),('Time','force_layers'))
TAU_Y_TEND=nc_ff.createVariable('TAU_Y_TEND',('f4'),('Time','force_layers'))

Z_FORCE_att={}
Z_FORCE_att['FieldType']=104.0
Z_FORCE_att['MemoryOrder']="Z "
Z_FORCE_att['description']="height of forcing time series"
Z_FORCE_att['units']=""
Z_FORCE_att['stagger']=""
#Z_FORCE_att['_FillValue']=  nf.default_fillvals['f4']
Z_FORCE.setncatts(Z_FORCE_att)

Z_FORCE_TEND_att={}
Z_FORCE_TEND_att['FieldType']=104.0
Z_FORCE_TEND_att['MemoryOrder']="Z "
Z_FORCE_TEND_att['description']="tendency height of forcing time series"
Z_FORCE_TEND_att['units']=""
Z_FORCE_TEND_att['stagger']=""
#Z_FORCE_TEND_att['_FillValue']=-999.0 #nf.default_fillvals['f4']
Z_FORCE_TEND.setncatts(Z_FORCE_TEND_att)

U_G_att={}
U_G_att['FieldType']=104.0
U_G_att['MemoryOrder']="Z "
U_G_att['description']="x-component geostrophic wind"
U_G_att['units']="m s-1"
U_G_att['stagger']=""
#U_G_att['_FillValue']=nf.default_fillvals['f4']
U_G.setncatts(U_G_att)

U_G_TEND_att={}
U_G_TEND_att['FieldType']=104
U_G_TEND_att['MemoryOrder']="Z "
U_G_TEND_att['description']="tendency x-component geostrophic wind"
U_G_TEND_att['units']="m s-2"
U_G_TEND_att['stagger']=""
#U_G_TEND_att['_FillValue']=nf.default_fillvals['f4']
U_G_TEND.setncatts(U_G_TEND_att)

V_G_att={}
V_G_att['FieldType']=104.0
V_G_att['MemoryOrder']="Z "
V_G_att['description']="y-component geostrophic wind"
V_G_att['units']="m s-1"
V_G_att['stagger']=""
#V_G_att['_FillValue']=nf.default_fillvals['f4']
V_G.setncatts(V_G_att)

V_G_TEND_att={}
V_G_TEND_att['FieldType']=104.0
V_G_TEND_att['MemoryOrder']="Z "
V_G_TEND_att['description']="tendency y-component geostrophic wind"
V_G_TEND_att['units']="m s-2"
V_G_TEND_att['stagger']=""
#V_G_TEND_att['_FillValue']=nf.default_fillvals['f4']
V_G_TEND.setncatts(V_G_TEND_att)


W_SUBS_att={}
W_SUBS_att['FieldType']=104.0
W_SUBS_att['MemoryOrder']="Z "
W_SUBS_att['description']="large-scale vertical motion (subsidence)"
W_SUBS_att['units']="m s-1"
W_SUBS_att['stagger']=""
#W_SUBS_att['_FillValue']=nf.default_fillvals['f4']
W_SUBS.setncatts(W_SUBS_att)

W_SUBS_TEND_att={}
W_SUBS_TEND_att['FieldType']=104.0
W_SUBS_TEND_att['MemoryOrder']="Z "
W_SUBS_TEND_att['description']="tendency large-scale vertical motion (subsidence)"
W_SUBS_TEND_att['units']="m s-2"
W_SUBS_TEND_att['stagger']=""
W_SUBS_TEND.setncatts(W_SUBS_TEND_att)

TH_UPSTREAM_X_att={}
TH_UPSTREAM_X_att['FieldType']=104.0
TH_UPSTREAM_X_att['MemoryOrder']="Z "
TH_UPSTREAM_X_att['description']="upstream theta x-advection"
TH_UPSTREAM_X_att['units']="K s-1"
TH_UPSTREAM_X_att['stagger']=""
TH_UPSTREAM_X.setncatts(TH_UPSTREAM_X_att)

TH_UPSTREAM_X_TEND_att={}
TH_UPSTREAM_X_TEND_att['FieldType']=104.0
TH_UPSTREAM_X_TEND_att['MemoryOrder']="Z "
TH_UPSTREAM_X_TEND_att['description']="tendency upstream theta x-advection"
TH_UPSTREAM_X_TEND_att['units']="K s-2"
TH_UPSTREAM_X_TEND_att['stagger']=""
TH_UPSTREAM_X_TEND.setncatts(TH_UPSTREAM_X_TEND_att)

TH_UPSTREAM_Y_att={}
TH_UPSTREAM_Y_att['FieldType']=104.0
TH_UPSTREAM_Y_att['MemoryOrder']="Z "
TH_UPSTREAM_Y_att['description']="upstream theta y-advection"
TH_UPSTREAM_Y_att['units']="K s-1"
TH_UPSTREAM_Y_att['stagger']=""
TH_UPSTREAM_Y.setncatts(TH_UPSTREAM_Y_att)

TH_UPSTREAM_Y_TEND_att={}
TH_UPSTREAM_Y_TEND_att['FieldType']=104.0
TH_UPSTREAM_Y_TEND_att['MemoryOrder']="Z "
TH_UPSTREAM_Y_TEND_att['description']="tendency upstream theta y-advection"
TH_UPSTREAM_Y_TEND_att['units']="K s-2"
TH_UPSTREAM_Y_TEND_att['stagger']=""
TH_UPSTREAM_Y_TEND.setncatts(TH_UPSTREAM_Y_TEND_att)


QV_UPSTREAM_X_att={}
QV_UPSTREAM_X_att['FieldType']=104.0
QV_UPSTREAM_X_att['MemoryOrder']="Z "
QV_UPSTREAM_X_att['description']="upstream qv x-advection"
QV_UPSTREAM_X_att['units']="kg kg-1 s-1"
QV_UPSTREAM_X_att['stagger']=""
QV_UPSTREAM_X.setncatts(QV_UPSTREAM_X_att)


QV_UPSTREAM_X_TEND_att={}
QV_UPSTREAM_X_TEND_att['FieldType']=104.0
QV_UPSTREAM_X_TEND_att['MemoryOrder']="Z "
QV_UPSTREAM_X_TEND_att['description']="tendency upstream qv x-advection"
QV_UPSTREAM_X_TEND_att['units']="kg kg-1 s-2"
QV_UPSTREAM_X_TEND_att['stagger']=""
QV_UPSTREAM_X_TEND.setncatts(QV_UPSTREAM_X_TEND_att)

QV_UPSTREAM_Y_att={}
QV_UPSTREAM_Y_att['FieldType']=104.0
QV_UPSTREAM_Y_att['MemoryOrder']="Z "
QV_UPSTREAM_Y_att['description']="upstream qv y-advection"
QV_UPSTREAM_Y_att['units']="kg kg-1 s-1"
QV_UPSTREAM_Y_att['stagger']=""
QV_UPSTREAM_Y.setncatts(QV_UPSTREAM_Y_att)


QV_UPSTREAM_Y_TEND_att={}
QV_UPSTREAM_Y_TEND_att['FieldType']=104.0
QV_UPSTREAM_Y_TEND_att['MemoryOrder']="Z "
QV_UPSTREAM_Y_TEND_att['description']="tendency upstream qv y-advection"
QV_UPSTREAM_Y_TEND_att['units']="kg kg-1 s-2"
QV_UPSTREAM_Y_TEND_att['stagger']=""
QV_UPSTREAM_Y_TEND.setncatts(QV_UPSTREAM_Y_TEND_att)


U_UPSTREAM_X_att={}
U_UPSTREAM_X_att['FieldType']=104.0
U_UPSTREAM_X_att['MemoryOrder']="Z "
U_UPSTREAM_X_att['description']="upstream U x-advection"
U_UPSTREAM_X_att['units']="m s-3"
U_UPSTREAM_X_att['stagger']=""
U_UPSTREAM_X.setncatts(U_UPSTREAM_X_att)

U_UPSTREAM_X_TEND_att={}
U_UPSTREAM_X_TEND_att['FieldType']=104.0
U_UPSTREAM_X_TEND_att['MemoryOrder']="Z "
U_UPSTREAM_X_TEND_att['description']="tendency upstream U x-advection"
U_UPSTREAM_X_TEND_att['units']="m s-3"
U_UPSTREAM_X_TEND_att['stagger']=""
U_UPSTREAM_X_TEND.setncatts(U_UPSTREAM_X_TEND_att)

U_UPSTREAM_Y_att={}
U_UPSTREAM_Y_att['FieldType']=104.0
U_UPSTREAM_Y_att['MemoryOrder']="Z "
U_UPSTREAM_Y_att['description']="upstream U y-advection"
U_UPSTREAM_Y_att['units']="m s-3"
U_UPSTREAM_Y_att['stagger']=""
U_UPSTREAM_Y.setncatts(U_UPSTREAM_Y_att)

U_UPSTREAM_Y_TEND_att={}
U_UPSTREAM_Y_TEND_att['FieldType']=104.0
U_UPSTREAM_Y_TEND_att['MemoryOrder']="Z "
U_UPSTREAM_Y_TEND_att['description']="tendency upstream U y-advection"
U_UPSTREAM_Y_TEND_att['units']="m s-3"
U_UPSTREAM_Y_TEND_att['stagger']=""
U_UPSTREAM_Y_TEND.setncatts(U_UPSTREAM_Y_TEND_att)

V_UPSTREAM_X_att={}
V_UPSTREAM_X_att['FieldType']=104.0
V_UPSTREAM_X_att['MemoryOrder']="Z "
V_UPSTREAM_X_att['description']="upstream V x-advection"
V_UPSTREAM_X_att['units']="m s-3"
V_UPSTREAM_X_att['stagger']=""
V_UPSTREAM_X.setncatts(V_UPSTREAM_X_att)

V_UPSTREAM_X_TEND_att={}
V_UPSTREAM_X_TEND_att['FieldType']=104.0
V_UPSTREAM_X_TEND_att['MemoryOrder']="Z "
V_UPSTREAM_X_TEND_att['description']="tendency upstream V x-advection"
V_UPSTREAM_X_TEND_att['units']="m s-3"
V_UPSTREAM_X_TEND_att['stagger']=""
V_UPSTREAM_X_TEND.setncatts(V_UPSTREAM_X_TEND_att)

V_UPSTREAM_Y_att={}
V_UPSTREAM_Y_att['FieldType']=104.0
V_UPSTREAM_Y_att['MemoryOrder']="Z "
V_UPSTREAM_Y_att['description']="upstream V y-advection"
V_UPSTREAM_Y_att['units']="m s-3"
V_UPSTREAM_Y_att['stagger']=""
V_UPSTREAM_Y.setncatts(V_UPSTREAM_Y_att)

V_UPSTREAM_Y_TEND_att={}
V_UPSTREAM_Y_TEND_att['FieldType']=104.0
V_UPSTREAM_Y_TEND_att['MemoryOrder']="Z "
V_UPSTREAM_Y_TEND_att['description']="tendency upstream V y-advection"
V_UPSTREAM_Y_TEND_att['units']="m s-3"
V_UPSTREAM_Y_TEND_att['stagger']=""
V_UPSTREAM_Y_TEND.setncatts(V_UPSTREAM_Y_TEND_att)

TAU_X_att={}
TAU_X_att['FieldType']=104.0
TAU_X_att['MemoryOrder']="Z "
TAU_X_att['description']="X-direction advective timescale"
TAU_X_att['units']="s"
TAU_X_att['stagger']=""
TAU_X.setncatts(TAU_X_att)

TAU_X_TEND_att={}
TAU_X_TEND_att['FieldType']=104.0
TAU_X_TEND_att['MemoryOrder']="Z "
TAU_X_TEND_att['description']="tendency X-direction advective timescale"
TAU_X_TEND_att['units']=""
TAU_X_TEND_att['stagger']=""
TAU_X_TEND.setncatts(TAU_X_TEND_att)

TAU_Y_att={}
TAU_Y_att['FieldType']=104.0
TAU_Y_att['MemoryOrder']="Z "
TAU_Y_att['description']="Y-direction advective timescale"
TAU_Y_att['units']="s"
TAU_Y_att['stagger']=""
TAU_Y.setncatts(TAU_Y_att)

TAU_Y_TEND_att={}
TAU_Y_TEND_att['FieldType']=104.0
TAU_Y_TEND_att['MemoryOrder']="Z "
TAU_Y_TEND_att['description']="tendency Y-direction advective timescale"
TAU_Y_TEND_att['units']=""
TAU_Y_TEND_att['stagger']=""
TAU_Y_TEND.setncatts(TAU_Y_TEND_att)

Z_FORCE[:]=np.array(z,dtype=float)
#Times[:]=Times_1
Times[:]=Times_3
U_G[:]=np.array(u_g,dtype=float) ; V_G[:]=np.array(v_g,dtype=float)
W_SUBS[:]=np.array(w,dtype=float) ;
TH_UPSTREAM_X=z_mt ; TH_UPSTREAM_Y=z_mt
QV_UPSTREAM_X=z_mt ; QV_UPSTREAM_Y=z_mt
U_UPSTREAM_X=z_mt ; V_UPSTREAM_Y=z_mt
Z_FORCE_TEND=z_mt ; U_G_TEND=z_mt
V_G_TEND=z_mt ; W_SUBS_TEND=z_mt
TH_UPSTREAM_X_TEND=z_mt ; TH_UPSTREAM_Y_TEND=z_mt
QV_UPSTREAM_X_TEND=z_mt ; QV_UPSTREAM_Y_TEND=z_mt
U_UPSTREAM_X_TEND=z_mt ; U_UPSTREAM_Y_TEND=z_mt
V_UPSTREAM_X_TEND=z_mt ; V_UPSTREAM_Y_TEND=z_mt
TAU_X=z_mt ; TAU_X_TEND=z_mt
TAU_Y=z_mt ; TAU_Y_TEND=z_mt
nc_ff.close()

##################################################################################################################################################


















