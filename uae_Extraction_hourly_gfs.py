import sys ; import numpy as np ; from scipy.interpolate import interp2d ; import datetime as dt ;  
import time ; import pygrib as pg ;from dateutil import rrule, tz
import pandas as pd ; import metpy.calc as mcalc ; from metpy.units import units

start_time = time.time()
#### path and input files are defined
main='/home/vkvalappil/Data/modelWRF/' ; scripts=main+'/scripts/' ; output='/home/vkvalappil/Data/oppModel/ARW/output/'
date=str(sys.argv[1]) ; fcs_leaddays=3 ; 
fcs_st_date=dt.datetime.strptime(date,'%Y%m%d%H') ; fcs_ed_date=(dt.datetime.strptime(date,'%Y%m%d%H')+dt.timedelta(days=fcs_leaddays));


lstFle='/home/vkvalappil/Data/oppModel/scripts/master_uae2.csv'
modFile='/home/vkvalappil/Data/modelWRF/input/gfs.'+date+'/gfs.'+date+'.grb2'
outFile=output+'gfs_hourly'+date+'.csv' ; 
#/home/Data/WRF-NMM18/vizsouthafrica/cnvgrib-1.4.0/./cnvgrib -nv -g12 inp out

grb_f=pg.open(modFile) ; 
lst_f=np.genfromtxt(lstFle,delimiter=',', dtype="S") ; lst_f=lst_f[lst_f[:,0].argsort(kind='mergesort')]  
points=lst_f[:,1:3].astype(float) ; noloc=points.shape[0] 

tmp=(np.array(grb_f.select(name="2 metre temperature"))) 
ugrd=(np.array(grb_f.select(name="10 metre U wind component"))) 
vgrd=(np.array(grb_f.select(name="10 metre V wind component"))) 
#LCDClcl=(np.array(grb_f.select(name="Low cloud cover"))) 
#MCDCmcl=(np.array(grb_f.select(name="Medium cloud cover"))) 
#HCDChcl=(np.array(grb_f.select(name="High cloud cover"))) 
RH=(np.array(grb_f.select(name="Relative humidity",typeOfLevel='heightAboveGround'))) 
dtmp=(np.array(grb_f.select(name="2 metre dewpoint temperature",))) 
PS=(np.array(grb_f.select(name="Surface pressure"))) 
SH=(np.array(grb_f.select(name="Specific humidity",typeOfLevel='heightAboveGround'))) 
#TSOIL=(np.array(grb_f.select(name="Soil Temperature"))) 
#SOILW=(np.array(grb_f.select(name=""))) 
#snowcsfc=(np.array(grb_f.select(name=""))) 
#APCPsfc=(np.array(grb_f.select(name="Total Precipitation"))) 
grb_f.close()

#lat, lon=tmp[1].latlons()
lat=tmp[0].data()[1] ; lon=tmp[0].data()[2] ; 
lat=np.vstack(lat[:,0]) ; lon=np.vstack(lon[0,:]) ; #lat=lat[::-1] 

stim=0 ; etim=12 ; tint=6
day_list=np.array(range(0,fcs_leaddays)) ; day_seq=np.vstack(np.tile(np.repeat(day_list,4),noloc)) 
#####  date day seq and time are arranged in req format #######################
tim_st=dt.datetime.strptime(date,'%Y%m%d%H') ; tim_ed=dt.datetime.strptime(date,'%Y%m%d%H')+dt.timedelta(days=1)
tim_list=[x.strftime('%H') for x in rrule.rrule(rrule.HOURLY,dtstart=tim_st,until=tim_ed)][0:-1][0::6]

#tim_list=(np.array(range(06,24,1) + range(00,06,1))) ; 

tim_list=[str(k).rjust(2, '0')for k in tim_list] 

tim_seq=np.vstack(np.tile(tim_list,3*noloc)) ;
c_date=np.vstack(np.repeat(fcs_st_date.strftime('%Y-%m-%d'),noloc*12))

from_zone = tz.gettz('UTC') ; to_zone = tz.gettz('Asia/Dubai') ; 
date_list=[(x.replace(tzinfo=from_zone)).astimezone(to_zone).strftime('%Y-%m-%d %H:%M:%S') for x in rrule.rrule(rrule.HOURLY,dtstart=fcs_st_date,until=fcs_ed_date)]
date_list=date_list[0::tint] ; date_list=np.delete(date_list,-1) ; f_date=np.vstack(np.tile(date_list,noloc)) ; 
tid=np.vstack(np.repeat(lst_f[:,0],12))  ; pre_mat=np.concatenate([tid,c_date,tim_seq,f_date,day_seq],axis=1)

temp=np.empty((0,noloc)) ; rhum=np.empty((0,noloc)) ; shum=np.empty((0,noloc)) ; cldfrn=np.empty((0,noloc)) ; 
mrio=np.empty((0,noloc)) ; spsf=np.empty((0,noloc)) ; uwnd=np.empty((0,noloc)) ; vwnd=np.empty((0,noloc)) ; dtemp=np.empty((0,noloc))

start_time = time.time()

for i in range(stim,etim):
    tmp1=tmp[i].data()
    tmp_f=interp2d(lon,lat, tmp1[0]-273.15,kind='linear',copy=False,bounds_error=True ) ; 
    #temp=np.concatenate([temp,(np.array([tmp_f(ii[1],ii[0]) for ii in points[:]])).T],axis=0)

    ugrd1=ugrd[i].data()    
    ugrd_f=interp2d(lon,lat, ugrd1[0],kind='linear',copy=False,bounds_error=True ) ; 
    
    vgrd1=vgrd[i].data()    
    vgrd_f=interp2d(lon,lat, vgrd1[0],kind='linear',copy=False,bounds_error=True ) ; 
    
    rhum1=RH[i].data() ;
    rhum_f=interp2d(lon,lat, rhum1[0],kind='linear',copy=False,bounds_error=True ) ;
    
    dtemp1=dtmp[i].data()
    dtmp_f=interp2d(lon,lat, dtemp1[0]-273.15,kind='linear',copy=False,bounds_error=True ) ;  
    
    SH1=SH[i].data()
    shum_f=interp2d(lon,lat,SH1[0],kind='linear',copy=False,bounds_error=True ) ;  
    
    PS1=PS[i].data()    
    spsf_f=interp2d(lon,lat,PS1[0],kind='linear',copy=False,bounds_error=True) ;   

    temp=np.concatenate([temp,(np.array([tmp_f(ii[1],ii[0]) for ii in points[:]])).T],axis=0)
    dtemp=np.concatenate([dtemp,(np.array([dtmp_f(ii[1],ii[0]) for ii in points[:]])).T],axis=0)
    cldfrn=np.concatenate([cldfrn,(np.array([dtmp_f(ii[1],ii[0]) for ii in points[:]])).T],axis=0)  
    rhum=np.concatenate([rhum,(np.array([rhum_f(ii[1],ii[0]) for ii in points[:]])).T],axis=0)  
    shum=np.concatenate([shum,(np.array([shum_f(ii[1],ii[0]) for ii in points[:]])).T],axis=0)      
    spsf=np.concatenate([spsf,(np.array([spsf_f(ii[1],ii[0]) for ii in points[:]])).T],axis=0)      
    mrio=np.concatenate([mrio,(np.array([rhum_f(ii[1],ii[0]) for ii in points[:]])).T],axis=0)      
    uwnd=np.concatenate([uwnd,(np.array([ugrd_f(ii[1],ii[0]) for ii in points[:]])).T],axis=0)
    vwnd=np.concatenate([vwnd,(np.array([vgrd_f(ii[1],ii[0]) for ii in points[:]])).T],axis=0)  

print("--- %s seconds ---" % (time.time() - start_time)) 
 
wdir=np.round(270-(np.arctan2(vwnd,uwnd))*(180/3.14)).astype(int)   ; wdir[wdir>360]=wdir[wdir>360]-360 
wspd=np.round(np.sqrt(np.square(uwnd)+np.square(vwnd))*3.6).astype(int) ; 

#### data is arranged in req format

temp=np.round(temp).astype(int) ; temp=np.vstack(temp.flatten('F'))
#temp=np.round(temp.reshape(fcs_leaddays,8,noloc)).astype(int) ;
dtemp=np.round(dtemp).astype(int) ; dtemp=np.vstack(dtemp.flatten('F'))
rhum=np.round(rhum).astype(int)   ; rhum=np.vstack(rhum.flatten('F'))
#rhum=np.round(rhum.reshape(fcs_leaddays,tint,noloc)).astype(int) ;
shum=(shum).astype(float)       ; shum=np.vstack(shum.flatten('F'))
spsf=np.round(spsf).astype(int) ; spsf=np.vstack(spsf.flatten('F'))
mrio=(mrio).astype(float)       ; mrio=np.vstack(mrio.flatten('F'))
wdir=np.round(wdir).astype(int) ; wdir=np.vstack(wdir.flatten('F'))
wspd=np.round(wspd).astype(int) ; wspd=np.vstack(wspd.flatten('F'))

cldfrn=(cldfrn.reshape(12,1,noloc)) ;  
meancld=cldfrn.mean(axis=1) ; meancld[meancld>8]=8 ; meancld[(meancld>0.1) & (meancld<0.6)]=1  ; 
meancld=np.round(meancld).astype(int) ; meancld=np.vstack(meancld.flatten('F'))


header='TEHSILID,DATE,TIME(GMT),localTime,DAY_SEQUENCE,TEMP,DTEMP,RH,CLOUD,SPHUM,MXRATIO,SURFPRES,WSPD,WDIR'
header=np.vstack(np.array(header.split(","))).T

fin_mat1=np.concatenate([pre_mat,temp,dtemp,rhum,meancld,shum,mrio,spsf,wdir,wspd],axis=1)
fin_mat=pd.DataFrame(fin_mat1,columns=header[0,:].T)
fin_mat.iloc[:,5:]=fin_mat.iloc[:,5:].apply(pd.to_numeric,errors='coerce')

m_tmp=np.array(fin_mat['TEMP']+273.15).astype(float)*units('K')
m_rh=np.array(fin_mat['RH']/100.0) 
press=np.array(fin_mat['SURFPRES'])*units('pascal')
mrio=mcalc.mixing_ratio_from_relative_humidity(m_rh,m_tmp,press)

fin_mat['MXRATIO']=mrio.m

fin_mat.to_csv(outFile,index=False)


print("--- %s seconds ---" % (time.time() - start_time))
quit() 




