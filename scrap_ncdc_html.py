
import pandas as pd ; 
import re
import urllib2 as urll2 ; import numpy as np ; import metpy.calc as mcalc ; from metpy.units import units

url='https://www.ncdc.noaa.gov/orders/isd/41194099999-2017-12_4887067749712dat.html'
#url='https://www.ncdc.noaa.gov/orders/isd/41194099999-2018-01_4887067749712dat.html'
web=urll2.Request(url) ; htm = urll2.urlopen(web) ; cont=htm.read()  

data=pd.read_html(cont)[1]
data.columns=data.iloc[0,:]

data_1=data.drop(['USAF','WBAN','CLG','SKC','GUS','L','M','H','MW','AW','W','STP','MAX','MIN','PCP01','PCP06','PCP24','PCPXX','SD'],axis=1)
data_1=data_1.drop([0,1],axis=0)
data_1['YR--MODAHRMN']=data_1['YR--MODAHRMN'].apply(pd.to_datetime, errors='ignore',format='%Y%m%d%H%M')
data_1.iloc[:,1:]=data_1.iloc[:,1:].apply(pd.to_numeric,errors='coerce') 


data_1['TEMP']=(data_1['TEMP']-32)/1.8 ; data_1['DEWP']=(data_1['DEWP']-32)/1.8 ; 


tmp=np.array(data_1['TEMP']+273.15)*units('K') ; dtmp=np.array(data_1['DEWP']+273.15)*units('K')

press=np.array(data_1['SLP'])*units('millibar')
rh=mcalc.relative_humidity_from_dewpoint(tmp,dtmp) 
mrio=mcalc.mixing_ratio_from_relative_humidity(rh,tmp,press)

data_1['RH']=rh*100 ; data_1['mrio']=mrio


data_2=pd.concat([data_1['YR--MODAHRMN'],data_1['SLP'],data_1['TEMP'],data_1['DEWP'],data_1['RH'],data_1['DIR'],data_1['SPD'],data_1['VSB'],data_1['mrio']],axis=1)
data_2.columns=['TIME','ALTM','TMP','DEW','RH','DIR','SPD','VIS','mrio']
stn=pd.DataFrame(['OMDB']*data_2.shape[0])
data_2.insert(0,'STN',stn)

#data_2.to_csv('/home/vkvalappil/Data/masdar_station_data/wyoming/ncdc_dubai_jan2018.csv') 

data_2.index=data_2.TIME

date_list=pd.DataFrame(data_2.index.strftime('%Y%m%d')).drop_duplicates()

for dte in date_list.iloc[:,0]:
    data_3=data_2[dte]
    data_3=data_3.reset_index(drop=True)
    data_3.to_csv('/home/vkvalappil/Data/masdar_station_data/wyoming/ncdc/Dubai_surf_mr'+dte+'.csv') 


